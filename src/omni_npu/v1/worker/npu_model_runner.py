# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Optional, Union, Any, cast

import torch
import torch.nn as nn
from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
)
from vllm.distributed.parallel_state import get_pp_group, prepare_communication_buffer_for_model
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    KVCacheConfig,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.logger import logger
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.spec_decode.eagle import EagleProposer

from omni_npu.v1.sample.sampler import NPUSamplerV1
from omni_npu.v1.sample.rejection_sampler import NPURejectionSampler
from omni_npu.compilation.acl_graph import ACLGraphWrapper, set_graph_params

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


@contextmanager
def switch_torch_device():
    origin_cuda = torch.cuda
    torch.cuda = torch.npu
    try:
        yield
    finally:
        torch.cuda = origin_cuda


class NPUModelRunner(GPUModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        with switch_torch_device():
            super().__init__(vllm_config, device)

        # NOTE:(runze) query_lens and seq_lens arguments need to be int64 in FIA op,
        # otherwise an implicit conversion would happen which might hurt performance.
        self.query_start_loc = self._make_buffer(self.max_num_reqs + 1,
                                                 dtype=torch.int64)
        self.seq_lens = self._make_buffer(self.max_num_reqs,
                                          dtype=torch.int64)

        # sampled_token_ids is int32 in npu, sampled_token_ids_pinned_cpu should
        # be same dtype to synchronize.
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_model_len, 1),
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory)

        # FIXME(runze): reusing VLLM's sampler fails, this sampler class is from omni_infer.
        # need to check why and try to remove it.
        self.sampler = NPUSamplerV1()

        if self.speculative_config and get_pp_group().is_last_rank:
            self.rejection_sampler = NPURejectionSampler()

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        kv_caches: dict[str, torch.Tensor] = {}
        has_attn, has_mamba = False, False
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                num_blocks = (raw_tensor.numel() //
                              kv_cache_spec.page_size_bytes)
                if isinstance(kv_cache_spec, AttentionSpec):
                    has_attn = True
                    kv_cache_tensors = attn_backend.reshape_kv_cache(
                        raw_tensor,
                        num_blocks,
                        kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        dtype=kv_cache_spec.dtype,
                    )
                    kv_caches[layer_name] = kv_cache_tensors
                elif isinstance(kv_cache_spec, MambaSpec):
                    raise NotImplementedError("Mamba functionality is in progress.")
                else:
                    raise NotImplementedError

        if has_attn and has_mamba:
            self._update_hybrid_attention_mamba_layout(kv_caches)

        return kv_caches

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        if self.vllm_config.model_config.use_mla and hasattr(self.vllm_config.model_config.hf_config, "index_topk"):
            indexer_head_size = self.vllm_config.model_config.hf_config.index_head_dim
            kv_cache_spec: dict[str, KVCacheSpec] = {}
            layer_type = cast(type[Any], AttentionLayerBase)
            attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
            for layer_name, attn_module in attn_layers.items():
                kv_cache_spec[layer_name] = MLAAttentionSpec(
                    block_size=self.vllm_config.cache_config.block_size,
                    num_kv_heads=1,
                    head_size=attn_module.head_size + indexer_head_size,
                    dtype=kv_cache_dtype_str_to_dtype(self.vllm_config.cache_config.cache_dtype, self.vllm_config.model_config),
                    cache_dtype_str=self.vllm_config.cache_config.cache_dtype
                )
            return kv_cache_spec
        else:
            return super().get_kv_cache_spec()

    # Note: used for model runner override.
    def _init_device_properties(self) -> None:
        """Initialize attributes from torch.npu.get_device_properties
        """
        self.device_properties = torch.npu.get_device_properties(self.device)
        self.num_sms = self.device_properties.multi_processor_count

    # Note: used for model runner override.
    def _sync_device(self) -> None:
        torch.npu.synchronize()

    def load_model(self, eep_scale_up: bool = False) -> None:
        """
        Args:
            eep_scale_up: the model loading is for elastic EP scale up.
        """
        super().load_model(eep_scale_up)

        if hasattr(self, "drafter") and isinstance(self.drafter, EagleProposer):
            prepare_communication_buffer_for_model(self.drafter.model)

        # wrap the model with full graph wrapper if needed.
        logger.debug(f"<<< {self.compilation_config.cudagraph_mode.has_full_cudagraphs()=}")
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.update_stream: torch.npu.Stream = torch.npu.Stream()
            set_graph_params(self.compilation_config.cudagraph_capture_sizes)
            self.model = ACLGraphWrapper(self.model.runnable,
                                         self.vllm_config,
                                         runtime_mode=CUDAGraphMode.FULL)
            logger.debug("<<< Wrapped original model with ACLGraphWrapper")

    def capture_model(self) -> int:
        logger.debug("<<< Capturing model in npu_model_runner")
        with switch_torch_device():
            super().capture_model()

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, AsyncModelRunnerOutput, IntermediateTensors]:
        with (switch_torch_device()
              if self.use_async_scheduling else nullcontext()):
            return super().execute_model(scheduler_output,
                                         intermediate_tensors)

    @torch.inference_mode
    def sample_tokens(self, grammar_output):
        with switch_torch_device():
            return super().sample_tokens(grammar_output)

    def get_model(self) -> nn.Module:
        # get raw model out of the aclgraph wrapper.
        if isinstance(self.model, ACLGraphWrapper):
            return self.model.unwrap()
        return self.model
