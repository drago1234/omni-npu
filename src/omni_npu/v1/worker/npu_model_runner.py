# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
from contextlib import contextmanager
import torch
from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
)
from vllm.distributed.parallel_state import get_pp_group
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    MambaSpec,
)
from vllm.logger import logger
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from omni_npu.v1.sample.sampler import NPUSamplerV1
from omni_npu.v1.sample.rejection_sampler import NPURejectionSampler
from omni_npu.compilation.acl_graph import ACLGraphWrapper, set_graph_params


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

        # FIXME(runze): reusing VLLM's sampler fails, this sampler class is from omni_infer.
        # need to check why and try to remove it.
        self.sampler = NPUSamplerV1()

        if self.speculative_config and get_pp_group().is_last_rank:
            self.rejection_sampler = NPURejectionSampler()

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
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

    # Note: used for model runner override.
    def _init_device_properties(self) -> None:
        """Initialize attributes from torch.npu.get_device_properties
        """
        self.device_properties = torch.npu.get_device_properties(self.device)
        self.num_sms = self.device_properties.multi_processor_count

    # Note: used for model runner override.
    def _sync_device(self) -> None:
        torch.npu.synchronize()

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        # TODO(tronzhang): error with event synchronize...
        return sampled_token_ids.tolist()


    def load_model(self, eep_scale_up: bool = False) -> None:
        """
        Args:
            eep_scale_up: the model loading is for elastic EP scale up.
        """
        super().load_model(eep_scale_up)

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