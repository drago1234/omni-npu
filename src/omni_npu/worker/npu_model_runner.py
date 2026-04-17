# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union, Any, cast, TypeAlias
import os

import torch
import numpy as np
import torch.nn as nn
from dataclasses import replace
from functools import wraps

from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    set_current_vllm_config,
)
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import (
    get_pp_group,
    prepare_communication_buffer_for_model,
)
from vllm.v1.attention.backend import (
    AttentionMetadata,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype, get_dtype_size
from vllm.forward_context import BatchDescriptor, set_forward_context, get_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.models.utils import extract_layer_index

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    KVCacheConfig,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.model_executor.models.interfaces import supports_mm_encoder_only
from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.ubatch_utils import UBatchSlices
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.compilation.cuda_graph import CUDAGraphStat

from omni_npu.worker.npu_mem_pool import NpuMemAllocator
from omni_npu.sample.sampler import NPUSamplerV1
from omni_npu.sample.rejection_sampler import NPURejectionSampler
from omni_npu.compilation.acl_graph import ACLGraphWrapper, set_graph_params
from omni_npu.plugin_decorators import (
    init_config_decorator,
    prepare_inputs_decorator,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# list when ubatching is enabled
PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict


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

        # is_mm_prefix_lm is used in _build_attention_metadata
        self.is_mm_prefix_lm = self.model_config.is_mm_prefix_lm

        # enable mtp acl graph mode
        if self.speculative_config and isinstance(self.drafter, EagleProposer):
            if self.compilation_config.mode == CompilationMode.VLLM_COMPILE:
                self.drafter.use_cuda_graph = self.compilation_config.cudagraph_mode.has_mode(CUDAGraphMode.FULL)
                self.drafter.batch_desc = None
                self.drafter.target_model_cuda_graph_mode = None

        # Overwrite num_accepted_tokens from GPUModelRunner to make it int32
        self.num_accepted_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        self.num_prompt_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )

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
            self.rejection_sampler = NPURejectionSampler(self.sampler)

        if vllm_config.additional_config is not None:
            from omni_npu.compilation.npugraph_ex_config import init_aclgraph_config
            init_aclgraph_config(vllm_config)
            self.use_rejection_sampler = vllm_config.additional_config.get("use_rejection_sampler", False)
            self.use_penalty = vllm_config.additional_config.get("use_penalty", False)
            self.total_step = vllm_config.additional_config.get("multi_step", 1)
            self.combine_block = vllm_config.additional_config.get("combine_block", 1)
            self.use_process_before_sample = vllm_config.additional_config.get("use_process_before_sample", False)
        else:
            self.use_rejection_sampler = False
            self.use_penalty = False
            self.total_step = 1
            self.combine_block = 1
            self.use_process_before_sample = False
        self.use_spec_decode = False
        num_tokens_per_reqs_decode = 1 if not self.use_spec_decode else (1 + self.speculative_config.num_speculative_tokens)
        self.block_size = vllm_config.cache_config.block_size
        self.max_num_blocks_per_req = cdiv(self.model_config.max_model_len,
                                           self.block_size*self.combine_block)*self.combine_block
        self.graph_block_tables = np.zeros(
            (self.max_num_reqs * num_tokens_per_reqs_decode,
             self.max_num_blocks_per_req),
            dtype=np.int32)
        val = getattr(self.model_config.hf_text_config, "router_sliding_window", 0)
        if isinstance(val, (int, float)):
            self.router_sliding_window = val
        else:
            self.router_sliding_window = 0
        if self.router_sliding_window > 0:
            self.req_cache_map = {self.max_num_reqs + 1: 0}
            self.cache_slot_id = torch.zeros(self.max_num_reqs,
                                    dtype=torch.long, device=self.device)

        self.batch_execution_and_padding_state: tuple[
            CUDAGraphMode,
            BatchDescriptor,
            torch.Tensor | None,
        ] | None = None

        self._is_mm_encoder_only = False


    def _build_conv_context(self, dummy:bool = False):
        forward_context = get_forward_context()
        if not dummy:
            keys_to_remove = [k for k in self.req_cache_map if k not in self.input_batch.req_ids]
            for k in keys_to_remove:
                del self.req_cache_map[k]
            for idx, req_id in enumerate(self.input_batch.req_ids):
                if req_id in self.req_cache_map:
                    cache_id = self.req_cache_map[req_id]
                    self.cache_slot_id[idx] = cache_id
                else:
                    self.cache_slot_id[idx] = 0
                self.req_cache_map[req_id] = idx + 1
            self.cache_slot_id[self.input_batch.num_reqs:] = 0
        forward_context.cache_slot_id = self.cache_slot_id

    def _model_forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> Any:
        """Helper method to call the model forward pass.

        This method can be overridden by subclasses for model execution.
        Motivation: We can inspect only this method versus
        the whole execute_model, which has additional logic.

        Args:
            input_ids: Input token IDs
            positions: Token positions
            intermediate_tensors: Tensors from previous pipeline stages
            inputs_embeds: Input embeddings (alternative to input_ids)
            **model_kwargs: Additional model arguments

        Returns:
            Model output tensor
        """
        if self.router_sliding_window > 1:
            self._build_conv_context()
        forward_context = get_forward_context()
        forward_context.capturing = False
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        kv_caches: dict[str, torch.Tensor] = {}
        has_tensor, has_tuple = False, False
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0, \
                    f"{kv_cache_spec=}, {raw_tensor.numel()=}, {kv_cache_spec.page_size_bytes=}"
                num_blocks = (raw_tensor.numel() //
                              kv_cache_spec.page_size_bytes)
                kwargs = {}
                kv_cache_tensors = attn_backend.reshape_kv_cache(
                    raw_tensor,
                    num_blocks,
                    kv_cache_spec,
                    **kwargs,
                )
                if isinstance(kv_cache_tensors, torch.Tensor) and kv_cache_tensors.is_contiguous():
                    has_tensor = True
                elif isinstance(kv_cache_tensors, tuple) and len(kv_cache_tensors) > 1:
                    has_tuple = True
                else:
                    raise RuntimeError(
                        f"Invalid case! Cache shouldn't be non-contiguous Tensor or single-element tuple."
                    )
                kv_caches[layer_name] = kv_cache_tensors

        if has_tensor and has_tuple:
            self._update_hybrid_attention_mamba_layout(kv_caches)

        return kv_caches

    def _update_hybrid_attention_mamba_layout(
        self, kv_caches: dict[str, Union[torch.Tensor, tuple[torch.Tensor, ...]]]
    ) -> None:
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                kv_cache = kv_caches[layer_name]
                if (
                    isinstance(kv_cache_spec, AttentionSpec)
                    and isinstance(kv_cache, torch.Tensor)
                    and kv_cache.shape[0] == 2
                ):
                    assert kv_cache.shape[1] != 2, (
                        "Fail to determine whether the layout is "
                        "(2, num_blocks, ...) or (num_blocks, 2, ...) for "
                        f"a tensor of shape {kv_cache.shape}"
                    )
                    hidden_size = kv_cache.shape[2:].numel()
                    kv_cache.as_strided_(
                        size=kv_cache.shape,
                        stride=(hidden_size, 2 * hidden_size, *kv_cache.stride()[2:]),
                    )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        is_pangu_hybrid = 'pangu_v2_hybrid' in os.getenv("OMNI_NPU_PATCHES_DIR", "")
        if not is_pangu_hybrid and self.vllm_config.model_config.use_mla and hasattr(self.vllm_config.model_config.hf_config, "index_topk"):
            indexer_head_size = self.vllm_config.model_config.hf_config.index_head_dim
            kv_cache_spec: dict[str, KVCacheSpec] = {}
            layer_type = cast(type[Any], AttentionLayerBase)
            attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
            for layer_name, attn_module in attn_layers.items():
                config = self.vllm_config.model_config.hf_config
                is_dsa = not hasattr(config, "dsa_layers") or extract_layer_index(layer_name) in config.dsa_layers
                head_size = (attn_module.head_size if hasattr(attn_module, 'head_size') else attn_module.head_dim) + \
                    (indexer_head_size if is_dsa else 0)
                if is_dsa and self.vllm_config.cache_config.cache_dtype in ["hif8_ds_mla"]:
                    # In the "HiF8 with scale" format, each token's KV cache is 656 Bytes
                    # reference vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py
                    kv_cache_spec[layer_name] = MLAAttentionSpec(
                        block_size=self.vllm_config.cache_config.block_size,
                        num_kv_heads=1,
                        head_size=656 + indexer_head_size + 4, # 4 bytes for one fp32 scale
                        dtype=kv_cache_dtype_str_to_dtype(self.vllm_config.cache_config.cache_dtype, self.vllm_config.model_config),
                        cache_dtype_str=self.vllm_config.cache_config.cache_dtype,
                    )
                elif not getattr(attn_module, 'sink_len', 0):
                    if int(os.getenv("ENABLE_OMNI_CACHE", "0")) and self.vllm_config.kv_transfer_config.kv_role == "kv_consumer":
                        head_size = indexer_head_size
                    # hif8_ds_mla kv quantization is only applied to DSA layers
                    if self.vllm_config.cache_config.cache_dtype in ["hif8_ds_mla"] and not is_dsa:
                        kv_dtype = kv_cache_dtype_str_to_dtype("auto", self.vllm_config.model_config)
                        kv_dtype_str = "auto"
                    else:  # keep original specified dtype
                        kv_dtype = kv_cache_dtype_str_to_dtype(self.vllm_config.cache_config.cache_dtype, self.vllm_config.model_config),
                        kv_dtype_str = self.vllm_config.cache_config.cache_dtype
                    kv_cache_spec[layer_name] = MLAAttentionSpec(
                        block_size=self.vllm_config.cache_config.block_size,
                        num_kv_heads=1,
                        head_size=head_size,
                        dtype=kv_dtype,
                        cache_dtype_str=kv_dtype_str,
                    )
                else:
                    from vllm.v1.kv_cache_interface import SinkMLAAttentionSpec
                    kv_cache_spec[layer_name] = SinkMLAAttentionSpec(
                        block_size=self.vllm_config.cache_config.block_size,
                        num_kv_heads=1,
                        head_size=head_size,
                        dtype=kv_cache_dtype_str_to_dtype(self.vllm_config.cache_config.cache_dtype, self.vllm_config.model_config),
                        cache_dtype_str=self.vllm_config.cache_config.cache_dtype,
                        sink_len=attn_module.sink_len,
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

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = True,
        force_eager: bool = False,
        # For cudagraph capture TODO(lucas): Refactor how we capture cudagraphs (will
        # be improved in model runner v2)
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        num_encoder_reqs: int = 0,
    ) -> tuple[
        CUDAGraphMode,
        BatchDescriptor,
        bool,
        torch.Tensor | None,
        CUDAGraphStat | None,
    ]:
        uniform_decode = self._is_uniform_decode(
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            uniform_decode_query_len=self.uniform_decode_query_len,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            force_uniform_decode=force_uniform_decode,
        )
        # Encoder-decoder models only support CG for decoder_step > 0 (no enc_output
        # is present). Also, chunked-prefill is disabled, so batch are uniform.
        has_encoder_output = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )

        has_lora = (
            len(self.input_batch.lora_id_to_lora_request) > 0
            if force_has_lora is None
            else force_has_lora
        )

        num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens)
        dispatch_cudagraph = (
            lambda num_tokens, disable_full: self.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens,
                has_lora=has_lora,
                uniform_decode=uniform_decode,
                disable_full=disable_full,
            )
            if not force_eager
            else (CUDAGraphMode.NONE, BatchDescriptor(num_tokens_padded))
        )

        cudagraph_mode, batch_descriptor = dispatch_cudagraph(
            num_tokens_padded, use_cascade_attn or has_encoder_output
        )
        num_tokens_padded = batch_descriptor.num_tokens
        if self.compilation_config.pass_config.enable_sp:
            assert (
                batch_descriptor.num_tokens
                % self.vllm_config.parallel_config.tensor_parallel_size
                == 0
            ), (
                "Sequence parallelism requires num_tokens to be "
                "a multiple of tensor parallel size"
            )

        # Extra coordination when running data-parallel since we need to coordinate
        # across ranks
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            # Disable DP padding when running eager to avoid excessive padding when
            # running prefills. This lets us set cudagraph_mode="NONE" on the prefiller
            # in a P/D setup and still use CUDA graphs (enabled by this padding) on the
            # decoder.

            # Adapt start: Add padding for EP
            allow_dp_padding = (
                self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
                or self.parallel_config.enable_expert_parallel
            )
            # Adapt end: Add padding for EP

            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    parallel_config=self.parallel_config,
                    allow_microbatching=allow_microbatching,
                    allow_dp_padding=allow_dp_padding,
                    num_tokens_padded=num_tokens_padded,
                    uniform_decode=uniform_decode,
                    num_scheduled_tokens_per_request=num_scheduled_tokens_np,
                    cudagraph_mode=cudagraph_mode.value,
                )
            )

            # Extract DP-synced values
            if num_tokens_across_dp is not None:
                dp_rank = self.parallel_config.data_parallel_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # Re-dispatch with DP padding so we have the correct batch_descriptor
                cudagraph_mode, batch_descriptor = dispatch_cudagraph(
                    num_tokens_padded,
                    disable_full=synced_cudagraph_mode <= CUDAGraphMode.PIECEWISE.value,
                )
                # Assert to make sure the agreed upon token count is correct otherwise
                # num_tokens_across_dp will no-longer be valid
                assert batch_descriptor.num_tokens == num_tokens_padded

        cudagraph_stats = None
        if self.vllm_config.observability_config.cudagraph_metrics:
            cudagraph_stats = CUDAGraphStat(
                num_unpadded_tokens=num_tokens,
                num_padded_tokens=batch_descriptor.num_tokens,
                num_paddings=batch_descriptor.num_tokens - num_tokens,
                runtime_mode=str(cudagraph_mode),
            )

        # Adapt start: MTP extra property.
        # Add `batch_descriptor` and `cudagraph_mode` for latter use in mtp.
        if self.speculative_config and isinstance(self.drafter, EagleProposer):
            self.batch_execution_and_padding_state = (
                cudagraph_mode,
                batch_descriptor,
                num_tokens_across_dp,
            )
        return (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        )

    def _hook_model_load_weights(self) -> None:
        model = self.get_model()
        if getattr(model, "_omni_npu_load_weights_hooked", False):
            return

        original_load_weights = getattr(model, "load_weights", None)
        if not callable(original_load_weights):
            logger.error("model.load_weights is not callable.")
            return

        @wraps(original_load_weights)
        def wrapped_load_weights(*args, **kwargs):
            logger.info("Before calling self.model.load_weights")
            try:
                allocator = NpuMemAllocator.get_instance()
                context = allocator.use_memory_pool(tag="weights")
                with context, set_current_vllm_config(self.vllm_config):
                    original_load_weights(*args, **kwargs)

                # this is for RL pause/resume scene, recapture the model after loading weights.
                if not self.model_config.enable_sleep_mode:
                    if not self.model_config.enforce_eager:
                        self.capture_model()
            finally:
                logger.info("After calling self.model.load_weights")

        model.load_weights = wrapped_load_weights
        model._omni_npu_load_weights_hooked = True

    def load_model(self, eep_scale_up: bool = False) -> None:
        """
        Args:
            eep_scale_up: the model loading is for elastic EP scale up.
        """
        logger.debug(f"<<< {self.vllm_config.npu_compilation_config.use_gegraph=}")
        if self.vllm_config.npu_compilation_config.use_gegraph:
            from vllm.model_executor.model_loader import get_model as original_get_model
            self.model = original_get_model(vllm_config=self.vllm_config)
            return
        super().load_model(eep_scale_up)

        if hasattr(self.model, "model"):
            prefetch_post_load_hook = getattr(self.model.model, "prefetch_post_load", None)
            if callable(prefetch_post_load_hook):
                prefetch_post_load_hook()

        if hasattr(self, "drafter") and isinstance(self.drafter, EagleProposer):
            prepare_communication_buffer_for_model(self.drafter.model)

        # wrap the model with full graph wrapper if needed.
        logger.debug(f"<<< {self.compilation_config.cudagraph_mode.has_full_cudagraphs()=}")
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            set_graph_params(self.compilation_config.cudagraph_capture_sizes)
            self.update_stream: torch.npu.Stream = torch.npu.Stream()
            
            attn_layer_names = set(get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys())
            if hasattr(self, "drafter") and isinstance(self.drafter, EagleProposer):
                attn_layer_names = attn_layer_names - set(self.drafter.attn_layer_names)
            attn_layer_names = list(attn_layer_names)

            self.model = ACLGraphWrapper(self.model.runnable,
                                         self.vllm_config,
                                         runtime_mode=CUDAGraphMode.FULL,
                                         update_stream=self.update_stream,
                                         attn_layer_names=attn_layer_names,
                                        )
            logger.debug("<<< Wrapped original model with ACLGraphWrapper")
            if hasattr(self, "drafter") and isinstance(self.drafter, EagleProposer):
                n_predict = getattr(self.drafter, "n_predict", 1)
                if n_predict == 1:
                    self.draft_update_stream: torch.npu.Stream = torch.npu.Stream()
                    self.drafter.model = ACLGraphWrapper(
                        self.drafter.model,
                        self.vllm_config,
                        runtime_mode=CUDAGraphMode.FULL,
                        update_stream=self.draft_update_stream,
                        attn_layer_names=self.drafter.attn_layer_names,
                    )
                    logger.debug("<<< Wrapped drafter model with ACLGraphWrapper")
                else:
                    mtp_start_layer_idx = self.drafter.model.config.num_hidden_layers
                    self.draft_update_stream: torch.npu.Stream = torch.npu.Stream()
                    
                    wrapped_layers = dict()
                    for i in range(n_predict):
                        mtp_layer_i = mtp_start_layer_idx + i
                        mtp_layer_i_attn_names = [
                            item for item in self.drafter.attn_layer_names
                            if self.drafter.model.get_spec_layer(item) == mtp_layer_i
                        ]
                        wrapped_layers[str(mtp_layer_i)] = ACLGraphWrapper(
                            self.drafter.model.model.layers[str(mtp_layer_i)],
                            self.vllm_config,
                            runtime_mode=CUDAGraphMode.FULL,
                            update_stream=self.draft_update_stream,
                            attn_layer_names=mtp_layer_i_attn_names,
                        )
                    self.drafter.model.model.wrapped_layers = wrapped_layers
                    logger.debug("<<< Wrapped multi mtp layers of drafter model with ACLGraphWrapper")
        self._is_mm_encoder_only = supports_mm_encoder_only(self.model)
        self._hook_model_load_weights()

    def capture_model(self) -> int:
        logger.debug("<<< Capturing model in npu_model_runner")
        if self.vllm_config.npu_compilation_config.use_gegraph:
            logger.info(f"<<< capture_model use gegraph, dummy_run max_num_reqs={self.max_num_reqs}")
            self._dummy_run(self.max_num_reqs, force_attention=True, uniform_decode=True)
            return
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

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
        activate_lora: bool = False,
        is_graph_capturing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a dummy forward pass to warm up/profile run or capture the
        CUDA graph for the model.

        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: used to control the behavior.
                - if not set will determine the cudagraph mode based on using
                    the self.cudagraph_dispatcher.
                - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
                - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                    needed.
            force_attention: If True, always create attention metadata. Used to
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
            create_mixed_batch: If True, create a mixed batch with both decode
                (1 token) and prefill (multiple tokens) requests.
            remove_lora: If False, dummy LoRAs are not destroyed after the run
            activate_lora: If False, dummy_run is performed without LoRAs.
        """
        if self._is_mm_encoder_only:
            # The current dummy run only covers LM execution, so we can skip it.
            # mm encoder dummy run may need to add in the future.
            return torch.tensor([]), torch.tensor([])

        assert (
            cudagraph_runtime_mode is None
            or cudagraph_runtime_mode.valid_runtime_modes()
        )

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            assert not uniform_decode
            # Create mixed batch:
            # first half decode tokens, second half one prefill
            num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1

            # Create decode requests (1 token each) followed by prefill request
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
            # Note: Overriding max_query_len to be the prefill tokens
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            assert not create_mixed_batch
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        num_tokens_unpadded = int(num_scheduled_tokens.sum())

        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

        _cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = (
            self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens,
                max_num_scheduled_tokens=max_query_len,
                use_cascade_attn=False,
                allow_microbatching=allow_microbatching,
                force_eager=is_profile
                or (cudagraph_runtime_mode == CUDAGraphMode.NONE),
                # `force_uniform_decode` is used for cudagraph capture; because for
                # capturing mixed prefill-decode batches, we sometimes use
                # num_tokens == num_reqs which looks like a uniform decode batch to the
                # dispatcher; but we actually want to capture a piecewise cudagraph
                force_uniform_decode=uniform_decode,
                # `force_has_lora` is used for cudagraph capture; because LoRA is
                # activated later in the context manager, but we need to know the
                # LoRA state when determining the batch descriptor for capture
                force_has_lora=activate_lora,
            )
        )
        # For dummy run，in execute_model , If cudagraph_runtime_mode is None, 
        # dummy run is in execute_model and no compilation is needed. 
        # If it is not None, dummy run is during model capture and compilation is required.
        need_compile = False
        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = _cudagraph_mode
        else:
            need_compile = True
            assert cudagraph_runtime_mode == _cudagraph_mode, (
                f"Cudagraph runtime mode mismatch in dummy_run. "
                f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
            )

        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = (
            batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        )
        ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
            should_ubatch,
            num_scheduled_tokens,
            num_tokens_padded,
            num_reqs_padded,
            self.vllm_config.parallel_config.num_ubatches,
        )
        logger.debug(
            "ubatch_slices: %s, ubatch_slices_padded: %s",
            ubatch_slices,
            ubatch_slices_padded,
        )

        attn_metadata: PerLayerAttnMetadata | None = None

        # If force_attention is True, we always capture attention. Otherwise,
        # it only happens for cudagraph_runtime_mode=FULL.
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            if create_mixed_batch:
                # In the mixed batch mode (used for FI warmup), we use
                # shorter sequence lengths to run faster.
                # TODO(luka) better system for describing dummy batches
                seq_lens = [1] * num_decode_tokens + [num_prefill_tokens + 1]
            else:
                seq_lens = max_query_len  # type: ignore[assignment]
            self.seq_lens.np[:num_reqs] = seq_lens
            self.seq_lens.np[num_reqs:] = 0
            self.seq_lens.copy_to_gpu()

            cum_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)
            self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
            #TODO check deepseek model 
            self.query_start_loc.np[num_reqs + 1 :].fill(cum_num_tokens[-1])
            self.query_start_loc.copy_to_gpu()

            pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
            attn_metadata, _ = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded,
                num_reqs=num_reqs_padded,
                max_query_len=max_query_len,
                ubatch_slices=ubatch_slices_padded if pad_attn else ubatch_slices,
                for_cudagraph_capture=is_graph_capturing,
            )

        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            activate_lora,
            remove_lora,
        ):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            model_kwargs = self._init_model_kwargs()
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)

                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
                model_kwargs = self._init_model_kwargs()
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions.gpu[:num_tokens_padded]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device,
                        )
                    )

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens_padded, None, False
                )

            if ubatch_slices_padded is not None:
                # Adjust values to reflect a single ubatch.
                # TODO(sage,lucas): this is cruft that should be addressed in
                #  the padding refactor.
                num_tokens_padded = ubatch_slices_padded[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_padded

            with (
                self.maybe_randomize_inputs(input_ids, inputs_embeds),
                set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_desc,
                    ubatch_slices=ubatch_slices_padded,
                ),
            ):
                if self.router_sliding_window > 1:
                    self._build_conv_context(dummy=True)
                forward_context = get_forward_context()
                forward_context.capturing = False
                forward_context.need_compile = need_compile
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            if self.speculative_config and self.speculative_config.use_eagle():
                assert isinstance(self.drafter, EagleProposer)
                # Adapt start: enable mtp acl graph mode
                use_cudagraphs = (
                    (
                        is_graph_capturing
                        and cudagraph_runtime_mode == CUDAGraphMode.FULL
                    )
                    or (
                        not is_graph_capturing
                        and cudagraph_runtime_mode != CUDAGraphMode.NONE
                    )
                ) and not self.speculative_config.enforce_eager
                # Adapt end: enable mtp acl graph mode

                # Note(gnovack) - We need to disable cudagraphs for one of the two
                # lora cases when cudagraph_specialize_lora is enabled. This is a
                # short term mitigation for issue mentioned in
                # https://github.com/vllm-project/vllm/issues/28334
                if self.compilation_config.cudagraph_specialize_lora and activate_lora:
                    use_cudagraphs = False

                # Adapt start: to pass attn_metadata
                self.drafter.dummy_run(
                    attn_metadata,
                    num_tokens,
                    use_cudagraphs=use_cudagraphs,
                    is_graph_capturing=is_graph_capturing,
                )
                # Adapt end: to pass attn_metadata

        # We register layerwise NVTX hooks here after the first dynamo tracing is
        # done to avoid nvtx operations in hook functions being traced by
        # torch dynamo and causing graph breaks.
        # Note that for DYNAMO_ONCE and VLLM_COMPILE mode,
        # compiled model's dynamo tracing is only done once and the compiled model's
        # __call__ function is replaced by calling the compiled function.
        # So it's safe to register hooks here. Hooks will be registered to
        # both compiled and uncompiled models but they will never
        # be called on the compiled model execution path.
        self._register_layerwise_nvtx_hooks()

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        return hidden_states, hidden_states[:num_reqs]

    @prepare_inputs_decorator
    def prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_tokens_after_padding: int,
    ) -> "InputBatch":
        input_batch = super().prepare_inputs(scheduler_output, num_tokens_after_padding)

        return input_batch

    @prepare_inputs_decorator
    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_tokens_after_padding: int,
    ) -> tuple:
        (logits_indices, spec_decode_metadata) = super()._prepare_inputs(scheduler_output, num_tokens_after_padding)

        return (logits_indices, spec_decode_metadata)

    @init_config_decorator
    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        super().initialize_kv_cache(kv_cache_config)

    def initialize_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> dict[str, torch.Tensor]:
        kv_caches = super().initialize_kv_cache_tensors(kv_cache_config, kernel_block_sizes)
        if has_kv_transfer_group():
            self.kv_caches_dict = kv_caches
        return kv_caches
    
    def kv_cache_after_wake_up(self) -> None:
        attn_layers = self.compilation_config.static_forward_context
        if self.model_config.enable_sleep_mode:
            from vllm.model_executor.layers.attention.static_sink_attention import StaticSinkAttention
            sink_mla_available = False
            try:
                from vllm.model_executor.layers.attention.static_sink_attention import StaticSinkMLAAttention
                sink_mla_available = True
            except ImportError:
                logger.warning("StaticSinkMLAAttention has not been defined, skipping...")
            for name, module in attn_layers.items():
                if isinstance(module, StaticSinkAttention) or (sink_mla_available 
                                                               and isinstance(module, StaticSinkMLAAttention)):
                    self._kv_cache_sink_attn_after_wake_up(module)

    def _kv_cache_sink_attn_after_wake_up(self, module) -> None:
        sink_kv_cache = getattr(module, "kv_cache")

        # populate_sink_kv in SinkAttention retrieves the `virtual_engine` value from `ForwardContext`
        # but this value is unavailable here.
        # Since `virtual_engine` is defaulted to 0 and will be deprecated, we directly set it to 0 here.
        self_kv_cache = sink_kv_cache[0]
        if self_kv_cache is not None and len(self_kv_cache) > 0:
            populate_sink_kv_method = getattr(module, "populate_sink_kv")
            populate_sink_kv_method(self_kv_cache[0], self_kv_cache[1])

        for kv_cache_group_id in range(len(self.kv_cache_config.kv_cache_groups)):
            for attn_group in self.attn_groups[kv_cache_group_id]:
                for attn_builder in attn_group.metadata_builders:
                    if hasattr(attn_builder, "reinit_block_table_with_sink"):
                        attn_builder.reinit_block_table_with_sink()

    def unregister_kv_caches(self):
        if self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.kv_connector == "LLMDataDistConnector":
            if has_kv_transfer_group():
                logger.info(f"unregister_kv_caches")
                get_kv_transfer_group().unregister_kv_caches()

    def reregister_kv_caches(self):
        if self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.kv_connector == "LLMDataDistConnector":
            if has_kv_transfer_group():
                logger.info(f"reregister_kv_caches")
                get_kv_transfer_group().register_kv_caches(self.kv_caches_dict)
