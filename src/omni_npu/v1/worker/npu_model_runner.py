# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union, Any, cast, TypeAlias

import torch
import numpy as np
import torch.nn as nn
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
)
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group, prepare_communication_buffer_for_model
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    KVCacheConfig,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.attention.backends.abstract import (
    AttentionMetadata,
)
from vllm.forward_context import set_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

from omni_npu.v1.sample.sampler import NPUSamplerV1
from omni_npu.v1.sample.rejection_sampler import NPURejectionSampler
from omni_npu.compilation.acl_graph import ACLGraphWrapper, set_graph_params

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

        # enable mtp acl graph mode
        if self.speculative_config and isinstance(self.drafter, EagleProposer):
            if self.compilation_config.mode == CompilationMode.VLLM_COMPILE:
                self.drafter.use_cuda_graph = self.compilation_config.cudagraph_mode.has_mode(CUDAGraphMode.PIECEWISE)

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
            self.rejection_sampler = NPURejectionSampler(self.sampler)

        if vllm_config.additional_config is not None:
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
        logger.debug(f"<<< {self.vllm_config.npu_compilation_config.use_gegraph=}")
        if self.vllm_config.npu_compilation_config.use_gegraph:
            from vllm.model_executor.model_loader import get_model as original_get_model
            self.model = original_get_model(vllm_config=self.vllm_config)
            return
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

        _cudagraph_mode, batch_desc, ubatch_slices, num_tokens_across_dp = (
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

        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = _cudagraph_mode
        else:
            assert cudagraph_runtime_mode == _cudagraph_mode, (
                f"Cudagraph runtime mode mismatch in dummy_run. "
                f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
            )

        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = (
            batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
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
            self.query_start_loc.copy_to_gpu()

            attn_metadata, _ = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs_padded,
                max_query_len=max_query_len,
                ubatch_slices=ubatch_slices,
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
            model_kwargs = self._init_model_kwargs(num_tokens_padded)
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
                model_kwargs = self._init_model_kwargs(num_tokens_padded)
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

            if ubatch_slices is not None:
                # Adjust values to reflect a single ubatch.
                # TODO(sage,lucas): this is cruft that should be addressed in
                #  the padding refactor.
                num_tokens_padded = ubatch_slices[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_padded

            with (
                self.maybe_randomize_inputs(input_ids),
                set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_desc,
                    ubatch_slices=ubatch_slices,
                ),
            ):
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
                # enable mtp acl graph mode
                use_cudagraphs = cudagraph_runtime_mode.has_mode(CUDAGraphMode.PIECEWISE)

                # Note(gnovack) - We need to disable cudagraphs for one of the two
                # lora cases when cudagraph_specialize_lora is enabled. This is a
                # short term mitigation for issue mentioned in
                # https://github.com/vllm-project/vllm/issues/28334
                if self.compilation_config.cudagraph_specialize_lora and activate_lora:
                    use_cudagraphs = False

                # Adapt start : to pass attn_metadata and batch_desc
                self.drafter.dummy_run(
                    attn_metadata,
                    num_tokens_padded,
                    use_cudagraphs=use_cudagraphs,
                    is_graph_capturing=is_graph_capturing,
                    batch_descriptor=batch_desc,
                )
                # Adapt end : to pass attn_metadata and batch_desc

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        logit_indices_device = torch.from_numpy(logit_indices).to(
            self.device, non_blocking=True
        )
        return hidden_states, hidden_states[logit_indices_device]

    def initialize_omni_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
        self.initialize_attn_backend(kv_cache_config)
        # The kernel block size for all KV cache groups. For example, if
        # kv_cache_manager uses block_size 256 for a given group, but the attention
        # backends for that group only supports block_size 64, we will return
        # kernel_block_size 64 and split the 256-token-block to 4 blocks with 64
        # tokens each.
        kernel_block_sizes = self._prepare_kernel_block_sizes(kv_cache_config)

        # create metadata builders
        self.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)

        # Reinitialize need to after initialize_attn_backend
        self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)

        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            # validate all draft model layers belong to the same kv cache
            # group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

        from omni_cache.cache.omni_cache_define import create_omni_cache
        create_omni_cache(
            kv_cache_config=self.kv_cache_config,
            vllm_config=self.vllm_config,
            runner=self,
        )
        from omni_cache.cache import omni_cache
        get_kv_transfer_group().register_kv_caches(
            omni_cache.MEMMAP_PATH,
            omni_cache.dtype,
            block_len_dtype=omni_cache.block_len_dtype,
            omni_cache=omni_cache
        )
        self.omni_cache = omni_cache
