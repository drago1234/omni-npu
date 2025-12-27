# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import numpy as np
import torch

import ast
from dataclasses import replace
from importlib.util import find_spec

import numpy as np
import torch
import torch.nn as nn

from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
)
from vllm.forward_context import set_forward_context

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.spec_decode.eagle import EagleProposer

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

@register_patch("TorchEagleProposer", EagleProposer)
class EagleProposerPatch(VLLMPatch):
    """Patch for vLLM's EagleProposer to support omni-npu compilation and execution.
    """

    _attr_names_to_apply = ['prepare_next_token_ids_padded', 'prepare_inputs_padded', 'dummy_run']

    def prepare_next_token_ids_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids and the number of valid sampled tokens
        for each request, considering the "discarded" requests whose next token
        is not sampled and comes from `request.get_token_id()` instead.
        It also accounts for the rejected tokens in `sampled_token_ids`.
        This function must use device functions to operate on the inputs, and
        should not introduce any blocking CPU-GPU synchronization.
        """
        # TODO(Ben): Combine this into a custom fused kernel

        # Precompute get_token_id for when there is no valid next token
        num_reqs = gpu_input_batch.num_reqs
        self.backup_next_token_ids.np[:num_reqs] = np.array(
            [
                requests[gpu_input_batch.req_ids[i]].get_token_id(
                    common_attn_metadata.seq_lens_cpu[i].item()
                )
                for i in range(num_reqs)
            ]
        )
        self.backup_next_token_ids.copy_to_gpu(num_reqs)

        # Mask out the sampled tokens indices that should not be sampled.
        discard_sampled_tokens_req_indices = torch.nonzero(
            discard_request_mask[:num_reqs]
        )

        valid_sampled_token_ids_gpu = sampled_token_ids.clone()
        valid_sampled_token_ids_gpu.index_fill_(
            0, discard_sampled_tokens_req_indices, -1
        )

        # Generate a mask for all valid tokens within those requests
        valid_mask = (valid_sampled_token_ids_gpu != -1) & (
            valid_sampled_token_ids_gpu < gpu_input_batch.vocab_size
        )

        # Count the number of valid tokens in each request
        valid_sampled_tokens_count = valid_mask.sum(dim=1)

        # Get the rightmost valid index per row
        last_valid_indices = valid_sampled_tokens_count - 1
        last_valid_indices_safe = torch.clamp(last_valid_indices, min=0)

        # Get last valid token from each row
        # (assume undefined state where there is no valid token)
        selected_tokens = torch.gather(
            valid_sampled_token_ids_gpu, 1, last_valid_indices_safe.unsqueeze(1)
        ).squeeze(1)

        # Use last token if valid, pre-computed backup if not
        batch_size = valid_sampled_token_ids_gpu.shape[0]
        next_token_ids = torch.where(
            last_valid_indices != -1,
            selected_tokens,
            self.backup_next_token_ids.gpu[:batch_size],
        )

        return next_token_ids, valid_sampled_tokens_count


    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding
        It updates the common_attn_metadata for speculative decoding,
        but does not consider the rejected tokens. Instead, all tokens
        are included as inputs to the speculator, with the rejected tokens
        used as padding and filtered out later by `token_indices_to_sample`.
        No blocking CPU operations should be introduced in this function.
        """
        num_draft_tokens_gpu = torch.cat(
            [
                spec_decode_metadata.cu_num_draft_tokens[0:1],
                spec_decode_metadata.cu_num_draft_tokens[1:]
                - spec_decode_metadata.cu_num_draft_tokens[:-1],
            ]
        )

        num_rejected_tokens_gpu = torch.where(
            num_draft_tokens_gpu > 0,
            num_draft_tokens_gpu + 1 - valid_sampled_tokens_count,
            torch.zeros_like(num_draft_tokens_gpu),
        )

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        total_num_tokens = query_start_loc_cpu[-1].item()
        token_indices = self.arange[:total_num_tokens]

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens_cpu=common_attn_metadata.seq_lens_cpu,
            num_computed_tokens_cpu=common_attn_metadata.num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=common_attn_metadata.seq_lens_cpu.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[token_indices],
            causal=True,
        )

        token_indices_to_sample = (
            common_attn_metadata.query_start_loc[1:] - 1 - num_rejected_tokens_gpu
        )

        return spec_common_attn_metadata, token_indices_to_sample

    @torch.inference_mode()
    def dummy_run(
        self,
        attn_metadata,
        num_tokens: int,
        use_cudagraphs=True,
        is_graph_capturing=False,
        batch_descriptor=None,
    ) -> None:
        # Adapt: new param attn_metadata and batch_descriptor
        # Determine if CUDA graphs should be used for this run.
        cudagraphs_enabled = use_cudagraphs and self.use_cuda_graph

        # FIXME: when using tree-based specdec, adjust number of forward-passes
        # according to the depth of the tree.
        for fwd_idx in range(
            self.num_speculative_tokens if not is_graph_capturing else 1
        ):
            if fwd_idx <= 1:
                num_tokens_dp_padded, num_tokens_across_dp = self._pad_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    num_tokens_padded=num_tokens,
                )
                if (
                    cudagraphs_enabled
                    and num_tokens_dp_padded
                    <= self.compilation_config.max_cudagraph_capture_size
                ):
                    num_input_tokens = self.vllm_config.pad_for_cudagraph(
                        num_tokens_dp_padded
                    )
                else:
                    num_input_tokens = num_tokens_dp_padded
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[self.dp_rank] = num_input_tokens

            # Adapt: pass attn_metadata and batch_descriptor to set_forward_context, change cudagraph_runtime_mode
            with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE
                if cudagraphs_enabled
                else CUDAGraphMode.NONE,
                batch_descriptor=batch_descriptor,
            ):
                if self.supports_mm_inputs:
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:num_input_tokens]
                else:
                    input_ids = self.input_ids[:num_input_tokens]
                    inputs_embeds = None

                self.model(
                    input_ids=input_ids,
                    positions=self._get_positions(num_input_tokens),
                    hidden_states=self.hidden_states[:num_input_tokens],
                    inputs_embeds=inputs_embeds,
                )
