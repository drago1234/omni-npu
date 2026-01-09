# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch_npu
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    ExpertTokensMetadata,
    FusedMoEActivationFormat,
    FusedMoEPrepareAndFinalize,
    PrepareResultType,
    TopKWeightAndReduce,
)
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform


class NpuMoEPrepareAndFinalize(FusedMoEPrepareAndFinalize):
    """
    NPU implementation of MoE prepare and finalize operations.
    Uses torch_npu.npu_moe_* ops.
    """

    def __init__(self, moe: FusedMoEConfig):
        super().__init__()
        self.moe = moe
        self.moe_all_to_all_group = get_ep_group().device_group
        self.moe_all_to_all_group_name = self.moe_all_to_all_group._get_backend(
                    torch.device(current_platform.device_type)).get_hccl_comm_name(
                    get_ep_group().rank_in_group)
        self.expand_idx = None
        self.ep_recv_counts = None
        self.tp_recv_counts = None

    def _get_mc2_mask(self, num_tokens: int) -> torch.Tensor | None:
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[next(iter(attn_metadata))]

        if hasattr(attn_metadata, 'decode') and attn_metadata.decode is not None:
            mc2_mask = attn_metadata.decode.mc2_mask[:num_tokens]
        else:
            mc2_mask = None
        return mc2_mask

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> PrepareResultType:
        if quant_config.use_int8_w8a8:
            quant_mode = 2  # Dynamic quantization
        else:
            quant_mode = 0  # No quantization

        self.num_experts = num_experts
        kwargs = {
            "x": a1,
            "expert_ids": topk_ids,  # [n*topk]
            "expert_shard_type": 0,  # Set it to 0 for now
            "shared_expert_rank_num": 0,  # 32
            "moe_expert_num": self.num_experts, #ENABLE_OMNI_PLANNER, 0 redundancy 256, 1 redundancy expert 320
            "global_bs": 0,  # 0 Default (all); all tokens can be set
            "scales": None,  # Quantization coefficient
            "quant_mode": quant_mode,
            "group_ep": self.moe_all_to_all_group_name,  # Unlike torch, it is obtained by name.
            "ep_world_size": self.moe.ep_size,
            "ep_rank_id": self.moe.ep_rank,
            "x_active_mask": self._get_mc2_mask(topk_ids.shape[0]),
        }

        output = torch_npu.npu_moe_distribute_dispatch_v2(**kwargs)
        expand_x, dynamic_scale, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = output[0:6]
        expert_tokens_meta = ExpertTokensMetadata(
            expert_num_tokens=expert_token_nums,
            expert_num_tokens_cpu=None # Can be filled if needed
        )
        self.expand_idx = expand_idx
        self.ep_recv_counts = ep_recv_counts
        self.tp_recv_counts = tp_recv_counts
        self.expert_token_nums = expert_token_nums
        return expand_x, dynamic_scale, expert_tokens_meta, None, None

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: TopKWeightAndReduce,
    ) -> None:
        kwargs = {
            "expand_x": fused_expert_output,
            "expert_ids": topk_ids,  # [n*topk]
            "assist_info_for_combine": self.expand_idx,
            "expert_scales": topk_weights.to(torch.float32),  # weight [n*topk]
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num":  self.num_experts, #ENABLE_OMNI_PLANNER, 0 redundancy 256, 1 redundancy expert 320
            "global_bs": 0,  # 0 Default (all); all tokens can be set
            "ep_send_counts": self.ep_recv_counts,  # dispatch's send_counts
            "group_ep": self.moe_all_to_all_group_name,  # Unlike torch, it is obtained by name.
            "ep_world_size": self.moe.ep_size,
            "ep_rank_id": self.moe.ep_rank,
            "tp_send_counts": self.tp_recv_counts,
            "x_active_mask": self._get_mc2_mask(topk_ids.shape[0]),
        }

        hidden_states_route = torch_npu.npu_moe_distribute_combine_v2(**kwargs)
        output.copy_(hidden_states_route)

    @property
    def activation_format(self) -> FusedMoEActivationFormat:
        return FusedMoEActivationFormat.Standard

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def num_dispatchers(self) -> int:
        return 1

    def output_is_reduced(self) -> bool:
        return False
