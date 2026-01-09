# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import torch.distributed as dist
import torch_npu

from vllm.platforms import current_platform
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context


@dataclass
class PreparePermuteResult:
    hidden_states_sorted_by_experts: torch.Tensor
    expert_tokens: torch.Tensor
    dynamic_scale: torch.Tensor
    avg_tokens_per_expert: Optional[torch.Tensor] = None


@dataclass
class All2AllPreparePermuteResult(PreparePermuteResult):
    input_splits: List[int] = field(default_factory=lambda: defaultdict(int))
    output_splits: List[int] = field(default_factory=lambda: defaultdict(int))
    expanded_x: torch.Tensor = None
    expanded_row_idx: torch.Tensor = None
    gathered_idxs_unsort: torch.Tensor = None


@dataclass
class DispatchCombinePreparePermuteResult(PreparePermuteResult):
    tp_recv_counts: torch.Tensor = None
    ep_recv_counts: torch.Tensor = None
    expand_idx: torch.Tensor = None


class FusedMoEPreparePermuteAndUnpermuteFinalize(ABC):
    """
    An abstract base class for the Fused MoE prepare + permute and unpermute + finalize.
    """
    def __init__(self, layer):
        self.num_experts = layer.global_num_experts
        self.num_local_experts = layer.local_num_experts
        self.ep_size = get_ep_group().world_size
        self.ep_rank = get_ep_group().rank
        self.moe_quant_config = layer.quant_method.moe_quant_config

    @abstractmethod
    def prepare_permute(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_ids: torch.Tensor
    ) -> PreparePermuteResult:
        raise NotImplementedError

    @abstractmethod
    def unpermute_finalize(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        prepare_and_permute_result: PreparePermuteResult
    ) -> None:
        raise NotImplementedError


class All2AllPrepPmtAndUnpmtFinal(FusedMoEPreparePermuteAndUnpermuteFinalize):
    def prepare_permute(
      self,
      layer: torch.nn.Module,
      x: torch.Tensor,
      topk_ids: torch.Tensor
    ) -> All2AllPreparePermuteResult:
        x = x.view(-1, x.shape[-1])
        topk_ids = topk_ids.int()
        expert_range = [0, self.num_experts]
        quant_mode = 1 if layer.quant_config is not None else -1

        expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
            x,
            expert_idx=topk_ids,
            scale=None,
            expert_num=self.num_experts,
            active_expert_range=expert_range,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_num=topk_ids.numel(),
            drop_pad_mode=0,
            row_idx_type=0,
            quant_mode=quant_mode
        )

        tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
        dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)  # (total_experts,) --> (total_ranks * n_routed_experts_per_rank)
        # combine tensors, do reduceSum and D2H toghter
        combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
        # view: EP, E//EP
        # sum: EP, the number of tokens each rank receives from other cards
        combine_tokens = combine_tokens.view(2, self.ep_size, -1).sum(2)
        all_tokens = combine_tokens[0].sum()
        combine_tokens_cpu = combine_tokens.cpu().tolist()
        # alltoall input splits, the total number of tokens routed from the current rank to other ranks
        input_splits = combine_tokens_cpu[1]
        # alltoall output splits, the number of tokens each rank receives from other cards
        output_splits = combine_tokens_cpu[0]
        # alltoall output, unfolded into one dimension, the size is the sum of the number of tokens routed from other cards to the current rank.
        gathered_tokens = expanded_x.new_empty(all_tokens.item(), expanded_x.shape[1])
        dist.all_to_all_single(gathered_tokens, expanded_x, output_splits, input_splits)

        if layer.quant_config is None:
            gathered_pertoken_scale = None
        else:
            gathered_pertoken_scale = pertoken_scale.new_empty(gathered_tokens.shape[0])
            dist.all_to_all_single(gathered_pertoken_scale, pertoken_scale, output_splits, input_splits)

        # reroute
        # Tokens merged by experts, scales merged by experts, indices for FinalizeRouting, number of tokens processed by each expert
        hidden_states_sorted_by_experts, gathered_pertoken_scale, gathered_idxs_unsort, tokens_per_local_expert = torch_npu.npu_moe_re_routing(
            gathered_tokens,
            tokens_per_expert_group.view(layer.ep_size, -1),
            per_token_scales=gathered_pertoken_scale
        )

        return All2AllPreparePermuteResult(
            gathered_idxs_unsort=gathered_idxs_unsort,
            expanded_x=expanded_x,
            expanded_row_idx=expanded_row_idx,
            input_splits=input_splits,
            output_splits=output_splits,
            hidden_states_sorted_by_experts=hidden_states_sorted_by_experts,
            expert_tokens=tokens_per_local_expert,
            dynamic_scale=gathered_pertoken_scale
        )

    def unpermute_finalize(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        all2all_prepare_permute_result: All2AllPreparePermuteResult
    ) -> torch.Tensor:
        gathered_idxs_unsort = all2all_prepare_permute_result.gathered_idxs_unsort
        expanded_x = all2all_prepare_permute_result.expanded_x
        input_splits = all2all_prepare_permute_result.input_splits
        output_splits = all2all_prepare_permute_result.output_splits
        expanded_row_idx = all2all_prepare_permute_result.expanded_row_idx

        new_x = torch.index_select(
            hidden_states,
            0,
            gathered_idxs_unsort.to(torch.float32).argsort().to(torch.int32)
        )

        gathered_tokens = new_x.new_empty(*expanded_x.shape)
        dist.all_to_all_single(gathered_tokens, new_x, input_splits, output_splits)

        hidden_states = torch_npu.npu_moe_finalize_routing(
            gathered_tokens,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights.to(gathered_tokens.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None,
            drop_pad_mode=2
        )
        return hidden_states


class DispatchCombinePrepPmtAndUnpmtFinal(FusedMoEPreparePermuteAndUnpermuteFinalize):
    def __init__(self, layer):
        super().__init__(layer)
        self.moe_all_to_all_group = get_ep_group().device_group
        self.moe_all_to_all_group_name = self.moe_all_to_all_group._get_backend(
            torch.device(current_platform.device_type)
        ).get_hccl_comm_name(get_ep_group().rank_in_group)

    def _get_mc2_mask(self, num_tokens: int) -> torch.Tensor | None:
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[next(iter(attn_metadata))]

        if hasattr(attn_metadata, 'decode') and attn_metadata.decode is not None:
            mc2_mask = attn_metadata.decode.mc2_mask[:num_tokens]
        else:
            mc2_mask = None
        return mc2_mask

    def prepare_permute(
      self,
      layer: torch.nn.Module,
      x: torch.Tensor,
      topk_ids: torch.Tensor
    ) -> DispatchCombinePreparePermuteResult:
        if self.moe_quant_config is not None and self.moe_quant_config.use_int8_w8a8:
            quant_mode = 2  # Dynamic quantization
        else:
            quant_mode = 0  # No quantization

        kwargs = {
            "x": x,
            "expert_ids": topk_ids,  # [n*topk]
            "expert_shard_type": 0,  # Set it to 0 for now
            "shared_expert_rank_num": 0,  # 32
            "moe_expert_num": self.num_experts,
            "global_bs": 0,  # 0 Default (all); all tokens can be set
            "scales": None,  # Quantization coefficient
            "quant_mode": quant_mode,
            "group_ep": self.moe_all_to_all_group_name,  # Unlike torch, it is obtained by name.
            "ep_world_size": self.ep_size,
            "ep_rank_id": self.ep_rank,
            "x_active_mask": self._get_mc2_mask(topk_ids.shape[0]),
        }
        output = torch_npu.npu_moe_distribute_dispatch_v2(**kwargs)
        expand_x, dynamic_scale, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = output[0:6]

        return DispatchCombinePreparePermuteResult(
            hidden_states_sorted_by_experts=expand_x,
            expert_tokens=expert_token_nums.to(torch.int64),
            tp_recv_counts=tp_recv_counts,
            ep_recv_counts=ep_recv_counts,
            expand_idx=expand_idx,
            dynamic_scale=dynamic_scale
        )

    def unpermute_finalize(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        dispatch_combine_prepare_permute_result: DispatchCombinePreparePermuteResult
    ) -> torch.Tensor:
        kwargs = {
            "expand_x": hidden_states,
            "expert_ids": topk_ids,  # [n*topk]
            "assist_info_for_combine": dispatch_combine_prepare_permute_result.expand_idx,
            "expert_scales": topk_weights.to(torch.float32),  # weight [n*topk]
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num":  self.num_experts,
            "global_bs": 0,  # 0 Default (all); all tokens can be set
            "ep_send_counts": dispatch_combine_prepare_permute_result.ep_recv_counts,  # dispatch's send_counts
            "group_ep": self.moe_all_to_all_group_name,  # Unlike torch, it is obtained by name.
            "ep_world_size": self.ep_size,
            "ep_rank_id": self.ep_rank,
            "tp_send_counts": dispatch_combine_prepare_permute_result.tp_recv_counts,
            "x_active_mask": self._get_mc2_mask(topk_ids.shape[0]),
        }
        hidden_states = torch_npu.npu_moe_distribute_combine_v2(**kwargs)
        return hidden_states
