# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Optional, Callable
import torch, torch_npu
import torch.distributed as dist
from vllm.logger import init_logger
from vllm.distributed import get_world_group, get_ep_group, get_tp_group, get_pp_group, tensor_model_parallel_all_reduce, GroupCoordinator
from vllm.platforms import current_platform
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoeWeightScaleSupported,
    FusedMoE,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.shared_fused_moe import SharedFusedMoE


logger = init_logger("vllm.omni_npu.layers.fused_moe.layer")


class AscendFusedMoE(FusedMoE):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def _load_per_channel_weight_scale(self, expert_data: torch.Tensor,
                                       shard_dim: int, shard_id: str,
                                       loaded_weight: torch.tensor,
                                       tp_rank: int):
        # adapt loaded_weight shape
        loaded_weight = loaded_weight.squeeze(-1)
        # adapt end
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(shard_id=shard_id,
                           shard_dim=shard_dim,
                           loaded_weight=loaded_weight,
                           expert_data=expert_data,
                           tp_rank=tp_rank)

    @staticmethod
    def select_experts(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: Optional[torch.Tensor] = None,
        e_score_correction_bias: Optional[torch.Tensor] = None,
    ):
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            # profile run, force load balance
            ep_rank = get_ep_group().rank
            global_num_experts = router_logits.shape[1]
            num_tokens = router_logits.shape[0]
            topk_ids = torch.arange(
                ep_rank * num_tokens * top_k,
                (ep_rank+1) * num_tokens * top_k,
                dtype=torch.int32,
                device=router_logits.device,
            ).view(num_tokens, top_k) % global_num_experts
            topk_weights = torch.rand_like(topk_ids)
            row_idx = torch.arange(topk_ids.numel(),
                                   device=topk_ids.device,
                                   dtype=torch.int32) \
                            .view(-1, num_tokens) \
                            .transpose(0, 1)
            return topk_weights, topk_ids, row_idx

        if use_grouped_topk and num_expert_group != 1:
            if topk_group is None:
                raise ValueError(f"Unsupported topk_group is None")
            if num_expert_group is None:
                raise ValueError(f"Unsupported num_expert_group is None")
            topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
                router_logits.float(),
                k=top_k,
                bias=e_score_correction_bias,
                k_group=topk_group,
                group_count=num_expert_group,
                group_select_mode=1,
                renorm=0,
                norm_type=1,
                routed_scaling_factor=routed_scaling_factor,
                eps=1e-20,
            )
            row_idx = torch.arange(topk_ids.numel(),
                                   device=current_platform.device_type,
                                   dtype=torch.int32) \
                            .view(-1, router_logits.shape[0]) \
                            .transpose(0, 1)
        elif custom_routing_function is None:
            topk_weights, topk_ids, row_idx = torch_npu.npu_moe_gating_top_k_softmax(router_logits, k=top_k)
            if renormalize:
                topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        else:
            topk_weights, topk_ids, row_idx = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)

        return topk_weights, topk_ids, row_idx


@SharedFusedMoE.register_oot
class AscendSharedFusedMoE(SharedFusedMoE, AscendFusedMoE):
    pass


def moe_infer_fusion(
    layer,
    x,
    topk_ids,
    topk_weight,
    comm_group: GroupCoordinator=None
):
    hidden_states = x.view(-1, x.shape[-1])
    max_num_deployed_expert = layer.moe_config.num_experts
    topk_ids = topk_ids.int()

    expert_range = [0, max_num_deployed_expert]
    quant_mode = 1 if layer.quant_config is not None else -1
    expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
        hidden_states,
        expert_idx=topk_ids,
        scale=None,
        expert_num=max_num_deployed_expert,
        active_expert_range=expert_range,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_num=topk_ids.numel(),
        drop_pad_mode=0,
        row_idx_type=0,
        quant_mode=quant_mode)

    tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
    group = comm_group.device_group if comm_group is not None else None
    dist.all_to_all_single(tokens_per_expert_group,
                           tokens_per_expert,
                           group=group)  # (total_experts,) --> (total_ranks * n_routed_experts_per_rank)

    # combine tensors, do reduceSum and D2H toghter
    combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
    # view: EP, E//EP
    # sum: EP, the number of tokens each rank receives from other cards
    ep_size = get_ep_group().world_size
    combine_tokens = combine_tokens.view(2, ep_size, -1).sum(2)
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
        dist.all_to_all_single(gathered_pertoken_scale, pertoken_scale, output_splits, input_splits, group=group)

    # reroute
    # Tokens merged by experts, scales merged by experts, indices for FinalizeRouting, number of tokens processed by each expert
    hidden_states_sorted_by_experts, gathered_pertoken_scale, gathered_idxs_unsort, tokens_per_local_expert = torch_npu.npu_moe_re_routing(
        gathered_tokens,
        tokens_per_expert_group.view(ep_size, -1),
        per_token_scales=gathered_pertoken_scale
    )
    group_list = tokens_per_local_expert.to(torch.int64)

    hidden_states_ordered_by_experts = layer.quant_method.gmm_expert(
        layer,
        hidden_states_sorted_by_experts,
        group_list,
        gathered_pertoken_scale,
        None,
    )

    new_x = torch.index_select(hidden_states_ordered_by_experts, 0,
                               gathered_idxs_unsort.to(torch.float32).argsort().to(torch.int32))
    gathered_tokens = new_x.new_empty(*expanded_x.shape)

    dist.all_to_all_single(gathered_tokens, new_x, input_splits, output_splits)

    hidden_states = torch_npu.npu_moe_finalize_routing(
        gathered_tokens,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weight.to(gathered_tokens.dtype),
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=None,
        drop_pad_mode=2
    )
    return hidden_states


def fused_experts(
    layer: torch.nn.Module,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
):
    row_idx = torch.arange(topk_ids.numel(), device=current_platform.device_type,
                           dtype=torch.int32).view(-1, x.shape[0]).transpose(0, 1)
    num_tokens, hidden_size = x.shape
    n_routed_experts = len(layer.w13_weight)
    sorted_tokens, expanded_src_to_dst_row, expanded_expert_idx = \
        torch_npu.npu_moe_init_routing(x, row_idx, topk_ids, num_tokens)
    expert_tokens = torch_npu.npu_moe_compute_expert_tokens(expanded_expert_idx, n_routed_experts).to(torch.int64)
    if layer.quant_config is None:
        pertoken_scale = None
    else:
        sorted_tokens, pertoken_scale = torch_npu.npu_dynamic_quant(sorted_tokens)

    out = layer.quant_method.gmm_expert(
        layer,
        sorted_tokens,
        expert_tokens,
        pertoken_scale
    )

    return torch_npu.npu_moe_finalize_routing(out, None, None, None, topk_weights,
                                              expanded_src_to_dst_row, topk_ids).to(torch.bfloat16)

def fused_experts_moe_dispatch_combine(layer: torch.nn.Module,
                                            hidden_states: torch.Tensor,
                                            topk_weights: torch.Tensor,
                                            topk_ids: torch.Tensor,
                                            max_num_deployed_expert: int,
                                            is_prefill: bool,
                                            is_route_expert: bool,
                                            ):
    expert_parallel_size = get_ep_group().world_size

    if expert_parallel_size > 1:
        # For vllm v1, metadata is a dict {layer_name: metadata}
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[next(iter(attn_metadata))]
        # mc2_mask = attn_metadata.decode.mc2_mask if attn_metadata is not None and getattr(attn_metadata, "decode", None) is not None else None
        mc2_mask = None # NOTE(Zuo Yuqi) Lazy Implemention
        global_bs = 0
        act_dtype = hidden_states.dtype
        # route
        shared_expert_rank_num = 0 # NOTE(Zuo Yuqi) Lazy Implemention
        kwargs = {
            "x": hidden_states,
            "expert_ids": topk_ids,  # [n*topk]
            "expert_shard_type": 0,
            "shared_expert_rank_num": shared_expert_rank_num,  # 32
            "moe_expert_num": max_num_deployed_expert,  # ENABLE_OMNI_PLANNER, 0 redundancy 256, 1 redundancy expert 320
            "global_bs": global_bs,  # 0 Default (all); all tokens can be set
        }
        experts_tp_size = layer.tp_size
        world_size = get_world_group().world_size
        # In fact, what we get is the die number, and the ep group is not adapted by default.
        # The default ep group is experts_num/die_num.
        global_rank = get_world_group().rank_in_group
        all_to_all_group_size = world_size // experts_tp_size

        ffn_dies = world_size

        moe_all_to_all_group_name = get_world_group().device_group._get_backend(
                    torch.device("npu")).get_hccl_comm_name(
                    get_world_group().rank_in_group)

        moe_rs_group_name = get_pp_group().device_group._get_backend(
                    torch.device("npu")).get_hccl_comm_name(
                    get_pp_group().rank_in_group)

        kwargs.update({
            "scales": None,  # Quantization coefficient
            "quant_mode": 2,  # 0: Non-quantization; 1: Static quantization; 2: Dynamic quantization, NOTE(Zuo Yuqi) Lazy Implemention: static quantization is not supported here.
            "group_ep": moe_all_to_all_group_name,  # Unlike torch, it is obtained by name.
            "ep_world_size": all_to_all_group_size,
            "ep_rank_id": global_rank // experts_tp_size,
            "group_tp": moe_rs_group_name,
            "tp_world_size": experts_tp_size,
            "tp_rank_id": global_rank % experts_tp_size,
            "x_active_mask": mc2_mask,
        })

        output = torch_npu.npu_moe_distribute_dispatch_v2(**kwargs)
        expand_x, dynamic_scale, expand_idx, expert_token_nums, ep_recv_counts = output[0:5]

        if global_rank < ffn_dies:
            group_list = expert_token_nums.to(torch.int64)

            if shared_expert_rank_num > 0 and global_rank // experts_tp_size < shared_expert_rank_num:
                x = {"x_int8": expand_x, "pertoken_scale": dynamic_scale}
                hidden_states_experts = layer(x)
            else:
                # cal experts
                group_list = group_list[
                            :len(layer.w13_weight)]  # Adapt to redundant and non-redundant layers, #ENABLE_OMNI_PLANNER
                hidden_states_experts = layer.quant_method.moe_expert_quant_forward(layer, expand_x, group_list, act_dtype, dynamic_scale)
        else:
            hidden_states = torch.zeros_like(expand_x).to(torch.bfloat16)
            ep_recv_counts = torch.zeros_like(ep_recv_counts)

        # moeCombine
        kwargs = {
            "expand_x": hidden_states_experts,
            "expert_ids": topk_ids,  # [n*topk]
            "assist_info_for_combine": expand_idx,
            "expert_scales": topk_weights.to(torch.float32),  # weight [n*topk]
            "expert_shard_type": 0,
            "shared_expert_rank_num": shared_expert_rank_num,
            "moe_expert_num": max_num_deployed_expert,  # ENABLE_OMNI_PLANNER, 0 redundancy 256, 1 redundancy expert 320
            "global_bs": global_bs,  # 0 Default (all); you can set all tokens
        }
        tp_recv_counts = output[5]
        stage3_kwargs = {
            "ep_send_counts": ep_recv_counts,  # dispatch's send_counts
            "group_ep": moe_all_to_all_group_name,  # Unlike torch, it is obtained by name.
            "ep_world_size": all_to_all_group_size,
            "ep_rank_id": global_rank // experts_tp_size,
            "tp_send_counts": tp_recv_counts,
            "group_tp": moe_rs_group_name,
            "tp_world_size": experts_tp_size,
            "tp_rank_id": global_rank % experts_tp_size,
            "x_active_mask": mc2_mask,
        }
        kwargs.update(stage3_kwargs)

        hidden_states_route = torch_npu.npu_moe_distribute_combine_v2(**kwargs)
    else:
        raise ValueError("ep number should be greater than 1.")
    return hidden_states_route
