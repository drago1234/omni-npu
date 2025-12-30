from typing import Optional
import torch
import torch.distributed as dist
import torch_npu
from vllm.platforms import current_platform
from vllm.distributed import get_ep_group

def fused_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    topk_weights, topk_ids, row_idx = torch_npu.npu_moe_gating_top_k_softmax(gating_output, k=topk)

    if renormalize:
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids, row_idx


def grouped_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None
):
    gating_output = gating_output.float()
    # scores = torch.softmax(gating_output, dim=-1)
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias.unsqueeze(0)
    num_token = scores.shape[0]
    group_scores = scores.view(num_token, num_expert_group,
                               -1).max(dim=-1).values  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1,
                           sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.shape[-1] // num_expert_group).reshape(num_token, -1)  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    topk_weights, topk_ids = torch.topk(tmp_scores,
                                        k=topk,
                                        dim=-1,
                                        sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    topk_ids = topk_ids.int()
    # adapt add row_idx
    row_idx = torch.arange(topk_ids.numel(), device=topk_ids.device, dtype=topk_ids.dtype)
    row_idx = row_idx.reshape(topk_ids.shape[1], topk_ids.shape[0]).transpose(1, 0).contiguous()
    # adapt end

    return topk_weights, topk_ids, row_idx


def fused_experts_allgather_ep_unquant(
    layer: torch.nn.Module,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
):
    batch_size, hidden_size = x.shape
    x = x.view(-1, hidden_size)
    n_total_expert = layer.global_num_experts

    experts_start_idx = layer.ep_rank * layer.local_num_experts  # ENABLE_OMNI_PLANNER
    experts_end_idx = experts_start_idx + layer.local_num_experts
    expert_range = [experts_start_idx, experts_end_idx]
    sorted_tokens, expanded_x_idx, expert_tokens, _ = torch_npu.npu_moe_init_routing_v2(
        x, topk_ids, active_num=topk_ids.numel(), expert_capacity=-1,
        expert_num=n_total_expert, drop_pad_mode=0, expert_tokens_num_type=1, expert_tokens_num_flag=True,
        quant_mode=-1, active_expert_range=expert_range, row_idx_type=1)
    token_sum = torch.sum(expert_tokens)

    gmm_out = layer.quant_method.gmm_expert(
        layer,
        sorted_tokens,
        expert_tokens,
        None
    )
    gmm_out[token_sum:].zero_()
    temp = torch.argsort(expanded_x_idx.to(torch.float), dim=-1).to(torch.int32)

    output = torch_npu.npu_moe_finalize_routing(
        gmm_out,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=temp,
        export_for_source_row=None,
        drop_pad_mode=2
    )
    return output


def fused_experts_allgather_ep(
    layer: torch.nn.Module,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    share_experts_output: torch.Tensor = None
):
    topk_weights = topk_weights.to(torch.float)
    batch_size, hidden_size = x.shape
    x = x.view(-1, hidden_size)
    n_total_expert = layer.global_num_experts
    experts_start_idx = layer.ep_rank * layer.local_num_experts  # ENABLE_OMNI_PLANNER
    experts_end_idx = experts_start_idx + layer.local_num_experts
    expert_range = [experts_start_idx, experts_end_idx]
    x, pertoken_scale = torch_npu.npu_dynamic_quant(x)

    output_dtype = torch.int32
    sorted_tokens, expanded_x_idx, expert_tokens, dynamic_quant_scale = torch_npu.npu_moe_init_routing_v2(
        x, topk_ids, scale=pertoken_scale, offset=None, active_num=topk_ids.numel(), expert_capacity=-1,
        expert_num=n_total_expert, drop_pad_mode=0, expert_tokens_num_type=1, expert_tokens_num_flag=True,
        quant_mode=-1, active_expert_range=expert_range, row_idx_type=1)

    range1 = torch.arange(0, expanded_x_idx.shape[0], dtype=torch.int32, device="npu")
    range2 = range1 * torch.tensor(991, dtype=torch.int32, device="npu")
    mask = (range1 >= torch.sum(expert_tokens)).to(torch.int32)
    expanded_x_idx += range2 * mask
    expanded_x_idx = expanded_x_idx % expanded_x_idx.shape[0]
    expanded_x_idx = torch.clamp(expanded_x_idx, min=0, max=expanded_x_idx.shape[0] - 1)
    sorted_topk_weight = torch.index_select(topk_weights.reshape(-1), 0, expanded_x_idx)
    row_index = expanded_x_idx // topk_ids.shape[-1]
    row_index = row_index.to(torch.int64)

    if share_experts_output is None:
        share_experts_output = torch.zeros((batch_size // layer.dp_size, hidden_size), dtype=torch.bfloat16,
                                           device=current_platform.device_type)
    gate_up_proj = torch_npu.npu_grouped_matmul([sorted_tokens], [layer.w13_weight], bias=None,
                                                group_list=expert_tokens,
                                                split_item=3, output_dtype=output_dtype, group_type=0,
                                                group_list_type=1)[0]
    quant_scale = torch.ones((layer.local_num_experts, layer.w13_weight_scale.shape[-1] // 2), dtype=torch.float32,
                                device=current_platform.device_type)
    gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(gate_up_proj,
                                                                      weight_scale=layer.w13_weight_scale,
                                                                      activation_scale=dynamic_quant_scale,
                                                                      bias=None,
                                                                      quant_scale=quant_scale,
                                                                      quant_offset=None,
                                                                      group_index=expert_tokens,
                                                                      activate_left=True,
                                                                      quant_mode=1)
    output = torch_npu.npu_grouped_matmul_finalize_routing(gate_up_proj,
                                                           layer.w2_weight,
                                                           expert_tokens,
                                                           scale=layer.w2_weight_scale.to(torch.float),
                                                           pertoken_scale=pertoken_scale,
                                                           shared_input=share_experts_output,
                                                           logit=sorted_topk_weight,
                                                           row_index=row_index,
                                                           output_bs=batch_size,
                                                           shared_input_weight=1.0,
                                                           group_list_type=1,
                                                           shared_input_offset=0).to(torch.bfloat16)
    return output


def fused_experts_tp(
    layer: torch.nn.Module,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
):
    row_idx = torch.arange(topk_ids.numel(), device=current_platform.device_type,
                           dtype=torch.int32).view(-1, x.shape[0]).transpose(0, 1)
    num_tokens, hidden_size = x.shape
    n_routed_experts = layer.global_num_experts
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


def moe_infer_fusion(
    layer: torch.nn.Module,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
):
    x = x.view(-1, x.shape[-1])
    max_num_deployed_expert = layer.w13_weight.shape[0] * get_ep_group().world_size
    topk_ids = topk_ids.int()

    expert_range = [0, max_num_deployed_expert]
    quant_mode = 1 if layer.quant_config is not None else -1
    expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
        x,
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
    dist.all_to_all_single(tokens_per_expert_group,
                           tokens_per_expert)  # (total_experts,) --> (total_ranks * n_routed_experts_per_rank)

    # combine tensors, do reduceSum and D2H toghter
    combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
    # view: EP, E//EP
    # sum: EP, the number of tokens each rank receives from other cards
    combine_tokens = combine_tokens.view(2, layer.ep_size, -1).sum(2)
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
    group_list = tokens_per_local_expert.to(torch.int64)

    if layer.enable_eplb:
        layer.planner.record_activation(layer.moe_layer_idx, group_list, support_multi_stream=False)

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
        scales=topk_weights.to(gathered_tokens.dtype),
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=None,
        drop_pad_mode=2
    )
    return hidden_states