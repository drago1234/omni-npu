# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
from typing import Optional
import torch
import torch.distributed as dist
import torch_npu

from vllm.logger import init_logger
from vllm.attention import AttentionMetadata
from vllm.platforms import current_platform
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import CompressedTensorsMoEMethod
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEQuantConfig, int8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.layer import FusedMoeWeightScaleSupported

from vllm.distributed import get_ep_group, GroupCoordinator, get_world_group
# from omni.layers.moe.fused_moe.fused_moe import (
#     fused_experts_moe_dispatch_combine,
#     moe_infer_fusion,
#     fused_experts_allgather_ep_a3,
#     fused_experts_allgather_ep_a2
# )


SEQ_SPLIT_LENGTH = 4096
# torch.npu.config.allow_internal_format = True
logger = init_logger("vllm.omni_npu.layers.quantization.compressed_tensors.compressed_tensors_moe")


class AscendCompressedTensorsW8A8Int8MoEMethod(CompressedTensorsMoEMethod):
    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.warm_up = True
        self.n_routed_experts = None
        self.smooth_scale = None

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig | None:
        return int8_w8a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            per_act_token_quant=True
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts,
                        2 * intermediate_size_per_partition,
                        hidden_size,
                        dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts,
                        hidden_size,
                        intermediate_size_per_partition,
                        dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})

        w13_scale = torch.nn.Parameter(
            torch.ones(num_experts,
                       2 * intermediate_size_per_partition,
                       dtype=torch.float32
                       if params_dtype == torch.float16 else torch.bfloat16),
            requires_grad=False,
        )
        w13_offset = torch.nn.Parameter(
            torch.zeros(num_experts,
                        2 * intermediate_size_per_partition,
                        dtype=torch.float32
                        if params_dtype == torch.float16 else torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        layer.register_parameter("w13_weight_offset", w13_offset)
        set_weight_attrs(w13_scale, extra_weight_attrs)
        set_weight_attrs(w13_offset, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(
            torch.ones(num_experts,
                       hidden_size,
                       dtype=torch.float32
                       if params_dtype == torch.float16 else torch.bfloat16),
            requires_grad=False,
        )
        w2_offset = torch.nn.Parameter(
            torch.zeros(num_experts,
                        hidden_size,
                        dtype=torch.float32
                        if params_dtype == torch.float16 else torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        layer.register_parameter("w2_weight_offset", w2_offset)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_offset, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight = torch.nn.Parameter(layer.w13_weight.transpose(1, 2).contiguous(), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(layer.w2_weight.transpose(1, 2).contiguous(), requires_grad=False)
        # if model_extra_config.operator_opt_config.gmm_nz:
        #     layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight, 29)
        #     layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight, 29)
        # if model_extra_config.operator_opt_config.pd_seperate_prefill:
        #     layer.w2_weight_scale = torch.nn.Parameter(layer.w2_weight_scale.to(torch.bfloat16), requires_grad=False)
        # elif not model_extra_config.operator_opt_config.opt_w2_scale_cast:
        #     layer.w2_weight_scale = torch.nn.Parameter(layer.w2_weight_scale.to(torch.float32), requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(layer.w13_weight_scale.to(torch.float32), requires_grad=False)
        self.n_routed_experts = len(layer.w13_weight)
        self.local_expert_indices_offset = (
            get_ep_group().rank_in_group * self.n_routed_experts
        )
        self.local_expert_indices = [
            self.local_expert_indices_offset + i for i in range(self.n_routed_experts)
        ]
        self.smooth_scale = torch.ones((self.n_routed_experts, layer.w13_weight_scale.shape[-1] // 2),
                                       dtype=torch.float32, device=current_platform.device_type)
        torch._dynamo.mark_static(self.smooth_scale)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        pertoken_scale: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        comm_group: Optional[GroupCoordinator] = None
    ) -> torch.Tensor:
        max_num_deployed_expert_per_rank = self.n_routed_experts
        if get_ep_group().world_size > 1:
            is_prefill = attn_metadata is None or attn_metadata.prefill is not None
            # if model_extra_config.operator_opt_config.prefill_moe_all_to_all or (model_extra_config.operator_opt_config.decode_moe_dispatch_combine and not is_prefill):
            if True:
                out = moe_infer_fusion(
                    layer,
                    x,
                    topk_ids,
                    topk_weights,
                    self.moe.num_experts,
                    self.warm_up,
                    is_prefill,
                    comm_group=comm_group,
                )
            else:
                out = fused_experts_moe_dispatch_combine(
                    layer,
                    x,
                    topk_weights,
                    topk_ids,
                    max_num_deployed_expert=max_num_deployed_expert_per_rank * get_ep_group().world_size,
                    is_prefill=is_prefill,
                    is_route_expert=True,
                )
            # else:
            #     if os.getenv("ASCEND_PLATFORM", "A3") == "A2":
            #         out = fused_experts_allgather_ep_a2(layer=layer,
            #                                              hidden_states=x,
            #                                              pertoken_scale=pertoken_scale,
            #                                              topk_weights=topk_weights,
            #                                              topk_ids=topk_ids,
            #                                              n_routed_experts=self.n_routed_experts,
            #                                              is_prefill=is_prefill,
            #                                              max_num_deployed_expert_per_rank=max_num_deployed_expert_per_rank,
            #                                              # ENABLE_OMNI_PLANNER
            #                                              smooth_scale=self.smooth_scale)
            #     else:
            #         out = fused_experts_allgather_ep_a3(layer=layer,
            #                                               hidden_states=x,
            #                                               pertoken_scale=pertoken_scale,
            #                                               topk_weights=topk_weights,
            #                                               topk_ids=topk_ids,
            #                                               n_routed_experts=self.n_routed_experts,
            #                                               is_prefill=is_prefill,
            #                                               max_num_deployed_expert_per_rank=max_num_deployed_expert_per_rank
            #                                               # ENABLE_OMNI_PLANNER
            #                                               )
            if self.warm_up:
                self.warm_up = False
            return out
        else:
            row_idx = torch.arange(topk_ids.numel(), device=current_platform.device_type,
                                   dtype=torch.int32).view(-1, x.shape[0]).transpose(0, 1)
            token_num = x.shape[0]
            # if token_num > SEQ_SPLIT_LENGTH:  # Split seq to reduce memory usage
            #     x_list = x.split(SEQ_SPLIT_LENGTH)
            #     topk_weights_list = topk_weights.split(SEQ_SPLIT_LENGTH)
            #     topk_ids_list = topk_ids.split(SEQ_SPLIT_LENGTH)
            #     out = []
            #     for i in range(len(x_list)):
            #         split_token, top_k = topk_weights_list[i].shape
            #         row_idx = torch.arange(split_token * top_k).to(torch.int32).view(
            #             (top_k, split_token)).T.contiguous().npu()
            #         out.append(fused_experts_w8a8(x_list[i],
            #                                       layer.w13_weight,
            #                                       layer.w2_weight,
            #                                       layer.w13_weight_scale,
            #                                       layer.w2_weight_scale,
            #                                       layer.w13_weight_offset,
            #                                       layer.w2_weight_offset,
            #                                       topk_weights_list[i],
            #                                       topk_ids_list[i],
            #                                       row_idx))
            #     return torch.concat(out)
            return fused_experts_w8a8(x,
                                      layer.w13_weight,
                                      layer.w2_weight,
                                      layer.w13_weight_scale,
                                      layer.w2_weight_scale,
                                      layer.w13_weight_offset,
                                      layer.w2_weight_offset,
                                      topk_weights,
                                      topk_ids,
                                      row_idx)


def fused_experts_w8a8(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w1_offset: torch.Tensor,
        w2_offset: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        row_idx: torch.Tensor,
    ):
    num_tokens, hidden_size = hidden_states.shape
    n_routed_experts = len(w1)
    sorted_tokens, expanded_src_to_dst_row, expanded_expert_idx = \
        torch_npu.npu_moe_init_routing(hidden_states, row_idx, topk_ids, num_tokens)
    expert_tokens = torch_npu.npu_moe_compute_expert_tokens(expanded_expert_idx, n_routed_experts).to(torch.int64)
    act_dtype = hidden_states.dtype
    w1_scale = w1_scale.to(torch.bfloat16)
    w2_scale = w2_scale.to(torch.bfloat16)
    sorted_tokens, pertoken_scale = torch_npu.npu_dynamic_quant(sorted_tokens)
    gate_up_proj = \
        torch_npu.npu_grouped_matmul([sorted_tokens], [w1], scale=[w1_scale], per_token_scale=[pertoken_scale],
                                     bias=None, group_list=expert_tokens, split_item=3, output_dtype=act_dtype,
                                     group_type=0,
                                     group_list_type=0)[0]

    gate_up_proj = torch_npu.npu_swiglu(gate_up_proj)
    gate_up_proj, pertoken_scale = torch_npu.npu_dynamic_quant(gate_up_proj)  # , smooth_scales=scale_2)

    out = torch_npu.npu_grouped_matmul([gate_up_proj], [w2], scale=[w2_scale], per_token_scale=[pertoken_scale],
                                       bias=None, group_list=expert_tokens, split_item=3, output_dtype=act_dtype,
                                       group_type=0,
                                       group_list_type=0)[0]
    out = out.float()
    return torch_npu.npu_moe_finalize_routing(out, None, None, None, topk_weights,
                                              expanded_src_to_dst_row, topk_ids).to(torch.bfloat16)


def moe_infer_fusion(
    layer,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    max_num_deployed_expert: int,
    warm_up: bool = False,
    is_prefill: bool = True,
    comm_group: GroupCoordinator = None,
):
    _, h = x.shape
    hidden_states = x.view(-1, h)
    topk_weight = topk_weight.to(x.dtype)
    # max_num_deployed_expert = layer.w13_weight.shape[0] * get_ep_group().world_size
    # if warm_up:
    #     # This is forced balancing, the goal is to reduce peak memory
    #     global_rank = get_world_group().rank_in_group
    #     step = hidden_states.shape[0] * 8  # topk 8 expert
    #     cur_topk_list = [
    #         (i + global_rank // 1) % max_num_deployed_expert for i in range(
    #             global_rank // 1 * step, (global_rank // 1 + 1) * step)]
    #     topk_ids = torch.Tensor(cur_topk_list).int().view(hidden_states.shape[0], -1).npu()
    # else:
    topk_ids = topk_ids.int()

    expert_range = [0, max_num_deployed_expert]
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
        quant_mode=1)

    tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
    group = comm_group.device_group if comm_group else None
    dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=group)

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
    gathered_tokens = expanded_x.new_empty(
        all_tokens.item(), expanded_x.shape[1]
    )
    gathered_pertoken_scale = pertoken_scale.new_empty(gathered_tokens.shape[0])

    dist.all_to_all_single(
        gathered_tokens,
        expanded_x,
        output_splits,
        input_splits,
        group=group
    )
    dist.all_to_all_single(
        gathered_pertoken_scale,
        pertoken_scale,
        output_splits,
        input_splits,
        group=group
    )

    # reroute
    # Tokens merged by experts, scales merged by experts, indices for FinalizeRouting, number of tokens processed by each expert
    hidden_states_sorted_by_experts, gathered_pertoken_scale, gathered_idxs_unsort, tokens_per_local_expert = torch_npu.npu_moe_re_routing(
        gathered_tokens,
        tokens_per_expert_group.view(ep_size, -1),
        per_token_scales=gathered_pertoken_scale
    )
    group_list = tokens_per_local_expert.to(torch.int64)
    hidden_states_ordered_by_experts = gmm_expert(
        layer,
        hidden_states_sorted_by_experts,
        tokens_per_local_expert.to(torch.int64),
        gathered_pertoken_scale,
        None,
    )

    new_x = torch.index_select(hidden_states_ordered_by_experts, 0,
                               gathered_idxs_unsort.to(torch.float32).argsort().to(torch.int32))
    gathered_tokens = new_x.new_empty(*expanded_x.shape)
    dist.all_to_all_single(
        gathered_tokens,
        new_x,
        input_splits,
        output_splits,
        group=group
    )
    return hidden_states, gathered_tokens, topk_weight, expanded_row_idx


def gmm_expert(layer, x, expert_tokens, dynamic_scale=None, avg_tokens_per_expert=None):
    # no need to transpose weight here if weight_nz enabled
    hidden_size = x.size(-1)
    h = x
    pertoken_scale = dynamic_scale

    if pertoken_scale.dim() > 1:
        pertoken_scale = pertoken_scale.reshape(-1)
        h = h.view(-1, hidden_size)
    # gmm1: gate_up
    avg_tokens_per_expert = avg_tokens_per_expert or [0]

    if layer.weight_num_bits == 8:
        mm1_mm3 = torch_npu.npu_grouped_matmul([h], [layer.w13_weight],
                                               group_list=expert_tokens, split_item=3,
                                               output_dtype=torch.int32, group_type=0,
                                               group_list_type=1, tuning_config=avg_tokens_per_expert)[0]
        # dequant_swiglu_quant
        scale_2 = torch.ones((expert_tokens.shape[0], layer.w13_weight_scale.shape[-1] // 2), dtype=torch.float32,
                             device=current_platform.device_type)
        intermediate_h, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
            mm1_mm3, weight_scale=layer.w13_weight_scale,
            activation_scale=pertoken_scale.squeeze(0), bias=None, quant_scale=scale_2, quant_offset=None,
            group_index=expert_tokens, activate_left=True, quant_mode=1)

        if pertoken_scale.dim() > 1:
            inter_size = intermediate_h.size(-1)
            pertoken_scale = pertoken_scale.reshape(-1)
            intermediate_h = intermediate_h.view(-1, inter_size)
        # gmm2: down
        out_dtype = torch.bfloat16
        w2_scale = layer.w2_weight_scale.to(torch.bfloat16)
        out_hidden = torch_npu.npu_grouped_matmul([intermediate_h], [layer.w2_weight], bias=None,
                                                  scale=[w2_scale], per_token_scale=[pertoken_scale],
                                                  group_list=expert_tokens, split_item=3,
                                                  output_dtype=out_dtype, group_type=0,
                                                  group_list_type=1, tuning_config=avg_tokens_per_expert)[0]
        return out_hidden
    elif layer.weight_num_bits == 4:
        mm1_mm3 = torch_npu.npu_grouped_matmul([h], [layer.w13_weight], bias=[layer.w13_weight_bias], scale=[layer.w13_weight_int4_scale],
                                               offset=None, antiquant_scale=None, antiquant_offset=None,
                                               per_token_scale=[pertoken_scale],
                                               group_list=expert_tokens,
                                               activation_input=None, activation_quant_scale=None,
                                               activation_quant_offset=None, split_item=3, group_type=0,
                                               group_list_type=1, act_type=0, output_dtype=torch.bfloat16)[0]

        fake_scale = torch.ones(layer.w13_weight_int4_scale.shape, dtype=torch.float32, device="npu").view(-1, layer.w13_weight_int4_scale.shape[-1])
        pertoken_scale = torch.ones(pertoken_scale.shape, dtype=torch.float32, device="npu")
        intermediate_h, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(mm1_mm3, weight_scale=fake_scale,
                                                                            activation_scale=pertoken_scale,
                                                                            bias=None, quant_scale=None,
                                                                            quant_offset=None,
                                                                            group_index=expert_tokens,
                                                                            activate_left=True, quant_mode=1)

        if pertoken_scale.dim() > 1:
            inter_size = intermediate_h.size(-1)
            pertoken_scale = pertoken_scale.reshape(-1)
            intermediate_h = intermediate_h.view(-1, inter_size)

        out_dtype = torch.bfloat16
        out_hidden = torch_npu.npu_grouped_matmul([intermediate_h], [layer.w2_weight], bias=[layer.w2_weight_bias],
                                                  scale=[layer.w2_weight_int4_scale], per_token_scale=[pertoken_scale],
                                                  group_list=expert_tokens, split_item=3,
                                                  output_dtype=out_dtype, group_type=0,
                                                  group_list_type=1, tuning_config=avg_tokens_per_expert)[0]
        return out_hidden
    else:
        raise NotImplementedError(f"Unsupported compress tensor type. num bits: {layer.weight_num_bits}")
