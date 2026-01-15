# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Optional, Callable, Union
from abc import ABC
import torch
import torch_npu
from vllm.platforms import current_platform
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import int8_w8a8_moe_quant_config, FusedMoEQuantConfig
from omni_npu.layers.fused_moe.layer import NPUFusedMoE
from omni_npu.layers.quantization.compressed_tensors.compressed_tensors_moe import NPUCompressedTensorsW8A8Int8MoEMethod
from omni_npu.v1.layers.fused_moe.fused_moe_prepare_permute_unpermute_finalize import (
    FusedMoEPreparePermuteAndUnpermuteFinalize,
    PreparePermuteResult
)


class NPUFusedMoEMethodBase(ABC):
    def __init__(self):
        self.prefill_prepare_permute_and_unpermute_finalize = None
        self.decode_prepare_permute_and_unpermute_finalize = None

    def set_prepare_permute_and_unpermute_finalize(
        self,
        prefill_prepare_permute_and_unpermute_finalize: FusedMoEPreparePermuteAndUnpermuteFinalize,
        decode_prepare_permute_and_unpermute_finalize: FusedMoEPreparePermuteAndUnpermuteFinalize,
    ):
        self.prefill_prepare_permute_and_unpermute_finalize = prefill_prepare_permute_and_unpermute_finalize
        self.decode_prepare_permute_and_unpermute_finalize = decode_prepare_permute_and_unpermute_finalize

    def apply_prepare_permute(
        self,
        prepare_permute_and_unpermute_finalize: FusedMoEPreparePermuteAndUnpermuteFinalize,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        return prepare_permute_and_unpermute_finalize.prepare_permute(layer, x, topk_ids)

    def apply_experts(
        self,
        layer: torch.nn.Module,
        prepare_permute_result: PreparePermuteResult,
    ):
        raise NotImplementedError

    def apply_unpermute_finalize(
        self,
        prepare_permute_and_unpermute_finalize: FusedMoEPreparePermuteAndUnpermuteFinalize,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        prepare_permute_result: PreparePermuteResult,
    ):
        return prepare_permute_and_unpermute_finalize.unpermute_finalize(hidden_states, topk_ids, topk_weights, prepare_permute_result)


class NPUCompressedTensorsW8A8Int8MoEMethodV1(NPUCompressedTensorsW8A8Int8MoEMethod, NPUFusedMoEMethodBase):
    def __init__(self, parent, layer):
        NPUCompressedTensorsW8A8Int8MoEMethod.__init__(self, parent, layer)
        NPUFusedMoEMethodBase.__init__(self)
        self.scale_2 = torch.ones(
            (layer.local_num_experts, layer.intermediate_size_per_partition), 
            dtype=torch.float32,
            device=current_platform.device_type
        )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> Optional[FusedMoEQuantConfig]:
        return int8_w8a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=None,
            a2_scale=None,
            per_act_token_quant=True,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        topk_weights, topk_ids, _ = NPUFusedMoE.select_experts(
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
        )

        if enable_eplb:
            _, topk_ids, _ = self.planner.plan(
                layer_idx_moe=self.moe_layer_idx,
                tokens=x,
                token_expert_ids=topk_ids,
                token_expert_scores=topk_weights,
                top_k=top_k,
                expert_mapping=self.expert_mapping,
            )
            layer.planner = self.planner
            layer.moe_layer_idx = self.moe_layer_idx

        attn_metadata = get_forward_context().attn_metadata
        is_prefill = attn_metadata is None or attn_metadata[next(iter(attn_metadata))].num_prefills > 0

        if is_prefill:
            prepare_permute_and_unpermute_finalize = self.prefill_prepare_permute_and_unpermute_finalize
        else:
            prepare_permute_and_unpermute_finalize = self.decode_prepare_permute_and_unpermute_finalize

        prepare_permute_result = self.apply_prepare_permute(
            prepare_permute_and_unpermute_finalize, layer, x, topk_ids
        )

        if enable_eplb:
            self.planner.record_activation(self.moe_layer_idx, prepare_permute_result.expert_tokens, support_multi_stream=False)

        hidden_states = self.apply_experts(layer, prepare_permute_result)

        return self.apply_unpermute_finalize(
            prepare_permute_and_unpermute_finalize, hidden_states, topk_ids, topk_weights, prepare_permute_result
        )

    def apply_experts(self, layer: torch.nn.Module, prepare_permute_result: PreparePermuteResult) -> torch.Tensor:
        hidden_states = prepare_permute_result.hidden_states_sorted_by_experts
        expert_tokens = prepare_permute_result.expert_tokens
        avg_tokens_per_expert = prepare_permute_result.avg_tokens_per_expert or [0]
        pertoken_scale = prepare_permute_result.dynamic_scale
        if pertoken_scale.dim() > 1:
            pertoken_scale = pertoken_scale.reshape(-1)
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        gate_up_proj = torch_npu.npu_grouped_matmul(
            [hidden_states],
            [layer.w13_weight],
            bias=None,
            scale=None,
            per_token_scale=None,
            group_list=expert_tokens,
            split_item=3,
            output_dtype=torch.int32,
            group_type=0,
            group_list_type=1)[0]

        intermediate_hidden_states, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
            gate_up_proj,
            weight_scale=layer.w13_weight_scale,
            activation_scale=pertoken_scale,
            bias=None,
            quant_offset=None,
            quant_scale=self.scale_2,
            group_index=expert_tokens,
            activate_left=True,
            quant_mode=1)

        hidden_states_experts = torch_npu.npu_grouped_matmul(
            [intermediate_hidden_states],
            [layer.w2_weight],
            scale=[layer.w2_weight_scale.to(torch.bfloat16)],
            per_token_scale=[pertoken_scale],
            bias=None,
            group_list=expert_tokens,
            split_item=3,
            output_dtype=torch.bfloat16,
            group_type=0,
            group_list_type=1,
            tuning_config=avg_tokens_per_expert)[0]
        
        return hidden_states_experts
