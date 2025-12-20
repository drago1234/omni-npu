# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Optional, Tuple

import torch
import torch_npu

from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    ExpertTokensMetadata,
    FusedMoEActivationFormat,
    FusedMoEPermuteExpertsUnpermute,
    TopKWeightAndReduce
)
from vllm.platforms import current_platform


class NPUFusedMoEPermuteExpertsUnpermute(FusedMoEPermuteExpertsUnpermute):
    def __init__(
        self,
        quant_config: FusedMoEQuantConfig,
        layer: torch.nn.Module
    ):
        super().__init__(quant_config)
        self.local_expert_num = len(layer.w13_weight)
        if self.quant_config.use_int8_w8a8:
            self.scale_2 = torch.ones((len(layer.w13_weight), layer.w13_weight_scale.shape[-1] // 2), dtype=torch.float32,
                                      device=current_platform.device_type)
        else:
            self.scale_2 = None

    @property
    def activation_formats(
        self,
    ) -> Tuple[FusedMoEActivationFormat, FusedMoEActivationFormat]:
        return (
            FusedMoEActivationFormat.Standard,
            FusedMoEActivationFormat.Standard,
        )

    def supports_chunking(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return True

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        assert w1.dim() == 3 and w2.dim() == 3
        E, N, _ = w1.size()
        K = a1.size(-1)
        M = topk_ids.size(0)

        assert topk_ids.dim() == 2
        topk = topk_ids.size(1)

        return E, M, N, K, topk

    def finalize_weight_and_reduce_impl(self) -> TopKWeightAndReduce:
        return None

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: Optional[ExpertTokensMetadata],
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], torch.dtype]:
        workspace13 = (M * min(topk, self.local_expert_num) * get_ep_group().world_size, N)
        workspace2 = (M * min(topk, self.local_expert_num) * get_ep_group().world_size, N // 2)
        output = (M * min(topk, self.local_expert_num) * get_ep_group().world_size, K)
        return (workspace13, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: Optional[ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
    ):
        if expert_tokens_meta is None:
            raise ValueError("expert_tokens_meta is required for NPU MoE")

        group_list = expert_tokens_meta.expert_num_tokens.to(torch.int64)
        if self.quant_config.use_int8_w8a8:
            # w8a8
            gate_up_proj = torch_npu.npu_grouped_matmul(
                [hidden_states],
                [w1], 
                bias=None,
                group_list=group_list,
                split_item=3,
                output_dtype=torch.int32,
                group_type=0,
                group_list_type=1)[0]
            gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
                gate_up_proj,
                weight_scale=self.quant_config.w1_scale,
                activation_scale=a1q_scale,
                bias=None,
                quant_offset=None,
                quant_scale=self.scale_2,
                group_index=group_list,
                activate_left=True,
                quant_mode=1)
            hidden_states_experts = torch_npu.npu_grouped_matmul(
                [gate_up_proj],
                [w2],
                scale=[self.quant_config.w2_scale.to(torch.bfloat16)],
                per_token_scale=[pertoken_scale],
                bias=None,
                group_list=group_list,
                split_item=3,
                output_dtype=torch.bfloat16,
                group_type=0,
                group_list_type=1)[0]
        else:
            # bf16
            gate_up_proj = torch_npu.npu_grouped_matmul(
                [hidden_states],
                [w1], 
                bias=None,
                group_list=group_list,
                split_item=3,
                group_type=0,
                group_list_type=1)[0]
            gate_up_proj = torch_npu.npu_swiglu(gate_up_proj)
            hidden_states_experts = torch_npu.npu_grouped_matmul(
                [gate_up_proj],
                [w2],
                bias=None,
                group_list=group_list,
                split_item=3,
                output_dtype=torch.bfloat16,
                group_type=0,
                group_list_type=1)[0]

        output.copy_(hidden_states_experts)