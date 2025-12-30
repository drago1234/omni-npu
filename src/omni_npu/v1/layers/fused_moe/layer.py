# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank
)
from omni_npu.layers.fused_moe.layer import NPUSharedFusedMoE
from omni_npu.v1.layers.fused_moe.fused_moe_prepare_permute_unpermute_finalize import (
    DispatchCombinePrepPmtAndUnpmtFinal,
    All2AllPrepPmtAndUnpmtFinal
)


class NPUHighPerfFusedMoE(NPUSharedFusedMoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ensure_moe_quant_config_init()
        self.make_prepare_permute_and_unpermute_finalize()

    def make_prepare_permute_and_unpermute_finalize(self):
        prefill_prepare_permute_and_unpermute_finalize = All2AllPrepPmtAndUnpmtFinal(self)
        decode_prepare_permute_and_unpermute_finalize = DispatchCombinePrepPmtAndUnpmtFinal(self)
        self.quant_method.set_prepare_permute_and_unpermute_finalize(
            prefill_prepare_permute_and_unpermute_finalize=prefill_prepare_permute_and_unpermute_finalize,
            decode_prepare_permute_and_unpermute_finalize=decode_prepare_permute_and_unpermute_finalize,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ):
        router_logits, _ = self.gate(hidden_states)

        share_expert_output = self.shared_experts(hidden_states)

        tp_size = get_tensor_model_parallel_world_size()  # attn tp size
        x = hidden_states
        if tp_size > 1:
            tp_rank = get_tensor_model_parallel_rank()
            t_ori = hidden_states.shape[0]
            t_pad = -(t_ori // -tp_size) * tp_size
            t_local = t_pad // tp_size
            num_pads = t_pad - t_ori
            # pad
            x = torch.nn.functional.pad(hidden_states, (0, 0, 0, num_pads), value=0)
            router_logits = torch.nn.functional.pad(router_logits, (0, 0, 0, num_pads), value=0)
            # deduplicate
            x = x[tp_rank * t_local: (tp_rank + 1) * t_local]
            router_logits = router_logits[tp_rank * t_local: (tp_rank + 1) * t_local]

        route_expert_output = self.quant_method.apply(
            layer=self,
            x=x,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map if not self.rocm_aiter_fmoe_enabled else self.expert_mask,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            enable_eplb=self.enable_eplb,
            expert_load_view=self.expert_load_view,
            logical_to_physical_map=self.logical_to_physical_map,
            logical_replica_count=self.logical_replica_count
        )

        if tp_size > 1:
            route_expert_output = tensor_model_parallel_all_gather(route_expert_output, dim=0)[:t_ori]

        return route_expert_output, share_expert_output
