# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank
)
from omni_npu.layers.fused_moe.layer import NPUSharedFusedMoE
from omni_npu.v1.layers.prefetch import PrefetcherBase
from omni_npu.v1.models.config_loader.loader import model_extra_config
from omni_npu.v1.layers.fused_moe.fused_moe_prepare_permute_unpermute_finalize import (
    DispatchCombinePrepPmtAndUnpmtFinal,
    All2AllPrepPmtAndUnpmtFinal
)


class NPUFusedMoEV1(NPUSharedFusedMoE, PrefetcherBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        PrefetcherBase.__init__(self)
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
        if model_extra_config.operator_opt_config.enable_prefetch and self.shared_experts is not None:
            prefetch_moe_fn = getattr(self, "prefetch_moe", None)
            if callable(prefetch_moe_fn):
                prefetch_moe_fn(trigger=hidden_states,
                                prefetch_experts=False,
                                prefetch_shared_experts=True)

        tp_size = get_tensor_model_parallel_world_size()  # attn tp size
        x = hidden_states
        if tp_size > 1:
            tp_rank = get_tensor_model_parallel_rank()
            t_ori = x.shape[0]
            t_pad = -(t_ori // -tp_size) * tp_size
            t_local = t_pad // tp_size
            num_pads = t_pad - t_ori
            if num_pads > 0:
                x = torch.nn.functional.pad(x, (0, 0, 0, num_pads), value=0)
            x = x[tp_rank * t_local: (tp_rank + 1) * t_local]

        if model_extra_config.operator_opt_config.enable_prefetch:
            prefetch_moe_fn = getattr(self, "prefetch_moe", None)
            if callable(prefetch_moe_fn):
                prefetch_moe_fn(trigger=x,
                                prefetch_experts=True,
                                prefetch_shared_experts=False)

        expert_output = self.quant_method.apply(
            layer=self,
            x=x,
            gate=self.gate,
            shared_experts=self.shared_experts,
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
            expert_output = tensor_model_parallel_all_gather(expert_output, dim=0)[:t_ori]

        if model_extra_config.operator_opt_config.enable_prefetch:
            self.prefetch_attention(trigger=hidden_states)
        return None, expert_output
