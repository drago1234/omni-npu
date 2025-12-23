# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Optional, Callable
import torch, torch_npu
from vllm.logger import init_logger
from vllm.distributed import (
    get_ep_group,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_gather,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)
from vllm.platforms import current_platform
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPermuteExpertsUnpermute,
    FusedMoEPrepareAndFinalize
)
from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod,
)
from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE
from omni_npu.layers.fused_moe.fused_moe import fused_experts_tp, moe_infer_fusion
from omni_npu.layers.fused_moe.npu_moe_prepare_finalize import NpuMoEPrepareAndFinalize
from omni_npu.layers.fused_moe.npu_moe_permute_unpermute import NPUFusedMoEPermuteExpertsUnpermute
torch.npu.config.allow_internal_format = True
logger = init_logger(__name__)


@UnquantizedFusedMoEMethod.register_oot
class NPUUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def forward_oot(
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
    ) -> torch.Tensor:
        tp_size = get_tensor_model_parallel_world_size()  # attn tp size
        hidden_states = x
        if layer.moe_parallel_config.use_ep and tp_size > 1:
            tp_rank = get_tensor_model_parallel_rank()
            t_ori = x.shape[0]
            t_pad = -(t_ori // -tp_size) * tp_size
            t_local = t_pad // tp_size
            num_pads = t_pad - t_ori
            # pad
            hidden_states = torch.nn.functional.pad(x, (0, 0, 0, num_pads), value=0)
            router_logits = torch.nn.functional.pad(router_logits, (0, 0, 0, num_pads), value=0)
            # deduplicate
            hidden_states = hidden_states[tp_rank * t_local: (tp_rank + 1) * t_local]
            router_logits = router_logits[tp_rank * t_local: (tp_rank + 1) * t_local]

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

        if self.moe.moe_parallel_config.use_ep:
            share_output = None
            if layer.shared_experts is not None:
                share_output = layer.shared_experts(x)
            attn_metadata = get_forward_context().attn_metadata
            use_all2all = attn_metadata is None or attn_metadata[next(iter(attn_metadata))].num_prefills > 0
            if use_all2all:
                output = moe_infer_fusion(
                    layer=layer,
                    x=hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids
                )
            else:
                output = self.fused_experts(
                    hidden_states=hidden_states,
                    w1=layer.w13_weight,
                    w2=layer.w2_weight,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    inplace=True,
                    activation=activation,
                    apply_router_weight_on_input=apply_router_weight_on_input,
                    global_num_experts=global_num_experts,
                    expert_map=expert_map,
                )
            if tp_size > 1:
                output = tensor_model_parallel_all_gather(output, dim=0)[:t_ori]
                if share_output is not None:
                    share_output = tensor_model_parallel_all_reduce(share_output)
            if share_output is not None:
                return share_output, output
            return output
        else:
            return fused_experts_tp(
                layer=layer,
                x=x,
                topk_ids=topk_ids,
                topk_weights=topk_weights
            )

    def maybe_make_prepare_finalize(self, routing_tables) -> Optional[FusedMoEPrepareAndFinalize]:
        return NpuMoEPrepareAndFinalize(self.moe)

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        return NPUFusedMoEPermuteExpertsUnpermute(self.moe_quant_config, layer)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        layer.w13_weight = torch.nn.Parameter(layer.w13_weight.transpose(1, 2).contiguous(), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(layer.w2_weight.transpose(1, 2).contiguous(), requires_grad=False)
        layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, 29)
        layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, 29)

    def gmm_expert(self, layer, h, expert_tokens, dynamic_scale=None, avg_tokens_per_expert=None):
        group_list_type = int(layer.moe_parallel_config.use_ep)
        mm1_mm3 = torch_npu.npu_grouped_matmul([h], [layer.w13_weight],
                                               group_list=expert_tokens, split_item=3, group_type=0,
                                               group_list_type=group_list_type)[0]
        intermediate_h = torch_npu.npu_swiglu(mm1_mm3)
        out_hidden = torch_npu.npu_grouped_matmul(
            [intermediate_h],
            [layer.w2_weight],
            bias=None,
            group_list=expert_tokens,
            split_item=3,
            group_type=0,
            group_list_type=group_list_type
        )[0]
        return out_hidden


@FusedMoE.register_oot
class NPUFusedMoE(FusedMoE):
    def maybe_all_reduce_tensor_model_parallel(self, final_hidden_states: torch.Tensor):
        """With NPU all-to-all, there is no need to perform all-reduce on final hidden states.
        """
        if self.moe_parallel_config.use_ep:
            return final_hidden_states

        return tensor_model_parallel_all_reduce(final_hidden_states)

    def maybe_init_modular_kernel(self) -> None:
        self.ensure_moe_quant_config_init()
        routing_tables = self._maybe_init_expert_routing_tables()
        prepare_finalize = self.quant_method.maybe_make_prepare_finalize(
            routing_tables=routing_tables
        )
        if prepare_finalize is not None:
            logger.debug(
                "%s for %s(%s)", prepare_finalize.__class__.__name__, self, id(self)
            )
            self.quant_method = FusedMoEModularMethod.make(
                self, self.quant_method, prepare_finalize, None
            )

    @staticmethod
    def select_experts(
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
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)

        return topk_weights, topk_ids, row_idx

@SharedFusedMoE.register_oot
class NPUSharedFusedMoE(SharedFusedMoE, NPUFusedMoE):
    pass


@FusedMoEModularMethod.register_oot
class NPUFusedMoEModularMethod(FusedMoEModularMethod):

    def __init__(self, old_quant_method, experts):
        super().__init__(old_quant_method, experts)
        self.old_quant_method.fused_experts = self.fused_experts

    def apply(self, *args, **kwargs):
        return self.old_quant_method.apply(*args, **kwargs)

    def gmm_expert(self, *args, **kwargs):
        return self.old_quant_method.gmm_expert(*args, **kwargs)
