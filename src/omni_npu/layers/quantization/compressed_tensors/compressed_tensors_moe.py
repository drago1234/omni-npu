# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
from typing import Callable, Optional
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch_npu

from vllm.logger import init_logger
from vllm.attention import AttentionMetadata
from vllm.platforms import current_platform
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import CompressedTensorsMoEMethod
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEQuantConfig, int8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.layer import FusedMoeWeightScaleSupported

from vllm.distributed import get_ep_group, GroupCoordinator, get_world_group, get_tensor_model_parallel_world_size

from omni_npu.layers.fused_moe.layer import moe_infer_fusion, fused_experts, fused_experts_moe_dispatch_combine
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
        return None

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
        # torch._dynamo.mark_static(self.smooth_scale)

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
    ) -> torch.Tensor:
        max_num_deployed_expert_per_rank = self.n_routed_experts
        if self.moe.moe_parallel_config.use_ep:

            tp_size = get_tensor_model_parallel_world_size()
            if tp_size > 1:
                from vllm.distributed import get_tp_group
                tp_rank = get_tp_group().rank
                t_ori = x.shape[0]
                t_pad = -(t_ori // -tp_size) * tp_size
                t_local = t_pad // tp_size
                num_pads = t_pad - t_ori
                # pad
                x = F.pad(x, (0, 0, 0, num_pads), value=0)
                router_logits = F.pad(router_logits, (0, 0, 0, num_pads), value=0)
                # deduplicate
                x = x[tp_rank * t_local:(tp_rank+1) * t_local]
                router_logits = router_logits[tp_rank*t_local:(tp_rank+1)*t_local]

                from omni_npu.layers.fused_moe.layer import AscendFusedMoE
                topk_weights, topk_ids, row_idx = AscendFusedMoE.select_experts(
                    x,
                    router_logits,
                    top_k=top_k,
                    use_grouped_topk=use_grouped_topk,
                    renormalize=renormalize,
                    topk_group=topk_group,
                    num_expert_group=num_expert_group,
                    custom_routing_function=custom_routing_function,
                    scoring_func=scoring_func,
                    routed_scaling_factor=routed_scaling_factor,
                    e_score_correction_bias=e_score_correction_bias
                )

            from vllm.forward_context import get_forward_context
            attn_metadata=get_forward_context().attn_metadata

            if attn_metadata is not None:
                attn_metadata = attn_metadata[next(iter(attn_metadata))]
            is_prefill = attn_metadata is None or getattr(attn_metadata, "prefill", None) is not None
            # if model_extra_config.operator_opt_config.prefill_moe_all_to_all or (model_extra_config.operator_opt_config.decode_moe_dispatch_combine and not is_prefill):
            # NOTE: Force prefill and decode use dispatch_combine cuz sth. wrong
            #  with a2a when compiling
            if is_prefill:
                out = moe_infer_fusion(
                    layer,
                    x,
                    topk_ids,
                    topk_weights,
                    comm_group=None
                )

            else:
                # logger.warning(f"<<< running in fused_experts_moe_dispatch_combine")
                out = fused_experts_moe_dispatch_combine(
                    layer,
                    x,
                    topk_weights,
                    topk_ids,
                    max_num_deployed_expert=max_num_deployed_expert_per_rank * get_ep_group().world_size,
                    is_prefill=is_prefill,
                    is_route_expert=True,
                )

            if tp_size > 1:
                out = get_tp_group().all_gather(out, dim=0)[:t_ori]
                out /= get_tp_group().world_size
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
            return fused_experts(layer, x, topk_ids, topk_weights)

    def gmm_expert(self, layer, h, expert_tokens, dynamic_scale=None, avg_tokens_per_expert=None):
        # no need to transpose weight here if weight_nz enabled
        hidden_size = h.size(-1)
        pertoken_scale = dynamic_scale
        if pertoken_scale.dim() > 1:
            pertoken_scale = pertoken_scale.reshape(-1)
            h = h.view(-1, hidden_size)

        if not layer.moe_parallel_config.use_ep:
            w1_scale = layer.w13_weight_scale.to(torch.bfloat16)
            w2_scale = layer.w2_weight_scale.to(torch.bfloat16)
            gate_up_proj = \
                torch_npu.npu_grouped_matmul([h], [layer.w13_weight], scale=[w1_scale],
                                             per_token_scale=[pertoken_scale],
                                             bias=None, group_list=expert_tokens, split_item=3,
                                             output_dtype=torch.bfloat16,
                                             group_type=0,
                                             group_list_type=0)[0]

            gate_up_proj = torch_npu.npu_swiglu(gate_up_proj)
            gate_up_proj, pertoken_scale = torch_npu.npu_dynamic_quant(gate_up_proj)  # , smooth_scales=scale_2)

            out_hidden = torch_npu.npu_grouped_matmul([gate_up_proj], [layer.w2_weight], scale=[w2_scale],
                                                      per_token_scale=[pertoken_scale],
                                                      bias=None, group_list=expert_tokens, split_item=3,
                                                      output_dtype=torch.bfloat16,
                                                      group_type=0,
                                                      group_list_type=0)[0]
        else:
            if layer.weight_num_bits != 8 and layer.weight_num_bits != 4:
                raise NotImplementedError(f"Unsupported compress tensor type. num bits: {layer.weight_num_bits}")

            avg_tokens_per_expert = avg_tokens_per_expert or [0]
            bias = None if layer.weight_num_bits == 8 else [layer.w13_weight_bias]
            scale = None if layer.weight_num_bits == 8 else [layer.w13_weight_int4_scale]
            per_token_scale = None if layer.weight_num_bits == 8 else [pertoken_scale]
            output_dtype = torch.int32 if layer.weight_num_bits == 8 else torch.bfloat16

            mm1_mm3 = torch_npu.npu_grouped_matmul([h], [layer.w13_weight],
                                                   bias=bias, scale=scale, per_token_scale=per_token_scale,
                                                   group_list=expert_tokens, split_item=3, group_type=0,
                                                   group_list_type=1, act_type=0, output_dtype=output_dtype)[0]

            weight_scale = layer.w13_weight_scale if layer.weight_num_bits == 8 else \
                torch.ones(layer.w13_weight_int4_scale.shape, dtype=torch.float32,
                           device="npu").view(-1, layer.w13_weight_int4_scale.shape[-1])
            pertoken_scale = pertoken_scale.squeeze(0) if layer.weight_num_bits == 8 else \
                torch.ones(pertoken_scale.shape, dtype=torch.float32, device="npu")
            # dequant_swiglu_quant
            quant_scale = torch.ones((expert_tokens.shape[0], layer.w13_weight_scale.shape[-1] // 2),
                                     dtype=torch.float32, device=current_platform.device_type)
            intermediate_h, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(mm1_mm3, weight_scale=weight_scale,
                                                                                activation_scale=pertoken_scale,
                                                                                bias=None, quant_scale=quant_scale,
                                                                                quant_offset=None,
                                                                                group_index=expert_tokens,
                                                                                activate_left=True, quant_mode=1)

            if pertoken_scale.dim() > 1:
                inter_size = intermediate_h.size(-1)
                pertoken_scale = pertoken_scale.reshape(-1)
                intermediate_h = intermediate_h.view(-1, inter_size)
            # gmm2: down
            w2_scale = [layer.w2_weight_scale.to(torch.bfloat16)] if layer.weight_num_bits == 8 else \
                [layer.w2_weight_int4_scale]
            out_hidden = torch_npu.npu_grouped_matmul([intermediate_h], [layer.w2_weight], bias=None,
                                                      scale=w2_scale, per_token_scale=[pertoken_scale],
                                                      group_list=expert_tokens, split_item=3,
                                                      output_dtype=torch.bfloat16, group_type=0,
                                                      group_list_type=1, tuning_config=avg_tokens_per_expert)[0]

        return out_hidden

    def moe_expert_quant_forward(self, layer, sorted_tokens, expert_tokens, act_dtype, dynamic_scale=None):
        pertoken_scale = dynamic_scale

        if layer.weight_num_bits == 8:
            gate_up_proj = torch_npu.npu_grouped_matmul([sorted_tokens], [layer.w13_weight], bias=None, group_list=expert_tokens,
                                            split_item=3, output_dtype=torch.int32, group_type=0, group_list_type=1)[0]

            scale_2 = torch.ones((len(layer.w13_weight), layer.w13_weight_scale.shape[-1] // 2), dtype=torch.float32,
                                device=current_platform.device_type)
            gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
                gate_up_proj, weight_scale=layer.w13_weight_scale, activation_scale=pertoken_scale, bias=None,
                quant_scale=scale_2, quant_offset=None,
                group_index=expert_tokens, activate_left=True, quant_mode=1)

            w2_scale = layer.w2_weight_scale.to(torch.bfloat16)

            out = torch_npu.npu_grouped_matmul([gate_up_proj], [layer.w2_weight], scale=[w2_scale],
                                            per_token_scale=[pertoken_scale], bias=None,
                                            group_list=expert_tokens, split_item=3, output_dtype=act_dtype,
                                            group_type=0,
                                            group_list_type=1)[0]
            return out
        elif layer.weight_num_bits == 4:  #NOTE(Zuo Yuqi) Lazy Implemention: INT4 Quantization is not supported now.
            raise NotImplementedError(f"Unsupported INT4 compress tensor type.")
        else:
            raise NotImplementedError(f"Unsupported compress tensor type. num bits: {layer.weight_num_bits}")
