# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Optional, Callable, Union

import torch
import torch_npu

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.forward_context import get_forward_context
from vllm.model_executor.utils import set_weight_attrs
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import CompressedTensorsW8A8Int8MoEMethod
from vllm.model_executor.layers.fused_moe.layer import FusedMoeWeightScaleSupported
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPermuteExpertsUnpermute,
    FusedMoEPrepareAndFinalize,
)
from vllm.distributed import get_world_group
from vllm.config import get_current_vllm_config

from omni_npu.layers.fused_moe.npu_moe_prepare_finalize import NpuMoEPrepareAndFinalize
from omni_npu.layers.fused_moe.npu_moe_permute_unpermute import NPUFusedMoEPermuteExpertsUnpermute
from omni_npu.layers.fused_moe.layer import NPUFusedMoE
from omni_npu.layers.fused_moe.fused_moe import moe_infer_fusion, fused_experts_tp, fused_experts_allgather_ep
from omni_npu.layers.fused_moe.config import int4_w4a8_moe_quant_config


torch.npu.config.allow_internal_format = True
logger = init_logger(__name__)


class NPUCompressedTensorsW8A8Int8MoEMethod(CompressedTensorsW8A8Int8MoEMethod):
    def __init__(
        self,
        parent,
        layer,
    ):
        super().__init__(parent, layer.moe_config)
        self.init_eplb(layer)
        device_name = torch_npu.npu.get_device_name(0)
        self.is_a2_device = device_name.startswith("Ascend910B")

    def init_eplb(self, layer):
        self.enable_eplb = layer.enable_eplb
        self.n_routed_experts = layer.moe_config.num_experts
        self.prefix = layer.layer_name
        self.vllm_config = get_current_vllm_config().model_config.hf_config
        self.num_of_redundant_experts = 0
        if self.enable_eplb:
            from omni_placement.omni_planner import OmniPlanner
            self.planner = OmniPlanner(config_file=None,
                            device="npu",
                            rank=get_world_group().rank_in_group,
                            world_size=get_world_group().world_size,
                            num_experts=self.n_routed_experts,
                            num_redundancy_shared_expert_rank=0)
            self.moe_layer_idx = OmniPlanner.get_deepseek_v3_moe_layer_idx(self.prefix, self.vllm_config.first_k_dense_replace)
            self.expert_mapping = self.planner.expert_mapping_on_current_layer(self.moe_layer_idx)
            self.num_of_redundant_experts = self.planner.get_num_of_redundant_experts(moe_layer_idx=self.moe_layer_idx,
                                                                                      num_expert_per_device_origin=layer.local_num_experts,
                                                                                      rank_device=get_world_group().rank_in_group)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        params_dtype = torch.int8
        num_experts = num_experts + self.num_of_redundant_experts
        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        assert not self.static_input_scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

        # WEIGHT_OFFSETS
        w13_weight_offset = torch.nn.Parameter(
            torch.zeros(num_experts,
                        2 * intermediate_size_per_partition,
                        1,
                        dtype=torch.float32
                        if params_dtype == torch.float16 else torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_offset", w13_weight_offset)
        w2_weight_offset = torch.nn.Parameter(
            torch.zeros(num_experts,
                        hidden_size,
                        1,
                        dtype=torch.float32
                        if params_dtype == torch.float16 else torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_offset", w2_weight_offset)
        set_weight_attrs(w13_weight_offset, extra_weight_attrs)
        set_weight_attrs(w2_weight_offset, extra_weight_attrs)
        
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(num_experts,
                            2 * intermediate_size_per_partition,
                            dtype=torch.bfloat16),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts,
                            hidden_size,
                            dtype=torch.bfloat16),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight = torch.nn.Parameter(layer.w13_weight.transpose(1, 2).contiguous(), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(layer.w2_weight.transpose(1, 2).contiguous(), requires_grad=False)

        layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, 29)
        layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, 29)

        layer.w13_weight_scale = torch.nn.Parameter(layer.w13_weight_scale.to(torch.float32).squeeze(-1), requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(layer.w2_weight_scale.to(torch.bfloat16).squeeze(-1), requires_grad=False)

    def maybe_make_prepare_finalize(self, routing_tables) -> Optional[FusedMoEPrepareAndFinalize]:
        return NpuMoEPrepareAndFinalize(self.moe)

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        return NPUFusedMoEPermuteExpertsUnpermute(self.moe_quant_config, layer)

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
        use_allgather_ep = self.is_a2_device and layer.moe_parallel_config.use_ep
        tp_size = get_tensor_model_parallel_world_size()  # attn tp size
        hidden_states = x
        global_num_experts = global_num_experts + get_world_group().world_size * self.num_of_redundant_experts

        if not use_allgather_ep and layer.moe_parallel_config.use_ep and tp_size > 1:
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

        topk_weights, topk_ids = NPUFusedMoE.select_experts(
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
        if use_allgather_ep:
            output = fused_experts_allgather_ep(
                layer=layer,
                x=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                is_prefill=is_prefill,
            )
            output = tensor_model_parallel_all_reduce(output)
            return output
        elif layer.moe_parallel_config.use_ep:
            share_output = None
            if layer.shared_experts is not None:
                share_output = layer.shared_experts(x)
            if is_prefill:
                output = moe_infer_fusion(
                    layer=layer,
                    x=hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
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
                if enable_eplb:
                    group_list = self.fused_experts.prepare_finalize.expert_token_nums
                    self.planner.record_activation(self.moe_layer_idx, group_list, support_multi_stream=False)

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
                topk_weights=topk_weights,
            )

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
            avg_tokens_per_expert = avg_tokens_per_expert or [0]
            bias = None
            scale = None
            per_token_scale = None
            output_dtype = torch.int32
            mm1_mm3 = torch_npu.npu_grouped_matmul([h], [layer.w13_weight],
                                                   bias=bias, scale=scale, per_token_scale=per_token_scale,
                                                   group_list=expert_tokens, split_item=3, group_type=0,
                                                   group_list_type=1, act_type=0, output_dtype=output_dtype)[0]

            weight_scale = layer.w13_weight_scale
            pertoken_scale = pertoken_scale.squeeze(0)
            # dequant_swiglu_quant
            quant_scale = torch.ones((expert_tokens.shape[0], weight_scale.shape[-1] // 2),
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
            w2_scale = [layer.w2_weight_scale.to(torch.bfloat16)]
            out_hidden = torch_npu.npu_grouped_matmul([intermediate_h], [layer.w2_weight], bias=None,
                                                      scale=w2_scale, per_token_scale=[pertoken_scale],
                                                      group_list=expert_tokens, split_item=3,
                                                      output_dtype=torch.bfloat16, group_type=0,
                                                      group_list_type=1, tuning_config=avg_tokens_per_expert)[0]

        return out_hidden

    @property
    def supports_eplb(self) -> bool:
        return True


class NPUCompressedTensorsW4A8Int4MoEMethod(NPUCompressedTensorsW8A8Int8MoEMethod):
    LAST_SEQ_LEN = None
    BEST_EXPERT_TOKENS = None

    def __init__(
        self,
        parent,
        layer,
    ):
        super().__init__(parent, layer)
        STORAGE_BITS_NPU = 8
        WEIGHT_BITS = 4
        self.warm_up = True
        self.n_total_experts = None
        self.pack_factor = STORAGE_BITS_NPU // WEIGHT_BITS
        self.smooth_scale = None

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        # Will transpose the loaded weight along the
        # intermediate and hidden dim sizes. Will
        # shard for TP along the transposed dims
        extra_weight_attrs.update({
            "is_transposed": False,
            "quant_method": FusedMoeWeightScaleSupported.CHANNEL.value
        })
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                    2 * intermediate_size_per_partition // self.pack_factor,
                                                    hidden_size,
                                                    dtype=torch.int8),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                   hidden_size // self.pack_factor,
                                                   intermediate_size_per_partition,
                                                   dtype=torch.int8),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_int4_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                                1,
                                                                2 * intermediate_size_per_partition,
                                                                dtype=torch.int64),
                                                    requires_grad=False)

        w2_weight_int4_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                                1,
                                                                hidden_size,
                                                                dtype=torch.int64),
                                                    requires_grad=False)

        w13_weight_bias = torch.nn.Parameter(torch.ones(num_experts,
                                                        2 * intermediate_size_per_partition,
                                                        dtype=torch.float32),
                                             requires_grad=False)

        w2_weight_bias = torch.nn.Parameter(torch.ones(num_experts,
                                                       hidden_size,
                                                       dtype=torch.float32),
                                            requires_grad=False)

        # INPUT_SCALES
        assert not self.static_input_scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

        layer.register_parameter("w13_weight_int4_scale", w13_weight_int4_scale)
        layer.register_parameter("w13_weight_bias", w13_weight_bias)
        set_weight_attrs(w13_weight_int4_scale, extra_weight_attrs)
        set_weight_attrs(w13_weight_bias, extra_weight_attrs)

        layer.register_parameter("w2_weight_int4_scale", w2_weight_int4_scale)
        layer.register_parameter("w2_weight_bias", w2_weight_bias)
        set_weight_attrs(w2_weight_int4_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight = torch.nn.Parameter(layer.w13_weight.transpose(1, 2).contiguous(), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(layer.w2_weight.transpose(1, 2).contiguous(), requires_grad=False)

        layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight, 29)            
        layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight, 29)

        layer.w13_weight.data = layer.w13_weight.data.view(torch.int32).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.view(torch.int32).contiguous()

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ):
        return int4_w4a8_moe_quant_config(
            w1_scale=layer.w13_weight_int4_scale,
            w2_scale=layer.w2_weight_int4_scale,
            w1_bias=layer.w13_weight_bias,
            w2_bias=layer.w2_weight_bias,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            per_act_token_quant=True,
        )

    def gmm_expert(self, layer, h, expert_tokens, dynamic_scale=None, avg_tokens_per_expert=None):
        # no need to transpose weight here if weight_nz enabled
        hidden_size = h.size(-1)
        pertoken_scale = dynamic_scale
        if pertoken_scale.dim() > 1:
            pertoken_scale = pertoken_scale.reshape(-1)
            h = h.view(-1, hidden_size)

        assert layer.moe_parallel_config.use_ep, "W4A8 only support ep"

        avg_tokens_per_expert = avg_tokens_per_expert or [0]
        tuning_config = None
        
        mm1_mm3 = torch_npu.npu_grouped_matmul([h], [layer.w13_weight], bias=[layer.w13_weight_bias], scale=[layer.w13_weight_int4_scale],
                                               offset=None, antiquant_scale=None, antiquant_offset=None,
                                               per_token_scale=[pertoken_scale],
                                               group_list=expert_tokens,
                                               activation_input=None, activation_quant_scale=None,
                                               activation_quant_offset=None, split_item=3, group_type=0,
                                               group_list_type=1, act_type=0, output_dtype=torch.bfloat16,
                                               tuning_config=tuning_config)[0]

        # dequant_swiglu_quant
        fake_scale = torch.ones(layer.w13_weight_int4_scale.shape, dtype=torch.float32, device="npu").view(-1, layer.w13_weight_int4_scale.shape[-1])
        pertoken_scale = torch.ones(pertoken_scale.shape, dtype=torch.float32, device="npu")
        intermediate_h, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(mm1_mm3, weight_scale=fake_scale,
                                                                            activation_scale=pertoken_scale,
                                                                            bias=None, quant_scale=None,
                                                                            quant_offset=None,
                                                                            group_index=expert_tokens,
                                                                            activate_left=True, quant_mode=1)

        out_hidden = torch_npu.npu_grouped_matmul([intermediate_h], [layer.w2_weight], bias=[layer.w2_weight_bias],
                                                  scale=[layer.w2_weight_int4_scale], per_token_scale=[pertoken_scale],
                                                  group_list=expert_tokens, split_item=3,
                                                  output_dtype=torch.bfloat16, group_type=0,
                                                  group_list_type=1, tuning_config=tuning_config)[0]

        return out_hidden
