# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import List, Optional, Tuple, Dict, Union

import torch
import torch_npu

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
from vllm.model_executor.parameter import ChannelQuantScaleParameter, ModelWeightParameter

from omni_npu.v1.distributed.communication_op_ext import layer_parallel_all2all_single, layer_parallel_all_gather
from omni_npu.v1.fused_mlp.layer import FusedMLPMethodBase
from omni_npu.v1.layers.linear import (
    FlashCommLinearMethodBase,
    layer_parallel_communication_op
)
from omni_npu.v1.layers.utils import get_npu_execution_type
from omni_npu.v1.utils import ACL_FORMAT_FRACTAL_NZ


logger = init_logger(__name__)


class W8A8Int8FCLinearMethod(FlashCommLinearMethodBase):
    """FlashComm Linear method for NPU W8A8.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: CompressedTensorsConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        weight_dtype = torch.int8

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=weight_dtype
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        logger.debug("NpuW8A8LinearMethod params_dtype=%s", params_dtype)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty(sum(output_partition_sizes), dtype=params_dtype),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        weight_offset = ChannelQuantScaleParameter(
            data=torch.empty(sum(output_partition_sizes), dtype=params_dtype),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_offset", weight_offset)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_data = torch_npu.npu_format_cast(
            layer.weight.data.t().contiguous(), ACL_FORMAT_FRACTAL_NZ
        )
        layer.weight = torch.nn.Parameter(weight_data, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            layer.weight_scale.data.view(-1), requires_grad=False
        )
        if hasattr(layer, 'weight_offset'):
            layer.weight_offset = torch.nn.Parameter(
                layer.weight_offset.data.view(-1).float(), requires_grad=False
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        bias: Optional[torch.Tensor] = None,
        x_transform: Optional[str] = None,
        x_dim: Optional[int] = 0,
        is_prefill: Optional[bool] = True,
    ) -> torch.Tensor:
        if isinstance(x, Dict):
            x_scale = x.get('pertoken_scale',None)
            x = x.get('x_int8',None)
        else:
            x, x_scale = torch_npu.npu_dynamic_quant(x)
        # TODO scale_parallel is not supported yet. scale_parallel = model_extra_config.operator_opt_config.enable_scale_parallel
        if x_transform == "AllGather":
            x_scale = layer_parallel_all_gather(
                x_scale, layer.layer_name_inside_block, "x", x_dim
            )
            x = layer_parallel_all_gather(x, layer.layer_name_inside_block, "x", x_dim)
        elif x_transform == "ALL2ALL":
            x_scale = layer_parallel_all2all_single(
                x_scale, layer.layer_name_inside_block, "x", x_dim
            )
            x = layer_parallel_all2all_single(
                x, layer.layer_name_inside_block, "x", x_dim
            )
        y = torch_npu.npu_quant_matmul(
            x1=x,
            x2=layer.weight,
            scale=layer.weight_scale,
            pertoken_scale=x_scale,
            bias=bias,
            output_dtype=layer.orig_dtype,
        )
        return y

class W8A8Int8MlpMethod(FusedMLPMethodBase):
    """Apply dequant_swiglu_quant fused kernel.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.gate_up_proj.weight_scale = torch.nn.Parameter(
            layer.gate_up_proj.weight_scale.data.float(), requires_grad=False
        )

    def apply_quant(
        self,
        x: torch.Tensor,
        x_transform: str = None,
        is_prefill: bool = True,
        stream_label: Optional[str | torch.npu.Stream] = None,
    ):
        with get_npu_execution_type(stream_label):
            x, x_scale = torch_npu.npu_dynamic_quant(x, smooth_scales=None)
        return x, x_scale

    def apply_part1_gate_up_on_stream(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        x_scale: torch.Tensor,
        stream_label: Optional[str | torch.npu.Stream] = None,
    ) -> torch.Tensor:
        # 多流设置
        with get_npu_execution_type(stream_label):
            y_int32 = torch_npu.npu_quant_matmul(
                x1=x,
                x2=layer.gate_up_proj.weight,
                scale=layer.gate_up_proj.weight_scale,
                pertoken_scale=x_scale,
                bias=None,
                output_dtype=layer.gate_up_proj.orig_dtype,
            )

        return y_int32

    def apply_part2_activation_on_stream(
        self,
        layer: torch.nn.Module,
        y_int32: torch.Tensor,
        x_scale: torch.Tensor,
        stream_label: Optional[str | torch.npu.Stream] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bias = layer.gate_up_proj.bias if not layer.gate_up_proj.skip_bias_add else None

        with get_npu_execution_type(stream_label):
            int_int32, int_scale = torch_npu.npu_dequant_swiglu_quant(
                y_int32,
                weight_scale=layer.gate_up_proj.weight_scale,
                activation_scale=x_scale,
                bias=bias,
                activate_left=True,
                quant_mode=1,
            )

        return int_int32, int_scale

    def apply_part3_down_on_stream(
        self,
        layer: torch.nn.Module,
        int_int32: torch.Tensor,
        int_scale: torch.Tensor,
        stream_label: Optional[str | torch.npu.Stream] = None,
    ) -> torch.Tensor:
        bias = (
            None
            if (layer.down_proj.tp_rank > 0 or layer.down_proj.skip_bias_add)
            else layer.down_proj.bias
        )
        with get_npu_execution_type(stream_label):
            output = torch_npu.npu_quant_matmul(
                x1=int_int32,
                x2=layer.down_proj.weight,
                scale=layer.down_proj.weight_scale,
                pertoken_scale=int_scale,
                bias=bias,
                output_dtype=layer.down_proj.orig_dtype,
            )

        return output

    def _layer_parallel_apply_x_transform(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        x_scale: torch.Tensor,
        x_transform: Optional[str] = None,
        x_dim: Optional[int] = 0,
        layer_name_inside_block: Optional[str] = None,
    ):
        # TODO scale_parallel is not supported yet. scale_parallel = model_extra_config.operator_opt_config.enable_scale_parallel
        if x_transform == "AllGather":
            x_scale = layer_parallel_all_gather(
                x_scale, layer_name_inside_block, "x", x_dim
            )
            x = layer_parallel_all_gather(x, layer.layer_name_inside_block, "x", x_dim)
        elif x_transform == "ALL2ALL":
            x_scale = layer_parallel_all2all_single(
                x_scale, layer_name_inside_block, "x", x_dim
            )
            x = layer_parallel_all2all_single(
                x, layer_name_inside_block, "x", x_dim
            )
        return x, x_scale

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        stream_label: Optional[str | torch.npu.Stream] = None,
    ) -> torch.Tensor:
        x, x_scale = self.apply_quant(x, layer.gate_up_proj.x_transform, stream_label)
        x, x_scale = self._layer_parallel_apply_x_transform(
            layer, x, x_scale, layer.gate_up_proj.x_transform, layer.gate_up_proj.x_dim,layer.gate_up_proj.layer_name_inside_block
        )
        y_gate_up_int32 = self.apply_part1_gate_up_on_stream(
            layer, x, x_scale, stream_label
        )
        y_gate_up_int32 = layer_parallel_communication_op(
            y_gate_up_int32,
            layer.gate_up_proj.y_transform,
            layer.gate_up_proj.layer_name_inside_block,
            "y",
            layer.gate_up_proj.y_dim,
        )
        int_int32, int_scale = self.apply_part2_activation_on_stream(
            layer, y_gate_up_int32, x_scale, stream_label
        )
        int_int32,int_scale = self._layer_parallel_apply_x_transform(
            layer,
            int_int32,
            int_scale,
            layer.down_proj.x_transform,
            layer.down_proj.x_dim,
            layer.down_proj.layer_name_inside_block,
        )
        output = self.apply_part3_down_on_stream(
            layer, int_int32, int_scale, stream_label
        )
        return layer_parallel_communication_op(
            output,
            layer.down_proj.y_transform,
            layer.down_proj.layer_name_inside_block,
            "y",
            layer.down_proj.y_dim,
        )