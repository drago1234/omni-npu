# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any, Callable, Dict, List, Optional
import os
import torch
import torch_npu
from vllm.logger import init_logger
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    ChannelQuantScaleParameter
)

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig

from omni_npu.v1.layers.linear import (
    FlashCommLinearMethodBase
)

from omni_npu.v1.utils import ACL_FORMAT_FRACTAL_NZ

from omni_npu.v1.distributed.communication_op_ext import (
    layer_parallel_all_gather,
    layer_parallel_all2all_single
)

# from omni_npu.models.config_loader.loader import model_extra_config

logger = init_logger(__name__)

class W8A8Int8FCLinearMethod(FlashCommLinearMethodBase):
    """FlashComm Linear method for NPU W8A8.
    
    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: CompressedTensorsConfig):
        self.quant_config = quant_config

    def create_weights(self,
                       layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int],
                       input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype,
                       **extra_weight_attrs):
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get('weight_loader')

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        weight_dtype = torch.int8

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=weight_dtype
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader
        )
        layer.register_parameter('weight', weight)

        logger.info('NpuW8A8LinearMethod params_dtype=%s', params_dtype)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty(sum(output_partition_sizes), dtype=params_dtype),
            output_dim=0,
            weight_loader=weight_loader
        )
        layer.register_parameter('weight_scale', weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_data = torch_npu.npu_format_cast(layer.weight.data.t().contiguous(), ACL_FORMAT_FRACTAL_NZ)
        layer.weight = torch.nn.Parameter(weight_data, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(layer.weight_scale.data.view(-1), requires_grad=False)

    def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
            x_transform: Optional[str] = None,
            x_dim: Optional[int] = 0,
            is_prefill: Optional[bool] = True
    ) -> torch.Tensor:
        x, x_scale = torch_npu.npu_dynamic_quant(x, smooth_scales=None)

        # TODO scale_parallel is not supported yet. scale_parallel = model_extra_config.operator_opt_config.enable_scale_parallel
        if x_transform == 'AllGather':
            x_scale = layer_parallel_all_gather(x_scale, layer.layer_name_inside_block, "x", x_dim)
            x = layer_parallel_all_gather(x, layer.layer_name_inside_block, "x", x_dim)
        elif x_transform == 'All2All':
            x_scale = layer_parallel_all2all_single(x_scale, layer.layer_name_inside_block, "x", x_dim)
            x = layer_parallel_all2all_single(x, layer.layer_name_inside_block, "x", x_dim)
        y = torch_npu.npu_quant_matmul(
            x1=x,
            x2=layer.weight,
            scale=layer.weight_scale,
            pertoken_scale=x_scale,
            bias=bias,
            output_dtype=layer.orig_dtype
        )
        return y