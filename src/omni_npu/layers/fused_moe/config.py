from dataclasses import dataclass

import torch
import torch_npu

from vllm.model_executor.layers.fused_moe.config import (FusedMoEQuantConfig, _quant_flags_to_group_shape, FusedMoEQuantDesc)


def use_int4_w4a8(self) -> bool:
    return self.quant_dtype == torch.int8 and self._w1.dtype == "int4"


def use_int8_w8a8(self) -> bool:
    return self.quant_dtype == torch.int8 and self._w1.dtype == torch.int8


FusedMoEQuantConfig.use_int4_w4a8 = property(use_int4_w4a8)
FusedMoEQuantConfig.use_int8_w8a8 = property(use_int8_w8a8)


def int4_w4a8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    a1_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    per_act_token_quant: bool = False,
):
    """
    Construct a quant config for int8 activations and int4 weights.
    """
    quant_dtype = torch.int8
    weight_dtype = "int4"
    a_shape, w_shape = _quant_flags_to_group_shape(
        quant_dtype, per_act_token_quant, False, None
    )
    quant_config = FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(quant_dtype, a_shape, a1_scale),
        _a2=FusedMoEQuantDesc(quant_dtype, a_shape, a2_scale),
        _w1=FusedMoEQuantDesc(weight_dtype, w_shape, w1_scale, None, None, w1_bias),
        _w2=FusedMoEQuantDesc(weight_dtype, w_shape, w2_scale, None, None, w2_bias),
    )
    assert quant_config.per_act_token_quant == per_act_token_quant
    return quant_config