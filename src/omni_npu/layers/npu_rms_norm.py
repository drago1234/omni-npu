# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
 
import torch
import torch_npu
from vllm.model_executor.layers.layernorm import RMSNorm


@RMSNorm.register_oot
class NPURMSNorm(RMSNorm):
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        quant_symbol=False
    ) -> torch.Tensor:
        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
            if quant_symbol:
                x_int8, pertoken_scale = torch_npu.npu_dynamic_quant(x)
                x = {"x_int8": x_int8, "pertoken_scale": pertoken_scale}
            return x, residual

        return torch_npu.npu_rms_norm(
            x,
            self.weight.data,
            self.variance_epsilon,
        )[0]