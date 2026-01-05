# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any

import torch
import torch_npu
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.layernorm import RMSNorm


@RMSNorm.register_oot
class NPURMSNorm(RMSNorm):
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        quant_symbol: bool = False,
        y_transform: str = "",
    ) -> torch.Tensor | tuple[torch.Tensor | dict[str, Any], torch.Tensor]:
        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
            if y_transform == "AG":
                x = get_tp_group().all_gather(x, dim=0)
            if quant_symbol:
                x_int8, pertoken_scale = torch_npu.npu_dynamic_quant(x)
                x = {"x_int8": x_int8, "pertoken_scale": pertoken_scale}
            return x, residual

        return torch_npu.npu_rms_norm(
            x,
            self.weight.data,
            self.variance_epsilon,
        )[0]