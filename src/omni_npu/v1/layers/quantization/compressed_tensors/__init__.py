# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from .npu_compressed_tensors_linear import W8A8Int8FCLinearMethod, W8A8Int8MlpMethod
from .npu_compressed_tensors_moe import NPUCompressedTensorsW8A8Int8MoEMethodV1

__all__ = [
    "W8A8Int8FCLinearMethod",
    "W8A8Int8MlpMethod",
    "NPUCompressedTensorsW8A8Int8MoEMethodV1",
]

