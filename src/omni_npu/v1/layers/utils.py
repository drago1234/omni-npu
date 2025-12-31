# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import logging

import torch
import torch_npu
import torchair
from contextlib import nullcontext

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

def get_npu_execution_type(stream_label):
    if stream_label is None:
        logger.info("entering default stream")
        return nullcontext()
    # Using strings to determine whether to include an item in the image, and later we will use logical differentiation based on parameters.
    elif isinstance(stream_label, str):
        return torchair.scope.npu_stream_switch(stream_label)  # Graph GE/ACL
    elif isinstance(stream_label,torch.npu.Stream):
        return torch.npu.stream(stream_label)  # eager
    return nullcontext()

def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0