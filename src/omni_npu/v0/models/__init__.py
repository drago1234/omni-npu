# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import torch_npu
from vllm import ModelRegistry

def register_model():
    is_A2 = torch_npu.npu.get_device_name(0).startswith("Ascend910B")
    ModelRegistry.register_model(
        "Qwen3MoeForCausalLM",
        "omni_npu.v0.models.qwen.qwen3_moe:Qwen3MoeForCausalLM")


