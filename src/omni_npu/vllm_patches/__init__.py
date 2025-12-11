# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# vllm_patches Reference: https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html


import importlib
import pkgutil
from .patch_manager import PatchManager

def auto_import_patches():
    from . import patches
    for _, module_name, _ in pkgutil.iter_modules(path=patches.__path__):
        importlib.import_module(f"omni_npu.vllm_patches.patches.{module_name}")

    # examples
    # for _, module_name, _ in pkgutil.iter_modules(path=patches.examples.__path__):
    #     importlib.import_module(f"omni_npu.vllm_patches.patches.examples.{module_name}")

manager = PatchManager()

def apply_patches():
    # auto import and register patches
    auto_import_patches()

    manager.apply_patches()
