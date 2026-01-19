# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# vllm_patches Reference: https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html

import importlib.util
import sys
from pathlib import Path

from omni_npu.vllm_patches import patches

from .patch_manager import PatchManager


def auto_import_patches():
    """
    Recursively import all patch modules under patches/,
    except patches/examples/*
    """

    root = Path(patches.__file__).parent
    base_pkg = patches.__name__

    for py_file in root.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue

        if "examples" in py_file.parts:
            continue

        rel_path = py_file.relative_to(root).with_suffix("")
        module_name = ".".join((base_pkg, *rel_path.parts))

        if module_name in sys.modules:
            continue

        spec = importlib.util.spec_from_file_location(module_name, py_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)


manager = PatchManager()


def apply_patches():
    # auto import and register patches
    auto_import_patches()

    manager.apply_patches()