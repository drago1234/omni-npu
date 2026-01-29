# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# vllm_patches Reference: https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html
import importlib.util
import json
import logging
import sys
from pathlib import Path

from omni_npu.vllm_patches import patches

from .patch_manager import PatchManager

try:
    MODEL_PATH = Path(sys.argv[2])
except IndexError:
    raise EnvironmentError("The model path must be passed as the second parameter through the command line.")
logger = logging.getLogger(__name__)

def import_patches_from_dir(root: Path, base_pkg: str):
    """
    Imports all.py files in the specified directory and sorts them by file name.
    """
    py_files = sorted(root.rglob("*.py"), key=lambda p: p.name)

    for py_file in py_files:
        if py_file.name == "__init__.py":
            continue

        rel_path = py_file.relative_to(root).with_suffix("")
        module_name = ".".join((base_pkg, *rel_path.parts))

        if module_name in sys.modules:
            continue

        spec = importlib.util.spec_from_file_location(module_name, py_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)


def get_model_type_from_config(model_path: Path) -> str:
    """
    read model_type from config.json
    """
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_type = config.get("model_type")
    if not model_type:
        raise ValueError("config.json 中缺少 model_type 字段")

    return model_type


def find_patch_dir_for_model(model_type: str, models_root: Path) -> Path:
    """
    Map to the specific directory under patches/models based on model_type.
    Supports:
    - Mapping table
    - Prefix matching
    - containment match
    """
    MODEL_PATCH_MAP = {
        "deepseek_v3": "deepseek",
        "deepseek_v32": "deepseek",
        "qwen3": "qwen",
    }

    patch_dir_name = MODEL_PATCH_MAP.get(model_type.lower())
    if patch_dir_name:
        candidate = models_root / patch_dir_name
        if candidate.exists():
            return candidate

    model_type_lower = model_type.lower()
    for subdir in models_root.iterdir():
        # Skip files, only process directories
        if not subdir.is_dir():
            continue

        subdir_name_lower = subdir.name.lower()
        # Match condition 1: model_type starts with subdirectory name (prefix match)
        # Match condition 2: subdirectory name is contained in model_type (containment match)
        if (model_type_lower.startswith(subdir_name_lower)
                or subdir_name_lower in model_type_lower):
            return subdir

    logger.warning(
        f"No patch directory found for model_type '{model_type}' in {models_root}"
    )
    return None


def auto_import_patches():
    """
    Automatically load patches:
        1. common directory
        2. model-specific directory (mapped according to model_type)
    The files within each directory are sorted by filename.
    """
    patches_root = Path(patches.__file__).parent
    base_pkg = patches.__name__

    common_dir = patches_root / "common"
    if common_dir.exists():
        import_patches_from_dir(common_dir, f"{base_pkg}.common")

    model_type = get_model_type_from_config(MODEL_PATH)
    models_root = patches_root / "models"
    model_dir = find_patch_dir_for_model(model_type, models_root)

    if model_dir and model_dir.exists():
        import_patches_from_dir(model_dir, f"{base_pkg}.models.{model_dir.name}")



manager = PatchManager()


def apply_patches():
    # auto import and register patches
    auto_import_patches()

    manager.apply_patches()