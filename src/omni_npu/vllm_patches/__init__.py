# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# vllm_patches Reference: https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html
import importlib.util
import json
import logging
import sys
import os
from pathlib import Path

from omni_npu.vllm_patches import patches

from .patch_manager import PatchManager

logger = logging.getLogger(__name__)

def get_model_type_from_args():
    try:
        model_type = sys.argv[2]
        if not model_type:
            raise ValueError("Command-line argument sys.argv[2] cannot be empty")
        return model_type
    except IndexError:
        raise ValueError("Model type not provided. Please ensure sys.argv[2] is passed")

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
        raise ValueError("model_type field is missing in config.json")

    return model_type


def _find_patch_dir_exact(model_type: str, models_root: Path):
    """
    Exact matching: Strictly match the lowercase model type with lowercase subdirectory name.
    Applicable: User manually sets OMNI_NPU_PATCHES_DIR environment variable.
    """
    model_type_lower = model_type.lower()
    for subdir in models_root.iterdir():
        if not subdir.is_dir():
            continue

        subdir_name_lower = subdir.name.lower()
        if subdir_name_lower == model_type_lower:
            logger.info(f"Exact match succeeded:'{model_type}'->'{subdir.name}'")
            return subdir

    logger.warning(f"Exact match failed: No directory for '{model_type}' in {models_root}")
    return None


def _find_patch_dir_fuzzy(model_type: str, models_root: Path) -> Path:
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
        logger.info(f"model_type is: {model_type}, subdir name is: {subdir_name_lower}")
        if (model_type_lower.startswith(subdir_name_lower)
                or subdir_name_lower in model_type_lower):
            return subdir

    logger.warning(
        f"No patch directory found for model_type '{model_type}' in {models_root}."
    )
    return None


def auto_import_patches():
    """
    Automatically load patches:
        1. common directory
        2. model-specific directory (mapped according to model_type)
    The files within each directory are sorted by filename.
    """
    # 1. Basic initialization
    patches_root = Path(patches.__file__).parent
    base_pkg = patches.__name__
    models_root = patches_root / "models"
    env_var_name = "OMNI_NPU_PATCHES_DIR"

    # 2. Load common patches first
    common_dir = patches_root / "common"
    if common_dir.exists():
        import_patches_from_dir(common_dir, f"{base_pkg}.common")

    # 3. Determine model type & whether it's user-manual env (NO extra marks)
    current_env_value = os.getenv(env_var_name)
    is_user_manual_env = False
    model_type = current_env_value

    if not model_type:
        # Case 1: No env var → Get from config & set env for subsequent calls
        model_path = Path(get_model_type_from_args())
        model_type = get_model_type_from_config(model_path)
        os.environ[env_var_name] = model_type  # Auto-set for recursive calls
    else:
        # Case 2: Env var exists → It's user-manual (not set by current process)
        is_user_manual_env = True

    # 4. Find model directory (Direct branch call, NO strategy assignment)
    model_dir = None
    if is_user_manual_env:
        # User manual env → Call exact match directly
        model_dir = _find_patch_dir_exact(model_type, models_root)
    else:
        # Auto-set / no env → Call fuzzy match directly
        model_dir = _find_patch_dir_fuzzy(model_type, models_root)

    if model_dir and model_dir.exists():
        import_patches_from_dir(model_dir, f"{base_pkg}.models.{model_dir.name}")
        logger.info(f"execute---> :{base_pkg}.models.{model_dir.name}")


manager = PatchManager()


def apply_patches():
    # auto import and register patches
    auto_import_patches()

    try:
        from omni_npu.v1.models import register_models
        register_models()
    except Exception:
        logger.exception("Failed to register omni-npu models.")

    manager.apply_patches()