# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import logging
from contextlib import nullcontext

import torch
import torch_npu
import torchair
from transformers import PretrainedConfig

from vllm.config import CacheConfig


def get_npu_execution_type(stream_label):
    if stream_label is None:
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

def named_stream(name): # not for graph
    if name == "current":
        return torch.npu.current_stream()
    if not hasattr(named_stream, name):
        setattr(named_stream, name, torch.npu.Stream())
    return getattr(named_stream, name)

def calculate_page_size_padded(
        cache_config: CacheConfig,
        cache_dtype_str: str | None,
        config: PretrainedConfig,
        mome_state_shapes: tuple | None,
        mome_state_dtypes: tuple | None,
        kernel_size: int = 0,
        num_speculative_tokens: int = 0,
    ) -> int | None:
    """
    Calculate page_size_padded for alignment across different attention mechanisms.

    Alignment priority:
    1. If DSA exists: align to DSA page size
    2. Otherwise: align to max(MOME page size, MLA/SWA page size)

    Args:
        cache_config: Cache configuration
        cache_dtype_str: Quantization dtype string (e.g., "fp8_ds_mla", "hif8_ds_mla", "int8_ds_mla")
        config: Model configuration

    Returns:
        page_size_padded in bytes, or None if no padding needed
    """
    from vllm.utils.torch_utils import get_dtype_size
    from math import prod

    block_size = cache_config.block_size
    dtype = torch.bfloat16  # Default dtype
    dtype_size = get_dtype_size(dtype)

    # Calculate MLA/SWA page size
    mla_head_size = config.kv_lora_rank + config.qk_rope_head_dim
    mla_page_dim = mla_head_size * dtype_size

    # Calculate DSA page size if DSA layer exists
    dsa_page_dim = None
    if hasattr(config, "index_topk") and config.index_topk > 0:
        index_head_dim = getattr(config, "index_head_dim", 0)
        # Non-quant case: standard attention format
        dsa_page_dim = (mla_head_size + index_head_dim) * dtype_size

    # Calculate MOME page size if MOME is enabled
    mome_page_size = None
    if getattr(config, "use_mome", False):
        num_total_tokens = kernel_size - 1 + num_speculative_tokens
        mome_page_size = sum(
            prod(shape) * get_dtype_size(dtype)
            for (shape, dtype) in zip(mome_state_shapes, mome_state_dtypes)
        ) * num_total_tokens

    if dsa_page_dim is None:
        denominator = mla_page_dim
    else:
        denominator = dsa_page_dim if mla_page_dim < dsa_page_dim else mla_page_dim

    if mome_page_size is not None:
        block_size = (mome_page_size / denominator + 16 - 1) // 16 * 16

    # Determine alignment priority
    if dsa_page_dim is not None:
        target_page_size = dsa_page_dim * block_size
    elif mome_page_size is not None:
        target_page_size = max(mome_page_size, mla_page_dim * block_size)
    else:
        block_size = cache_config.block_size
        target_page_size = mla_page_dim * block_size

    return int(target_page_size), int(block_size)