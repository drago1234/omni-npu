# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)
NPU_ATTENTION_BACKEND = {}


def _maybe_padded_raw_tensor_to_strided_caches(
    raw_tensor: torch.Tensor,
    num_blocks: int,
    block_size: int,
    shapes: tuple[tuple[int, ...], ...],
    dtypes: tuple[torch.dtype, ...],
    page_size_bytes: int,
) -> tuple[torch.Tensor, ...]:
    """
    Creates strided views of a raw memory tensor to represent heterogeneous,
    padded KV cache blocks.

    This function maps a flat 1D raw tensor into multiple multi-dimensional
    cache tensors. It assumes the raw tensor is partitioned into `num_blocks`
    pages of `page_size_bytes`. Within each page, the sub-tensors defined by
    `shapes` and `dtypes` are packed sequentially, followed by potential padding
    to fill the rest of the page.

    Memory Layout per Page (Block):
    [ Tensor 0 ] [ Tensor 1 ] ... [ Tensor N ] [ Padding (optional) ]
    |<- bytes ->|<- bytes ->|    |<- bytes ->|
    |<------------------- page_size_bytes ------------------------->|

    Args:
        raw_tensor (torch.Tensor): The underlying 1D memory pool.
        num_blocks (int): The total number of memory blocks (pages) allocated.
        block_size (int): The number of tokens each block can hold.
        shapes (tuple[tuple[int, ...], ...]): A tuple containing the trailing
            dimensions for each sub-tensor (excluding num_blocks and block_size).
        dtypes (tuple[torch.dtype, ...]): The data types corresponding to each shape.
        page_size_bytes (int): The fixed size of each memory block in bytes.

    Returns:
        cache_tensors (tuple[torch.Tensor, ...]): A tuple of strided tensors sharing the same
            underlying storage as `raw_tensor`. Each tensor has the shape:
            (num_blocks, block_size, *shape).

    Raises:
        AssertionError: If shapes and dtypes lengths mismatch, if type sizes don't
            align, if the raw tensor is too small, or if the page size is insufficient.
    """
    assert len(shapes) == len(dtypes), f"Error! {len(shapes)=} while {len(dtypes)=}."

    # Ensure the raw tensor has enough physical memory
    total_required_bytes = num_blocks * page_size_bytes
    actual_bytes = raw_tensor.numel() * raw_tensor.element_size()
    assert actual_bytes >= total_required_bytes, (
        f"Error! Raw tensor has {actual_bytes} bytes, "
        f"but {total_required_bytes} bytes are required."
    )

    cache_tensors = []
    storage_offset_bytes = 0

    for shape, dtype in zip(shapes, dtypes):
        dtype_size = dtype.itemsize
        assert page_size_bytes % dtype_size == 0, (
            f"Error! Page size {page_size_bytes} is not a multiple of dtype size {dtype_size}."
        )
        assert storage_offset_bytes % dtype_size == 0, (
            f"Error! Offset {storage_offset_bytes} is not aligned for dtype size {dtype_size}."
        )

        num_element_per_page = page_size_bytes // dtype_size
        target_shape = (num_blocks, block_size, *shape)

        # Get contiguous strides. stride[0] will equal the total number
        # of elements this specific tensor occupies within a single block.
        stride = torch.empty(target_shape).stride()

        # Override the 0th stride to jump by the full page size
        target_stride = (num_element_per_page, *stride[1:])

        tensor = torch.as_strided(
            raw_tensor.view(dtype),
            size=target_shape,
            stride=target_stride,
            storage_offset=storage_offset_bytes // dtype_size,
        )
        cache_tensors.append(tensor)

        # Advance the byte offset by the size this tensor takes up in one block
        storage_offset_bytes += stride[0] * dtype_size

    # The crucial missing check: Did we exceed the allocated page size?
    assert storage_offset_bytes <= page_size_bytes, (
        f"Error! Sub-tensors require {storage_offset_bytes} bytes per block, "
        f"which exceeds the allocated page_size_bytes of {page_size_bytes}."
    )

    return tuple(cache_tensors)


def register_attention_backend(backend: str):

    def decorator(cls: type) -> type:
        if not hasattr(cls, "reshape_kv_cache") or not callable(getattr(cls, "reshape_kv_cache")):
            raise NotImplementedError(
                f"Cannot register '{backend}': {cls.__name__} must implement the "
                "`reshape_kv_cache` static method required by NPU backends."
            )

        attn_module = f"{cls.__module__}.{cls.__qualname__}"
        logger.debug("Register attention %s with module %s", backend,
                     attn_module)
        NPU_ATTENTION_BACKEND[backend] = attn_module
        return cls

    return decorator
