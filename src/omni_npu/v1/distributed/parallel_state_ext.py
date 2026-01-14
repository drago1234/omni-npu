# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

"""Layer-parallel group helpers for vLLM (omni-npu extension).

This module reads `layer_parallel_config` from `vllm_config.additional_config` and
creates per-layer/per-tensor-tag process groups (via vLLM's `GroupCoordinator`).

It also provides utility helpers to split inputs evenly across the layer-parallel
world size, with optional padding + all-gather unpadding.
"""

import os
from typing import Any

import torch
import torch.distributed as dist

from vllm.config import get_current_vllm_config
from vllm.platforms import current_platform

from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_tp_group,
    get_world_group as _get_world_group,
    init_model_parallel_group,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "ensure_layer_parallel_initialized",
    "initialize_local_world_group",
    "get_local_world_group",
    "get_world_group",
    "get_layer_parallel_group",
    "get_layer_transform_type",
    "get_layer_dim",
    "get_layer_parallel_world_size",
    "get_layer_parallel_rank",
    "is_layer_parallel_input_split_enabled",
    "maybe_pad_and_slice",
    "maybe_unpad_and_all_gather",
]

_LOCAL_WORLD = None

# Layer-parallel communication registry (fixed name: _LAYER_COMM_DICT).
# key: layer name inside a block in layer_parallel_config (e.g., "self_attn.q_proj")
# value:
#   {
#     "parallel_group": GroupCoordinator | None,
#     "x_transform": dict[str, Any] | None,  # {"type": str, "dim": int, "parallel_group": GroupCoordinator | None}
#     "y_transform": dict[str, Any] | None,  # {"type": str, "dim": int, "parallel_group": GroupCoordinator | None}
#   }
_LAYER_COMM_DICT: dict[str, dict[str, Any]] | None = None

# Global keys under layer_parallel_config (currently only input_split).
_LAYER_PARALLEL_GLOBAL_CFG: dict[str, Any] | None = None

# Supported tensor transform keys.
_TENSOR_TRANSFORM_KEYS = ("x_transform", "y_transform")


def is_layer_parallel_input_split_enabled() -> bool:
    """Return global input_split flag under layer_parallel_config (default: False)."""
    if _LAYER_PARALLEL_GLOBAL_CFG is None:
        return False
    return bool(_LAYER_PARALLEL_GLOBAL_CFG.get("input_split", False))


def get_world_group():
    return _get_world_group()


def calculate_effective_local_size(local_size: int, world_size: int) -> int:
    effective_local_size = min(local_size, world_size)
    if effective_local_size < local_size:
        logger.info(
            "Note: Using only %s of %s available NPU devices",
            effective_local_size,
            local_size,
        )
    if world_size % effective_local_size != 0:
        raise AssertionError(
            f"world_size ({world_size}) must be divisible by "
            f"effective_local_size ({effective_local_size})"
        )
    return effective_local_size


def initialize_local_world_group(backend: str | None = None) -> None:
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")

    global _LOCAL_WORLD
    if _LOCAL_WORLD is not None:
        return

    world_size: int = dist.get_world_size()
    if int(os.getenv("NO_NPU_MOCK", "0")):
        visible = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "")
        local_size = len(visible.split(",")) if visible else 1
    else:
        local_size = torch.npu.device_count()
    local_size = calculate_effective_local_size(local_size, world_size)

    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    num_local_groups: int = world_size // local_size
    group_ranks = []
    for i in range(num_local_groups):
        ranks = list(range(i * local_size, (i + 1) * local_size))
        group_ranks.append(ranks)

    _LOCAL_WORLD = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="world_local",
    )


def get_local_world_group():
    if _LOCAL_WORLD is None:
        raise RuntimeError("local world group is not initialized")
    return _LOCAL_WORLD


def ensure_layer_parallel_initialized(
    backend: str | None = None,
) -> None:
    """Initialize layer-parallel groups from vLLM config (idempotent).

    Safe to call multiple times. If the vLLM config or `layer_parallel_config`
    is missing, initialization becomes a no-op and defaults are used.
    """
    global _LAYER_COMM_DICT, _LAYER_PARALLEL_GLOBAL_CFG
    if _LAYER_COMM_DICT is not None:
        return

    if not dist.is_initialized():
        logger.warning(
            "Distributed is not initialized, skipping layer parallel initialization"
        )
        # Mark as initialized to avoid repeated warnings and repeated attempts.
        _LAYER_COMM_DICT = {}
        _LAYER_PARALLEL_GLOBAL_CFG = {"input_split": False}
        return

    initialize_local_world_group(backend=backend)

    vllm_config, layer_parallel_config = _load_layer_parallel_config_from_vllm()
    if vllm_config is None:
        logger.debug("vllm_config is None, layer parallel skip initialization")
        _LAYER_COMM_DICT = {}
        _LAYER_PARALLEL_GLOBAL_CFG = {"input_split": False}
        return

    if not layer_parallel_config:
        _LAYER_COMM_DICT = {}
        _LAYER_PARALLEL_GLOBAL_CFG = {"input_split": False}
        return

    _LAYER_COMM_DICT = {}
    _LAYER_PARALLEL_GLOBAL_CFG = {
        "input_split": bool(layer_parallel_config.get("input_split", False)),
    }

    parallel_config = getattr(vllm_config, "parallel_config", None)
    local_rank = getattr(parallel_config, "local_rank", 0)
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    backend = backend or getattr(current_platform, "dist_backend", "hccl")

    for layer_name_inside_block, config in layer_parallel_config.items():
        # Skip global fields under layer_parallel_config.
        if layer_name_inside_block == "input_split":
            continue
        if not isinstance(config, dict):
            continue

        layer_cfg: dict[str, Any] = {
            "parallel_group": _create_group_from_tp_size_or_ranks(
                config.get("tp_size_or_ranks"),
                local_rank,
                backend,
                f"layer_{layer_name_inside_block}",
            )
        }

        for tag in _TENSOR_TRANSFORM_KEYS:
            transform_cfg = _parse_tensor_transform_cfg(
                config.get(tag),
                local_rank,
                backend,
                f"layer_{layer_name_inside_block}_{tag}",
            )
            if transform_cfg is not None:
                layer_cfg[tag] = transform_cfg

        _LAYER_COMM_DICT[layer_name_inside_block] = layer_cfg


def get_layer_parallel_group(
    layer_name_inside_block: str,
    tensor_tag: str | None = None,
) -> GroupCoordinator | None:
    """Get the effective `GroupCoordinator` for a layer and optional tensor tag.

    Fallback order (most specific to least specific):
    - x/y transform group (when `tensor_tag` is "x" or "y")
    - per-layer group
    - global tensor-parallel group (`get_tp_group()`)

    Notes:
    - If the global TP group is not initialized, this function returns `None`
      (instead of raising an assertion from vLLM's `get_tp_group()`).
    """
    try:
        default_tp_group: GroupCoordinator | None = get_tp_group()
    except AssertionError:
        default_tp_group = None

    # If layer-parallel is not initialized, fall back to TP group (or None).
    if _LAYER_COMM_DICT is None:
        return default_tp_group

    layer_cfg: dict[str, Any] | None = _LAYER_COMM_DICT.get(layer_name_inside_block)

    # Fallback order: x/y transform group -> layer group -> global TP group.
    if layer_cfg and tensor_tag in ("x", "y"):
        transform_key = f"{tensor_tag}_transform"
        transform_cfg: dict[str, Any] | None = layer_cfg.get(transform_key)
        if transform_cfg is not None:
            transform_group = transform_cfg.get("parallel_group")
            if transform_group is not None:
                return transform_group

    if layer_cfg is not None:
        layer_group = layer_cfg.get("parallel_group")
        if layer_group is not None:
            return layer_group

    return default_tp_group


def get_layer_transform_type(layer_name_inside_block: str, tensor_tag: str | None = None) -> str:
    """Return the configured transform op type for a layer's tensor tag (default: 'NoOp')."""
    if _LAYER_COMM_DICT is None:
        return "NoOp"
    layer_cfg = _LAYER_COMM_DICT.get(layer_name_inside_block)
    if not layer_cfg:
        return "NoOp"
    if tensor_tag in ("x", "y"):
        transform_key = f"{tensor_tag}_transform"
        transform_cfg = layer_cfg.get(transform_key)
        if transform_cfg:
            return transform_cfg.get("type", "NoOp")
    return "NoOp"


def get_layer_dim(layer_name_inside_block: str, tensor_tag: str | None = None) -> int:
    """Return the configured dim for a layer's tensor tag (default: 0)."""
    if _LAYER_COMM_DICT is None:
        return 0
    layer_cfg = _LAYER_COMM_DICT.get(layer_name_inside_block)
    if not layer_cfg:
        return 0
    if tensor_tag in ("x", "y"):
        transform_key = f"{tensor_tag}_transform"
        transform_cfg = layer_cfg.get(transform_key)
        if transform_cfg:
            return transform_cfg.get("dim", 0)
    return 0


def get_layer_parallel_world_size(layer_name_inside_block: str, tensor_tag: str | None = None) -> int:
    """Return world size for the effective group (default: 1)."""
    group = get_layer_parallel_group(layer_name_inside_block, tensor_tag)
    return group.world_size if group else 1


def get_layer_parallel_rank(layer_name_inside_block: str, tensor_tag: str | None = None) -> int:
    """Return rank within the effective group (default: 0)."""
    group = get_layer_parallel_group(layer_name_inside_block, tensor_tag)
    return group.rank_in_group if group else 0


def maybe_pad_and_slice(
    input: torch.Tensor,
    dim: int = 0,
    layer_name_inside_block: str | None = None,
) -> tuple[torch.Tensor, int]:
    """Optionally pad `input` on `dim` and return the rank-local slice.

    Returns `(slice_tensor, original_length_on_dim)`.
    """
    if dim < 0:
        dim += input.dim()
    if dim < 0 or dim >= input.dim():
        raise ValueError(f"Invalid dim={dim} for tensor with dim={input.dim()}")
    if layer_name_inside_block is None:
        return input, input.shape[dim]

    world_size = get_layer_parallel_world_size(layer_name_inside_block)
    if world_size <= 1:
        return input, input.shape[dim]

    rank = get_layer_parallel_rank(layer_name_inside_block)
    orig_size = input.shape[dim]
    pad_size = (world_size - (orig_size % world_size)) % world_size

    if pad_size > 0:
        padding = [0] * (2 * input.dim())
        # torch.nn.functional.pad uses reversed dimension order.
        padding_idx = (input.dim() - 1 - dim) * 2 + 1
        padding[padding_idx] = pad_size
        input = torch.nn.functional.pad(input, padding)

    chunk_size = input.shape[dim] // world_size
    start = rank * chunk_size
    end = (rank + 1) * chunk_size

    slices = [slice(None)] * input.dim()
    slices[dim] = slice(start, end)
    output = input[tuple(slices)]

    return output, orig_size


def maybe_unpad_and_all_gather(
    input: torch.Tensor,
    actual_length: int,
    dim: int = 0,
    layer_name_inside_block: str | None = None,
) -> torch.Tensor:
    """All-gather `input` along `dim` and remove padding based on `actual_length`."""
    if dim < 0:
        dim += input.dim()
    if dim < 0 or dim >= input.dim():
        raise ValueError(f"Invalid dim={dim} for tensor with dim={input.dim()}")
    if layer_name_inside_block is None:
        return input

    group = get_layer_parallel_group(layer_name_inside_block)
    if group is None or group.world_size <= 1:
        return input

    output = group.all_gather(input, dim=dim)

    if output.shape[dim] > actual_length:
        slices = [slice(None)] * output.dim()
        slices[dim] = slice(0, actual_length)
        output = output[tuple(slices)]

    return output


def _load_layer_parallel_config_from_vllm() -> tuple[Any | None, dict[str, Any]]:
    """Fetch vllm_config from framework context and extract layer_parallel_config."""
    vllm_config = None
    try:
        vllm_config = get_current_vllm_config()
    except Exception as e:
        logger.debug(f"Failed to get vllm_config from framework: {e}")

    if vllm_config is None:
        return None, {}

    additional_config = getattr(vllm_config, "additional_config", {})
    if not isinstance(additional_config, dict):
        additional_config = {}

    layer_parallel_config = additional_config.get("layer_parallel_config", {})
    if not isinstance(layer_parallel_config, dict):
        layer_parallel_config = {}

    return vllm_config, layer_parallel_config


_CANONICAL_COMM_OP_TYPE_ALIASES: dict[str, str] = {
    "noop": "NoOp",
    "none": "NoOp",
    "no": "NoOp",
    "all2all": "ALL2ALL",
    "all2allv": "ALL2ALL",
    "alltoall": "ALL2ALL",
    "alltoallv": "ALL2ALL",
    "allreduce": "AllReduce",
    "allgather": "AllGather",
    "reducescatter": "ReduceScatter",
}
_CANONICAL_COMM_OP_TYPES: set[str] = {
    "NoOp",
    "ALL2ALL",
    "AllReduce",
    "AllGather",
    "ReduceScatter",
}


def _normalize_comm_op_type(op_type: Any) -> str:
    """Normalize op type strings to canonical values."""
    if not isinstance(op_type, str):
        return "NoOp"
    normalized = op_type.strip()
    if not normalized:
        return "NoOp"
    key = normalized.replace("_", "").replace("-", "").replace(" ", "").lower()
    # allow canonical names without case sensitivity
    for canonical in _CANONICAL_COMM_OP_TYPES:
        if normalized.lower() == canonical.lower():
            return canonical
    return _CANONICAL_COMM_OP_TYPE_ALIASES.get(key, "NoOp")


def _parse_tensor_transform_cfg(
    transform_cfg: Any,
    local_rank: int,
    backend: str,
    group_name: str,
) -> dict[str, Any] | None:
    """Parse x_transform / y_transform into a uniform dict."""
    if not isinstance(transform_cfg, dict):
        return None
    t_type = _normalize_comm_op_type(transform_cfg.get("type", "NoOp"))
    dim = int(transform_cfg.get("dim", 0) or 0)
    return {
        "type": t_type,
        "dim": dim,
        "parallel_group": _create_group_from_tp_size_or_ranks(
            transform_cfg.get("tp_size_or_ranks"),
            local_rank,
            backend,
            group_name,
        ),
    }


def _create_group_from_tp_size_or_ranks(
    tp_size_or_ranks: Any,
    local_rank: int,
    backend: str,
    group_name: str,
) -> GroupCoordinator | None:
    """Parse tp_size_or_ranks and create a GroupCoordinator."""
    group_ranks = _tp_size_or_ranks_to_group_ranks(tp_size_or_ranks, group_name)
    if group_ranks is None:
        return None
    return init_model_parallel_group(
        group_ranks=group_ranks,
        local_rank=local_rank,
        backend=backend,
        group_name=group_name,
    )


def _tp_size_or_ranks_to_group_ranks(
    spec: Any,
    group_name: str,
) -> list[list[int]] | None:
    """Convert tp_size_or_ranks into group_ranks for init_model_parallel_group."""
    if not spec:
        return None

    if isinstance(spec, list):
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before parsing tp_size_or_ranks."
            )
        world = dist.get_world_size()

        # Normalize `spec` into `list[list[int]]`.
        # - list[int] -> [list[int]]
        # - list[list[int]] -> list[list[int]]
        if all(isinstance(x, list) for x in spec):
            group_ranks = spec
        elif all(isinstance(x, int) for x in spec):
            group_ranks = [spec]
        else:
            raise RuntimeError(
                f"Invalid tp_size_or_ranks={spec!r} for {group_name}: "
                "expected list[int] or list[list[int]]."
            )

        flat: list[int] = []
        for grp in group_ranks:
            if not isinstance(grp, list) or not grp:
                raise RuntimeError(
                    f"Invalid tp_size_or_ranks={spec!r} for {group_name}: "
                    "each group must be a non-empty list of global ranks."
                )
            seen_in_grp: set[int] = set()
            for r in grp:
                if not isinstance(r, int):
                    raise RuntimeError(
                        f"Invalid tp_size_or_ranks={spec!r} for {group_name}: "
                        "rank ids must be integers."
                    )
                if r < 0 or r >= world:
                    raise RuntimeError(
                        f"Invalid rank id {r} in tp_size_or_ranks for {group_name}: "
                        f"expected 0 <= rank < world_size({world})."
                    )
                if r in seen_in_grp:
                    raise RuntimeError(
                        f"Duplicate rank {r} within a single group in tp_size_or_ranks "
                        f"for {group_name}."
                    )
                seen_in_grp.add(r)
                flat.append(r)

        if len(set(flat)) != len(flat):
            raise RuntimeError(
                f"tp_size_or_ranks for {group_name} contains duplicate ranks across groups."
            )
        if set(flat) != set(range(world)):
            raise RuntimeError(
                f"tp_size_or_ranks for {group_name} must cover all ranks in the "
                f"default process group. Got {len(set(flat))}/{world} ranks. "
                "If you are using DP/PP/ExternalDP, provide a list-of-lists that "
                "partitions the full world."
            )
        return group_ranks

    if isinstance(spec, int):
        cfg = get_current_vllm_config()
        if cfg is None:
            raise RuntimeError(
                "tp_size_or_ranks=int requires vllm_config to be available "
                "via get_current_vllm_config()."
            )
        pc = cfg.parallel_config
        dp = int(getattr(pc, "data_parallel_size", 1))
        pp = int(getattr(pc, "pipeline_parallel_size", 1))
        tp = int(getattr(pc, "tensor_parallel_size", 1))
        world = dist.get_world_size()

        denom = dp * pp * tp
        if denom <= 0 or world % denom != 0:
            raise RuntimeError(
                f"Invalid parallel layout for subgroup derivation: "
                f"world_size={world}, dp={dp}, pp={pp}, tp={tp}."
            )
        if tp % spec != 0:
            raise RuntimeError(
                f"tp_size_or_ranks={spec} must divide tensor_parallel_size={tp}."
            )

        external_dp = world // denom
        tp_groups: list[list[int]] = (
            torch.arange(world)
            .reshape(external_dp, dp, pp, tp)
            .reshape(-1, tp)
            .tolist()
        )
        return [g[i:i + spec] for g in tp_groups for i in range(0, tp, spec)]

    logger.warning(f"Unsupported tp_size_or_ranks type: {type(spec)} for {group_name}")
    return None
