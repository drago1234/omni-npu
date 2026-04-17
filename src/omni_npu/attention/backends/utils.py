# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import torch
import numpy as np
from importlib.metadata import entry_points

from vllm.utils.math_utils import cdiv
from vllm.distributed import GroupCoordinator
from vllm.logger import init_logger
from vllm.platforms import current_platform


logger = init_logger(__name__)
NPU_ATTENTION_BACKEND = {}

# NOTE: Cache for entry points to avoid repeated lookups
_PLUGIN_BACKEND_CACHE = None  # type: dict[str, type] | None


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

def _load_plugin_backends_map() -> dict[str, type]:
    """NOTE: Scan entry points once and build a name -> class map.

    Each entry point under group ``omni.attention_backends`` is loaded
    and indexed by the value returned by its ``get_name()`` method.
    The result is cached so subsequent calls are free.
    """
    global _PLUGIN_BACKEND_CACHE
    if _PLUGIN_BACKEND_CACHE is not None:
        return _PLUGIN_BACKEND_CACHE

    plugin_map: dict[str, type] = {}
    try:
        eps = entry_points(group="omni.attention_backends")
    except Exception:
        _PLUGIN_BACKEND_CACHE = plugin_map
        return plugin_map

    for ep in eps:
        try:
            backend_cls = ep.load()
            name = backend_cls.get_name()
            plugin_map[name] = backend_cls
            logger.debug("Found plugin backend %s from entry point %s", name, ep.name)
        except Exception as e:
            logger.warning("Failed to load backend plugin %s: %s", ep.name, e)

    _PLUGIN_BACKEND_CACHE = plugin_map
    return plugin_map

def _is_plugin_disabled(backend: str) -> bool:
    """NOTE: Check whether a backend name is listed in the
    ``DISABLE_PLUGIN_BACKENDS`` environment variable.

    The env var is a comma-separated list of backend names that should
    **not** be replaced by their plugin counterparts.  For example::

        DISABLE_PLUGIN_BACKENDS=NPUDSA,NPUMLA

    means the decorator will return the original base class for NPUDSA
    and NPUMLA even if a plugin is available.
    """
    disabled = os.environ.get("DISABLE_PLUGIN_BACKENDS", "")
    if not disabled:
        return False
    disabled_names = [s.strip() for s in disabled.split(",") if s.strip()]
    return backend in disabled_names

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

def get_attention_backend(name: str) -> str | None:
    """Get a registered attention backend path string by name."""
    return NPU_ATTENTION_BACKEND.get(name)

def apply_plugin_overrides():
    """NOTE: Replace registered backends with their plugin counterparts.

    This must be called **after** all base backends have been imported
    (and thus registered), to avoid circular-import issues — plugin
    modules import base classes from omni_npu, so we cannot load them
    while the base modules are still being imported.

    For each registered backend name, we check whether an entry point
    under ``omni.attention_backends`` provides a class whose
    ``get_name()`` matches.  If it does and the backend is not listed
    in the ``DISABLE_PLUGIN_BACKENDS`` environment variable, the
    plugin class replaces the base class in ``NPU_ATTENTION_BACKEND``
    and is also returned so the caller can rebind module-level names.

    Returns:
        A tuple ``(overrides, base_paths)`` where *overrides* maps
        backend name to the plugin class (only for backends that were
        actually overridden) and *base_paths* is the snapshot of
        ``NPU_ATTENTION_BACKEND`` taken before any plugin was loaded.
    """
    overrides: dict[str, type] = {}

    # Snapshot base paths before loading plugins, because loading
    # plugin modules triggers their @register_attention_backend
    # decorators which would overwrite the base paths.
    base_paths = dict(NPU_ATTENTION_BACKEND)

    # Build the plugin map (cached after first call).
    # This loads plugin modules via entry points, which triggers
    # their @register_attention_backend decorators.
    plugin_map = _load_plugin_backends_map()

    for backend_name, base_module in base_paths.items():
        if _is_plugin_disabled(backend_name):
            # If disabled, restore the base path in case a plugin
            # decorator already overwrote it during _load_plugin_backends_map.
            NPU_ATTENTION_BACKEND[backend_name] = base_module
            logger.debug("Plugin override for %s disabled by env var", backend_name)
            continue

        plugin_cls = plugin_map.get(backend_name)
        if plugin_cls is None:
            continue

        plugin_module = f"{plugin_cls.__module__}.{plugin_cls.__qualname__}"
        if plugin_module == base_module:
            continue  # same class, skip

        logger.info(
            "Plugin backend replaces %s: %s -> %s",
            backend_name, base_module, plugin_module,
        )
        NPU_ATTENTION_BACKEND[backend_name] = plugin_module
        overrides[backend_name] = plugin_cls

    return overrides, base_paths

def get_available_backends() -> list[str]:
    """List all registered backend names."""
    return list(NPU_ATTENTION_BACKEND.keys())


class SPManager:
    """
    This module handles sequence parallelism (SP).

    SP here refers to standard SP splitting, i.e., splitting tokens with ceil division
    based on sp_size.

    CP refers to zigzag-form SP splitting, aimed at adjusting query distribution to
    balance attention computation across ranks.

    Module initialization defaults to standard SP operations; CP-related operations
    require explicit init_zigzag flag.
    """

    def __init__(
        self,
        sp_group: GroupCoordinator,
        blk_table_ref: torch.Tensor,   # ref only
        slot_mapping_ref: torch.Tensor,
        cumlens: torch.Tensor,         # [B + 1]
        cumlens_np: np.ndarray = None, # [B + 1]
        init_zigzag: bool = False,
        pg: int = 128,                 # page size
        mome_kernel_width: int = 3,
    ):
        self.sp_group = sp_group
        self.sp_size = sp_group.world_size
        self.sp_rank = sp_group.rank_in_group
        self.sp_comm = sp_group.device_group
        assert self.sp_size > 0

        if cumlens_np is None:
            cumlens_np = np.array(cumlens.tolist(), dtype=np.int32)
        assert cumlens.dim() == 1 and cumlens_np.ndim == 1
        assert cumlens_np.size == cumlens.size(0)
        assert cumlens.size(0) > 1 # at least 2 elements in a valid batch, [B + 1]

        assert blk_table_ref.dim() == 2
        self.table_size = blk_table_ref.size(1)
        self.pg = pg
        self.init_zigzag = init_zigzag
        self.mome_prefix_size = mome_kernel_width - 1

        self._scheme_sp_ctrl(cumlens_np)
        if init_zigzag:
            self._scheme_cp_attn(cumlens, blk_table_ref, slot_mapping_ref)
            self._scheme_cp_slice(cumlens_np)
            self._scheme_cp_reorg(cumlens_np)
            if self.mome_prefix_size > 0:
                self._scheme_cp_mome(cumlens_np)

    def _scheme_sp_ctrl(self, cumlens: np.ndarray):
        self.tok = int(cumlens[-1])
        self.sp_len = cdiv(self.tok, self.sp_size)
        self.sp_align = self.sp_len * self.sp_size
        a = min(self.tok, self.sp_rank * self.sp_len)
        b = min(self.tok, a + self.sp_len)
        self.slice_domain = (a, b)

    def align_tokens(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(0) == self.tok
        if self.tok == self.sp_align:
            return x
        y = x.new_zeros(self.sp_align, *x.shape[1:])
        y[:self.tok] = x
        return y

    def slice_tokens(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(0) >= self.tok
        a, b = self.slice_domain
        if b == a + self.sp_len:
            return x[a:b]
        y = x.new_zeros(self.sp_len, *x.shape[1:])
        if b > a:
            y[:b - a] = x[a:b]
        return y

    def ag_tokens(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(0) == self.sp_len
        return self.sp_group.all_gather(x, dim=0)[:self.tok]


    def _scheme_cp_reorg(self, cumlens: np.ndarray): # [B + 1]
        frag_num = self.sp_size * 2
        frag_lens = cdiv(np.diff(cumlens), frag_num)

        ends = cumlens[1:].repeat(2)
        frags = frag_lens.repeat(2)
        frags_base = frags.cumsum() - frags
        sp_len, cp_len = self.sp_len, int(frags.sum())

        class slicer:
            def __init__(self, src, dst, cp, raw):
                self.src, self.dst = int(src), int(dst)
                self.cp, self.raw = int(cp), int(raw)
                self.len = 0
            def slice_src(self, x: torch.Tensor):
                return x[self.src : self.src + self.raw]
            def slice_dst(self, x: torch.Tensor):
                return x[self.dst : self.dst + self.raw]

        def _recv(cp: int, sends: list):
            prev = slicer(0, 0, cp, 0)
            sends[0].append(prev)
            left = np.stack([
                cumlens[:-1] + cp * frag_lens,
                cumlens[:-1] + (frag_num - 1 - cp) * frag_lens,
            ]).transpose().flatten()
            right = np.clip(left + frags, a_max=ends, a_min=None)
            for p, a, b in zip(frags_base, left, right):
                while(a < b):
                    inc = sp_len - a % sp_len
                    sect = slicer(a % sp_len, p, cp, raw=min(inc, b - a))
                    sends[a // sp_len].append(sect)
                    prev.len = sect.dst - prev.dst
                    prev = sect; a += inc; p += inc
            prev.len = cp_len - prev.dst

        parallel = range(self.sp_size)
        sends = [[] for sp in parallel] # [sp, *]
        for cp in parallel: _recv(cp, sends)
        sends[0] = [it for it in sends[0] if it.len > 0]

        a2a_map = [] # [sp, cp]
        for sp_sects in sends:
            send_split = [0 for cp in parallel]
            dst = 0
            for sect in sp_sects:
                sect.dst = dst # post-a2a -> pre-a2a
                dst += sect.len
                send_split[sect.cp] += sect.len
            a2a_map.append(send_split)

        sends = sends[self.sp_rank]
        sp_split = a2a_map[self.sp_rank]
        cp_split = [it[self.sp_rank] for it in a2a_map]

        self.cp_reorg_metadata = (sends, sp_split, cp_split, sp_len, cp_len)

    def sp_to_cp(self, sp: torch.Tensor) -> torch.Tensor:
        assert self.init_zigzag
        sends, sp_split, cp_split, sp_len, cp_len = self.cp_reorg_metadata
        assert sp.size(0) == sp_len
        tmp = sp.new_zeros(sum(sp_split), *sp.shape[1:])
        for it in sends:
            it.slice_dst(tmp).copy_(it.slice_src(sp))
        cp = sp.new_empty(cp_len, *sp.shape[1:])
        # split could be all 0, for send-only or recv-only case
        torch.distributed.all_to_all_single(
            cp, tmp, cp_split, sp_split,
            group=self.sp_comm)
        return cp # output zigzag

    def cp_to_sp(self, cp: torch.Tensor) -> torch.Tensor:
        assert self.init_zigzag
        sends, sp_split, cp_split, sp_len, cp_len = self.cp_reorg_metadata
        assert cp.size(0) == cp_len
        tmp = cp.new_empty(sum(sp_split), *cp.shape[1:])
        # split could be all 0, for send-only or recv-only case
        torch.distributed.all_to_all_single(
            tmp, cp, sp_split, cp_split,
            group=self.sp_comm)
        sp = cp.new_zeros(sp_len, *cp.shape[1:])
        for it in sends:
            it.slice_src(sp).copy_(it.slice_dst(tmp))
        return sp


    def _scheme_cp_slice(self, cumlens: np.ndarray):
        frag_num = self.sp_size * 2
        frag_lens = cdiv(np.diff(cumlens), frag_num)

        ends = cumlens[1:].repeat(2)
        frags = frag_lens.repeat(2)
        frags_base = frags.cumsum() - frags

        left = np.stack([
            cumlens[:-1] + self.sp_rank * frag_lens,
            cumlens[:-1] + (frag_num - 1 - self.sp_rank) * frag_lens,
        ]).transpose().flatten()
        right = np.clip(left + frags, a_max=ends, a_min=None)

        sects = [(dst, src, end - src) for dst, src, end
            in zip(frags_base, left, right) if src < end]
        cp_len = int(frags.sum())
        self.cp_slice_metadata = (sects, cp_len)

    def cp_slice(self, x: torch.Tensor) -> torch.Tensor:
        assert self.init_zigzag
        assert x.size(0) >= self.tok
        sects, cp_len = self.cp_slice_metadata
        y = x.new_zeros(cp_len, *x.shape[1:])
        for dst, src, n in sects:
            y[dst : dst + n] = x[src : src + n]
        return y

    def _scheme_cp_attn(self, cumlens: torch.Tensor, blk_table_ref: torch.Tensor, slot_mapping_ref: torch.Tensor):
        frag_num = self.sp_size * 2
        cumlens = cumlens.to(torch.int32)    # [B + 1]
        seq_lens = cumlens.diff()            # [B]
        frag_lens = cdiv(seq_lens, frag_num) # [B]
        seq_blks = cdiv(seq_lens, self.pg)   # [B]

        kv_lens_1 = frag_lens * (self.sp_rank + 1)                # [B]
        kv_lens_2 = frag_lens * (frag_num - self.sp_rank)         # [B]
        frag_cumlens = frag_lens.cumsum(dim=0, dtype=torch.int32) # [B]
        self.cp_attn_metadata_half = (frag_cumlens, frag_cumlens, kv_lens_1, kv_lens_2)

        q_lens = frag_lens.repeat_interleave(2, dim=0)      # [2B]
        q_cumlens = q_lens.cumsum(dim=0, dtype=torch.int32) # [2B]
        kv_lens = torch.stack([kv_lens_1, kv_lens_2])       # [2, B]
        kv_lens = kv_lens.transpose(0, 1).flatten()         # [2B]

        blk_base = seq_blks.cumsum(dim=0, dtype=torch.int32) - seq_blks
        table0 = torch.arange(self.table_size, dtype=torch.int32, device=cumlens.device)
        blk_table = (table0.view(1, -1) + blk_base.repeat_interleave(2, dim=0).view(-1, 1))
        cp_block_table = blk_table_ref.repeat_interleave(2, dim=0)
        self.cp_attn_metadata = (q_cumlens, kv_lens, blk_table, cp_block_table)
        self.cp_slot_mapping = slot_mapping_ref.clone()

        seq_lens = seq_lens.tolist()
        align_lens = (seq_blks * self.pg).tolist()
        self.blk_align_metadata = (seq_lens, align_lens)

    def cp_attn_meta(self) -> tuple:
        # FA once:
        #
        assert self.init_zigzag
        return self.cp_attn_metadata # q_cumlens, kv_lens, blk_table

    def cp_attn_meta_2(self) -> tuple:
        assert self.init_zigzag
        # return q_cumlens_1, q_cumlens_2, kv_lens_1, kv_lens_2
        return self.cp_attn_metadata_half

    def page_align(self, x: torch.Tensor) -> torch.Tensor:
        assert self.init_zigzag
        seq_lens, align_lens = self.blk_align_metadata
        assert x.size(0) == self.tok
        y = x.new_empty(sum(align_lens), *x.shape[1:])
        torch.split_with_sizes_copy(x, seq_lens,
            out=[it[:n] for it, n in
                zip(y.split(align_lens), seq_lens)])
        return y.view(-1, self.pg, 1, x.size(-1))

    def _scheme_cp_mome(self, cumlens: np.ndarray):
        cp_query_start_loc = cdiv(cumlens, self.sp_size)
        if self.sp_rank == 0:
            offset = self.mome_kernel_width - 1
        else:
            offset = (self.mome_kernel_width - 1) * 2
        mome_query_start_loc = cp_query_start_loc + offset
        mome_query_start_loc[0] = cp_query_start_loc[0]
        mome_query_start_loc_tensor = torch.tensor(mome_query_start_loc, dtype=torch.int32, device=current_platform.device_type)
        self.cp_mome_metadata = (mome_query_start_loc_tensor, )

    def _scheme_cp_mome(self, cumlens: np.ndarray):
        """
        TP4, req_num = 2 for example, 8 chunks per request in total
        r0 means req0, c0 means chunk0
        before mome hidden_states layout:
        rank0: | r0 c0 | r0 c7 | r1 c0 | r1 c7 |
        rank1: | r0 c1 | r0 c6 | r1 c1 | r1 c6 |
        before mome suffix exchange suffix layout:
        rank0: | r0 c0 suffix(2) | r0 c7 suffix(2) | r1 c0 suffix(2) | r1 c7 suffix(2) |
        rank1: | r0 c1 suffix(2) | r0 c6 suffix(2) | r1 c1 suffix(2) | r1 c6 suffix(2) |
        after mome suffix exchange suffix layout:
        rank0: | r0 c6 suffix(2) | r1 c6 suffix(2) |
        rank1: | r0 c0 suffix(2) | r0 c5 suffix(2) | r1 c0 suffix(2) | r1 c5 suffix(2) |
        after rank0 mome suffix broadcast hidden_states layout:
        rank0: | r0 c0 | r0 c6 suffix(2) | r0 c7 | r1 c0 | r1 c6 suffix(2) | r1 c7 |
        rank1: | r0 c0 suffix(2) | r0 c1 | r0 c5 suffix(2) | r0 c6 | r0 c7 suffix(4) | r1 c0 suffix(2) | r1 c1 | r1 c5 suffix(2) | r1 c6 | r1 c7 suffix(4) |
        after mome suffix exchange hidden_states layout:
        rank0: | r0 c0 | r0 c6 suffix(2) | r0 c7 | r1 c0 | r1 c6 suffix(2) | r1 c7 |
        rank1: | r0 c0 suffix(2) | r0 c1 | r0 c5 suffix(2) | r0 c6 | r0 c7 suffix(4) | r1 c0 suffix(2) | r1 c1 | r1 c5 suffix(2) | r1 c6 | r1 c7 suffix(4) |
        rank 0 start_query_loc = [0, (2 *r0_chunk_len + 2), (2 * r0_chunk_len + 2) + (2 * r1_chunk_len + 2)]
        rank 1 start_query_loc = [0, (2 * (r0_chunk_len + 2) + 4), (2 * (r0_chunk_len + 2) + 4) + (2 * (r1_chunk_len + 2) + 4)]
        after mome restore layout:
        rank0: | r0 c0 | r0 c7 | r1 c0 | r1 c7 |
        rank1: | r0 c1 | r0 c6 | r1 c1 | r1 c6 | 
        """

        cp_query_split_lens = cdiv(np.diff(cumlens), self.sp_size * 2)
        self.cp_mome_phase_split_sizes = tuple(int(req_len) for req_len in cp_query_split_lens.tolist())
        num_reqs = len(self.cp_mome_phase_split_sizes)

        factor = 1 if self.sp_rank == 0 else 2
        core_merged_query_lens = 2 * cp_query_split_lens + factor * self.mome_prefix_size
        # Non-rank0 appends rank0's per-request 4 tail tokens after suffix exchange.
        tail_append_len = 0 if self.sp_rank == 0 else 2 * self.mome_prefix_size
        merged_query_lens = core_merged_query_lens + tail_append_len
        mome_query_start_loc = np.zeros(num_reqs + 1, dtype=cp_query_split_lens.dtype)
        mome_query_start_loc[1:] = np.cumsum(merged_query_lens)
        self.cp_mome_query_start_loc = torch.tensor(
            mome_query_start_loc,
            dtype=torch.int32,
            device=current_platform.device_type,
        )

        self.cp_mome_req_split_sizes = tuple(
            2 * req_len for req_len in self.cp_mome_phase_split_sizes
        )
        self.cp_mome_merged_core_split_sizes = tuple(
            int(merged_len) for merged_len in core_merged_query_lens.tolist()
        )
        self.cp_mome_merged_split_sizes = tuple(int(merged_len) for merged_len in merged_query_lens.tolist())
        self.cp_mome_suffix_block_len = num_reqs * self.mome_prefix_size
        self.cp_mome_local_suffix_len = self.cp_mome_suffix_block_len * 2

    def mome_suffix_exchange(self, x: torch.Tensor) -> torch.Tensor:
        # x layout is req-prior: [req0 chunk0][req0 chunk7][req1 chunk0][req1 chunk7]...
        suffix_size = self.mome_prefix_size
        req_chunks = x.split(self.cp_mome_req_split_sizes, dim=0)
        phase0_chunks = [
            req_chunk[:req_len]
            for req_chunk, req_len in zip(req_chunks, self.cp_mome_phase_split_sizes)
        ]
        phase1_chunks = [
            req_chunk[req_len:]
            for req_chunk, req_len in zip(req_chunks, self.cp_mome_phase_split_sizes)
        ]
        local_suffixes = torch.cat(
            [req_chunk[-suffix_size:] for req_chunk in phase0_chunks]
            + [req_chunk[-suffix_size:] for req_chunk in phase1_chunks],
            dim=0,
        )
        phase0_local_suffix = local_suffixes.narrow(0, 0, self.cp_mome_suffix_block_len)
        all_suffixes = self.sp_group.all_gather(local_suffixes, dim=0)

        local_suffix_len = self.cp_mome_local_suffix_len
        suffix_block_len = self.cp_mome_suffix_block_len

        if self.sp_rank == 0:
            phase0_suffix_chunks = ()
            phase1_suffix = all_suffixes.narrow(0, local_suffix_len + suffix_block_len, suffix_block_len)
        else:
            prev_rank_off = (self.sp_rank - 1) * local_suffix_len
            phase0_suffix = all_suffixes.narrow(0, prev_rank_off, suffix_block_len)
            phase0_suffix_chunks = phase0_suffix.split(suffix_size, dim=0)
            if self.sp_rank == self.sp_size - 1:
                phase1_suffix = phase0_local_suffix
            else:
                next_rank_off = (self.sp_rank + 1) * local_suffix_len + suffix_block_len
                phase1_suffix = all_suffixes.narrow(0, next_rank_off, suffix_block_len)

        phase1_suffix_chunks = phase1_suffix.split(suffix_size, dim=0)
        merged_pieces = []
        for idx, (phase0_chunk, phase1_suffix_chunk, phase1_chunk) in enumerate(zip(phase0_chunks, phase1_suffix_chunks, phase1_chunks)):
            if self.sp_rank != 0:
                merged_pieces.append(phase0_suffix_chunks[idx])
            merged_pieces.extend([phase0_chunk, phase1_suffix_chunk, phase1_chunk])
        return torch.cat(merged_pieces, dim=0)

    def broadcast_mome_req_tails_from_rank0(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        tail_len = self.mome_prefix_size + self.mome_prefix_size
        req_core_sizes = self.cp_mome_merged_core_split_sizes
        num_reqs = len(req_core_sizes)
        tail_shape = (num_reqs, tail_len, *x.shape[1:])
        tail_scratch = torch.empty(
            tail_shape,
            dtype=x.dtype,
            device=x.device,
        )

        if self.sp_rank == 0:
            for req_idx, (req_chunk, req_len) in enumerate(zip(x.split(req_core_sizes, dim=0), req_core_sizes)):
                assert req_len >= tail_len, f"Expected req_len >= tail_len, got req_len={req_len}, tail_len={tail_len}"
                tail_scratch[req_idx].copy_(req_chunk[-tail_len:])

        torch.distributed.broadcast(
            tail_scratch,
            src=0,
            group=self.sp_comm,
        )
        if self.sp_rank == 0:
            return x

        req_chunks = x.split(req_core_sizes, dim=0)
        merged_pieces = []
        for idx, req_chunk in enumerate(req_chunks):
            merged_pieces.extend([req_chunk, tail_scratch[idx]])
        return torch.cat(merged_pieces, dim=0)

    def mome_split_and_cat(self, merged_output: torch.Tensor) -> torch.Tensor:
        merged_chunks = merged_output.split(self.cp_mome_merged_split_sizes, dim=0)
        restored_chunks = []
        phase0_start = 0 if self.sp_rank == 0 else self.mome_prefix_size
        phase1_base = self.mome_prefix_size if self.sp_rank == 0 else 2 * self.mome_prefix_size
        for req_len, merged_chunk in zip(self.cp_mome_phase_split_sizes, merged_chunks):
            phase0_end = phase0_start + req_len
            phase0_chunk = merged_chunk[phase0_start:phase0_end]
            phase1_start = phase1_base + req_len
            phase1_chunk = merged_chunk[phase1_start:phase1_start + req_len]
            restored_chunks.extend([phase0_chunk, phase1_chunk])
        return torch.cat(restored_chunks, dim=0)

    def __repr__(self):
        return f"""SPManager(
            sp_size={self.sp_size}, 
            sp_rank={self.sp_rank}, 
            init_zigzag={self.init_zigzag},
            cp_mome_query_start_loc={self.cp_mome_query_start_loc},
            cp_mome_req_split_sizes={self.cp_mome_req_split_sizes},
        )"""


class DummySPManager:

    def __init__(self, sp_group: GroupCoordinator):
        self.sp_size = sp_group.world_size
        # in dummy_run, we assert token_num divisible by sp_size

    def sp_to_cp(self, x: torch.Tensor):
        return x

    def cp_to_sp(self, x: torch.Tensor):
        return x

    def align_tokens(self, x: torch.Tensor):
        return x

    def page_align(self, x: torch.Tensor):
        return x

    def slice_tokens(self, x: torch.Tensor):
        return x[: x.size(0) // self.sp_size].clone()

    def cp_slice(self, x: torch.Tensor):
        return x[: x.size(0) // self.sp_size].clone()

    def ag_tokens(self, x: torch.Tensor):
        return torch.cat([x] * self.sp_size)

    def cp_attn_meta(self) -> tuple:
        return None, None, None, None

    def mome_suffix_exchange(self, x: torch.Tensor):
        return x

    def broadcast_mome_req_tails_from_rank0(self, x: torch.Tensor):
        return x

    def mome_split_and_cat(self, x: torch.Tensor):
        return x


def lazy_init_cos_sin(
    sp_manager: SPManager | DummySPManager,
    cos: torch.Tensor,
    sin: torch.Tensor,
    init_zigzag: bool = False,
):
    if hasattr(sp_manager, "sp_cos"):
        return
    if init_zigzag:
        sp_manager.cp_cos = sp_manager.cp_slice(cos)
        sp_manager.cp_sin = sp_manager.cp_slice(sin)
    sp_manager.sp_cos = sp_manager.slice_tokens(cos)
    sp_manager.sp_sin = sp_manager.slice_tokens(sin)
