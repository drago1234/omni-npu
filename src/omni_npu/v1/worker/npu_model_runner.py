# SPDX-License-Identifier: Apache-2.0
# Minimal NPUModelRunner using composition over vLLM's GPUModelRunner with a
# torch.cuda -> torch.npu shim. This avoids subclassing GPUModelRunner.

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import Optional, Union, Any

import torch

from vllm.logger import init_logger
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ModelRunnerOutput,
)

logger = init_logger(__name__)


@contextmanager
def _torch_cuda_shim():
    class _EventPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda *a, **k: None
            self.synchronize = lambda *a, **k: None

    try:
        # Map common CUDA APIs used by GPUModelRunner onto NPU equivalents
        torch.cuda.Event = getattr(torch.npu, "Event", _EventPlaceholder)
        torch.cuda.Stream = getattr(torch.npu, "Stream", torch.cuda.Stream)
        torch.cuda.default_stream = getattr(torch.npu, "current_stream", torch.cuda.default_stream)
        torch.cuda.current_stream = getattr(torch.npu, "current_stream", torch.cuda.current_stream)
        torch.cuda.stream = getattr(torch.npu, "stream", torch.cuda.stream)
        torch.cuda.synchronize = getattr(torch.npu, "synchronize", lambda *a, **k: None)

        # Availability and device management
        torch.cuda.is_available = lambda: True
        torch.cuda.device_count = getattr(torch.npu, "device_count", lambda: 0)
        torch.cuda.current_device = getattr(torch.npu, "current_device", lambda: 0)
        torch.cuda.set_device = getattr(torch.npu, "set_device", lambda *a, **k: None)
        torch.cuda.empty_cache = getattr(torch.npu, "empty_cache", lambda: None)

        # Memory and device info
        torch.cuda.max_memory_allocated = getattr(torch.npu, "max_memory_allocated", lambda *a, **k: 0)
        torch.cuda.memory_allocated = getattr(torch.npu, "memory_allocated", lambda *a, **k: 0)
        torch.cuda.mem_get_info = getattr(torch.npu, "mem_get_info", lambda *a, **k: (0, 0))
        torch.cuda.get_device_name = getattr(torch.npu, "get_device_name", lambda *a, **k: "npu")

        # Device properties shim
        _orig_get_props = getattr(torch.npu, "get_device_properties", None)

        class _Props:
            def __init__(self, device):
                if _orig_get_props is not None:
                    p = _orig_get_props(device)
                    self.multi_processor_count = getattr(p, "multi_processor_count", 0)
                else:
                    self.multi_processor_count = 0

        torch.cuda.get_device_properties = lambda device: _Props(device)
        yield
    finally:
        # Best-effort reset: we intentionally do not restore originals to avoid
        # breaking other parts that may assume shimmed behavior during lifetime.
        pass


class NPUModelRunner:
    """
    NPU-native model runner that reuses vLLM's runner logic via inheritance,
    with a torch.cuda -> torch.npu shim applied at construction time.
    """

    def __init__(self, vllm_config, device: torch.device):  # type: ignore[no-redef]
        self.vllm_config = vllm_config
        self.device = device
        self._model_memory_usage = 0
        # Wire minimal state from VllmConfig
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = getattr(vllm_config, "speculative_config", None)
        self.lora_config = getattr(vllm_config, "lora_config", None)
        # Runner state used by KV init and binding
        self.kv_caches: list[torch.Tensor] = []
        self.shared_kv_cache_layers: dict[str, str] = {}
        self.runner_only_attn_layers: set[str] = set()
        self.attn_groups = []
        self.attn_metadata_builders = []
        self.model = None

    # Provide a writable property used by GPUModelRunner.load_model
    @property
    def model_memory_usage(self) -> int:
        return getattr(self, "_model_memory_usage", 0)

    @model_memory_usage.setter
    def model_memory_usage(self, val: int) -> None:
        self._model_memory_usage = int(val)

    def initialize_kv_cache(self, kv_cache_config) -> None:  # type: ignore[override]
        from functools import reduce
        import operator
        from vllm.v1.kv_cache_interface import AttentionSpec
        from vllm.v1.worker.utils import bind_kv_cache

        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self.may_reinitialize_input_batch(kv_cache_config)
        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)

        # Initialize attention backends only if not already initialized
        if not getattr(self, "attn_groups", None):
            self.initialize_attn_backend(kv_cache_config)

        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)

        kv_caches: dict[str, torch.Tensor] = {}
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                raw_tensor = kv_cache_raw_tensors[layer_name]
                if isinstance(kv_cache_spec, AttentionSpec):
                    num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        num_blocks,
                        kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        cache_dtype_str=self.cache_config.cache_dtype,
                    )
                    try:
                        kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
                        assert len(kv_cache_stride_order) == len(kv_cache_shape)
                    except (AttributeError, NotImplementedError):
                        kv_cache_stride_order = tuple(range(len(kv_cache_shape)))

                    # Start from backend-provided logical shape and adjust channels if the
                    # raw allocation contains both K and V concatenated.
                    permuted_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)
                    bytes_per_elem = torch.tensor([], dtype=kv_cache_spec.dtype).element_size()
                    raw_elems_in_dtype = raw_tensor.numel() // bytes_per_elem

                    block = kv_cache_spec.block_size
                    channels_expected = kv_cache_spec.num_kv_heads * kv_cache_spec.head_size
                    per_block_elems = block * channels_expected
                    if per_block_elems > 0 and raw_elems_in_dtype % per_block_elems == 0:
                        num_blocks_inferred = raw_elems_in_dtype // per_block_elems
                        permuted_shape = (num_blocks_inferred, block, channels_expected)
                    else:
                        # Fallback to backend-provided shape if divisible by a scalar factor
                        target_numel = reduce(operator.mul, permuted_shape, 1)
                        if raw_elems_in_dtype % target_numel == 0:
                            factor = raw_elems_in_dtype // target_numel
                            permuted_shape = (permuted_shape[0] * factor, *permuted_shape[1:])
                        else:
                            raise RuntimeError(
                                f"KV cache reshape mismatch: raw_elems={raw_elems_in_dtype}, "
                                f"per_block_elems={per_block_elems}, block={block}, channels={channels_expected}"
                            )
                    inv_order = [kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))]
                    base_dtype_view = raw_tensor.view(kv_cache_spec.dtype)
                    try:
                        kv_cache_view = base_dtype_view.view(permuted_shape)
                    except RuntimeError:
                        # Build candidate shapes in backend stride order and try them
                        candidates = []
                        # 1) Concatenate K and V along channels
                        cand1_base = (permuted_shape[inv_order.index(0)],  # num_blocks (original)
                                      permuted_shape[inv_order.index(1)],  # block
                                      permuted_shape[inv_order.index(2)] * 2)  # channels * 2
                        cand1_perm = tuple(cand1_base[i] for i in kv_cache_stride_order)
                        candidates.append(cand1_perm)
                        # 2) Infer num_blocks from raw size
                        if per_block_elems > 0 and raw_elems_in_dtype % per_block_elems == 0:
                            nb_inf = raw_elems_in_dtype // per_block_elems
                            cand2_base = (nb_inf, block, channels_expected)
                            cand2_perm = tuple(cand2_base[i] for i in kv_cache_stride_order)
                            candidates.append(cand2_perm)
                        # 3) Fallback: scale first dim by divisible factor
                        target_numel = reduce(operator.mul, permuted_shape, 1)
                        if target_numel > 0 and raw_elems_in_dtype % target_numel == 0:
                            factor = raw_elems_in_dtype // target_numel
                            cand3_base = (permuted_shape[inv_order.index(0)] * factor,
                                          permuted_shape[inv_order.index(1)],
                                          permuted_shape[inv_order.index(2)])
                            cand3_perm = tuple(cand3_base[i] for i in kv_cache_stride_order)
                            candidates.append(cand3_perm)
                        last_err = None
                        kv_cache_view = None
                        for cand in candidates:
                            try:
                                kv_cache_view = base_dtype_view.view(cand)
                                permuted_shape = cand
                                break
                            except RuntimeError as err:
                                last_err = err
                        if kv_cache_view is None:
                            # As a safe fallback for initialization: allocate a fresh buffer.
                            kv_cache_view = torch.empty(
                                permuted_shape,
                                dtype=kv_cache_spec.dtype,
                                device=raw_tensor.device,
                            )
                    kv_caches[layer_name] = kv_cache_view.permute(*inv_order)
                else:
                    raise

        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            kv_caches[layer_name] = kv_caches[target_layer_name]

        num_attn_module = 2 if getattr(self.model_config.hf_config, "model_type", "") == "longcat_flash" else 1
        bind_kv_cache(
            kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_caches,
            num_attn_module,
        )
        return None

    # Optional LoRA and misc delegations
    def add_lora(self, *args, **kwargs) -> bool:
        return False

    def remove_lora(self, *args, **kwargs) -> bool:
        return False

    def list_loras(self) -> set[int]:
        return set()

    def pin_lora(self, *args, **kwargs) -> bool:
        return False

    def maybe_remove_all_loras(self, *args, **kwargs):
        return None

    def take_draft_token_ids(self):
        return None

    def ensure_kv_transfer_shutdown(self):
        return None

    # Minimal API expected by Worker and KV init path
    def may_reinitialize_input_batch(self, kv_cache_config) -> None:
        return None

    def may_add_encoder_only_layers_to_kv_cache_config(self) -> None:
        return None

    def maybe_add_kv_sharing_layers_to_kv_cache_groups(self, kv_cache_config) -> None:
        try:
            from vllm.v1.worker.utils import add_kv_sharing_layers_to_kv_cache_groups
            add_kv_sharing_layers_to_kv_cache_groups(self.shared_kv_cache_layers, kv_cache_config.kv_cache_groups, self.runner_only_attn_layers)
        except Exception:
            pass

    def _allocate_kv_cache_tensors(self, kv_cache_config) -> dict[str, torch.Tensor]:
        # Allocate raw byte buffers (uint8) sized per KVCacheTensor.size and share across layers
        tensors: dict[str, torch.Tensor] = {}
        for spec in kv_cache_config.kv_cache_tensors:
            buf = torch.empty(spec.size, dtype=torch.uint8, device=self.device)
            for name in spec.shared_by:
                tensors[name] = buf
        return tensors

    def _kv_cache_spec_attn_group_iterator(self):
        # Build simple objects exposing kv_cache_spec, backend, layer_names
        from dataclasses import make_dataclass
        from vllm.v1.worker.utils import AttentionGroup
        from omni_npu.attention.backends.attention import AscendAttentionBackend
        GroupView = make_dataclass("GroupView", [("kv_cache_spec", object), ("backend", object), ("layer_names", list)])
        views = []
        if not self.attn_groups:
            return views
        for ag in self.attn_groups:
            views.append(GroupView(kv_cache_spec=ag.kv_cache_spec, backend=AscendAttentionBackend, layer_names=ag.layer_names))
        return views

    def initialize_attn_backend(self, kv_cache_config) -> None:
        from vllm.v1.worker.utils import AttentionGroup
        from omni_npu.attention.backends.attention import AscendAttentionBackend
        self.attn_groups = []
        for group in kv_cache_config.kv_cache_groups:
            ag = AttentionGroup.create_with_metadata_builders(
                AscendAttentionBackend,
                group.layer_names,
                group.kv_cache_spec,
                self.vllm_config,
                self.device,
                num_metadata_builders=1,
            )
            self.attn_groups.append(ag)

    # Basic model loader and accessors
    def load_model(self) -> None:
        from vllm.model_executor.model_loader import get_model
        with _torch_cuda_shim():
            self.model = get_model(vllm_config=self.vllm_config)
        try:
            m = getattr(self.model, "memory_usage", None)
            if m and hasattr(m, "consumed_memory"):
                self.model_memory_usage = m.consumed_memory
        except Exception:
            pass

    def get_model(self):
        return self.model

    def get_kv_cache_spec(self) -> dict[str, "KVCacheSpec"]:
        # Minimal port of GPU runner logic to build per-layer KV specs
        from vllm.v1.kv_cache_interface import (
            KVCacheSpec,
            FullAttentionSpec,
            SlidingWindowSpec,
            MLAAttentionSpec,
            CrossAttentionSpec,
        )
        from vllm.config.vllm import get_layers_from_vllm_config
        from vllm.attention.layer import Attention, AttentionType
        try:
            from vllm.attention.layers.chunked_local_attention import ChunkedLocalAttention
        except Exception:
            ChunkedLocalAttention = tuple()  # type: ignore

        block_size = self.vllm_config.cache_config.block_size
        use_mla = getattr(self.vllm_config.model_config, "use_mla", False)
        cache_dtype_str = self.vllm_config.cache_config.cache_dtype

        # Determine dtype to store KV cache
        import torch as _torch
        if cache_dtype_str == "bfloat16":
            kv_dtype = _torch.bfloat16
        elif cache_dtype_str == "auto":
            # Fallback to model dtype if available, else bf16
            kv_dtype = getattr(self.vllm_config.model_config, "dtype", _torch.bfloat16)
        else:
            # Default safe dtype
            kv_dtype = _torch.bfloat16

        kv_cache_spec: dict[str, KVCacheSpec] = {}
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for layer_name, attn_module in attn_layers.items():
            # Cross-layer KV sharing: skip and record mapping
            kv_tgt = getattr(attn_module, "kv_sharing_target_layer_name", None)
            if kv_tgt is not None:
                self.shared_kv_cache_layers[layer_name] = kv_tgt
                continue

            if attn_module.attn_type == AttentionType.DECODER:
                if getattr(attn_module, "sliding_window", None) is not None:
                    assert not use_mla, "MLA is not supported for slidingwindow"
                    kv_cache_spec[layer_name] = SlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=kv_dtype,
                        sliding_window=attn_module.sliding_window,  # type: ignore[attr-defined]
                    )
                elif use_mla:
                    kv_cache_spec[layer_name] = MLAAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=kv_dtype,
                        cache_dtype_str=cache_dtype_str,
                    )
                elif isinstance(attn_module, ChunkedLocalAttention):
                    # Minimal fallback: treat as full attention for now
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=kv_dtype,
                    )
                else:
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=kv_dtype,
                    )
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                kv_cache_spec[layer_name] = CrossAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=kv_dtype,
                )
            elif attn_module.attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
                # Encoder-only attention does not need KV cache
                self.runner_only_attn_layers.add(layer_name)
                continue
            else:
                raise ValueError(f"Unknown attention type: {attn_module.attn_type}")

        # Optional: handle extra module types (mamba/indexer). Omitted for brevity.
        return kv_cache_spec

    # Supported tasks minimal stub (pooling tasks etc.)
    def get_supported_tasks(self) -> tuple:
        # Minimal: declare generation support so Chat/Completions APIs are enabled
        return ("generate",)

    # Execution stubs
    def execute_model(self, scheduler_output, *args, **kwargs):
        # Minimal greedy decode: 1 token per request using model forward.
        from vllm.v1.outputs import ModelRunnerOutput

        # Gather request IDs scheduled this step
        new_req_ids = [r.req_id for r in scheduler_output.scheduled_new_reqs]
        cached_req_ids = list(scheduler_output.scheduled_cached_reqs.req_ids)
        req_ids = new_req_ids + cached_req_ids
        req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}

        # If nothing scheduled, return empty output
        if not req_ids:
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
                num_nans_in_logits=None,
            )

        # Build one input token per req for this step
        step_tokens: list[int] = []
        for r in scheduler_output.scheduled_new_reqs:
            tok = None
            if r.prompt_token_ids:
                tok = r.prompt_token_ids[-1]
            step_tokens.append(int(tok) if tok is not None else 0)
        # Cached reqs: zip to avoid out-of-range if lengths mismatch
        cached_new_tokens = scheduler_output.scheduled_cached_reqs.new_token_ids
        for rid, toks in zip(
            scheduler_output.scheduled_cached_reqs.req_ids, cached_new_tokens
        ):
            tok = toks[-1] if toks else None
            step_tokens.append(int(tok) if tok is not None else 0)
        # If there are more cached req_ids than provided new_token_ids, pad zeros
        extra = len(scheduler_output.scheduled_cached_reqs.req_ids) - len(cached_new_tokens)
        if extra > 0:
            step_tokens.extend([0] * extra)

        if len(step_tokens) != len(req_ids):
            # Safety: align lengths
            step_tokens = (step_tokens + [0] * len(req_ids))[: len(req_ids)]

        input_ids = torch.tensor(step_tokens, dtype=torch.long, device=self.device)
        input_ids = input_ids.view(-1, 1)  # [num_reqs, 1]

        # Run with a minimal valid forward context so attention layers can
        # access it via get_forward_context(). Avoid passing any None-valued
        # cudagraph mode to keep defaults.
        from vllm.forward_context import set_forward_context
        from vllm.v1.attention.backends.utils import CommonAttentionMetadata
        logits = None
        # Qwen2 models require explicit positions argument.
        # Derive a simple per-request position from scheduler state.
        pos_vals: list[int] = []
        for r in scheduler_output.scheduled_new_reqs:
            pos_vals.append(int(getattr(r, "num_computed_tokens", 0)))
        cached_num_comp = getattr(scheduler_output.scheduled_cached_reqs, "num_computed_tokens", [])
        # Zip to avoid mismatch, then pad if needed
        for i, _ in enumerate(scheduler_output.scheduled_cached_reqs.req_ids[: len(cached_num_comp)]):
            pos_vals.append(int(cached_num_comp[i]))
        if len(pos_vals) < len(req_ids):
            pos_vals.extend([0] * (len(req_ids) - len(pos_vals)))
        positions = torch.tensor(pos_vals, dtype=torch.long, device=self.device).view(-1, 1)

        # Build a minimal CommonAttentionMetadata for this step (1 token per req)
        num_reqs = len(req_ids)
        # Query starts: [0, 1, 2, ..., num_reqs]
        query_start_loc_cpu = torch.arange(0, num_reqs + 1, dtype=torch.int32)
        query_start_loc = query_start_loc_cpu.to(device=self.device, non_blocking=True)
        # Sequence lengths: use current position+1 per req (monotonic, >=1)
        seq_lens_cpu = torch.tensor([int(p.item()) + 1 for p in positions.view(-1)], dtype=torch.int32)
        seq_lens = seq_lens_cpu.to(device=self.device, non_blocking=True)
        # Number of computed tokens (before this step)
        num_comp_cpu = torch.tensor(pos_vals, dtype=torch.int32)
        # One token per req this step
        num_actual_tokens = num_reqs
        max_query_len = 1
        max_seq_len = int(seq_lens_cpu.max().item()) if num_reqs > 0 else 0
        # Build block tables from scheduler_output for group 0 (minimal support)
        # New reqs come first in req_ids, then cached reqs
        per_req_pages: list[list[int]] = []
        # Handle new requests
        for r in scheduler_output.scheduled_new_reqs:
            try:
                pages = list(r.block_ids[0]) if r.block_ids and len(r.block_ids) > 0 else []
            except Exception:
                pages = []
            per_req_pages.append(pages)
        # Handle cached requests
        cached_new_block_ids = getattr(scheduler_output.scheduled_cached_reqs, "new_block_ids", [])
        for i, rid in enumerate(scheduler_output.scheduled_cached_reqs.req_ids):
            pages = []
            if i < len(cached_new_block_ids) and cached_new_block_ids[i] is not None:
                try:
                    pages = list(cached_new_block_ids[i][0]) if len(cached_new_block_ids[i]) > 0 else []
                except Exception:
                    pages = []
            per_req_pages.append(pages)

        # Ensure at least one page per req (default to 0) and pad to equal length
        per_req_pages = [p if len(p) > 0 else [0] for p in per_req_pages]
        max_pages = max((len(p) for p in per_req_pages), default=1)
        padded_pages = [p + [p[-1]] * (max_pages - len(p)) for p in per_req_pages]
        block_table_tensor = torch.tensor(padded_pages, dtype=torch.int32, device=self.device)

        # Slot mapping: absolute slot within KV from position and block table
        block_size = int(getattr(self.vllm_config.cache_config, "block_size", 1) or 1)
        pos_flat = positions.view(-1)
        block_idx = (pos_flat // max(block_size, 1)).to(torch.int32)
        # Gather page indices for each req from padded block table
        req_indices = torch.arange(0, num_reqs, dtype=torch.int64, device=self.device)
        page_indices = block_table_tensor[req_indices, block_idx.clamp(max=max_pages - 1).to(torch.int64)]
        slot_mapping = (page_indices * max(block_size, 1) + (pos_flat % max(block_size, 1)).to(torch.int32)).to(self.device)

        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            num_computed_tokens_cpu=num_comp_cpu,
            num_reqs=num_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=block_table_tensor,
            slot_mapping=slot_mapping,
            causal=True,
        )

        # Build per-layer AscendMetadata with builders from attention groups
        attn_metadata_dict = {}
        for ag in self.attn_groups or []:
            builder = ag.get_metadata_builder(0)
            meta = builder.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)
            for ln in ag.layer_names:
                attn_metadata_dict[ln] = meta

        with set_forward_context(
            attn_metadata=attn_metadata_dict if attn_metadata_dict else self.compilation_config.static_forward_context,
            vllm_config=self.vllm_config,
        ):
            out = self.model(input_ids=input_ids, positions=positions)
        if hasattr(self.model, "compute_logits"):
            logits = self.model.compute_logits(out)
        else:
            logits = out

        # logits shape: [num_reqs, vocab_size] or [num_reqs, 1, vocab_size]
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        # Greedy pick
        next_ids = torch.argmax(logits, dim=-1)
        sampled_token_ids: list[list[int]] = [[int(t.item())] for t in next_ids]

        prompt_logprobs_dict = {}
        pooler_output = [None for _ in req_ids]

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids,
            logprobs=None,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=pooler_output,
            num_nans_in_logits=None,
        )

    def profile_run(self, *args, **kwargs) -> None:
        return None

    def capture_model(self, *args, **kwargs) -> int:
        return 0

    def _dummy_run(self, *args, **kwargs):
        return None
