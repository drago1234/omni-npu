import types
import importlib

import pytest
import torch
from vllm.v1 import kv_cache_interface as kv_iface

if not hasattr(kv_iface, "SinkFullAttentionSpec"):
    class _CompatSinkFullAttentionSpec(kv_iface.FullAttentionSpec):
        # Compatibility shim for older/newer vLLM APIs used during module import.
        sink_len: int = 0

        def __init__(self, *args, sink_len=0, **kwargs):
            super().__init__(*args, **kwargs)
            self.sink_len = sink_len
            self.head_size_v = None

        def set_head_size_v(self, value):
            self.head_size_v = value

    kv_iface.SinkFullAttentionSpec = _CompatSinkFullAttentionSpec

sink_patch_mod = importlib.import_module(
    "omni_npu.vllm_patches.patches.models.openpangu_vl.patch_m_static_sink_attention"
)
StaticSinkAttentionPatch = sink_patch_mod.StaticSinkAttentionPatch
create_static_sink_attention_backendPatch = sink_patch_mod.create_static_sink_attention_backendPatch
maybe_populate_sinkPatch = sink_patch_mod.maybe_populate_sinkPatch
from vllm.attention.backends.abstract import AttentionType


def test_create_static_sink_attention_backend_build_keeps_zero_len_unchanged(monkeypatch):
    create_static_sink_attention_backendPatch.create_static_sink_attention_backend.cache_clear()

    class _FakeUnderlyingBuilder:
        def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
            self.vllm_config = vllm_config
            self.device = device

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            return common_attn_metadata

    class _FakeUnderlyingBackend:
        @staticmethod
        def get_builder_cls():
            return _FakeUnderlyingBuilder

    monkeypatch.setattr(
        sink_patch_mod,
        "subclass_attention_backend",
        lambda name_prefix, attention_backend_cls, builder_cls: types.SimpleNamespace(
            builder_cls=builder_cls
        ),
    )

    backend = create_static_sink_attention_backendPatch.create_static_sink_attention_backend(
        _FakeUnderlyingBackend,
        sink_len=4,
    )
    builder = backend.builder_cls(
        kv_cache_spec=None,
        layer_names=[],
        vllm_config=types.SimpleNamespace(cache_config=types.SimpleNamespace(block_size=2)),
        device=torch.device("cpu"),
    )

    metadata = types.SimpleNamespace(
        seq_lens=torch.tensor([0, 2, 5], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([0, 2, 5], dtype=torch.int32),
        max_seq_len=5,
    )
    out = builder.build(0, metadata)
    assert out.seq_lens.tolist() == [0, 6, 9]
    assert out.seq_lens_cpu.tolist() == [0, 6, 9]
    assert out.max_seq_len == 9


def test_create_static_sink_attention_backend_build_adds_all_when_no_zero(monkeypatch):
    create_static_sink_attention_backendPatch.create_static_sink_attention_backend.cache_clear()

    class _FakeUnderlyingBuilder:
        def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
            pass

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            return common_attn_metadata

    class _FakeUnderlyingBackend:
        @staticmethod
        def get_builder_cls():
            return _FakeUnderlyingBuilder

    monkeypatch.setattr(
        sink_patch_mod,
        "subclass_attention_backend",
        lambda name_prefix, attention_backend_cls, builder_cls: types.SimpleNamespace(
            builder_cls=builder_cls
        ),
    )

    backend = create_static_sink_attention_backendPatch.create_static_sink_attention_backend(
        _FakeUnderlyingBackend,
        sink_len=3,
    )
    builder = backend.builder_cls(
        kv_cache_spec=None,
        layer_names=[],
        vllm_config=types.SimpleNamespace(cache_config=types.SimpleNamespace(block_size=2)),
        device=torch.device("cpu"),
    )

    metadata = types.SimpleNamespace(
        seq_lens=torch.tensor([1, 2], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([1, 2], dtype=torch.int32),
        max_seq_len=0,
    )
    out = builder.build(0, metadata)
    assert out.seq_lens.tolist() == [4, 5]
    assert out.seq_lens_cpu.tolist() == [4, 5]
    assert out.max_seq_len == 0


def test_maybe_populate_sink_only_populates_once(monkeypatch):
    class _Layer:
        def __init__(self):
            self.sink_populated = False
            self.calls = 0

        def populate_sink_kv(self, self_k_cache, self_v_cache):
            self.calls += 1

    layer = _Layer()
    monkeypatch.setattr(
        sink_patch_mod,
        "get_forward_context",
        lambda: types.SimpleNamespace(no_compile_layers={"layer.0": layer}),
    )

    k_cache = torch.ones(2, 2)
    v_cache = torch.ones(2, 2)

    maybe_populate_sinkPatch.maybe_populate_sink(k_cache, v_cache, "layer.0")
    assert layer.calls == 1

    layer.sink_populated = True
    maybe_populate_sinkPatch.maybe_populate_sink(k_cache, v_cache, "layer.0")
    assert layer.calls == 1


def test_maybe_populate_sink_skips_when_cache_empty(monkeypatch):
    class _Layer:
        def __init__(self):
            self.sink_populated = False
            self.calls = 0

        def populate_sink_kv(self, self_k_cache, self_v_cache):
            self.calls += 1

    layer = _Layer()
    monkeypatch.setattr(
        sink_patch_mod,
        "get_forward_context",
        lambda: types.SimpleNamespace(no_compile_layers={"layer.0": layer}),
    )

    empty = torch.empty(0)
    maybe_populate_sinkPatch.maybe_populate_sink(empty, empty, "layer.0")
    assert layer.calls == 0


def test_populate_sink_kv_writes_expected_slots(monkeypatch):
    attn = StaticSinkAttentionPatch.StaticSinkAttention.__new__(
        StaticSinkAttentionPatch.StaticSinkAttention
    )
    attn.block_size = 2
    attn.sink_len = 6
    attn.sink_populated = False
    attn.sink_key = torch.arange(18, dtype=torch.float32).reshape(6, 1, 3)
    attn.sink_value = (torch.arange(18, dtype=torch.float32) + 100).reshape(6, 1, 3)

    calls = []

    def _fake_scatter(dst, indices, updates):
        calls.append((dst.shape, indices.clone(), updates.clone()))
        return dst

    monkeypatch.setattr(
        sink_patch_mod.current_platform,
        "current_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(sink_patch_mod.torch_npu, "npu_scatter_nd_update_", _fake_scatter)

    k_cache = torch.zeros(8, 1, 3)
    v_cache = torch.zeros(8, 1, 3)
    attn.populate_sink_kv(k_cache, v_cache)

    assert len(calls) == 2
    expected_slots = torch.arange(2, 8, dtype=torch.long).unsqueeze(1)
    assert attn.sink_populated is True


def test_get_kv_cache_spec_uses_sink_full_attention_spec(monkeypatch):
    class _FakeSpec:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.head_size_v = None

        def set_head_size_v(self, value):
            self.head_size_v = value

    monkeypatch.setattr(sink_patch_mod, "SinkFullAttentionSpec", _FakeSpec)

    attn = StaticSinkAttentionPatch.StaticSinkAttention.__new__(
        StaticSinkAttentionPatch.StaticSinkAttention
    )
    attn.attn_type = AttentionType.DECODER
    attn.num_kv_heads = 8
    attn.head_size = 64
    attn.sink_len = 32
    attn.kv_cache_torch_dtype = torch.float16
    attn.head_size_v = 80

    cfg = types.SimpleNamespace(cache_config=types.SimpleNamespace(block_size=16))
    spec = attn.get_kv_cache_spec(cfg)

    assert spec.kwargs["block_size"] == 16
    assert spec.kwargs["num_kv_heads"] == 8
    assert spec.kwargs["head_size"] == 64
    assert spec.kwargs["sink_len"] == 32
    assert spec.kwargs["dtype"] == torch.float16
    assert spec.head_size_v == 80


@pytest.mark.parametrize("k_numel, sink_populated, expected_calls", [(0, False, 0), (4, True, 0), (4, False, 1)])
def test_maybe_populate_sink_branch_matrix(monkeypatch, k_numel, sink_populated, expected_calls):
    class _Layer:
        def __init__(self, populated):
            self.sink_populated = populated
            self.calls = 0

        def populate_sink_kv(self, self_k_cache, self_v_cache):
            self.calls += 1

    layer = _Layer(sink_populated)
    monkeypatch.setattr(
        sink_patch_mod,
        "get_forward_context",
        lambda: types.SimpleNamespace(no_compile_layers={"layer.0": layer}),
    )

    if k_numel == 0:
        k_cache = torch.empty(0)
        v_cache = torch.empty(0)
    else:
        k_cache = torch.ones(2, 2)
        v_cache = torch.ones(2, 2)

    maybe_populate_sinkPatch.maybe_populate_sink(k_cache, v_cache, "layer.0")
    assert layer.calls == expected_calls
