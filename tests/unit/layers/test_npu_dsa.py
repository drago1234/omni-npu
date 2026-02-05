import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import omni_npu.v1.layers.attention.npu_dsa as npu_dsa_mod


# -----------------------------
# Helpers: minimal fake modules
# -----------------------------
class _FakeLinear:
    """Mimic FlashCommLinear-like modules: call(x) -> (y, None)."""

    def __init__(self, out_features: int, *, return_dtype=None):
        self.out_features = int(out_features)
        self.return_dtype = return_dtype

    def __call__(self, x: torch.Tensor):
        y = torch.zeros(
            (x.shape[0], self.out_features),
            device=x.device,
            dtype=(self.return_dtype or x.dtype),
        )
        return y, None


class _FakeLinearWithWeight(_FakeLinear):
    """Used for _forward_mlaprolog path: must expose .weight (and optionally .weight_scale)."""

    def __init__(self, in_features: int, out_features: int, *, return_dtype=torch.float32, has_weight_scale=False):
        super().__init__(out_features, return_dtype=return_dtype)
        self.weight = torch.zeros((out_features, in_features), dtype=return_dtype)
        if has_weight_scale:
            self.weight_scale = torch.ones((out_features,), dtype=torch.float32)


class _FakeLayerNorm:
    def __call__(self, x: torch.Tensor):
        return x


class _FakeRMSNorm:
    """Need to be callable AND provide .weight and .variance_epsilon."""

    def __init__(self, dim: int, eps: float = 1e-6, dtype=torch.float32):
        self.weight = torch.ones((dim,), dtype=dtype)
        self.variance_epsilon = eps

    def __call__(self, x: torch.Tensor):
        return x


def _make_decode_meta(bs: int, topk_tokens: int):
    class _Decode:
        def __init__(self):
            self.query_cumlens = torch.arange(1, bs + 1, dtype=torch.int32)
            self.seq_lens = torch.full((bs,), 10, dtype=torch.int32)
            self.block_table = torch.zeros((bs, 4), dtype=torch.int32)

    return _Decode()


def _make_prefill_meta(bs: int):
    class _Prefill:
        def __init__(self):
            self.query_start_loc = torch.tensor([0, bs], dtype=torch.int64)
            self.query_cumlens = torch.arange(1, bs + 1, dtype=torch.int32)
            self.seq_lens = torch.full((bs,), 10, dtype=torch.int32)
            self.block_table = torch.zeros((bs, 4), dtype=torch.int32)
            self.chunked_context = None

    return _Prefill()


# ======================================================================
# Indexer tests
# ======================================================================
class TestIndexer(unittest.TestCase):
    def _make_indexer_stub(
        self,
        *,
        hidden_size=16,
        q_lora_rank=12,
        topk_tokens=8,
        n_head=4,
        head_dim=8,
        rope_dim=4,
        dtype=torch.float32,
    ):
        idx = SimpleNamespace()
        idx.topk_tokens = topk_tokens
        idx.n_head = n_head
        idx.head_dim = head_dim
        idx.rope_dim = rope_dim
        idx.q_lora_rank = q_lora_rank

        idx.wq_b = _FakeLinear(n_head * head_dim, return_dtype=dtype)
        idx.wk = _FakeLinear(head_dim, return_dtype=dtype)
        idx.k_norm = _FakeLayerNorm()
        idx.weights_proj = _FakeLinear(n_head, return_dtype=dtype)

        idx._apply_lightning_indexer = lambda q, weights, attn_metadata, kv_cache: (
            npu_dsa_mod.Indexer._apply_lightning_indexer(idx, q, weights, attn_metadata, kv_cache)
        )

        return idx

    def test_indexer_apply_lightning_indexer_prefill_or_decode_selects_metadata(self):
        idx = self._make_indexer_stub(topk_tokens=8, n_head=4, head_dim=8, rope_dim=4)
        bs = 3

        q = torch.randn((bs, idx.n_head, idx.head_dim), dtype=torch.float32)
        weights = torch.randn((bs, idx.n_head), dtype=torch.float32)
        kv_cache = (None, None, torch.randn((16, 1, idx.head_dim), dtype=torch.float32))

        attn_metadata = SimpleNamespace(prefill=_make_prefill_meta(bs), decode=None)

        def _fake_lightning_indexer(**kwargs):
            self.assertEqual(kwargs.get("layout_query"), "TND")
            self.assertEqual(kwargs.get("layout_key"), "PA_BSND")
            self.assertEqual(kwargs.get("sparse_count"), idx.topk_tokens)
            self.assertEqual(kwargs.get("sparse_mode"), 3)
            return torch.zeros((bs, 1, idx.topk_tokens), dtype=torch.int32)

        with patch.object(npu_dsa_mod.torch.ops, "custom", create=True) as custom_ns:
            custom_ns.npu_lightning_indexer = MagicMock(side_effect=_fake_lightning_indexer)
            out = npu_dsa_mod.Indexer._apply_lightning_indexer(idx, q, weights, attn_metadata, kv_cache)
        self.assertEqual(tuple(out.shape), (bs, 1, idx.topk_tokens))

        attn_metadata = SimpleNamespace(prefill=None, decode=_make_decode_meta(bs, idx.topk_tokens))
        with patch.object(npu_dsa_mod.torch.ops, "custom", create=True) as custom_ns:
            custom_ns.npu_lightning_indexer = MagicMock(side_effect=_fake_lightning_indexer)
            out2 = npu_dsa_mod.Indexer._apply_lightning_indexer(idx, q, weights, attn_metadata, kv_cache)
        self.assertEqual(tuple(out2.shape), (bs, 1, idx.topk_tokens))

    def test_indexer_forward_applies_rotary_and_calls_lightning_indexer_and_returns_k(self):
        T = 5
        hidden_size = 16
        q_lora_rank = 12

        idx = self._make_indexer_stub(
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            topk_tokens=8,
            n_head=4,
            head_dim=8,
            rope_dim=4,
            dtype=torch.float32,
        )

        hidden_states = torch.randn((T, hidden_size), dtype=torch.float32)
        qr = torch.randn((T, q_lora_rank), dtype=torch.float32)

        cos = torch.zeros((T, 1, 1, idx.rope_dim), dtype=torch.float32)
        sin = torch.zeros((T, 1, 1, idx.rope_dim), dtype=torch.float32)

        attn_metadata = SimpleNamespace()
        attn_metadata.prefill = None
        attn_metadata.decode = _make_decode_meta(T, idx.topk_tokens)
        attn_metadata.slot_mapping = torch.arange(T, dtype=torch.int64)

        k_dim = idx.head_dim
        kv_cache = (None, None, torch.zeros((T, 1, k_dim), dtype=torch.float32))

        rotary_calls = {"n": 0}
        lightning_calls = {"n": 0}
        scatter_calls = {"n": 0}

        def _fake_rotary_mul(x, cos_arg, sin_arg):
            rotary_calls["n"] += 1
            return x

        def _fake_scatter(dst, indices, src):
            scatter_calls["n"] += 1
            return dst

        def _fake_lightning_indexer(**kwargs):
            lightning_calls["n"] += 1
            return torch.zeros((T, 1, idx.topk_tokens), dtype=torch.int32)

        with patch.object(npu_dsa_mod.torch_npu, "npu_rotary_mul", side_effect=_fake_rotary_mul, create=True), \
             patch.object(npu_dsa_mod.torch_npu, "npu_scatter_nd_update_", side_effect=_fake_scatter, create=True), \
             patch.object(npu_dsa_mod.torch.ops, "custom", create=True) as custom_ns:
            custom_ns.npu_lightning_indexer = MagicMock(side_effect=_fake_lightning_indexer)

            topk_indices, k_out = npu_dsa_mod.Indexer.forward(
                idx,
                hidden_states=hidden_states,
                qr=qr,
                cos=cos,
                sin=sin,
                attn_metadata=attn_metadata,
                kv_cache=kv_cache,
            )

        self.assertEqual(rotary_calls["n"], 2)
        self.assertEqual(lightning_calls["n"], 1)
        self.assertEqual(scatter_calls["n"], 1)
        self.assertEqual(tuple(topk_indices.shape), (T, 1, idx.topk_tokens))
        self.assertEqual(tuple(k_out.shape), (T, 1, idx.head_dim))

    def test_indexer_forward_when_kv_cache_indexer_none_skips_scatter(self):
        T = 4
        idx = self._make_indexer_stub(topk_tokens=8, n_head=4, head_dim=8, rope_dim=4)
        hidden_states = torch.randn((T, 16), dtype=torch.float32)
        qr = torch.randn((T, idx.q_lora_rank), dtype=torch.float32)
        cos = torch.zeros((T, 1, 1, idx.rope_dim), dtype=torch.float32)
        sin = torch.zeros((T, 1, 1, idx.rope_dim), dtype=torch.float32)

        attn_metadata = SimpleNamespace(
            prefill=None,
            decode=_make_decode_meta(T, idx.topk_tokens),
            slot_mapping=torch.arange(T, dtype=torch.int64),
        )

        kv_cache = (None, None, None)

        scatter_calls = {"n": 0}

        def _fake_scatter(*args, **kwargs):
            scatter_calls["n"] += 1
            return args[0]

        with patch.object(npu_dsa_mod.torch_npu, "npu_rotary_mul", side_effect=lambda x, c, s: x, create=True), \
             patch.object(npu_dsa_mod.torch_npu, "npu_scatter_nd_update_", side_effect=_fake_scatter, create=True), \
             patch.object(npu_dsa_mod.torch.ops, "custom", create=True) as custom_ns:
            custom_ns.npu_lightning_indexer = MagicMock(return_value=torch.zeros((T, 1, idx.topk_tokens), dtype=torch.int32))
            topk, k = npu_dsa_mod.Indexer.forward(idx, hidden_states, qr, cos, sin, attn_metadata, kv_cache)

        self.assertEqual(scatter_calls["n"], 0)
        self.assertEqual(tuple(topk.shape), (T, 1, idx.topk_tokens))
        self.assertEqual(tuple(k.shape), (T, 1, idx.head_dim))


# ======================================================================
# forward routing
# ======================================================================
class TestNPUDeepseekSparseAttentionForwardRouting(unittest.TestCase):
    def _make_attn_stub(self):
        m = SimpleNamespace()
        m.prefix = "layers.0"
        m._forward_prefill = MagicMock(return_value=torch.tensor([111]))
        m._forward_decode = MagicMock(return_value=torch.tensor([222]))
        return m

    def test_forward_routes_none_and_prefill_and_decode(self):
        m = self._make_attn_stub()
        hidden_states = torch.randn((3, 16), dtype=torch.float32)
        cos = torch.zeros((3, 1, 1, 4), dtype=torch.float32)
        sin = torch.zeros((3, 1, 1, 4), dtype=torch.float32)

        fc = SimpleNamespace(attn_metadata=None, virtual_engine=0)
        with patch.object(npu_dsa_mod, "get_forward_context", return_value=fc):
            out = npu_dsa_mod.NPUDeepseekSparseAttention.forward(m, hidden_states, cos, sin)
        self.assertTrue(torch.equal(out, torch.tensor([111])))

        m._forward_prefill.reset_mock()
        m._forward_decode.reset_mock()
        meta = SimpleNamespace(prefill=object(), decode=None)
        fc = SimpleNamespace(attn_metadata=meta, virtual_engine=0)
        with patch.object(npu_dsa_mod, "get_forward_context", return_value=fc):
            out = npu_dsa_mod.NPUDeepseekSparseAttention.forward(m, hidden_states, cos, sin)
        self.assertTrue(torch.equal(out, torch.tensor([111])))

        m._forward_prefill.reset_mock()
        m._forward_decode.reset_mock()
        meta = SimpleNamespace(prefill=None, decode=object())
        fc = SimpleNamespace(attn_metadata=meta, virtual_engine=0)
        with patch.object(npu_dsa_mod, "get_forward_context", return_value=fc):
            out = npu_dsa_mod.NPUDeepseekSparseAttention.forward(m, hidden_states, cos, sin)
        self.assertTrue(torch.equal(out, torch.tensor([222])))

    def test_forward_attn_metadata_dict_extracts_by_prefix_attn(self):
        m = self._make_attn_stub()
        m.prefix = "layers.7"
        hidden_states = torch.randn((2, 16), dtype=torch.float32)
        cos = torch.zeros((2, 1, 1, 4), dtype=torch.float32)
        sin = torch.zeros((2, 1, 1, 4), dtype=torch.float32)

        meta = SimpleNamespace(prefill=None, decode=object())
        fc = SimpleNamespace(attn_metadata={f"{m.prefix}.attn": meta}, virtual_engine=0)

        with patch.object(npu_dsa_mod, "get_forward_context", return_value=fc):
            out = npu_dsa_mod.NPUDeepseekSparseAttention.forward(m, hidden_states, cos, sin)

        self.assertTrue(torch.equal(out, torch.tensor([222])))
        self.assertEqual(m._forward_decode.call_count, 1)


# ======================================================================
# Core methods
# ======================================================================
class TestNPUDeepseekSparseAttentionCore(unittest.TestCase):
    def _make_core_stub(self):
        m = SimpleNamespace()
        m.num_local_heads = 2
        m.kv_lora_rank = 3
        m.v_head_dim = 4
        m.scaling = 0.5

        impl = SimpleNamespace()
        impl.W_UV = torch.randn((m.num_local_heads, m.kv_lora_rank, m.v_head_dim), dtype=torch.float32)
        m.attn = SimpleNamespace(impl=impl)

        m.o_proj = MagicMock(side_effect=lambda x: (torch.ones((x.shape[0], 8), dtype=x.dtype), None))
        return m

    def test_apply_attention_returns_zeros_when_metadata_none(self):
        m = self._make_core_stub()
        q_nope = torch.randn((5, m.num_local_heads, m.kv_lora_rank), dtype=torch.float32)
        q_pe = torch.randn((5, m.num_local_heads, 4), dtype=torch.float32)
        k_nope = torch.randn((5, 1, 1, m.kv_lora_rank), dtype=torch.float32)
        k_pe = torch.randn((5, 1, 1, 4), dtype=torch.float32)

        out = npu_dsa_mod.NPUDeepseekSparseAttention._apply_attention(
            m,
            topk_indices=torch.zeros((5, 1, 8), dtype=torch.int32),
            q_nope=q_nope,
            q_pe=q_pe,
            k_nope=k_nope,
            k_pe=k_pe,
            attn_metadata=None,
        )
        self.assertEqual(tuple(out.shape), (5, m.num_local_heads, m.kv_lora_rank))
        self.assertTrue(torch.all(out == 0))

    def test_apply_attention_calls_custom_sparse_flash_attention_with_decode(self):
        m = self._make_core_stub()
        meta = SimpleNamespace(prefill=None, decode=_make_decode_meta(3, 8))

        q_nope = torch.randn((3, m.num_local_heads, m.kv_lora_rank), dtype=torch.float32)
        q_pe = torch.randn((3, m.num_local_heads, 4), dtype=torch.float32)
        k_nope = torch.randn((3, 1, 1, m.kv_lora_rank), dtype=torch.float32)
        k_pe = torch.randn((3, 1, 1, 4), dtype=torch.float32)
        topk_indices = torch.zeros((3, 1, 8), dtype=torch.int32)

        def _fake_sparse_flash_attention(**kwargs):
            B = kwargs["query"].shape[0]
            N = kwargs["query"].shape[1]
            return torch.zeros((B, N, m.kv_lora_rank), dtype=torch.float32)

        with patch.object(npu_dsa_mod.torch.ops, "custom", create=True) as custom_ns:
            custom_ns.npu_sparse_flash_attention = MagicMock(side_effect=_fake_sparse_flash_attention)
            out = npu_dsa_mod.NPUDeepseekSparseAttention._apply_attention(
                m,
                topk_indices=topk_indices,
                q_nope=q_nope,
                q_pe=q_pe,
                k_nope=k_nope,
                k_pe=k_pe,
                attn_metadata=meta,
            )

        self.assertEqual(tuple(out.shape), (3, m.num_local_heads, m.kv_lora_rank))
        self.assertEqual(custom_ns.npu_sparse_flash_attention.call_count, 1)

    def test_apply_attention_calls_custom_sparse_flash_attention_with_prefill(self):
        m = self._make_core_stub()
        meta = SimpleNamespace(prefill=_make_prefill_meta(3), decode=None)

        q_nope = torch.randn((3, m.num_local_heads, m.kv_lora_rank), dtype=torch.float32)
        q_pe = torch.randn((3, m.num_local_heads, 4), dtype=torch.float32)
        k_nope = torch.randn((3, 1, 1, m.kv_lora_rank), dtype=torch.float32)
        k_pe = torch.randn((3, 1, 1, 4), dtype=torch.float32)
        topk_indices = torch.zeros((3, 1, 8), dtype=torch.int32)

        with patch.object(npu_dsa_mod.torch.ops, "custom", create=True) as custom_ns:
            custom_ns.npu_sparse_flash_attention = MagicMock(
                return_value=torch.zeros((3, m.num_local_heads, m.kv_lora_rank), dtype=torch.float32)
            )
            out = npu_dsa_mod.NPUDeepseekSparseAttention._apply_attention(
                m,
                topk_indices=topk_indices,
                q_nope=q_nope,
                q_pe=q_pe,
                k_nope=k_nope,
                k_pe=k_pe,
                attn_metadata=meta,
            )
        self.assertEqual(tuple(out.shape), (3, m.num_local_heads, m.kv_lora_rank))
        self.assertEqual(custom_ns.npu_sparse_flash_attention.call_count, 1)

    def test_mla_epilog_matmul_then_calls_o_proj_and_returns_first(self):
        m = self._make_core_stub()
        B = 6
        attn_output = torch.randn((B, m.num_local_heads, m.kv_lora_rank), dtype=torch.float32)
        out = npu_dsa_mod.NPUDeepseekSparseAttention._mla_epilog(m, attn_output)
        self.assertEqual(tuple(out.shape), (B, 8))
        self.assertTrue(m.o_proj.called)
        self.assertTrue(torch.all(out == 1))


# ======================================================================
# Prefill / Decode high-coverage tests
# ======================================================================
class TestNPUDeepseekSparseAttentionPrefillDecode(unittest.TestCase):
    def _make_attn_impl_stub(self, *, use_omni_cache=False, use_mlaprolog=False, q_lora_rank=12, quant_symbol=False):
        m = SimpleNamespace()

        m.hidden_size = 16
        m.num_local_heads = 2
        m.qk_nope_head_dim = 4
        m.qk_rope_head_dim = 4
        m.qk_head_dim = m.qk_nope_head_dim + m.qk_rope_head_dim
        m.v_head_dim = 4
        m.kv_lora_rank = 3
        m.scaling = 0.5

        m.q_lora_rank = q_lora_rank
        m.use_omni_cache = use_omni_cache
        m.use_mlaprolog = use_mlaprolog
        m.layer_idx = 0

        m.quant_symbol = quant_symbol

        if m.q_lora_rank is not None:
            m.q_a_proj = _FakeLinear(m.q_lora_rank, return_dtype=torch.float32)
            m.q_proj = None
        else:
            m.q_a_proj = None
            m.q_proj = _FakeLinear(m.num_local_heads * m.qk_head_dim, return_dtype=torch.float32)

        m.kv_a_proj_with_mqa = _FakeLinear(m.kv_lora_rank + m.qk_rope_head_dim, return_dtype=torch.float32)
        m.q_a_layernorm = _FakeLayerNorm()

        m.kv_a_layernorm = _FakeRMSNorm(m.kv_lora_rank, eps=1e-6, dtype=torch.float32)
        m.q_b_proj = _FakeLinear(m.num_local_heads * m.qk_head_dim, return_dtype=torch.float32)
        m.o_proj = MagicMock(side_effect=lambda x: (torch.zeros((x.shape[0], 8), dtype=x.dtype), None))

        if use_mlaprolog:
            m.q_a_proj = _FakeLinearWithWeight(m.hidden_size, m.q_lora_rank, has_weight_scale=False)
            m.q_b_proj = _FakeLinearWithWeight(m.q_lora_rank, m.num_local_heads * m.qk_head_dim, has_weight_scale=quant_symbol)
            m.kv_a_proj_with_mqa = _FakeLinearWithWeight(m.hidden_size, m.kv_lora_rank + m.qk_rope_head_dim, has_weight_scale=False)
            m.q_a_layernorm = _FakeRMSNorm(m.q_lora_rank, eps=1e-6, dtype=torch.float32)

        impl = SimpleNamespace()
        impl.W_UK_T = torch.randn((m.num_local_heads, m.qk_nope_head_dim, m.kv_lora_rank), dtype=torch.float32)
        impl.W_UV = torch.randn((m.num_local_heads, m.kv_lora_rank, m.v_head_dim), dtype=torch.float32)

        if use_omni_cache:
            m.attn = SimpleNamespace(impl=impl, kv_cache=[None])
        else:
            kv0 = torch.zeros((64, 1, 1, m.kv_lora_rank), dtype=torch.float32)
            kv1 = torch.zeros((64, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)
            m.attn = SimpleNamespace(impl=impl, kv_cache=[(kv0, kv1)])

        def _fake_indexer(hidden_states, q_norm, cos, sin, attn_metadata, kv_cache):
            bs = hidden_states.shape[0]
            return (
                torch.zeros((bs, 1, 8), dtype=torch.int32),
                torch.zeros((bs, 1, m.qk_rope_head_dim), dtype=torch.float32),
            )

        m.indexer = MagicMock(side_effect=_fake_indexer)

        m._apply_attention = lambda *args, **kwargs: npu_dsa_mod.NPUDeepseekSparseAttention._apply_attention(m, *args, **kwargs)
        m._mla_epilog = lambda attn_output: npu_dsa_mod.NPUDeepseekSparseAttention._mla_epilog(m, attn_output)
        m._forward_prefill = lambda *args, **kwargs: npu_dsa_mod.NPUDeepseekSparseAttention._forward_prefill(m, *args, **kwargs)
        m._forward_decode = lambda *args, **kwargs: npu_dsa_mod.NPUDeepseekSparseAttention._forward_decode(m, *args, **kwargs)
        m._forward_mlaprolog = lambda *args, **kwargs: npu_dsa_mod.NPUDeepseekSparseAttention._forward_mlaprolog(m, *args, **kwargs)

        m.actual_seq_lengths = {
            1: torch.tensor([1], dtype=torch.int64),
            2: torch.tensor([1, 2], dtype=torch.int64),
            3: torch.tensor([1, 2, 3], dtype=torch.int64),
            4: torch.tensor([1, 2, 3, 4], dtype=torch.int64),
        }
        m.num_speculative_tokens = 0
        m.tp_size = 1

        return m

    def test_forward_prefill_attn_metadata_none_hits_zeros_attention_and_epilog(self):
        m = self._make_attn_impl_stub(use_omni_cache=False, use_mlaprolog=False, q_lora_rank=12)
        bs = 4
        hidden_states = torch.randn((bs, m.hidden_size), dtype=torch.float32)
        cos = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)
        sin = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)

        fc = SimpleNamespace(attn_metadata=None, virtual_engine=0)

        with patch.object(npu_dsa_mod, "get_forward_context", return_value=fc), \
             patch.object(npu_dsa_mod.torch_npu, "npu_interleave_rope", side_effect=lambda x, c, s: x, create=True):
            out = m._forward_prefill(hidden_states, cos, sin, attn_metadata=None)

        self.assertEqual(tuple(out.shape), (bs, 8))
        self.assertEqual(m.indexer.call_count, 0)

    def test_forward_prefill_with_metadata_calls_kv_rmsnorm_rope_cache_and_indexer_and_attention(self):
        m = self._make_attn_impl_stub(use_omni_cache=False, use_mlaprolog=False, q_lora_rank=12)
        bs = 3
        hidden_states = torch.randn((bs, m.hidden_size), dtype=torch.float32)
        cos = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)
        sin = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)

        attn_metadata = SimpleNamespace(
            prefill=_make_prefill_meta(bs),
            decode=None,
            slot_mapping=torch.arange(bs, dtype=torch.int64),
        )
        fc = SimpleNamespace(attn_metadata=attn_metadata, virtual_engine=0)

        def _fake_kv_rmsnorm_rope_cache(*args, **kwargs):
            k_pe = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)
            k_nope = torch.zeros((bs, 1, 1, m.kv_lora_rank), dtype=torch.float32)
            return k_pe, k_nope, None, None

        def _fake_sparse_flash_attention(**kwargs):
            B = kwargs["query"].shape[0]
            N = kwargs["query"].shape[1]
            return torch.zeros((B, N, m.kv_lora_rank), dtype=torch.float32)

        with patch.object(npu_dsa_mod, "get_forward_context", return_value=fc), \
             patch.object(npu_dsa_mod.torch_npu, "npu_interleave_rope", side_effect=lambda x, c, s: x, create=True), \
             patch.object(npu_dsa_mod.torch_npu, "npu_kv_rmsnorm_rope_cache", side_effect=_fake_kv_rmsnorm_rope_cache, create=True), \
             patch.object(npu_dsa_mod.torch.ops, "custom", create=True) as custom_ns:
            custom_ns.npu_sparse_flash_attention = MagicMock(side_effect=_fake_sparse_flash_attention)
            out = m._forward_prefill(hidden_states, cos, sin, attn_metadata=attn_metadata)

        self.assertEqual(tuple(out.shape), (bs, 8))
        self.assertEqual(m.indexer.call_count, 1)
        self.assertEqual(custom_ns.npu_sparse_flash_attention.call_count, 1)

    def test_forward_prefill_with_omni_cache_triggers_synchronize_d2h(self):
        m = self._make_attn_impl_stub(use_omni_cache=True, use_mlaprolog=False, q_lora_rank=12)
        bs = 2
        hidden_states = torch.randn((bs, m.hidden_size), dtype=torch.float32)
        cos = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)
        sin = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)

        attn_metadata = SimpleNamespace(
            prefill=_make_prefill_meta(bs),
            decode=None,
            slot_mapping=torch.arange(bs, dtype=torch.int64),
        )
        fc = SimpleNamespace(attn_metadata=attn_metadata, virtual_engine=0)

        fake_omni_cache_obj = SimpleNamespace(
            device_cache=(
                torch.zeros((64, 1, 1, m.kv_lora_rank), dtype=torch.float32),
                torch.zeros((64, 1, 1, m.qk_rope_head_dim), dtype=torch.float32),
            ),
            synchronize_d2h=MagicMock(),
        )
        fake_cache_mod = types.ModuleType("omni_cache.cache")
        fake_cache_mod.omni_cache = fake_omni_cache_obj

        class _FakeEvent:
            def __init__(self, *args, **kwargs):
                pass

            def record(self, stream):
                return None

        def _fake_sparse_flash_attention(**kwargs):
            B = kwargs["query"].shape[0]
            N = kwargs["query"].shape[1]
            return torch.zeros((B, N, m.kv_lora_rank), dtype=torch.float32)

        with patch.object(npu_dsa_mod, "get_forward_context", return_value=fc), \
             patch.object(npu_dsa_mod, "current_stream", return_value=object(), create=True), \
             patch.object(torch, "npu", create=True) as torch_npu_ns, \
             patch.object(npu_dsa_mod.torch_npu, "npu_interleave_rope", side_effect=lambda x, c, s: x, create=True), \
             patch.object(npu_dsa_mod.torch.ops, "custom", create=True) as custom_ns, \
             patch.dict(sys.modules, {"omni_cache.cache": fake_cache_mod}):
            torch_npu_ns.Event = _FakeEvent
            custom_ns.npu_sparse_flash_attention = MagicMock(side_effect=_fake_sparse_flash_attention)
            out = m._forward_prefill(hidden_states, cos, sin, attn_metadata=attn_metadata)

        self.assertEqual(tuple(out.shape), (bs, 8))
        self.assertTrue(fake_omni_cache_obj.synchronize_d2h.called)

    def test_forward_prefill_use_omni_cache_true_attn_metadata_none_does_not_import_omni_cache_and_skips_indexer(self):
        m = self._make_attn_impl_stub(use_omni_cache=True, use_mlaprolog=False, q_lora_rank=12)
        bs = 3
        hidden_states = torch.randn((bs, m.hidden_size), dtype=torch.float32)
        cos = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)
        sin = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)

        fc = SimpleNamespace(attn_metadata=None, virtual_engine=0)

        class _Boom(types.ModuleType):
            def __getattr__(self, item):
                raise AssertionError("should not import omni_cache when attn_metadata is None")

        with patch.object(npu_dsa_mod, "get_forward_context", return_value=fc), \
             patch.object(npu_dsa_mod.torch_npu, "npu_interleave_rope", side_effect=lambda x, c, s: x, create=True), \
             patch.dict(sys.modules, {"omni_cache.cache": _Boom("omni_cache.cache")}):
            out = m._forward_prefill(hidden_states, cos, sin, attn_metadata=None)

        self.assertEqual(tuple(out.shape), (bs, 8))
        self.assertEqual(m.indexer.call_count, 0)

    def test_forward_decode_use_mlaprolog_false_q_lora_rank_not_none_path(self):
        m = self._make_attn_impl_stub(use_omni_cache=False, use_mlaprolog=False, q_lora_rank=12)
        bs = 3
        hidden_states = torch.randn((bs, m.hidden_size), dtype=torch.float32)
        cos = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)
        sin = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)

        attn_metadata = SimpleNamespace(
            prefill=None,
            decode=_make_decode_meta(bs, 8),
            slot_mapping=torch.arange(bs, dtype=torch.int64),
        )
        fc = SimpleNamespace(attn_metadata=attn_metadata, virtual_engine=0)

        def _fake_kv_rmsnorm_rope_cache(*args, **kwargs):
            k_pe = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)
            k_nope = torch.zeros((bs, 1, 1, m.kv_lora_rank), dtype=torch.float32)
            return k_pe, k_nope, None, None

        def _fake_sparse_flash_attention(**kwargs):
            B = kwargs["query"].shape[0]
            N = kwargs["query"].shape[1]
            return torch.zeros((B, N, m.kv_lora_rank), dtype=torch.float32)

        with patch.object(npu_dsa_mod, "get_forward_context", return_value=fc), \
             patch.object(npu_dsa_mod.torch_npu, "npu_kv_rmsnorm_rope_cache", side_effect=_fake_kv_rmsnorm_rope_cache, create=True), \
             patch.object(npu_dsa_mod.torch_npu, "npu_interleave_rope", side_effect=lambda x, c, s: x, create=True), \
             patch.object(npu_dsa_mod.torch.ops, "custom", create=True) as custom_ns:
            custom_ns.npu_sparse_flash_attention = MagicMock(side_effect=_fake_sparse_flash_attention)
            out = m._forward_decode(hidden_states, cos, sin, attn_metadata=attn_metadata)

        self.assertEqual(tuple(out.shape), (bs, 8))
        self.assertTrue(m.indexer.called)
        self.assertEqual(custom_ns.npu_sparse_flash_attention.call_count, 1)

    def test_forward_decode_use_mlaprolog_true_quant_symbol_true_wraps_q_norm_dict(self):
        m = self._make_attn_impl_stub(use_omni_cache=False, use_mlaprolog=True, q_lora_rank=12, quant_symbol=True)
        bs = 3
        hidden_states = torch.randn((bs, m.hidden_size), dtype=torch.float32)
        cos = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)
        sin = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)

        attn_metadata = SimpleNamespace(
            prefill=None,
            decode=_make_decode_meta(bs, 8),
            slot_mapping=torch.arange(bs, dtype=torch.int64),
        )
        fc = SimpleNamespace(attn_metadata=attn_metadata, virtual_engine=0)

        def _fake_mla_prolog_v3(**kwargs):
            q_nope = torch.zeros((bs, m.num_local_heads, m.kv_lora_rank), dtype=torch.float32)
            q_pe = torch.zeros((bs, m.num_local_heads, m.qk_rope_head_dim), dtype=torch.float32)
            dequant_scale_q_nope = torch.ones((bs,), dtype=torch.float32)
            q_norm = torch.zeros((bs, m.q_lora_rank), dtype=torch.int8)
            dequant_scale_q_norm = torch.ones((bs,), dtype=torch.float32)
            return q_nope, q_pe, dequant_scale_q_nope, q_norm, dequant_scale_q_norm

        def _fake_sparse_flash_attention(**kwargs):
            B = kwargs["query"].shape[0]
            N = kwargs["query"].shape[1]
            return torch.zeros((B, N, m.kv_lora_rank), dtype=torch.float32)

        def _indexer_assert_q_norm_dict(hidden_states, q_norm, cos, sin, attn_metadata, kv_cache):
            self.assertIsInstance(q_norm, dict)
            self.assertIn("x_int8", q_norm)
            self.assertIn("pertoken_scale", q_norm)
            bs_local = hidden_states.shape[0]
            return (
                torch.zeros((bs_local, 1, 8), dtype=torch.int32),
                torch.zeros((bs_local, 1, m.qk_rope_head_dim), dtype=torch.float32),
            )

        m.indexer = MagicMock(side_effect=_indexer_assert_q_norm_dict)

        with patch.object(npu_dsa_mod, "get_forward_context", return_value=fc), \
             patch.object(npu_dsa_mod.torch.ops, "custom", create=True) as custom_ns:
            custom_ns.npu_mla_prolog_v3 = MagicMock(side_effect=_fake_mla_prolog_v3)
            custom_ns.npu_sparse_flash_attention = MagicMock(side_effect=_fake_sparse_flash_attention)

            out = m._forward_decode(hidden_states, cos, sin, attn_metadata=attn_metadata)

        self.assertEqual(tuple(out.shape), (bs, 8))
        self.assertTrue(custom_ns.npu_mla_prolog_v3.called)
        self.assertTrue(m.indexer.called)
        self.assertEqual(custom_ns.npu_sparse_flash_attention.call_count, 1)

    def test_forward_decode_use_mlaprolog_true_quant_symbol_false_passes_tensor_q_norm(self):
        m = self._make_attn_impl_stub(use_omni_cache=False, use_mlaprolog=True, q_lora_rank=12, quant_symbol=False)
        bs = 2
        hidden_states = torch.randn((bs, m.hidden_size), dtype=torch.float32)
        cos = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)
        sin = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)

        attn_metadata = SimpleNamespace(
            prefill=None,
            decode=_make_decode_meta(bs, 8),
            slot_mapping=torch.arange(bs, dtype=torch.int64),
        )
        fc = SimpleNamespace(attn_metadata=attn_metadata, virtual_engine=0)

        def _fake_mla_prolog_v3(**kwargs):
            q_nope = torch.zeros((bs, m.num_local_heads, m.kv_lora_rank), dtype=torch.float32)
            q_pe = torch.zeros((bs, m.num_local_heads, m.qk_rope_head_dim), dtype=torch.float32)
            dequant_scale_q_nope = torch.ones((bs,), dtype=torch.float32)
            q_norm = torch.zeros((bs, m.q_lora_rank), dtype=torch.float32)
            dequant_scale_q_norm = torch.ones((bs,), dtype=torch.float32)
            return q_nope, q_pe, dequant_scale_q_nope, q_norm, dequant_scale_q_norm

        def _fake_sparse_flash_attention(**kwargs):
            B = kwargs["query"].shape[0]
            N = kwargs["query"].shape[1]
            return torch.zeros((B, N, m.kv_lora_rank), dtype=torch.float32)

        def _indexer_assert_q_norm_tensor(hidden_states, q_norm, cos, sin, attn_metadata, kv_cache):
            self.assertIsInstance(q_norm, torch.Tensor)
            bs_local = hidden_states.shape[0]
            return (
                torch.zeros((bs_local, 1, 8), dtype=torch.int32),
                torch.zeros((bs_local, 1, m.qk_rope_head_dim), dtype=torch.float32),
            )

        m.indexer = MagicMock(side_effect=_indexer_assert_q_norm_tensor)

        with patch.object(npu_dsa_mod, "get_forward_context", return_value=fc), \
             patch.object(npu_dsa_mod.torch.ops, "custom", create=True) as custom_ns:
            custom_ns.npu_mla_prolog_v3 = MagicMock(side_effect=_fake_mla_prolog_v3)
            custom_ns.npu_sparse_flash_attention = MagicMock(side_effect=_fake_sparse_flash_attention)

            out = m._forward_decode(hidden_states, cos, sin, attn_metadata=attn_metadata)

        self.assertEqual(tuple(out.shape), (bs, 8))
        self.assertTrue(custom_ns.npu_mla_prolog_v3.called)
        self.assertTrue(m.indexer.called)
        self.assertEqual(custom_ns.npu_sparse_flash_attention.call_count, 1)


# ======================================================================
# __init__ coverage tests (FIXED)
# ======================================================================
class TestNPUDeepseekSparseAttentionInit(unittest.TestCase):
    def _fake_config(self, rope_type="default", apply_yarn=True):
        cfg = SimpleNamespace()
        cfg.index_topk = 8
        cfg.index_n_heads = 4
        cfg.index_head_dim = 8
        cfg.qk_rope_head_dim = 4
        cfg.rms_norm_eps = 1e-6
        cfg.rope_parameters = {
            "rope_type": rope_type,
            "apply_yarn_scaling": apply_yarn,
            "factor": 2.0,
            "mscale_all_dim": False,
        }
        return cfg

    def _fake_vllm_config(self):
        return SimpleNamespace(
            speculative_config=None,
            npu_compilation_config=SimpleNamespace(decode_gear_list=[1]),
        )

    @patch("omni_npu.v1.layers.attention.npu_dsa.ReplicatedFlashCommLinear")
    @patch("omni_npu.v1.layers.attention.npu_dsa.LayerNorm")
    def test_indexer_init_basic(self, mock_ln, mock_linear):
        mock_linear.return_value = MagicMock()
        mock_ln.return_value = MagicMock()

        idx = npu_dsa_mod.Indexer(
            vllm_config=self._fake_vllm_config(),
            config=self._fake_config(),
            hidden_size=16,
            q_lora_rank=12,
            quant_config=None,
            cache_config=None,
            prefix="layers.0.indexer",
        )

        self.assertEqual(idx.topk_tokens, 8)
        self.assertEqual(idx.n_head, 4)
        self.assertEqual(idx.head_dim, 8)
        self.assertTrue(hasattr(idx, "wq_b"))
        self.assertTrue(hasattr(idx, "wk"))
        self.assertTrue(hasattr(idx, "k_norm"))
        self.assertTrue(hasattr(idx, "weights_proj"))

    @patch.object(npu_dsa_mod, "get_tensor_model_parallel_world_size", return_value=1)
    @patch("omni_npu.v1.layers.attention.npu_dsa.MLAAttention")
    @patch("omni_npu.v1.layers.attention.npu_dsa.get_rope")
    @patch("omni_npu.v1.layers.attention.npu_dsa.Indexer")
    @patch("omni_npu.v1.layers.attention.npu_dsa.RowParallelFlashCommLinear")
    @patch("omni_npu.v1.layers.attention.npu_dsa.ColumnParallelFlashCommLinear")
    @patch("omni_npu.v1.layers.attention.npu_dsa.ReplicatedFlashCommLinear")
    @patch("omni_npu.v1.layers.attention.npu_dsa.RMSNorm")
    def test_attention_init_q_lora_not_none(
        self,
        mock_rms,
        mock_rep,
        mock_col,
        mock_row,
        mock_indexer,
        mock_rope,
        mock_attn,
        mock_tp,
    ):
        mock_rep.return_value = MagicMock()
        mock_col.return_value = MagicMock()
        mock_row.return_value = MagicMock()
        mock_rms.return_value = MagicMock()
        mock_indexer.return_value = MagicMock()
        mock_attn.return_value = MagicMock()
        mock_rope.return_value = MagicMock()

        m = npu_dsa_mod.NPUDeepseekSparseAttention(
            vllm_config=self._fake_vllm_config(),
            config=self._fake_config(),
            hidden_size=16,
            num_heads=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            q_lora_rank=12,
            kv_lora_rank=3,
            cache_config=None,
            quant_config=None,
            prefix="layers.0",
        )

        self.assertTrue(hasattr(m, "q_a_proj"))
        self.assertTrue(hasattr(m, "q_b_proj"))
        self.assertTrue(hasattr(m, "kv_a_proj_with_mqa"))
        self.assertTrue(hasattr(m, "rotary_emb"))
        self.assertTrue(hasattr(m, "indexer"))
        self.assertTrue(hasattr(m, "attn"))

    @patch.object(npu_dsa_mod, "get_tensor_model_parallel_world_size", return_value=1)
    @patch("omni_npu.v1.layers.attention.npu_dsa.yarn_get_mscale", return_value=2.0)
    @patch("omni_npu.v1.layers.attention.npu_dsa.MLAAttention")
    @patch("omni_npu.v1.layers.attention.npu_dsa.get_rope")
    @patch("omni_npu.v1.layers.attention.npu_dsa.Indexer")
    @patch("omni_npu.v1.layers.attention.npu_dsa.RowParallelFlashCommLinear")
    @patch("omni_npu.v1.layers.attention.npu_dsa.ColumnParallelFlashCommLinear")
    @patch("omni_npu.v1.layers.attention.npu_dsa.ReplicatedFlashCommLinear")
    @patch("omni_npu.v1.layers.attention.npu_dsa.RMSNorm")
    def test_attention_init_q_lora_none_and_yarn_scaling(
        self,
        mock_rms,
        mock_rep,
        mock_col,
        mock_row,
        mock_indexer,
        mock_rope,
        mock_attn,
        mock_mscale,
        mock_tp,
    ):
        mock_rep.return_value = MagicMock()
        mock_col.return_value = MagicMock()
        mock_row.return_value = MagicMock()
        mock_rms.return_value = MagicMock()
        mock_indexer.return_value = MagicMock()
        mock_attn.return_value = MagicMock()
        mock_rope.return_value = MagicMock()

        m = npu_dsa_mod.NPUDeepseekSparseAttention(
            vllm_config=self._fake_vllm_config(),
            config=self._fake_config(rope_type="deepseek_yarn"),
            hidden_size=16,
            num_heads=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            q_lora_rank=None,
            kv_lora_rank=3,
            cache_config=None,
            quant_config=None,
            prefix="layers.1",
        )

        self.assertTrue(hasattr(m, "q_proj"))
        self.assertFalse(hasattr(m, "q_a_proj"))
        self.assertGreater(m.scaling, 0)


if __name__ == "__main__":
    unittest.main()