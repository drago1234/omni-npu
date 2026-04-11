import pytest
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import omni_npu.v1.layers.attention.npu_mla as npu_mla_mod


class _FakeLinear:
    """Mimic linear modules: call(x) -> (y, None)."""

    def __init__(self, out_features: int, *, return_dtype=torch.float32):
        self.out_features = int(out_features)
        self.return_dtype = return_dtype

    def __call__(self, x):
        if isinstance(x, dict):
            x = x["x_int8"]
        y = torch.zeros((x.shape[0], self.out_features), dtype=self.return_dtype, device=x.device)
        return y, None

    def forward(self, x):
        return self.__call__(x)


class _FakeLayerNorm:
    def __init__(self, dim: int, eps: float = 1e-6):
        self.weight = torch.ones((dim,), dtype=torch.float32)
        self.variance_epsilon = eps

    def __call__(self, x):
        return x


class _FakeAggregateConv:
    def __call__(self, x, only_prefill=False, force_decode=False):
        return x


class _Fake_even_odd_indexing():
    def __call__(self, x):
        return x


class _Fake_insert_tensor_by_start_loc():
    def __call__(self, x, insert_segment=None, start_loc=None):
        return x


class _FakeStream:
    def wait_stream(self, _other):
        return None


def _make_decode_meta(bs: int):
    return SimpleNamespace(
        query_cumlens=torch.arange(1, bs + 1, dtype=torch.int32),
        seq_lens=torch.full((bs,), 8, dtype=torch.int32),
        block_table=torch.zeros((bs, 4), dtype=torch.int32),
        slot_mapping=torch.arange(bs, dtype=torch.int64)
    )


def _make_prefill_meta(bs: int, *, max_query_len=2):
    return SimpleNamespace(
        seq_lens=torch.arange(1, bs + 1, dtype=torch.int32),
        query_cumlens=torch.arange(1, bs + 1, dtype=torch.int32),
        max_query_len=max_query_len,
        query_start_loc=[0] + torch.arange(1, bs + 1, dtype=torch.int32).tolist(),
        slot_mapping=torch.arange(bs, dtype=torch.int64)
    )


class TestNPUMLAForwardRouting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the mock for npu_mla_forward to work on CPU backend."""
        # Mock torch.ops.vllm.npu_mla_forward to call the actual npu_mla_forward function
        torch.ops.vllm.npu_mla_forward = npu_mla_mod.npu_mla_forward
    
    def _make_stub(self):
        m = SimpleNamespace()
        m.prefix = "layers.0"
        m.quant_symbol = False
        m._forward_prefill = MagicMock(return_value=torch.tensor([1]))
        m._forward_decode = MagicMock(return_value=torch.tensor([2]))
        m.param_sink_number = 128
        m.param_sink_with_value = True
        m.kv_lora_rank = 512
        m.qk_rope_head_dim = 64

        sink_k_pe = torch.zeros((128, m.kv_lora_rank), dtype=torch.bfloat16)
        sink_compressed_kv = torch.zeros((128, m.qk_rope_head_dim), dtype=torch.bfloat16)
        m.attn = SimpleNamespace(sink_k_pe=sink_k_pe,
            sink_compressed_kv=sink_compressed_kv,
            sink_populated=True,
        )
        return m

    def test_forward_routes_prefill_and_decode(self):
        m = self._make_stub()
        hs = torch.randn((3, 16), dtype=torch.float32)
        cos = torch.zeros((3, 1, 1, 4), dtype=torch.float32)
        sin = torch.zeros((3, 1, 1, 4), dtype=torch.float32)

        fc = SimpleNamespace(attn_metadata=None, virtual_engine=0, no_compile_layers={m.prefix: m}, capturing=False)
        with patch.object(npu_mla_mod, "get_forward_context", return_value=fc):
            out = npu_mla_mod.NPUDeepseekMLAAttention.forward(m, hs, cos, sin)
        self.assertTrue(torch.equal(out, torch.tensor([1])))

        m._forward_prefill.reset_mock()
        prefill_meta = _make_prefill_meta(3, max_query_len=2)
        fc = SimpleNamespace(attn_metadata=SimpleNamespace(prefill=prefill_meta, decode=None,
            num_actual_tokens=3, num_decodes=0, num_prefills=1, num_decode_tokens=0,
            slot_mapping=torch.arange(3, dtype=torch.int64)),
            virtual_engine=0, no_compile_layers={m.prefix: m}, capturing=False)
        with patch.object(npu_mla_mod, "get_forward_context", return_value=fc):
            out = npu_mla_mod.NPUDeepseekMLAAttention.forward(m, hs, cos, sin)
        self.assertTrue(torch.equal(out, torch.tensor([1])))

        m._forward_decode.reset_mock()
        decode_meta = _make_decode_meta(2)
        fc = SimpleNamespace(attn_metadata=SimpleNamespace(prefill=None, decode=decode_meta,
            num_actual_tokens=2, num_decodes=2, num_prefills=0, num_decode_tokens=2,
            slot_mapping=torch.arange(2, dtype=torch.int64)),
            virtual_engine=0, no_compile_layers={m.prefix: m}, capturing=False)
        with patch.object(npu_mla_mod, "get_forward_context", return_value=fc):
            out = npu_mla_mod.NPUDeepseekMLAAttention.forward(m, hs, cos, sin)
        self.assertTrue(torch.equal(out, torch.tensor([2])))

    def test_forward_attn_metadata_dict_extracts_prefix(self):
        m = self._make_stub()
        m.prefix = "layers.7"
        hs = torch.randn((2, 16), dtype=torch.float32)
        cos = torch.zeros((2, 1, 1, 4), dtype=torch.float32)
        sin = torch.zeros((2, 1, 1, 4), dtype=torch.float32)

        decode_meta = _make_decode_meta(2)
        meta = SimpleNamespace(prefill=None, decode=decode_meta, num_actual_tokens=2, 
            num_decodes=2, num_prefills=0, num_decode_tokens=2, 
            slot_mapping=torch.arange(2, dtype=torch.int64))
        fc = SimpleNamespace(attn_metadata={f"{m.prefix}.attn": meta}, virtual_engine=0, no_compile_layers={m.prefix: m}, capturing=False)
        with patch.object(npu_mla_mod, "get_forward_context", return_value=fc):
            out = npu_mla_mod.NPUDeepseekMLAAttention.forward(m, hs, cos, sin)

        self.assertTrue(torch.equal(out, torch.tensor([2])))
        self.assertEqual(m._forward_decode.call_count, 1)

    def test_forward_quant_symbol_wraps_hidden_states(self):
        m = self._make_stub()
        m.quant_symbol = True
        hs = torch.randn((2, 16), dtype=torch.float32)
        cos = torch.zeros((2, 1, 1, 4), dtype=torch.float32)
        sin = torch.zeros((2, 1, 1, 4), dtype=torch.float32)
        fc = SimpleNamespace(attn_metadata=None, virtual_engine=0, no_compile_layers={m.prefix: m}, capturing=False)

        with patch.object(npu_mla_mod, "get_forward_context", return_value=fc), patch.object(
            npu_mla_mod.torch_npu,
            "npu_dynamic_quant",
            side_effect=lambda x: (x.to(torch.int8), torch.ones((x.shape[0],), dtype=torch.float32)),
            create=True,
        ):
            npu_mla_mod.NPUDeepseekMLAAttention.forward(m, hs, cos, sin)

        passed_hs = m._forward_prefill.call_args[0][0]
        self.assertIsInstance(passed_hs, dict)
        self.assertIn("x_int8", passed_hs)
        self.assertIn("pertoken_scale", passed_hs)

    def test_insert_tensor_by_start_loc(self):
        raw = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.int32)
        insert = torch.tensor([[9], [8]], dtype=torch.int32)
        start_loc = [0, 2, 5]
        fc = SimpleNamespace(attn_metadata=None, virtual_engine=0)
        with patch.object(npu_mla_mod, "get_forward_context", return_value=fc):
            out = npu_mla_mod.NPUDeepseekMLAAttention._insert_tensor_by_start_loc(raw, insert, start_loc)
        expected = torch.tensor([[9], [8], [1], [2], [9], [8], [3], [4], [5]], dtype=torch.int32)
        self.assertTrue(torch.equal(out, expected))

    def test_even_odd_indexing(self):
        x = torch.tensor([
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ])
        expected = torch.tensor([
            [1, 3, 2, 4],
            [5, 7, 6, 8]
        ])
        
        fc = SimpleNamespace(attn_metadata=None, virtual_engine=0)
        with patch.object(npu_mla_mod, "get_forward_context", return_value=fc):
            out = npu_mla_mod.NPUDeepseekMLAAttention.even_odd_indexing(x)
        self.assertTrue(torch.equal(out, expected))


class TestNPUMLAPrefillDecode(unittest.TestCase):
    def _make_stub(self):
        m = SimpleNamespace()
        m.prefix = "layers.0"
        m.hidden_size = 16
        m.q_lora_rank = 12
        # Align with implementation's fixed latent_cache last dim (576).
        m.kv_lora_rank = 544
        m.num_local_heads = 2
        m.qk_nope_head_dim = 4
        m.qk_rope_head_dim = 32
        m.qk_head_dim = m.qk_nope_head_dim + m.qk_rope_head_dim
        m.v_head_dim = 8
        m.scaling = 0.5
        m.quant_symbol = False
        m.param_sink_number = 128
        m.param_sink_with_value = True
        m.sliding_window = 512
        m.rope_interleaved = False
        m.merge_q_kv_conv = False

        m.q_a_proj = _FakeLinear(m.q_lora_rank)
        m.kv_a_proj_with_mqa = _FakeLinear(m.kv_lora_rank + m.qk_rope_head_dim)
        m.q_a_layernorm = _FakeLayerNorm(m.q_lora_rank)
        m.kv_a_layernorm = _FakeLayerNorm(m.kv_lora_rank)
        m.q_b_proj = _FakeLinear(m.num_local_heads * m.qk_head_dim)
        m.kv_b_proj = _FakeLinear(m.num_local_heads * (m.qk_nope_head_dim + m.v_head_dim))
        m.o_proj = _FakeLinear(16)
        m.qa_conv = _FakeAggregateConv()
        m.compresskv_conv = _FakeAggregateConv()
        m.merge_conv = None
        m.o_conv = _FakeAggregateConv()
        m.even_odd_indexing = _Fake_even_odd_indexing()
        m._insert_tensor_by_start_loc = _Fake_insert_tensor_by_start_loc()

        impl = SimpleNamespace(
            W_UK_T=torch.zeros((m.num_local_heads, m.qk_nope_head_dim, m.kv_lora_rank), dtype=torch.float32),
            W_UV=torch.zeros((m.num_local_heads, m.kv_lora_rank, m.v_head_dim), dtype=torch.float32),
            SHARE_MASK_TRIL_SPARSE=torch.ones((1,), dtype=torch.bool),
        )
        kv0 = torch.zeros((2, 128, m.kv_lora_rank), dtype=torch.float32)
        kv1 = torch.zeros((2, 128, m.qk_rope_head_dim), dtype=torch.float32)
        sink_k_pe = torch.zeros((128, m.kv_lora_rank), dtype=torch.bfloat16)
        sink_compressed_kv = torch.zeros((128, m.qk_rope_head_dim), dtype=torch.bfloat16)
        m.attn = SimpleNamespace(impl=impl, kv_cache=[(kv0, kv1)], sink_k_pe=sink_k_pe,
                                sink_compressed_kv=sink_compressed_kv,
                                sink_populated=True)
        m.config = self._fake_cfg()
        return m

    def _fake_cfg(self, model_type="openpangu_v2"):
        return SimpleNamespace(
            model_type=model_type,
        )

    def test_forward_prefill_metadata_none_returns_zeroed_path(self):
        m = self._make_stub()
        bs = 4
        hs = torch.randn((bs, m.hidden_size), dtype=torch.float32)
        cos = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)
        sin = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)

        with patch.object(
            npu_mla_mod.torch_npu, "npu_interleave_rope", side_effect=lambda x, c, s: x, create=True
        ):
            out = npu_mla_mod.NPUDeepseekMLAAttention._forward_prefill(m, hs, cos, sin, attn_metadata=None)

        self.assertEqual(tuple(out.shape), (bs, 16))

    def test_forward_prefill_merge_conv(self):
        m = self._make_stub()
        m.merge_q_kv_conv = True
        m.merge_conv = _FakeAggregateConv()
        bs = 3
        hs = torch.randn((bs, m.hidden_size), dtype=torch.float32)
        cos = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)
        sin = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)
        meta = SimpleNamespace(
            prefill=_make_prefill_meta(bs, max_query_len=2),
            decode=None,
            slot_mapping=torch.arange(bs, dtype=torch.int64),
        )
        fc = SimpleNamespace(attn_metadata=meta, virtual_engine=0, capturing=False)
        fake_stream = _FakeStream()

        def _fake_kv_rmsnorm_rope_cache(*args, **kwargs):
            k_pe = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)
            kv_a = torch.zeros((bs, 1, 1, m.kv_lora_rank), dtype=torch.float32)
            return None, None, k_pe, kv_a

        def _fake_fused_attention(*args, **kwargs):
            q_arg = args[0]
            return torch.zeros((q_arg.shape[0], q_arg.shape[1], m.v_head_dim), dtype=torch.float32), None

        with patch.object(npu_mla_mod, "get_forward_context", return_value=fc), patch.object(
            npu_mla_mod, "named_stream", return_value=fake_stream
        ), patch.object(
            npu_mla_mod.torch, "npu", create=True
        ) as torch_npu_ns, patch.object(
            npu_mla_mod.torch_npu,
            "npu_dynamic_quant",
            side_effect=lambda x: (x.to(torch.int8), torch.ones((x.shape[0],), dtype=torch.float32)),
            create=True,
        ), patch.object(
            npu_mla_mod.torch_npu, "npu_kv_rmsnorm_rope_cache", side_effect=_fake_kv_rmsnorm_rope_cache, create=True
        ), patch.object(
            npu_mla_mod.torch_npu, "npu_interleave_rope", side_effect=lambda x, c, s: x, create=True
        ), patch.object(
            npu_mla_mod.torch.ops.npu, "npu_fused_infer_attention_score", side_effect=_fake_fused_attention, create=True
        ), patch.object(
            npu_mla_mod.torch.ops.custom, "npu_fused_infer_attention_sink", side_effect=_fake_fused_attention, create=True
        ):
            torch_npu_ns.current_stream = lambda: fake_stream
            torch_npu_ns.stream = lambda _s: nullcontext()
            out = npu_mla_mod.NPUDeepseekMLAAttention._forward_prefill(m, hs, cos, sin, attn_metadata=meta.prefill)

        self.assertEqual(tuple(out.shape), (bs, 16))

    def test_forward_decode_merge_conv(self):
        m = self._make_stub()
        m.merge_q_kv_conv = True
        m.merge_conv = _FakeAggregateConv()
        bs = 3
        hs = torch.randn((bs, m.hidden_size), dtype=torch.float32)
        cos = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)
        sin = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)
        meta = SimpleNamespace(
            prefill=None,
            decode=_make_decode_meta(bs),
            slot_mapping=torch.arange(bs, dtype=torch.int64),
        )
        fc = SimpleNamespace(attn_metadata=meta, virtual_engine=0, capturing=False)

        def _fake_kv_rmsnorm_rope_cache(*args, **kwargs):
            k_rope = torch.zeros((2, 1, 128, m.qk_rope_head_dim), dtype=torch.float32)
            k_nope = torch.zeros((2, 1, 128, m.kv_lora_rank), dtype=torch.float32)
            return k_rope, k_nope, None, None

        def _fake_tbm(x, w, perm_y=None):
            return torch.zeros((x.shape[0], x.shape[1], w.shape[2]), dtype=torch.float32)

        with patch.object(npu_mla_mod, "get_forward_context", return_value=fc), patch.object(
            npu_mla_mod.torch_npu, "npu_kv_rmsnorm_rope_cache", side_effect=_fake_kv_rmsnorm_rope_cache, create=True
        ), patch.object(
            npu_mla_mod.torch_npu, "npu_interleave_rope", side_effect=lambda x, c, s: x, create=True
        ), patch.object(
            npu_mla_mod.torch_npu, "npu_transpose_batchmatmul", side_effect=_fake_tbm, create=True
        ), patch.object(
            npu_mla_mod.NPUMLAImpl, "ensure_decode_attn_mask", return_value=None
        ), patch.object(
            npu_mla_mod.NPUMLAImpl, "DECORE_ATTN_MASK", torch.ones((1,), dtype=torch.bool), create=True
        ), patch.object(
            npu_mla_mod.torch.ops.npu,
            "npu_fused_infer_attention_score",
            return_value=(torch.zeros((m.num_local_heads, bs, m.kv_lora_rank), dtype=torch.float32),),
            create=True,
        ), patch.object(
            npu_mla_mod.torch.ops.custom,
            "npu_fused_infer_attention_sink",
            return_value=(torch.zeros((bs, m.num_local_heads, m.kv_lora_rank), dtype=torch.float32),),
            create=True,
         ):
            out = npu_mla_mod.NPUDeepseekMLAAttention._forward_decode(m, hs, cos, sin, attn_metadata=meta.decode)

        self.assertEqual(tuple(out.shape), (bs, 16))

    def test_forward_prefill_with_metadata_runs_attention_and_epilog(self):
        m = self._make_stub()
        bs = 3
        hs = torch.randn((bs, m.hidden_size), dtype=torch.float32)
        cos = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)
        sin = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)
        meta = SimpleNamespace(
            prefill=_make_prefill_meta(bs, max_query_len=2),
            decode=None,
            slot_mapping=torch.arange(bs, dtype=torch.int64),
        )
        fc = SimpleNamespace(attn_metadata=meta, virtual_engine=0, capturing=False)
        fake_stream = _FakeStream()

        def _fake_kv_rmsnorm_rope_cache(*args, **kwargs):
            k_pe = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)
            kv_a = torch.zeros((bs, 1, 1, m.kv_lora_rank), dtype=torch.float32)
            return None, None, k_pe, kv_a

        def _fake_fused_attention(*args, **kwargs):
            q_arg = args[0]
            return torch.zeros((q_arg.shape[0], q_arg.shape[1], m.v_head_dim), dtype=torch.float32), None

        with patch.object(npu_mla_mod, "get_forward_context", return_value=fc), patch.object(
            npu_mla_mod, "named_stream", return_value=fake_stream
        ), patch.object(
            npu_mla_mod.torch, "npu", create=True
        ) as torch_npu_ns, patch.object(
            npu_mla_mod.torch_npu,
            "npu_dynamic_quant",
            side_effect=lambda x: (x.to(torch.int8), torch.ones((x.shape[0],), dtype=torch.float32)),
            create=True,
        ), patch.object(
            npu_mla_mod.torch_npu, "npu_kv_rmsnorm_rope_cache", side_effect=_fake_kv_rmsnorm_rope_cache, create=True
        ), patch.object(
            npu_mla_mod.torch_npu, "npu_interleave_rope", side_effect=lambda x, c, s: x, create=True
        ), patch.object(
            npu_mla_mod.torch.ops.npu, "npu_fused_infer_attention_score", side_effect=_fake_fused_attention, create=True
        ), patch.object(
            npu_mla_mod.torch.ops.custom, "npu_fused_infer_attention_sink", side_effect=_fake_fused_attention, create=True
        ):
            torch_npu_ns.current_stream = lambda: fake_stream
            torch_npu_ns.stream = lambda _s: nullcontext()
            out = npu_mla_mod.NPUDeepseekMLAAttention._forward_prefill(m, hs, cos, sin, attn_metadata=meta.prefill)

        self.assertEqual(tuple(out.shape), (bs, 16))

    def test_forward_decode_runs_decode_attention_path(self):
        m = self._make_stub()
        bs = 3
        hs = torch.randn((bs, m.hidden_size), dtype=torch.float32)
        cos = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)
        sin = torch.zeros((bs, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)
        meta = SimpleNamespace(
            prefill=None,
            decode=_make_decode_meta(bs),
            slot_mapping=torch.arange(bs, dtype=torch.int64),
        )
        fc = SimpleNamespace(attn_metadata=meta, virtual_engine=0, capturing=False)

        def _fake_kv_rmsnorm_rope_cache(*args, **kwargs):
            k_rope = torch.zeros((2, 1, 128, m.qk_rope_head_dim), dtype=torch.float32)
            k_nope = torch.zeros((2, 1, 128, m.kv_lora_rank), dtype=torch.float32)
            return k_rope, k_nope, None, None

        def _fake_tbm(x, w, perm_y=None):
            return torch.zeros((x.shape[0], x.shape[1], w.shape[2]), dtype=torch.float32)

        with patch.object(npu_mla_mod, "get_forward_context", return_value=fc), patch.object(
            npu_mla_mod.torch_npu, "npu_kv_rmsnorm_rope_cache", side_effect=_fake_kv_rmsnorm_rope_cache, create=True
        ), patch.object(
            npu_mla_mod.torch_npu, "npu_interleave_rope", side_effect=lambda x, c, s: x, create=True
        ), patch.object(
            npu_mla_mod.torch_npu, "npu_transpose_batchmatmul", side_effect=_fake_tbm, create=True
        ), patch.object(
            npu_mla_mod.NPUMLAImpl, "ensure_decode_attn_mask", return_value=None
        ), patch.object(
            npu_mla_mod.torch.ops.npu,
            "npu_fused_infer_attention_score",
            return_value=(torch.zeros((m.num_local_heads, bs, m.kv_lora_rank), dtype=torch.float32),),
            create=True,
        ), patch.object(
            npu_mla_mod.torch.ops.custom,
            "npu_fused_infer_attention_sink",
            return_value=(torch.zeros((bs, m.num_local_heads, m.kv_lora_rank), dtype=torch.float32),),
            create=True,
         ):
            out = npu_mla_mod.NPUDeepseekMLAAttention._forward_decode(m, hs, cos, sin, attn_metadata=meta.decode)

        self.assertEqual(tuple(out.shape), (bs, 16))

    def test_forward_prefill_decode_mixed_both_paths(self):
        """Test the case when both prefill and decode metadata are present."""
        m = self._make_stub()
        m._forward_prefill = MagicMock(return_value=torch.randn((3, 16), dtype=torch.float32))
        m._forward_decode = MagicMock(return_value=torch.randn((2, 16), dtype=torch.float32))

        num_decode_tokens = 2
        num_prefill_tokens = 3
        num_actual_tokens = num_decode_tokens + num_prefill_tokens

        hs = torch.randn((num_actual_tokens, m.hidden_size), dtype=torch.float32)
        cos = torch.zeros((num_actual_tokens, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)
        sin = torch.zeros((num_actual_tokens, 1, 1, m.qk_rope_head_dim), dtype=torch.float32)

        # Create metadata with both decode and prefill
        prefill_meta = _make_prefill_meta(3, max_query_len=2)
        decode_meta = _make_decode_meta(2)

        meta = SimpleNamespace(
            prefill=prefill_meta,
            decode=decode_meta,
            num_actual_tokens=num_actual_tokens,
            num_decodes=2,
            num_prefills=3,
            num_decode_tokens=num_decode_tokens,
            slot_mapping=torch.arange(num_actual_tokens, dtype=torch.int64),
        )
        fc = SimpleNamespace(attn_metadata=meta, virtual_engine=0, no_compile_layers={m.prefix: m}, capturing=False)

        with patch.object(npu_mla_mod, "get_forward_context", return_value=fc):
            out = npu_mla_mod.NPUDeepseekMLAAttention.forward(m, hs, cos, sin)

        # Verify that both forward_prefill and forward_decode were called with pd_mixed_flag=True
        self.assertEqual(m._forward_prefill.call_count, 1)
        self.assertEqual(m._forward_decode.call_count, 1)

        # Check that prefill was called with the correct hidden_states and flags
        prefill_call_args = m._forward_prefill.call_args
        prefill_hs = prefill_call_args[0][0]
        self.assertEqual(prefill_hs.shape[0], num_prefill_tokens)
        pd_mixed_flag_prefill = prefill_call_args[1].get("pd_mixed_flag", False)
        self.assertTrue(pd_mixed_flag_prefill)

        # Check that decode was called with the correct hidden_states and flags
        decode_call_args = m._forward_decode.call_args
        decode_hs = decode_call_args[0][0]
        self.assertEqual(decode_hs.shape[0], num_decode_tokens)
        pd_mixed_flag_decode = decode_call_args[1].get("pd_mixed_flag", False)
        self.assertTrue(pd_mixed_flag_decode)

        # Verify output shape combines both outputs
        self.assertEqual(out.shape, (num_actual_tokens, m.hidden_size))

class TestNPUMLAMergeConvInit(unittest.TestCase):
    """Test merge_conv initialization logic in __init__ method."""

    def _fake_cfg(self, rope_type="default", apply_yarn=True, torch_dtype=torch.bfloat16, use_mome=True):
        return SimpleNamespace(
            rms_norm_eps=1e-6,
            rope_parameters={
                "rope_type": rope_type,
                "apply_yarn_scaling": apply_yarn,
                "factor": 2.0,
                "mscale_all_dim": False,
            },
            torch_dtype=torch_dtype,
            use_mome=use_mome,
        )

    @pytest.mark.usefixtures("default_vllm_config")
    @patch("omni_npu.model_config.config_loader.loader.model_extra_config")
    @patch("omni_npu.v1.layers.attention.npu_mla.RMSNorm")
    @patch("omni_npu.v1.layers.attention.npu_mla.ReplicatedLinear")
    @patch("omni_npu.v1.layers.attention.npu_mla.ColumnParallelFlashCommLinear")
    @patch("omni_npu.v1.layers.attention.npu_mla.RowParallelFlashCommLinear")
    @patch("omni_npu.v1.layers.attention.npu_mla.get_rope")
    @patch("omni_npu.v1.layers.attention.npu_mla.MLAAttention")
    @patch.object(npu_mla_mod, "get_tensor_model_parallel_world_size", return_value=1)
    def test_init_merge_conv_created_when_merge_q_kv_conv_true(
        self,
        mock_tp,
        mock_attn,
        mock_rope,
        mock_row,
        mock_col,
        mock_rep,
        mock_rms,
        mock_model_extra_config,
    ):
        """Test that merge_conv is created when merge_q_kv_conv is True."""
        # Setup mock for AggregateConv since it's imported via try/except
        mock_aggregate_conv = MagicMock()
        npu_mla_mod.AggregateConv = mock_aggregate_conv

        mock_model_extra_config.operator_opt_config = SimpleNamespace(merge_q_kv_conv=True)
        mock_aggregate_conv.return_value = MagicMock()
        mock_rms.return_value = MagicMock()
        mock_rep.return_value = MagicMock()
        mock_col.return_value = MagicMock()
        mock_row.return_value = MagicMock()
        mock_rope.return_value = MagicMock()
        mock_attn.return_value = MagicMock()

        m = npu_mla_mod.NPUDeepseekMLAAttention(
            vllm_config=SimpleNamespace(),
            config=self._fake_cfg(),
            hidden_size=16,
            num_heads=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            q_lora_rank=12,
            kv_lora_rank=8,
            cache_config=None,
            quant_config=None,
            prefix="layers.0",
        )

        # Verify merge_conv was created with correct arguments
        self.assertTrue(hasattr(m, "merge_conv"))
        self.assertTrue(mock_aggregate_conv.called)

    @pytest.mark.usefixtures("default_vllm_config")
    @patch("omni_npu.model_config.config_loader.loader.model_extra_config")
    @patch("omni_npu.v1.layers.attention.npu_mla.RMSNorm")
    @patch("omni_npu.v1.layers.attention.npu_mla.ReplicatedLinear")
    @patch("omni_npu.v1.layers.attention.npu_mla.ColumnParallelFlashCommLinear")
    @patch("omni_npu.v1.layers.attention.npu_mla.RowParallelFlashCommLinear")
    @patch("omni_npu.v1.layers.attention.npu_mla.get_rope")
    @patch("omni_npu.v1.layers.attention.npu_mla.MLAAttention")
    @patch.object(npu_mla_mod, "get_tensor_model_parallel_world_size", return_value=1)
    def test_init_merge_conv_is_none_when_merge_q_kv_conv_false(
        self,
        mock_tp,
        mock_attn,
        mock_rope,
        mock_row,
        mock_col,
        mock_rep,
        mock_rms,
        mock_model_extra_config,
    ):
        """Test that merge_conv is None when merge_q_kv_conv is False."""
        # Setup mock for AggregateConv since it's imported via try/except
        mock_aggregate_conv = MagicMock()
        npu_mla_mod.AggregateConv = mock_aggregate_conv

        mock_model_extra_config.operator_opt_config = SimpleNamespace(merge_q_kv_conv=False)
        mock_aggregate_conv.return_value = MagicMock()
        mock_rms.return_value = MagicMock()
        mock_rep.return_value = MagicMock()
        mock_col.return_value = MagicMock()
        mock_row.return_value = MagicMock()
        mock_rope.return_value = MagicMock()
        mock_attn.return_value = MagicMock()

        m = npu_mla_mod.NPUDeepseekMLAAttention(
            vllm_config=SimpleNamespace(),
            config=self._fake_cfg(),
            hidden_size=16,
            num_heads=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            q_lora_rank=12,
            kv_lora_rank=8,
            cache_config=None,
            quant_config=None,
            prefix="layers.0",
        )

        # Verify merge_conv is None
        self.assertTrue(hasattr(m, "merge_conv"))
        self.assertIsNone(m.merge_conv)
        self.assertEqual(m.merge_q_kv_conv, False)

        # Cleanup
        delattr(npu_mla_mod, "AggregateConv")

    @pytest.mark.usefixtures("default_vllm_config")
    @patch("omni_npu.model_config.config_loader.loader.model_extra_config")
    @patch("omni_npu.v1.layers.attention.npu_mla.RMSNorm")
    @patch("omni_npu.v1.layers.attention.npu_mla.ReplicatedLinear")
    @patch("omni_npu.v1.layers.attention.npu_mla.ColumnParallelFlashCommLinear")
    @patch("omni_npu.v1.layers.attention.npu_mla.RowParallelFlashCommLinear")
    @patch("omni_npu.v1.layers.attention.npu_mla.get_rope")
    @patch("omni_npu.v1.layers.attention.npu_mla.MLAAttention")
    @patch.object(npu_mla_mod, "get_tensor_model_parallel_world_size", return_value=1)
    def test_init_merge_conv_is_none_when_use_mome_false(
        self,
        mock_tp,
        mock_attn,
        mock_rope,
        mock_row,
        mock_col,
        mock_rep,
        mock_rms,
        mock_model_extra_config,
    ):
        """Test that merge_conv is None when use_mome is False, regardless of merge_q_kv_conv."""
        # Setup mock for AggregateConv since it's imported via try/except
        mock_aggregate_conv = MagicMock()
        npu_mla_mod.AggregateConv = mock_aggregate_conv

        mock_model_extra_config.operator_opt_config = SimpleNamespace(merge_q_kv_conv=True)
        mock_rms.return_value = MagicMock()
        mock_rep.return_value = MagicMock()
        mock_col.return_value = MagicMock()
        mock_row.return_value = MagicMock()
        mock_rope.return_value = MagicMock()
        mock_attn.return_value = MagicMock()

        cfg = self._fake_cfg(use_mome=False)

        m = npu_mla_mod.NPUDeepseekMLAAttention(
            vllm_config=SimpleNamespace(),
            config=cfg,
            hidden_size=16,
            num_heads=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            q_lora_rank=12,
            kv_lora_rank=8,
            cache_config=None,
            quant_config=None,
            prefix="layers.0",
        )

        # Verify merge_conv is None and merge_q_kv_conv is False when use_mome is False
        self.assertIsNone(m.merge_conv)
        self.assertEqual(m.merge_q_kv_conv, False)

        # Cleanup
        delattr(npu_mla_mod, "AggregateConv")


class TestNPUMLAInit(unittest.TestCase):
    def _fake_cfg(self, rope_type="default", apply_yarn=True, torch_dtype=torch.bfloat16):
        return SimpleNamespace(
            rms_norm_eps=1e-6,
            rope_parameters={
                "rope_type": rope_type,
                "apply_yarn_scaling": apply_yarn,
                "factor": 2.0,
                "mscale_all_dim": False,
            },
            torch_dtype=torch_dtype,
        )

    @patch.object(npu_mla_mod, "get_tensor_model_parallel_world_size", return_value=1)
    @patch("omni_npu.v1.layers.attention.npu_mla.MLAAttention")
    @patch("omni_npu.v1.layers.attention.npu_mla.get_rope")
    @patch("omni_npu.v1.layers.attention.npu_mla.RowParallelFlashCommLinear")
    @patch("omni_npu.v1.layers.attention.npu_mla.ColumnParallelFlashCommLinear")
    @patch("omni_npu.v1.layers.attention.npu_mla.ReplicatedLinear")
    @patch("omni_npu.v1.layers.attention.npu_mla.RMSNorm")
    @pytest.mark.usefixtures("default_vllm_config")
    def test_init_basic_default_rope(
        self, mock_rms, mock_rep, mock_col, mock_row, mock_rope, mock_attn, mock_tp
    ):
        mock_rms.return_value = MagicMock()
        mock_rep.return_value = MagicMock()
        mock_col.return_value = MagicMock()
        mock_row.return_value = MagicMock()
        mock_rope.return_value = MagicMock()
        mock_attn.return_value = MagicMock()

        m = npu_mla_mod.NPUDeepseekMLAAttention(
            vllm_config=SimpleNamespace(),
            config=self._fake_cfg(),
            hidden_size=16,
            num_heads=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            q_lora_rank=12,
            kv_lora_rank=8,
            cache_config=None,
            quant_config=None,
            prefix="layers.0",
        )
        self.assertEqual(m.num_local_heads, 4)
        self.assertTrue(hasattr(m, "attn"))
        self.assertTrue(hasattr(m, "rotary_emb"))

    @patch.object(npu_mla_mod, "get_tensor_model_parallel_world_size", return_value=1)
    @patch("omni_npu.v1.layers.attention.npu_mla.yarn_get_mscale", return_value=2.0)
    @patch("omni_npu.v1.layers.attention.npu_mla.MLAAttention")
    @patch("omni_npu.v1.layers.attention.npu_mla.get_rope")
    @patch("omni_npu.v1.layers.attention.npu_mla.RowParallelFlashCommLinear")
    @patch("omni_npu.v1.layers.attention.npu_mla.ColumnParallelFlashCommLinear")
    @patch("omni_npu.v1.layers.attention.npu_mla.ReplicatedLinear")
    @patch("omni_npu.v1.layers.attention.npu_mla.RMSNorm")
    @pytest.mark.usefixtures("default_vllm_config")
    def test_init_non_default_rope_rewrites_type_and_applies_yarn_scaling(
        self, mock_rms, mock_rep, mock_col, mock_row, mock_rope, mock_attn, mock_mscale, mock_tp
    ):
        mock_rms.return_value = MagicMock()
        mock_rep.return_value = MagicMock()
        mock_col.return_value = MagicMock()
        mock_row.return_value = MagicMock()
        mock_rope.return_value = MagicMock()
        mock_attn.return_value = MagicMock()

        cfg = self._fake_cfg(rope_type="not_default", apply_yarn=True)
        m = npu_mla_mod.NPUDeepseekMLAAttention(
            vllm_config=SimpleNamespace(),
            config=cfg,
            hidden_size=16,
            num_heads=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            q_lora_rank=12,
            kv_lora_rank=8,
            cache_config=None,
            quant_config=None,
            prefix="layers.1",
        )
        self.assertEqual(cfg.rope_parameters["rope_type"], "deepseek_yarn")
        self.assertGreater(m.scaling, 0)
        self.assertTrue(mock_mscale.called)


if __name__ == "__main__":
    unittest.main()
