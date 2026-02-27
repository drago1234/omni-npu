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


class _FakeStream:
    def wait_stream(self, _other):
        return None


def _make_decode_meta(bs: int):
    return SimpleNamespace(
        query_cumlens=torch.arange(1, bs + 1, dtype=torch.int32),
        seq_lens=torch.full((bs,), 8, dtype=torch.int32),
        block_table=torch.zeros((bs, 4), dtype=torch.int32),
    )


def _make_prefill_meta(bs: int, *, max_query_len=2):
    return SimpleNamespace(
        seq_lens=torch.arange(1, bs + 1, dtype=torch.int32),
        query_cumlens=torch.arange(1, bs + 1, dtype=torch.int32),
        max_query_len=max_query_len,
    )


class TestNPUMLAForwardRouting(unittest.TestCase):
    def _make_stub(self):
        m = SimpleNamespace()
        m.prefix = "layers.0"
        m.quant_symbol = False
        m._forward_prefill = MagicMock(return_value=torch.tensor([1]))
        m._forward_decode = MagicMock(return_value=torch.tensor([2]))
        return m

    def test_forward_routes_prefill_and_decode(self):
        m = self._make_stub()
        hs = torch.randn((3, 16), dtype=torch.float32)
        cos = torch.zeros((3, 1, 1, 4), dtype=torch.float32)
        sin = torch.zeros((3, 1, 1, 4), dtype=torch.float32)

        fc = SimpleNamespace(attn_metadata=None, virtual_engine=0)
        with patch.object(npu_mla_mod, "get_forward_context", return_value=fc):
            out = npu_mla_mod.NPUDeepseekMLAAttention.forward(m, hs, cos, sin)
        self.assertTrue(torch.equal(out, torch.tensor([1])))

        m._forward_prefill.reset_mock()
        fc = SimpleNamespace(attn_metadata=SimpleNamespace(prefill=object(), decode=None), virtual_engine=0)
        with patch.object(npu_mla_mod, "get_forward_context", return_value=fc):
            out = npu_mla_mod.NPUDeepseekMLAAttention.forward(m, hs, cos, sin)
        self.assertTrue(torch.equal(out, torch.tensor([1])))

        m._forward_decode.reset_mock()
        fc = SimpleNamespace(attn_metadata=SimpleNamespace(prefill=None, decode=object()), virtual_engine=0)
        with patch.object(npu_mla_mod, "get_forward_context", return_value=fc):
            out = npu_mla_mod.NPUDeepseekMLAAttention.forward(m, hs, cos, sin)
        self.assertTrue(torch.equal(out, torch.tensor([2])))

    def test_forward_attn_metadata_dict_extracts_prefix(self):
        m = self._make_stub()
        m.prefix = "layers.7"
        hs = torch.randn((2, 16), dtype=torch.float32)
        cos = torch.zeros((2, 1, 1, 4), dtype=torch.float32)
        sin = torch.zeros((2, 1, 1, 4), dtype=torch.float32)

        meta = SimpleNamespace(prefill=None, decode=object())
        fc = SimpleNamespace(attn_metadata={f"{m.prefix}.attn": meta}, virtual_engine=0)
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
        fc = SimpleNamespace(attn_metadata=None, virtual_engine=0)

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


class TestNPUMLAPrefillDecode(unittest.TestCase):
    def _make_stub(self):
        m = SimpleNamespace()
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

        m.q_a_proj = _FakeLinear(m.q_lora_rank)
        m.kv_a_proj_with_mqa = _FakeLinear(m.kv_lora_rank + m.qk_rope_head_dim)
        m.q_a_layernorm = _FakeLayerNorm(m.q_lora_rank)
        m.kv_a_layernorm = _FakeLayerNorm(m.kv_lora_rank)
        m.q_b_proj = _FakeLinear(m.num_local_heads * m.qk_head_dim)
        m.kv_b_proj = _FakeLinear(m.num_local_heads * (m.qk_nope_head_dim + m.v_head_dim))
        m.o_proj = _FakeLinear(16)

        impl = SimpleNamespace(
            W_UK_T=torch.zeros((m.num_local_heads, m.qk_nope_head_dim, m.kv_lora_rank), dtype=torch.float32),
            W_UV=torch.zeros((m.num_local_heads, m.kv_lora_rank, m.v_head_dim), dtype=torch.float32),
            SHARE_MASK_TRIL_SPARSE=torch.ones((1,), dtype=torch.bool),
        )
        kv0 = torch.zeros((2, 128, m.kv_lora_rank), dtype=torch.float32)
        kv1 = torch.zeros((2, 128, m.qk_rope_head_dim), dtype=torch.float32)
        m.attn = SimpleNamespace(impl=impl, kv_cache=[(kv0, kv1)])
        return m

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
        fc = SimpleNamespace(attn_metadata=meta, virtual_engine=0)
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
        ):
            torch_npu_ns.current_stream = lambda: fake_stream
            torch_npu_ns.stream = lambda _s: nullcontext()
            out = npu_mla_mod.NPUDeepseekMLAAttention._forward_prefill(m, hs, cos, sin, attn_metadata=meta)

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
        fc = SimpleNamespace(attn_metadata=meta, virtual_engine=0)

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
        ):
            out = npu_mla_mod.NPUDeepseekMLAAttention._forward_decode(m, hs, cos, sin, attn_metadata=meta)

        self.assertEqual(tuple(out.shape), (bs, 16))


class TestNPUMLAInit(unittest.TestCase):
    def _fake_cfg(self, rope_type="default", apply_yarn=True):
        return SimpleNamespace(
            rms_norm_eps=1e-6,
            rope_parameters={
                "rope_type": rope_type,
                "apply_yarn_scaling": apply_yarn,
                "factor": 2.0,
                "mscale_all_dim": False,
            },
        )

    @patch.object(npu_mla_mod, "get_tensor_model_parallel_world_size", return_value=1)
    @patch("omni_npu.v1.layers.attention.npu_mla.MLAAttention")
    @patch("omni_npu.v1.layers.attention.npu_mla.get_rope")
    @patch("omni_npu.v1.layers.attention.npu_mla.RowParallelFlashCommLinear")
    @patch("omni_npu.v1.layers.attention.npu_mla.ColumnParallelFlashCommLinear")
    @patch("omni_npu.v1.layers.attention.npu_mla.ReplicatedLinear")
    @patch("omni_npu.v1.layers.attention.npu_mla.RMSNorm")
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
