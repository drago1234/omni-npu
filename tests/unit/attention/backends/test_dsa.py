import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import torch

import omni_npu.attention.backends.dsa as mla_mod


class TestNPUDSABackend(unittest.TestCase):
    def test_reshape_kv_cache_splits_into_three_expected_shapes(self):
        num_blocks = 2
        block_size = 3
        shapes = [
            (num_blocks, block_size, 1, 512),
            (num_blocks, block_size, 1, 64),
            (num_blocks, block_size, 1, 128),
        ]
        total = sum(int(torch.tensor(s).prod().item()) for s in shapes)

        raw = torch.zeros((total,), dtype=torch.bfloat16)

        out = mla_mod.NPUDSABackend.reshape_kv_cache(
            raw_tensor=raw,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
        )
        self.assertEqual(len(out), 3)
        self.assertEqual(tuple(out[0].shape), shapes[0])
        self.assertEqual(tuple(out[1].shape), shapes[1])
        self.assertEqual(tuple(out[2].shape), shapes[2])

    def test_reshape_kv_cache_raises_when_numel_mismatch(self):
        raw = torch.zeros((123,), dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            _ = mla_mod.NPUDSABackend.reshape_kv_cache(
                raw_tensor=raw,
                num_blocks=1,
                block_size=1,
                num_kv_heads=1,
                head_size=128,
                dtype=torch.bfloat16,
            )


class TestNPUDSAMetadataBuilder(unittest.TestCase):
    def _new_builder_minimal(self):
        """
        Avoid calling real MLACommonMetadataBuilder.__init__.
        We'll create via __new__ and fill only attributes used by methods we test.
        """
        b = mla_mod.NPUDSAMetadataBuilder.__new__(mla_mod.NPUDSAMetadataBuilder)
        b.uniform_decode_query_len = 1
        b.mc2_mask = torch.zeros(16, dtype=torch.bool)
        b.vllm_config = MagicMock()
        b.vllm_config.kv_transfer_config = None
        b.vllm_config.speculative_config = None
        b.vllm_config.scheduler_config.max_num_seqs = 16
        b.dcp_world_size = 1
        b._use_fi_prefill = False
        b._use_cudnn_prefill = False
        b.aot_schedule = False
        return b

    def test_generate_activate_mask_sets_prefix_true(self):
        b = self._new_builder_minimal()
        mask = b._generate_activate_mask(actual_seqs_num=5)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertTrue(torch.all(mask[:5]))
        self.assertTrue(torch.all(~mask[5:]))

    def test_align_slot_mapping_pads_to_num_reqs_times_query_len(self):
        b = self._new_builder_minimal()
        b.uniform_decode_query_len = 3
        num_reqs = 4
        target = num_reqs * b.uniform_decode_query_len  # 12

        slot_mapping = torch.tensor([1, 2, 3], dtype=torch.long)
        with patch.object(mla_mod, "PAD_SLOT_ID", 999, create=True):
            out = b._align_slot_mapping(slot_mapping, num_reqs=num_reqs)

        self.assertEqual(out.numel(), target)
        self.assertTrue(torch.equal(out[:3], slot_mapping))
        self.assertTrue(torch.all(out[3:] == 999))

    def test_build_prefill_populates_seq_lens_and_query_cumlens(self):
        b = self._new_builder_minimal()

        class _Prefill:
            def __init__(self):
                self.query_start_loc = torch.tensor([0, 2, 5], dtype=torch.long)
                self.chunked_context = None
                self.seq_lens = None
                self.query_cumlens = None

        class _Meta:
            def __init__(self):
                self.prefill = _Prefill()
                self.decode = None
                self.num_actual_tokens = 0
                self.num_reqs = 0
                self.slot_mapping = torch.tensor([], dtype=torch.long)

        fake_meta = _Meta()

        with patch.object(mla_mod.MLACommonMetadataBuilder, "build", return_value=fake_meta):
            out = b.build(common_prefix_len=0, common_attn_metadata=MagicMock(), fast_build=False)

        self.assertIs(out, fake_meta)
        self.assertTrue(torch.equal(out.prefill.seq_lens, torch.tensor([2, 3], dtype=torch.long)))
        self.assertTrue(torch.equal(out.prefill.query_cumlens, torch.tensor([2, 5], dtype=torch.long)))

    def test_build_raises_on_chunked_prefill(self):
        b = self._new_builder_minimal()

        class _Prefill:
            def __init__(self):
                self.query_start_loc = torch.tensor([0, 1], dtype=torch.long)
                self.chunked_context = object()  # not None triggers error
                self.seq_lens = None
                self.query_cumlens = None

        class _Meta:
            def __init__(self):
                self.prefill = _Prefill()
                self.decode = None
                self.num_actual_tokens = 0
                self.num_reqs = 0
                self.slot_mapping = torch.tensor([], dtype=torch.long)

        fake_meta = _Meta()

        with patch.object(mla_mod.MLACommonMetadataBuilder, "build", return_value=fake_meta):
            with self.assertRaises(RuntimeError):
                _ = b.build(common_prefix_len=0, common_attn_metadata=MagicMock(), fast_build=False)


class TestNPUDSAImplInit(unittest.TestCase):
    def test_init_raises_for_unsupported_features(self):
        with patch.object(mla_mod.MLACommonBaseImpl, "__init__", return_value=None), \
             patch.object(mla_mod.MLACommonMetadataBuilder, "determine_chunked_prefill_workspace_size", return_value=8), \
             patch.object(mla_mod, "get_current_vllm_config", return_value=MagicMock()):
            with self.assertRaises(NotImplementedError):
                _ = mla_mod.NPUDSAImpl(
                    num_heads=2, head_size=8, scale=1.0, num_kv_heads=2,
                    alibi_slopes=[1.0], sliding_window=None, kv_cache_dtype="bf16",
                    logits_soft_cap=None, attn_type=mla_mod.AttentionType.DECODER,
                    kv_sharing_target_layer_name=None,
                )

    def test_init_raises_when_attn_type_not_decoder(self):
        with patch.object(mla_mod.MLACommonBaseImpl, "__init__", return_value=None), \
             patch.object(mla_mod.MLACommonMetadataBuilder, "determine_chunked_prefill_workspace_size", return_value=8), \
             patch.object(mla_mod, "get_current_vllm_config", return_value=MagicMock()):
            with self.assertRaises(NotImplementedError):
                _ = mla_mod.NPUDSAImpl(
                    num_heads=2, head_size=8, scale=1.0, num_kv_heads=2,
                    alibi_slopes=None, sliding_window=None, kv_cache_dtype="bf16",
                    logits_soft_cap=None, attn_type="NOT_DECODER",
                    kv_sharing_target_layer_name=None,
                )


class TestNPUDSAImplForward(unittest.TestCase):
    def _new_impl_stub(self):
        """
        IMPORTANT:
        In this environment, calling torch.nn.Module.__init__ (or nn.Module.init)
        on an object created via __new__ triggers `super(type, obj)` errors.
        So we do NOT make a real nn.Module instance here.

        Instead we build a plain object stub containing only attributes/methods
        that NPUDSAImpl.forward() uses.
        """
        impl = SimpleNamespace()

        # attributes used by forward
        impl.num_heads = 2
        impl.scale = 1.0
        impl.qk_nope_head_dim = 4
        impl.qk_rope_head_dim = 4
        impl.v_head_dim = 4
        impl.kv_lora_rank = 3
        impl.chunked_prefill_workspace_size = 7

        # weights used by _absorb_prolog and _v_up_proj
        impl.W_UK_T = torch.randn((impl.num_heads, impl.qk_nope_head_dim, 5), dtype=torch.float32)
        impl.W_UV = torch.randn((impl.num_heads, impl.kv_lora_rank, impl.v_head_dim), dtype=torch.float32)

        # indexer used by _apply_sparse_attention (only accessed for sparse_indices slicing)
        idx = SimpleNamespace()
        idx.topk_tokens = 8
        idx.topk_indices_buffer = torch.zeros((16, idx.topk_tokens), dtype=torch.int32)
        impl.indexer = idx

        # bind the real helper methods from the class onto this stub via lambda wrappers
        # (so they receive `impl` as self)
        impl._absorb_prolog = lambda q: mla_mod.NPUDSAImpl._absorb_prolog(impl, q)
        impl._v_up_proj = lambda x, out: mla_mod.NPUDSAImpl._v_up_proj(impl, x, out)

        impl._apply_sparse_attention = MagicMock()

        return impl

    def test_forward_profile_run_metadata_none_returns_zeroed_output(self):
        impl = self._new_impl_stub()

        T = 6
        # output must be provided
        output = torch.randn((T, impl.num_heads, impl.qk_nope_head_dim + impl.v_head_dim), dtype=torch.float32)
        q = torch.randn((T, impl.num_heads, impl.qk_nope_head_dim + impl.qk_rope_head_dim), dtype=torch.float32)
        k_c_normed = torch.randn((T, impl.qk_nope_head_dim + impl.v_head_dim), dtype=torch.float32)
        k_pe = torch.randn((T, 1, impl.qk_rope_head_dim), dtype=torch.float32)
        kv_cache = (torch.ones((1, 1)), torch.ones((1, 1)), torch.ones((1, 1)))

        out = mla_mod.NPUDSAImpl.forward(
            impl,
            layer=MagicMock(),
            q=q,
            k_c_normed=k_c_normed,
            k_pe=k_pe,
            kv_cache=kv_cache,
            attn_metadata=None,
            output=output,
        )
        self.assertIs(out, output)
        self.assertTrue(torch.all(out == 0), "profile run should fill output with zeros")

    def test_forward_raises_when_output_scale_provided(self):
        impl = self._new_impl_stub()
        output = torch.zeros((2, 1, 1), dtype=torch.float32)

        with self.assertRaises(NotImplementedError):
            _ = mla_mod.NPUDSAImpl.forward(
                impl,
                layer=MagicMock(),
                q=torch.zeros((2, 1, 8), dtype=torch.float32),
                k_c_normed=torch.zeros((2, 8), dtype=torch.float32),
                k_pe=torch.zeros((2, 1, 4), dtype=torch.float32),
                kv_cache=(torch.ones((1, 1)), torch.ones((1, 1)), torch.ones((1, 1))),
                attn_metadata=MagicMock(),
                output=output,
                output_scale=torch.tensor(1.0),
            )

    def test_forward_calls_scatter_and_runs_prefill_and_decode_paths(self):
        impl = self._new_impl_stub()

        class _Decode:
            def __init__(self):
                self.query_cumlens = torch.tensor([1, 2], dtype=torch.int32)
                self.seq_lens = torch.tensor([5, 6], dtype=torch.int32)
                self.block_table = torch.zeros((2, 4), dtype=torch.int32)
                self.dcp_tot_seq_lens = None
                self.mc2_mask = None

        class _Prefill:
            def __init__(self):
                self.query_cumlens = torch.tensor([2], dtype=torch.int32)
                self.seq_lens = torch.tensor([7], dtype=torch.int32)
                self.block_table = torch.zeros((1, 4), dtype=torch.int32)
                self.query_start_loc = torch.tensor([0, 2], dtype=torch.long)
                self.chunked_context = None

        meta = SimpleNamespace()
        meta.num_actual_tokens = 5
        meta.num_decodes = 1
        meta.num_prefills = 1
        meta.num_decode_tokens = 2
        meta.decode = _Decode()
        meta.prefill = _Prefill()
        meta.slot_mapping = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

        # inputs
        q = torch.randn((6, impl.num_heads, impl.qk_nope_head_dim + impl.qk_rope_head_dim), dtype=torch.float32)
        k_c_normed = torch.randn((6, impl.qk_nope_head_dim + impl.v_head_dim), dtype=torch.float32)
        k_pe = torch.randn((6, 1, impl.qk_rope_head_dim), dtype=torch.float32)

        # kv cache
        k_nope = torch.zeros((10, impl.qk_nope_head_dim + impl.v_head_dim), dtype=torch.float32)
        k_rope = torch.zeros((10, impl.qk_rope_head_dim), dtype=torch.float32)
        kv_cache = (k_nope, k_rope, torch.zeros((1,), dtype=torch.float32))

        # padded output (2D)
        output = torch.zeros((8, impl.num_heads * impl.v_head_dim), dtype=torch.float32)

        # patch scatter
        scatter_calls = {"n": 0}

        def _fake_scatter(dst, slots, src):
            scatter_calls["n"] += 1
            return dst

        # attention returns latent with shape that _v_up_proj expects logically:
        # _v_up_proj transposes and views -> (num_heads, -1, kv_lora_rank) then bmm with W_UV
        def _fake_attn(q_nope, q_pe, kv_cache_arg, attn_meta_arg):
            return torch.randn((q_nope.shape[0], impl.num_heads, impl.kv_lora_rank), dtype=torch.float32)

        impl._apply_sparse_attention = MagicMock(side_effect=_fake_attn)

        with patch.object(mla_mod.torch_npu, "npu_scatter_nd_update_", side_effect=_fake_scatter):
            out = mla_mod.NPUDSAImpl.forward(
                impl,
                layer=MagicMock(),
                q=q,
                k_c_normed=k_c_normed,
                k_pe=k_pe,
                kv_cache=kv_cache,
                attn_metadata=meta,
                output=output,
            )

        self.assertEqual(scatter_calls["n"], 2, "should scatter into k_nope and k_rope once each")
        self.assertEqual(impl._apply_sparse_attention.call_count, 2, "prefill + decode should call attention twice")
        self.assertIs(out, output, "should return padded output tensor")


if __name__ == "__main__":
    unittest.main()