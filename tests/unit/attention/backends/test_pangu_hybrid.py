# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock, patch
import torch
import pytest
import os
import sys
import types
from types import SimpleNamespace

# Ensure torch.fp8 exists for the test environment to avoid AttributeError
# when pangu_hybrid.py references it dynamically.
if not hasattr(torch, 'fp8'):
    torch.fp8 = getattr(torch, 'float8_e4m3fn', torch.int8)

# --- MOCK vLLM kv_cache_interface before any omni_npu imports ---
# This is needed because pangu_hybrid.py tries to import MomeSpec from vllm.v1.kv_cache_interface
# which only works after patches are applied.
from omni_npu.vllm_patches.patches.models.pangu_v2_hybrid.patch_kv_cache_interface import (
    DSAAttentionSpec, ShareKVSlidingWindowSpec, MomeSpec, UniformTypeKVCacheSpecsPatch
)

import vllm.v1.kv_cache_interface as kv_cache_interface_mod
# Manually "apply" the patch for the test environment
kv_cache_interface_mod.MomeSpec = MomeSpec
kv_cache_interface_mod.DSAAttentionSpec = DSAAttentionSpec
kv_cache_interface_mod.ShareKVSlidingWindowSpec = ShareKVSlidingWindowSpec

from vllm.v1.kv_cache_interface import KVCacheSpec

from omni_npu.vllm_patches.patches.models.pangu_v2_hybrid.patch_single_type_kv_cache_manager import (
    MomeManager
)
from omni_npu.vllm_patches.patches.models.pangu_v2_hybrid.patch_worker_utils import (
    bind_kv_cache_patched
)
from omni_npu.vllm_patches.patches.models.pangu_v2_hybrid.patch_kv_cache_utils import (
    _get_kv_cache_groups_uniform_page_size_patched
)
from omni_npu.attention.backends.utils import _maybe_padded_raw_tensor_to_strided_caches
from omni_npu.attention.backends.dsa import NPUDSABackend
from omni_npu.attention.backends.mla import NPUMLABackend
from omni_npu.attention.backends.mome import (
    NPUPanguMomeBackend,
    NPUMomeAttentionMetadataBuilder,
    NPUMomeAttentionMetadata,
)

import omni_npu.attention.backends.dsa as dsa_module
import omni_npu.attention.backends.mla as mla_module


@pytest.mark.unit
class TestPanguKVCacheSpecs(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')

    def test_dsa_attention_spec_real_page_size(self):
        # Default/Non-quant
        spec = DSAAttentionSpec(num_kv_heads=1, head_size=128, block_size=16, dtype=torch.bfloat16)
        self.assertEqual(spec.real_page_size_bytes, 16 * 1 * 128 * 2)

        # Quant fp8
        spec_fp8 = DSAAttentionSpec(num_kv_heads=1, head_size=128, block_size=16, dtype=torch.bfloat16, cache_dtype_str="fp8_ds_mla")
        self.assertEqual(spec_fp8.real_page_size_bytes, 16 * (656 + 128 + 4))

        # Quant int8
        spec_int8 = DSAAttentionSpec(num_kv_heads=1, head_size=128, block_size=16, dtype=torch.bfloat16, cache_dtype_str="int8_ds_mla")
        self.assertEqual(spec_int8.real_page_size_bytes, 16 * (656 + 128 + 2))

    def test_dsa_attention_spec_validation(self):
        # Assertions in __post_init__
        with self.assertRaises(AssertionError):
            DSAAttentionSpec(num_kv_heads=2, head_size=128, block_size=16, dtype=torch.bfloat16)

        with self.assertRaises(AssertionError):
            DSAAttentionSpec(num_kv_heads=1, head_size=128, block_size=16, dtype=torch.bfloat16, page_size_padded=1024)

        with self.assertRaises(AssertionError):
            DSAAttentionSpec(num_kv_heads=1, head_size=128, block_size=16, dtype=torch.bfloat16, sliding_window=100)

    def test_dsa_attention_spec_merge(self):
        spec1 = DSAAttentionSpec(num_kv_heads=1, head_size=128, block_size=16, dtype=torch.bfloat16, cache_dtype_str="fp8_ds_mla")
        spec2 = DSAAttentionSpec(num_kv_heads=1, head_size=128, block_size=16, dtype=torch.bfloat16, cache_dtype_str="fp8_ds_mla")

        merged = DSAAttentionSpec.merge([spec1, spec2])
        self.assertEqual(merged.cache_dtype_str, "fp8_ds_mla")
        self.assertEqual(merged.block_size, 16)

        spec3 = DSAAttentionSpec(num_kv_heads=1, head_size=128, block_size=16, dtype=torch.bfloat16, cache_dtype_str="int8_ds_mla")
        with self.assertRaises(AssertionError):
            DSAAttentionSpec.merge([spec1, spec3])

    def test_share_kv_sliding_window_spec(self):
        # SlidingWindowSpec requires sliding_window
        spec = ShareKVSlidingWindowSpec(num_kv_heads=1, head_size=512, block_size=16, dtype=torch.bfloat16, sliding_window=100)
        self.assertEqual(spec.real_page_size_bytes, 16 * 1 * 512 * 2)

        # Invalid num_kv_heads
        with self.assertRaises(AssertionError):
            ShareKVSlidingWindowSpec(num_kv_heads=2, head_size=512, block_size=16, dtype=torch.bfloat16, sliding_window=100)

        # Invalid head_size
        with self.assertRaises(AssertionError):
            ShareKVSlidingWindowSpec(num_kv_heads=1, head_size=128, block_size=16, dtype=torch.bfloat16, sliding_window=100)

    def test_mome_spec(self):
        shapes = ((10,), (20,), (30,))
        dtypes = (torch.float32, torch.bfloat16, torch.int8)
        # MambaSpec does not have num_kv_heads or head_size
        spec = MomeSpec(
            block_size=16,
            shapes=shapes, dtypes=dtypes, kernel_size=4, num_spec_tokens=2
        )
        # kernel_size - 1 + num_spec_tokens
        self.assertEqual(spec.num_total_tokens, 4 - 1 + 2)

        # Expected size calculation: sum(prod(shape) * dtype_size) * num_total_tokens
        # (10 * 4 + 20 * 2 + 30 * 1) = 40 + 40 + 30 = 110. 110 * 5 = 550
        expected_page_size = 110 * 5
        self.assertEqual(spec.page_size_bytes, expected_page_size)

        with self.assertRaises(ValueError):
            MomeSpec(block_size=16, shapes=((10,),), dtypes=(torch.float32,), kernel_size=4)

        with self.assertRaises(ValueError):
            MomeSpec(block_size=16, shapes=shapes, dtypes=dtypes, kernel_size=0)


@pytest.mark.unit
class TestMomeManager(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')

    def test_mome_manager_skipped_and_prefix(self):
        spec = MomeSpec(
            block_size=16,
            shapes=((10,), (10,), (10,)), dtypes=(torch.float32, torch.float32, torch.float32),
            kernel_size=4, num_spec_tokens=0
        )
        mock_pool = MagicMock()
        manager = MomeManager(kv_cache_spec=spec, block_pool=mock_pool, enable_caching=True, kv_cache_group_id=0)

        # max(0, num_computed_tokens - self.kernel_size + 1)
        self.assertEqual(manager.get_num_skipped_tokens(5), 2)
        self.assertEqual(manager.get_num_common_prefix_blocks("req_1"), 0)

    def test_find_longest_cache_hit_right_to_left(self):
        spec = MomeSpec(
            block_size=16,
            shapes=((10,), (10,), (10,)), dtypes=(torch.float32, torch.float32, torch.float32),
            kernel_size=4, num_spec_tokens=0
        )
        mock_pool = MagicMock()
        mock_pool.null_block = "NULL_BLOCK"

        manager = MomeManager(kv_cache_spec=spec, block_pool=mock_pool, enable_caching=True, kv_cache_group_id=0)

        def mock_get_cached_block(block_hash, group_ids):
            if block_hash == "hit":
                return ["HIT_BLOCK"]
            return None
        mock_pool.get_cached_block.side_effect = mock_get_cached_block

        block_hashes = ["miss1", "miss2", "hit", "miss3"]
        computed = manager.find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=64, # 4 blocks * 16
            kv_cache_group_ids=[0],
            block_pool=mock_pool,
            kv_cache_spec=spec,
            use_eagle=False,
            alignment_tokens=16
        )

        # Should pad front with NULL_BLOCK and include HIT_BLOCK at the end
        self.assertEqual(len(computed[0]), 3)
        self.assertEqual(computed[0][0], "NULL_BLOCK")
        self.assertEqual(computed[0][1], "NULL_BLOCK")
        self.assertEqual(computed[0][2], "HIT_BLOCK")


@pytest.mark.unit
class TestMaybePaddedRawTensorToStridedCaches(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')

    def test_valid_strided_caches(self):
        num_blocks = 2
        block_size = 16
        page_size_bytes = 1024 # 1 block page
        raw = torch.zeros(num_blocks * page_size_bytes, dtype=torch.uint8, device=self.device)

        shapes = ((10,), (4,))
        dtypes = (torch.float32, torch.int8)

        caches = _maybe_padded_raw_tensor_to_strided_caches(
            raw, num_blocks, block_size, shapes, dtypes, page_size_bytes
        )

        self.assertEqual(len(caches), 2)
        c1, c2 = caches
        self.assertEqual(c1.shape, (2, 16, 10))
        self.assertEqual(c2.shape, (2, 16, 4))
        self.assertEqual(c1.dtype, torch.float32)
        self.assertEqual(c2.dtype, torch.int8)

        # verify strides: memory offset per block
        # target_stride[0] should jump to the next block (page_size / dtype_size)
        self.assertEqual(c1.stride(0), 1024 // 4)
        self.assertEqual(c2.stride(0), 1024 // 1)

    def test_insufficient_raw_tensor(self):
        raw = torch.zeros(100, dtype=torch.uint8, device=self.device) # Too small
        with self.assertRaises(AssertionError):
            _maybe_padded_raw_tensor_to_strided_caches(raw, 2, 16, ((10,),), (torch.float32,), 1024)

    def test_exceeding_page_size(self):
        raw = torch.zeros(2 * 1024, dtype=torch.uint8, device=self.device)
        shapes = ((300,),) # 300 * 4 = 1200 bytes per block, exceeds 1024 page size limit
        # torch.as_strided can raise RuntimeError if it exceeds storage
        with self.assertRaises((AssertionError, RuntimeError)):
            _maybe_padded_raw_tensor_to_strided_caches(raw, 2, 16, shapes, (torch.float32,), 1024)

    def test_mismatched_shapes_and_dtypes(self):
        raw = torch.zeros(1024, dtype=torch.uint8, device=self.device)
        with self.assertRaises(AssertionError):
            _maybe_padded_raw_tensor_to_strided_caches(raw, 1, 16, ((10,),), (torch.float32, torch.int8), 1024)


@pytest.mark.unit
class TestNPUPanguBackends(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')

    def test_mla_backend(self):
        self.assertEqual(NPUMLABackend.get_name(), "NPUMLA")
        spec = MagicMock()
        spec.block_size = 16
        spec.dtype = torch.bfloat16
        spec.page_size_bytes = 16 * (512 + 64) * 2
        raw = torch.zeros(2 * spec.page_size_bytes, dtype=torch.uint8, device=self.device)

        with patch.object(
            mla_module.model_extra_config,
            "operator_opt_config",
            SimpleNamespace(use_noncontiguous_kv=True),
        ):
            caches = NPUMLABackend.reshape_kv_cache(raw, 2, spec)
        self.assertEqual(len(caches), 2)
        self.assertEqual(caches[0].shape, (2, 16, 512))
        self.assertEqual(caches[1].shape, (2, 16, 64))

    def test_dsa_backend_fp8(self):
        self.assertEqual(NPUDSABackend.get_name(), "NPUDSA")
        spec = MagicMock()
        spec.block_size = 16
        spec.cache_dtype_str = "fp8_ds_mla"
        spec.page_size_bytes = 16 * (656 * 1 + 132 * 1) # Approximation of required bytes
        raw = torch.zeros(2 * spec.page_size_bytes, dtype=torch.uint8, device=self.device)

        with patch.object(
            dsa_module.model_extra_config,
            "operator_opt_config",
            SimpleNamespace(use_noncontiguous_kv=True),
        ):
            caches = NPUDSABackend.reshape_kv_cache(raw, 2, spec)
        self.assertEqual(caches[0].shape, (2, 16, 656))
        self.assertEqual(caches[1].shape, (2, 16, 132))

    def test_dsa_backend_bf16(self):
        spec = MagicMock()
        spec.block_size = 16
        spec.cache_dtype_str = "bfloat16"
        spec.page_size_bytes = 16 * (576 * 2 + 128 * 2)
        raw = torch.zeros(2 * spec.page_size_bytes, dtype=torch.uint8, device=self.device)

        with patch.object(
            dsa_module.model_extra_config,
            "operator_opt_config",
            SimpleNamespace(use_noncontiguous_kv=True),
        ):
            caches = NPUDSABackend.reshape_kv_cache(raw, 2, spec)
        self.assertEqual(caches[0].shape, (2, 16, 576))
        self.assertEqual(caches[1].shape, (2, 16, 128))

    def test_mome_backend(self):
        self.assertEqual(NPUPanguMomeBackend.get_name(), "NPUPanguMome")
        self.assertEqual(NPUPanguMomeBackend.get_builder_cls(), NPUMomeAttentionMetadataBuilder)

        spec = MagicMock()
        spec.num_total_tokens = 5
        spec.shapes = ((10,), (20,))
        spec.dtypes = (torch.float32, torch.bfloat16)
        spec.page_size_bytes = 5 * (10 * 4 + 20 * 2)
        raw = torch.zeros(2 * spec.page_size_bytes, dtype=torch.uint8, device=self.device)

        caches = NPUPanguMomeBackend.reshape_kv_cache(raw, 2, spec)
        self.assertEqual(len(caches), 2)
        self.assertEqual(caches[0].shape, (2, 5, 10))
        self.assertEqual(caches[1].shape, (2, 5, 20))


@pytest.mark.unit
class TestNPUMomeAttentionMetadataBuilder(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.spec = MagicMock(spec=MomeSpec)
        self.spec.block_size = 16

        self.vllm_config = MagicMock()
        self.vllm_config.speculative_config = None
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.compilation_config.max_cudagraph_capture_size = 16
        self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs.return_value = True
        self.vllm_config.cache_config.enable_prefix_caching = True
        self.vllm_config.model_config.max_model_len = 1024
        self.vllm_config.parallel_config.decode_context_parallel_size = 1

    def test_builder_init(self):
        builder = NPUMomeAttentionMetadataBuilder(self.spec, ["layer1"], self.vllm_config, self.device)
        self.assertEqual(builder.mome_block_size, 16)
        self.assertEqual(builder.decode_cudagraph_max_bs, 16)
        # 16 is decode_cudagraph_max_bs limit, 1024/16 = 64 blocks max
        self.assertEqual(builder.cache_indices_tensor.shape, (16, 64))

    @patch('omni_npu.attention.backends.mome.split_decodes_and_prefills')
    def test_builder_build_decode_with_cudagraph(self, mock_split):
        mock_split.return_value = (4, 0, 4, 0) # num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens

        builder = NPUMomeAttentionMetadataBuilder(self.spec, ["layer1"], self.vllm_config, self.device)

        common_meta = MagicMock()
        common_meta.num_reqs = 4
        common_meta.query_start_loc = torch.tensor([0, 1, 2, 3, 4], device=self.device)
        common_meta.max_query_len = 1
        common_meta.block_table_tensor = torch.ones((4, 64), dtype=torch.int32, device=self.device)
        common_meta.compute_num_computed_tokens.return_value = torch.tensor([10, 20, 30, 40], device=self.device)
        common_meta.seq_lens = torch.tensor([11, 21, 31, 41], device=self.device)

        meta = builder.build(common_prefix_len=0, common_attn_metadata=common_meta)

        self.assertIsInstance(meta, NPUMomeAttentionMetadata)
        self.assertEqual(meta.num_decodes, 4)
        self.assertEqual(meta.num_prefills, 0)
        self.assertEqual(meta.B_size, 16)

        # Test CUDAGraph persistence copying mechanism works appropriately
        self.assertTrue(torch.equal(builder.cache_indices_tensor[:4], common_meta.block_table_tensor[:4]))
        self.assertTrue(torch.equal(meta.cache_indices, builder.cache_indices_tensor[:4]))

    @patch('omni_npu.attention.backends.mome.split_decodes_and_prefills')
    def test_update_block_table(self, mock_split):
        mock_split.return_value = (4, 0, 4, 0)
        builder = NPUMomeAttentionMetadataBuilder(self.spec, ["layer1"], self.vllm_config, self.device)

        metadata = MagicMock()
        metadata.num_prefills = 0

        new_table = torch.full((4, 64), 2, dtype=torch.int32, device=self.device)

        updated_meta = builder.update_block_table(metadata, new_table, torch.tensor([], device=self.device))

        # Should persist the updated table into the tensor buffer
        self.assertTrue(torch.equal(builder.cache_indices_tensor[:4], new_table))
        self.assertTrue(torch.equal(updated_meta.cache_indices, new_table))

    @patch('omni_npu.attention.backends.mome.split_decodes_and_prefills')
    def test_build_for_cudagraph_capture(self, mock_split):
        mock_split.return_value = (2, 0, 2, 0)
        builder = NPUMomeAttentionMetadataBuilder(self.spec, ["layer1"], self.vllm_config, self.device)

        common_meta = MagicMock()
        common_meta.num_reqs = 2
        common_meta.num_actual_tokens = 2
        common_meta.query_start_loc = torch.tensor([0, 1, 2], device=self.device)
        common_meta.compute_num_computed_tokens.return_value = torch.tensor([10, 20], device=self.device)
        common_meta.seq_lens = torch.tensor([11, 21], device=self.device)
        common_meta.block_table_tensor = torch.ones((2, 64), dtype=torch.int32, device=self.device)

        with patch.object(builder, 'build') as mock_build:
            builder.build_for_cudagraph_capture(common_meta)
            mock_build.assert_called_once()
            args, kwargs = mock_build.call_args
            # Diff of query_start_loc -> num_accepted_tokens
            self.assertTrue(torch.equal(args[2], torch.tensor([1, 1], dtype=torch.int32, device=self.device)))


@pytest.mark.unit
class TestPanguPatches(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')

    def test_uniform_type_kv_cache_specs_patch(self):
        spec1 = MomeSpec(block_size=16, shapes=((10,),)*3, dtypes=(torch.float32,)*3, kernel_size=4)
        spec2 = MomeSpec(block_size=16, shapes=((10,),)*3, dtypes=(torch.float32,)*3, kernel_size=4)

        specs = {"layer1": spec1, "layer2": spec2}
        self.assertTrue(UniformTypeKVCacheSpecsPatch.is_uniform_type(specs))

    def test_worker_utils_patch_bind_kv_cache(self):
        kv_caches = {"model.layers.0.attn.kv": torch.zeros(1, device=self.device), "model.layers.1.attn.kv": torch.zeros(2, device=self.device)}
        forward_context = {k: MagicMock() for k in kv_caches}
        runner_kv_caches = []

        bind_kv_cache_patched(kv_caches, forward_context, runner_kv_caches)
        self.assertEqual(len(runner_kv_caches), 2)
        for k, v in kv_caches.items():
            self.assertEqual(forward_context[k].kv_cache, [v])

    def test_override_group_size_patch(self):
        spec1 = MagicMock(spec=KVCacheSpec)
        spec1.block_size = 16
        kv_cache_spec = {"layer0": spec1, "layer1": spec1, "layer2": spec1}

        with patch('omni_npu.vllm_patches.patches.models.pangu_v2_hybrid.patch_kv_cache_utils.create_kv_cache_group_specs') as mock_create, \
             patch.dict(os.environ, {"HYBRID_ATTN_GROUP_SIZE": "2"}):
            _get_kv_cache_groups_uniform_page_size_patched(kv_cache_spec)
            mock_create.assert_called_once()
            args = mock_create.call_args[0]
            # 3 layers split into groups of 2 -> 2 groups
            self.assertEqual(len(args[1]), 2)


if __name__ == '__main__':
    unittest.main()
