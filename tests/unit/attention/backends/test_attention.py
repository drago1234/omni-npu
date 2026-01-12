# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for NPUAttentionBackendImpl that do NOT require actual NPU hardware.
These tests use mocking to verify the logic and API contracts.
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import torch
import pytest
from typing import Generic, TypeVar, List, Any, Type
import types
from abc import ABC, abstractmethod

from vllm.attention.backends.abstract import (
    AttentionLayer,
    AttentionType,
    AttentionBackend,
    AttentionImpl
) 

import torch_npu
utils_mod_patcher = None
distributed_mod_patcher = None
forward_context_mod_patcher = None
patcher_dcp = None
patcher_pcp = None
NPUAttentionBackendImpl = None
NPUMetadata = None
NPUAttentionBackend = None
NPUAttentionMetadataBuilder = None

def setup_module():
    try:
        global NPUAttentionBackendImpl, NPUMetadata, NPUAttentionBackend, NPUAttentionMetadataBuilder 
        # Define a TypeVar for the metadata type
        MetadataT = TypeVar('MetadataT')

        # Make the mock class generic
        class AttentionMetadataBuilder(Generic[MetadataT]):
            def __init__(self, *args, **kwargs):
                if len(args) == 4:
                    kv_cache_spec, layer_names, vllm_config, device = args
                else:
                    kv_cache_spec = kwargs['kv_cache_spe']
                    layer_names = kwargs['layer_names']
                    vllm_config = kwargs['vllm_config']
                    device = kwargs['device']
                self.kv_cache_spec = kv_cache_spec
                self.layer_names = layer_names
                self.vllm_config = vllm_config
                self.device = device
            def _init_reorder_batch_threshold(self, reorder_batch_threshold, default_threshold):
                    self.reorder_batch_threshold = max(reorder_batch_threshold, 0)

        utils_mod = types.ModuleType('vllm.v1.attention.backends.utils')
        utils_mod.split_decodes_and_prefills = MagicMock(return_value=(1, 0, 1, 0))
        utils_mod.AttentionMetadataBuilder = AttentionMetadataBuilder
        utils_mod.CommonAttentionMetadata = MagicMock()
        utils_mod.AttentionCGSupport = MagicMock()
        utils_mod_patcher = patch.dict('sys.modules', {
                'vllm.v1.attention.backends.utils': utils_mod
        })
        utils_mod_patcher.start()
        
        fake_dcp = MagicMock()
        fake_dcp.world_size = 1
        fake_dcp.rank_in_group = 0

        fake_pcp = MagicMock()
        fake_pcp.world_size = 1
        fake_pcp.rank_in_group = 0

        patcher_dcp = patch('vllm.distributed.parallel_state.get_dcp_group', return_value=fake_dcp)
        patcher_pcp = patch('vllm.distributed.parallel_state.get_pcp_group', return_value=fake_pcp)
        
        distributed_mod_patcher = patch.dict('sys.modules', {
            'vllm.distributed': MagicMock(),
            'vllm.distributed.eplb': MagicMock(),
            'vllm.distributed.eplb.eplb_state': MagicMock(),
        })
        
        patcher_dcp.start()
        patcher_pcp.start()
        distributed_mod_patcher.start()


        forward_ctx_mod = types.ModuleType('vllm.forward_context')
        forward_ctx_mod.get_forward_context = MagicMock(
            batch_descriptor=None
        )
        forward_context_mod_patcher = patch.dict('sys.modules', {
            'vllm.forward_context': forward_ctx_mod
        })
        forward_context_mod_patcher.start()
        # Now it's safe to import omni_npu — its backend will inherit from REAL base classes
        from omni_npu.attention.backends import (
            NPUAttentionBackendImpl as _impl,
            NPUMetadata as _meta,
            NPUAttentionBackend as _backend,
            NPUAttentionMetadataBuilder as _builder,
        )
        NPUAttentionBackendImpl = _impl
        NPUMetadata = _meta
        NPUAttentionBackend = _backend
        NPUAttentionMetadataBuilder = _builder
    except Exception as e:
        print(f"❌ FAILED to import omni_npu classes: {e}")
        import traceback
        traceback.print_exc()
        raise

def teardown_module():
    global utils_mod_patcher, distributed_mod_patcher, forward_context_mod_patcher
    if utils_mod_patcher is not None:
        utils_mod_patcher.stop()
    if distributed_mod_patcher is not None:
        distributed_mod_patcher.stop()
    if forward_context_mod_patcher is not None:
        forward_context_mod_patcher.stop()
    if patcher_dcp:
        patcher_dcp.stop()
    if patcher_dcp:
        patcher_pcp.stop()
    
@pytest.mark.unit
class TestNPUAttentionBackendDefault(unittest.TestCase):

    def setUp(self):
        self.impl_interface_cls = NPUAttentionBackend

    def test_backend_properties(self):
        backend = self.impl_interface_cls()
        self.assertIn(torch.float16, backend.get_supported_dtypes())
        self.assertEqual(backend.get_name(), "VLLM_NPU_ATTN")
        self.assertIs(backend.get_impl_cls(), NPUAttentionBackendImpl)
        self.assertIs(backend.get_metadata_cls(), NPUMetadata)
        self.assertIs(backend.get_builder_cls(), NPUAttentionMetadataBuilder)

    def test_kv_cache_shape_and_reshape(self):
        shape = self.impl_interface_cls.get_kv_cache_shape(
            num_blocks=10,
            block_size=16,
            num_kv_heads=4,
            head_size=128
        )
        self.assertEqual(shape, (10, 16, 512))  # 4 * 128 = 512

        raw = torch.randn(2 * 10 * 16 * 512, dtype=torch.bfloat16)
        k_cache, v_cache = self.impl_interface_cls.reshape_kv_cache(
            raw, num_blocks=10, block_size=16, num_kv_heads=4, head_size=128, dtype=torch.bfloat16
        )
        self.assertEqual(k_cache.shape, (10, 16, 512))
        self.assertEqual(v_cache.shape, (10, 16, 512))
        self.assertTrue(torch.equal(raw[:10*16*512].view(10,16,512), k_cache))

@pytest.mark.unit
class TestNPUAttentionBackendDefaultMetadataBuilder(unittest.TestCase):

    def setUp(self):
        self.metadata_builder_cls = NPUAttentionMetadataBuilder
        
    def test_metadata_builder(self):
            # Define a minimal CommonAttentionMetadata (normally from vLLm)
            class CommonAttentionMetadata:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            spec = MagicMock()
            spec.block_size = 16
            vllm_config = MagicMock()
            vllm_config.reorder_batch_threshold = 0
            builder = self.metadata_builder_cls(
                kv_cache_spec=spec,
                layer_names=["test"],
                vllm_config=vllm_config,
                device=torch.device("npu")
            )

            common_meta = CommonAttentionMetadata(
                num_actual_tokens=20,
                query_start_loc=torch.tensor([0, 10, 20]),
                seq_lens=torch.tensor([10, 10]),
                max_query_len=10,
                block_table_tensor=torch.randint(0, 100, (2, 10)),
                slot_mapping=torch.arange(20),
                context_lens=None,
                max_context_len=None,
                qkv_format="TND",
            )

            with patch('vllm.v1.attention.backends.utils.split_decodes_and_prefills', return_value=(0, 2, 0, 20)):
                meta = builder.build(common_prefix_len=0, common_attn_metadata=common_meta)

            self.assertIsInstance(meta, NPUMetadata)
            self.assertEqual(meta.num_actual_tokens, 20)
            self.assertEqual(meta.num_prefills, 0)
            self.assertEqual(meta.query_cumlens, [10, 20])
            self.assertEqual(meta.seq_lens, [10, 10])
            self.assertEqual(meta.max_query_len, 10)

@pytest.mark.unit
class TestNPUAttentionBackendDefaultImpl(unittest.TestCase):

    def setUp(self):
        self.impl_cls = NPUAttentionBackendImpl
        self.metadata_cls = NPUMetadata
    
    def test_init_success(self):
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=1.0,
            num_kv_heads=4,
            attn_type=AttentionType.DECODER,
        )
        self.assertEqual(impl.num_heads, 8)
        self.assertEqual(impl.num_kv_heads, 4)
        self.assertEqual(impl.head_size, 128)

    def test_init_invalid_attn_type_raises(self):
        with self.assertRaises(NotImplementedError):
            self.impl_cls(
                num_heads=8,
                head_size=128,
                scale=1.0,
                attn_type="ENCODER",
            )

    def test_init_num_heads_not_divisible_by_kv_heads_raises(self):
        with self.assertRaises(RuntimeError):
            self.impl_cls(
                num_heads=7,
                head_size=128,
                scale=1.0,
                num_kv_heads=3,
                attn_type=AttentionType.DECODER,
            )

    def test_forward_decode_path_calls_npu_fused_infer_attention_score(self):
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=0.125,
            num_kv_heads=4,
            attn_type=AttentionType.DECODER,
        )

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        batch_size = 12
        query = torch.randn(batch_size, 8 * 128)
        key = torch.randn(batch_size, 4 * 128)
        value = torch.randn(batch_size, 4 * 128)
        kv_cache = (torch.zeros(batch_size ** 2, 16, 4 * 128), torch.zeros(100, 16, 4 * 128))

        metadata = self.metadata_cls(
            num_actual_tokens=10,
            block_tables=torch.randint(0, 100, (2, 10)),
            query_cumlens=[10],
            seq_lens=[10],
            max_query_len=1,
            slot_mapping=torch.arange(10),
            num_prefills=0,
            num_decode_tokens=8,
            num_decodes=2,
        )

        attn_output = torch.randn(batch_size, 8, 128)
        output = torch.empty_like(attn_output)
    
        def fake_scatter_nd_update_(tensor, indices, updates):
            if indices.ndim == 2 and indices.shape[1] == 1:
                indices = indices.squeeze(1)
            elif indices.ndim > 1:
                raise NotImplementedError("Only 1D or [N,1] indices supported in mock")

            num_indices = indices.shape[0]
            if updates.shape[0] != num_indices:
                updates = updates[:num_indices]

            tensor[indices] = updates
            return tensor

        with patch('torch_npu.npu_scatter_nd_update_', side_effect=fake_scatter_nd_update_), \
         patch('torch_npu.npu_fused_infer_attention_score', return_value=(output,)) as mock_decode:
            
            result = impl.forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )

            # self.assertEqual(mock_scatter.call_count, 2)
            mock_decode.assert_called_once()
            args, kwargs = mock_decode.call_args
            self.assertEqual(kwargs['num_heads'], 8)
            self.assertEqual(kwargs['num_key_value_heads'], 4)
            self.assertEqual(kwargs['input_layout'], "BSND")
            self.assertAlmostEqual(kwargs['scale'], 0.125)
            self.assertIs(result, output)

    def test_forward_prefill_path_calls_npu_fused_infer_attention_score_v2(self):
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=0.125,
            num_kv_heads=4,
            attn_type=AttentionType.DECODER,
        )

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        query = torch.randn(20, 8 * 128)  # [20, 1024]
        key = torch.randn(20, 4 * 128)    # [20, 512]
        value = torch.randn(20, 4 * 128)
        kv_cache = (torch.zeros(100, 16, 4 * 128), torch.zeros(100, 16, 4 * 128))
        output = torch.empty_like(query)  # [20, 1024]

        metadata = self.metadata_cls(
            num_actual_tokens=20,
            block_tables=torch.randint(0, 100, (2, 10)),
            query_cumlens=[10, 20],
            seq_lens=[10, 10],
            max_query_len=10,
            slot_mapping=torch.arange(20),
            num_prefills=2,
        )

        prefill_output = output.clone()  # [20, 1024]
        def fake_scatter_nd_update_(tensor, indices, updates):
            if indices.ndim == 2 and indices.shape[1] == 1:
                indices = indices.squeeze(1)
            elif indices.ndim > 1:
                raise NotImplementedError("Only 1D or [N,1] indices supported in mock")

            num_indices = indices.shape[0]
            if updates.shape[0] != num_indices:
                updates = updates[:num_indices]

            tensor[indices] = updates
            return tensor

        with patch('torch_npu.npu_scatter_nd_update_', side_effect=fake_scatter_nd_update_), \
         patch('torch_npu.npu_fused_infer_attention_score_v2', return_value=(prefill_output,)) as mock_decode:
            result = impl.forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )

            self.assertIs(result, output)

    def test_forward_requires_output_tensor(self):
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=1.0,
            attn_type=AttentionType.DECODER,
        )
        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        query = torch.randn(1, 1024)
        key = value = torch.randn(1, 512)
        kv_cache = (torch.zeros(10, 16, 512), torch.zeros(10, 16, 512))
        metadata = self.metadata_cls(
            num_actual_tokens=1,
            block_tables=torch.zeros(1, 1, dtype=torch.int32),
            query_cumlens=[1],
            seq_lens=[1],
            slot_mapping=torch.tensor([0], dtype=torch.int64),
            num_prefills=0,
        )

        with self.assertRaises(AssertionError):
            impl.forward(layer, query, key, value, kv_cache, metadata, output=None)

    def test_forward_k_v_scale_not_one_raises(self):
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=1.0,
            attn_type=AttentionType.DECODER,
        )
        layer = MagicMock()
        layer._k_scale_float = 0.5
        layer._v_scale_float = 1.0

        query = torch.randn(1, 1024)
        key = value = torch.randn(1, 512)
        kv_cache = (torch.zeros(10, 16, 512), torch.zeros(10, 16, 512))
        output = torch.empty_like(query)
        metadata = self.metadata_cls(
            num_actual_tokens=1,
            block_tables=torch.zeros(1, 1, dtype=torch.int32),
            query_cumlens=[1],
            seq_lens=[1],
            slot_mapping=torch.tensor([0], dtype=torch.int64),
            num_prefills=0,
        )

        with self.assertRaises(RuntimeError):
            impl.forward(layer, query, key, value, kv_cache, metadata, output=output)
