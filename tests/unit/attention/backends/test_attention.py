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

# =============================================================================
# STEP 1: Define minimal real abstract base classes (do NOT mock these!)
# We need real classes so that NPUAttentionBackend can properly inherit them.
# =============================================================================

class AttentionType:
    DECODER = "DECODER"

class AttentionImpl(ABC, Generic[TypeVar('T')]):
    pass

class AttentionLayer(ABC):
    pass

class AttentionBackend(ABC):
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        ...

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> Type[AttentionImpl]:
        ...

    @staticmethod
    def get_supported_dtypes() -> List[Any]:
        return []

    @staticmethod
    def get_kv_cache_shape(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def reshape_kv_cache(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metadata_cls():
        raise NotImplementedError

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError

# =============================================================================
# STEP 2: Inject real abstract classes into vLLm's module namespace
# This ensures 'from vllm.attention.backends.abstract import ...' works correctly.
# =============================================================================

abstract_mod = types.ModuleType('vllm.attention.backends.abstract')
abstract_mod.AttentionType = AttentionType
abstract_mod.AttentionImpl = AttentionImpl
abstract_mod.AttentionLayer = AttentionLayer
abstract_mod.AttentionBackend = AttentionBackend
sys.modules['vllm.attention.backends.abstract'] = abstract_mod

# =============================================================================
# STEP 3: Safely mock ONLY the submodules that don't affect class inheritance
# Use REAL module objects to avoid import/attribute errors
# =============================================================================

# --- vllm.platforms ---
platforms_mod = types.ModuleType('vllm.platforms')
current_platform = MagicMock()
current_platform.device_type = "npu"
platforms_mod.current_platform = current_platform
sys.modules['vllm.platforms'] = platforms_mod

# --- vllm.forward_context ---
forward_ctx_mod = types.ModuleType('vllm.forward_context')
forward_ctx_mod.get_forward_context = MagicMock(return_value=None)
sys.modules['vllm.forward_context'] = forward_ctx_mod

# --- vllm.v1 module hierarchy ---
sys.modules['vllm.v1'] = types.ModuleType('vllm.v1')
sys.modules['vllm.v1.attention'] = types.ModuleType('vllm.v1.attention')
sys.modules['vllm.v1.attention.backends'] = types.ModuleType('vllm.v1.attention.backends')

# --- vllm.v1.attention.backends.utils ---
utils_mod = types.ModuleType('vllm.v1.attention.backends.utils')
utils_mod.split_decodes_and_prefills = MagicMock(return_value=(1, 0, 1, 0))

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
        self._init_reorder_batch_threshold = getattr(vllm_config, 'reorder_batch_threshold', 0)
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device
utils_mod.AttentionMetadataBuilder = AttentionMetadataBuilder
utils_mod.CommonAttentionMetadata = MagicMock()
utils_mod.AttentionCGSupport = MagicMock()
sys.modules['vllm.v1.attention.backends.utils'] = utils_mod

# --- vllm.v1.kv_cache_interface ---
kv_cache_mod = types.ModuleType('vllm.v1.kv_cache_interface')
kv_cache_mod.AttentionSpec = MagicMock()
sys.modules['vllm.v1.kv_cache_interface'] = kv_cache_mod

# Ensure top-level 'vllm' exists
if 'vllm' not in sys.modules:
    sys.modules['vllm'] = types.ModuleType('vllm')

# =============================================================================
# STEP 4: Mock torch_npu before importing omni_npu
# =============================================================================

torch_npu_mock = MagicMock()
sys.modules['torch_npu'] = torch_npu_mock

# Now it's safe to import omni_npu — its backend will inherit from REAL base classes
from omni_npu.attention.backends import (
    NPUAttentionBackendImpl,
    NPUMetadata,
    NPUAttentionBackend,
    NPUAttentionMetadataBuilder,
)

# =============================================================================
# Unit Test Class
# =============================================================================

@pytest.mark.unit
class TestNPUAttentionBackendDefault(unittest.TestCase):

    def setUp(self):
        self.impl_interface_cls = NPUAttentionBackend
        self.torch_npu_mock = torch_npu_mock

    def tearDown(self):
        # Reset mocks but DO NOT delete modules (avoids import instability)
        self.torch_npu_mock.reset_mock()

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
        self.torch_npu_mock = torch_npu_mock

    def tearDown(self):
        # Reset mocks but DO NOT delete modules (avoids import instability)
        self.torch_npu_mock.reset_mock()
        
    def test_metadata_builder(self):
            # Define a minimal CommonAttentionMetadata (normally from vLLm)
            class CommonAttentionMetadata:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            spec = MagicMock()
            spec.block_size = 16

            builder = self.metadata_builder_cls(
                kv_cache_spec=spec,
                layer_names=["test"],
                vllm_config=MagicMock(),
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
        self.torch_npu_mock = torch_npu_mock

    def tearDown(self):
        # Reset mocks but DO NOT delete modules (avoids import instability)
        self.torch_npu_mock.reset_mock()
    
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

        query = torch.randn(10, 8 * 128)
        key = torch.randn(10, 4 * 128)
        value = torch.randn(10, 4 * 128)
        kv_cache = (torch.zeros(100, 16, 4 * 128), torch.zeros(100, 16, 4 * 128))
        output = torch.empty_like(query)  # [10, 1024]

        metadata = self.metadata_cls(
            num_actual_tokens=10,
            block_tables=torch.randint(0, 100, (2, 10)),
            query_cumlens=[10],
            seq_lens=[10],
            max_query_len=1,
            slot_mapping=torch.arange(10),
            num_prefills=0,
        )

        decode_output = output.unsqueeze(1).clone()  # [10, 1, 1024]

        with patch.object(self.torch_npu_mock, 'npu_scatter_nd_update_', wraps=self.torch_npu_mock.npu_scatter_nd_update_) as mock_scatter, \
             patch.object(self.torch_npu_mock, 'npu_fused_infer_attention_score', return_value=(decode_output,)) as mock_decode:

            result = impl.forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )

            self.assertEqual(mock_scatter.call_count, 2)
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

        with patch.object(self.torch_npu_mock, 'npu_scatter_nd_update_') as mock_scatter, \
             patch.object(self.torch_npu_mock, 'npu_fused_infer_attention_score_v2', return_value=(prefill_output,)) as mock_prefill:

            result = impl.forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )

            self.assertEqual(mock_scatter.call_count, 2)
            mock_prefill.assert_called_once()
            args, kwargs = mock_prefill.call_args
            self.assertEqual(kwargs['num_query_heads'], 8)
            self.assertEqual(kwargs['num_key_value_heads'], 4)
            self.assertEqual(kwargs['input_layout'], "TND")
            self.assertAlmostEqual(kwargs['softmax_scale'], 0.125)
            self.assertEqual(kwargs['sparse_mode'], 3)
            self.assertIn('atten_mask', kwargs)
            self.assertEqual(kwargs['actual_seq_qlen'], [10, 20])
            self.assertEqual(kwargs['actual_seq_kvlen'], [10, 10])
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


if __name__ == '__main__':
    unittest.main()