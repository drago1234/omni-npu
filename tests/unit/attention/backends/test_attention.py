# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for NPUAttentionBackendImpl that do NOT require actual NPU hardware.
These tests use mocking to verify the logic and API contracts.
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import importlib
import torch
import pytest
from typing import Generic, TypeVar
import types


import torch_npu

from vllm.v1.attention.backend import AttentionBackend, AttentionImpl, AttentionLayer, AttentionType
from vllm.v1.kv_cache_interface import AttentionSpec


def create_mock_modules(name: str, pure_mock: bool = False):
    if pure_mock:
        res_mod = MagicMock()
    else:
        res_mod = types.ModuleType(name)
    res_mod.__path__ = []
    res_mod.__spec__ = None

    return res_mod


@pytest.fixture
def npu_attention_classes(monkeypatch):
    """
    Module-scoped fixture that sets up NPU attention backend classes.
    Uses patches to mock vLLM dependencies and provides clean isolation.
    """
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

    attn_backend_mod = types.ModuleType('vllm.v1.attention.backend')
    attn_backend_mod.AttentionMetadataBuilder = AttentionMetadataBuilder
    attn_backend_mod.CommonAttentionMetadata = MagicMock()
    attn_backend_mod.AttentionCGSupport = MagicMock()

    # Add missing imports to attn_backend_mod
    attn_backend_mod.AttentionBackend = AttentionBackend
    attn_backend_mod.AttentionImpl = AttentionImpl
    attn_backend_mod.AttentionLayer = AttentionLayer
    attn_backend_mod.AttentionType = AttentionType

    # Create a real class for AttentionMetadata to avoid metaclass conflict
    # When vllm.v1.attention.backends.mla.common.MLACommonMetadata inherits from it.
    class AttentionMetadata:
        pass

    attn_backend_mod.AttentionMetadata = AttentionMetadata

    utils_mod = types.ModuleType('vllm.v1.attention.backends.utils')
    utils_mod.split_decodes_and_prefills = MagicMock(return_value=(1, 0, 1, 0))
    utils_mod.PAD_SLOT_ID = -1

    monkeypatch.setattr("vllm.v1.attention.backend", attn_backend_mod)
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends.utils",
                        utils_mod)

    vllm_distributed_mod = create_mock_modules("vllm.distributed")
    monkeypatch.setitem(sys.modules, "vllm.distributed", vllm_distributed_mod)
    monkeypatch.setitem(sys.modules, "vllm.distributed.eplb", MagicMock())
    monkeypatch.setitem(sys.modules, "vllm.distributed.eplb.eplb_state",
                        MagicMock())
    monkeypatch.setitem(sys.modules,
                        "vllm.distributed.get_tensor_model_parallel_rank",
                        MagicMock())
    monkeypatch.setitem(
        sys.modules, "vllm.distributed.get_tensor_model_parallel_world_size",
        MagicMock())

    vllm_device_comm_mod = create_mock_modules(
        "vllm.distributed.device_communicators")
    monkeypatch.setitem(sys.modules, "vllm.distributed.device_communicators",
                        vllm_device_comm_mod)

    vllm_device_comm_shm_mod = create_mock_modules(
        "vllm.distributed.device_communicators.shm_object_storage", True)
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.device_communicators.shm_object_storage",
        vllm_device_comm_shm_mod)

    vllm_parallel_state_mod = create_mock_modules(
        "vllm.distributed.parallel_state", True)
    fake_dcp = MagicMock()
    fake_dcp.world_size = 1
    fake_dcp.rank_in_group = 0
    fake_pcp = MagicMock()
    fake_pcp.world_size = 1
    fake_pcp.rank_in_group = 0
    vllm_parallel_state_mod.get_dcp_group = lambda: fake_dcp
    vllm_parallel_state_mod.get_pcp_group = lambda: fake_pcp
    monkeypatch.setitem(sys.modules,
                        "vllm.distributed.parallel_state.get_dcp_group",
                        lambda: fake_dcp)
    monkeypatch.setitem(sys.modules,
                        "vllm.distributed.parallel_state.get_pcp_group",
                        lambda: fake_pcp)

    mock_forward_ctx = MagicMock()
    mock_forward_ctx.capturing = False
    mock_forward_ctx.batch_descriptor = None

    forward_ctx_mod = types.ModuleType('vllm.forward_context')
    forward_ctx_mod.get_forward_context = MagicMock(return_value=mock_forward_ctx)
    forward_ctx_mod.BatchDescriptor = MagicMock()
    forward_ctx_mod.capturing = False
    monkeypatch.setitem(sys.modules, "vllm.forward_context", forward_ctx_mod)

    try:
        import omni_npu.attention.backends.attention as attn_mod
        import omni_npu.attention.backends as backends_mod
        importlib.reload(attn_mod)
        importlib.reload(backends_mod)

        # Now it's safe to import omni_npu — its backend will inherit from REAL base classes
        from omni_npu.attention.backends import (
            NPUAttentionBackendImpl as _impl,
            NPUMetadata as _meta,
            NPUAttentionBackend as _backend,
            NPUAttentionMetadataBuilder as _builder,
        )

        _builder._init_reorder_batch_threshold = lambda *args, **kwargs: None

        # Yield a dictionary with all the classes
        yield {
            'NPUAttentionBackendImpl': _impl,
            'NPUMetadata': _meta,
            'NPUAttentionBackend': _backend,
            'NPUAttentionMetadataBuilder': _builder,
            'AttentionType': AttentionType,
        }
    except Exception as e:
        print(f"❌ FAILED to import omni_npu classes: {e}")
        import traceback
        traceback.print_exc()
        raise


@pytest.mark.unit
class TestNPUAttentionBackendDefault(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def setup_classes(self, npu_attention_classes):
        """Inject the attention classes from the module-scoped fixture."""
        self.impl_interface_cls = npu_attention_classes['NPUAttentionBackend']
        self.npu_attention_classes = npu_attention_classes

    def test_backend_properties(self):
        backend = self.impl_interface_cls()
        self.assertIn(torch.float16, backend.get_supported_dtypes())
        self.assertEqual(backend.get_name(), "VLLM_NPU_ATTN")
        self.assertIs(backend.get_impl_cls(), self.npu_attention_classes['NPUAttentionBackendImpl'])
        self.assertIs(backend.get_metadata_cls(), self.npu_attention_classes['NPUMetadata'])
        self.assertIs(backend.get_builder_cls(), self.npu_attention_classes['NPUAttentionMetadataBuilder'])

    def test_kv_cache_shape_and_reshape(self):
        shape = self.impl_interface_cls.get_kv_cache_shape(
            num_blocks=10,
            block_size=16,
            num_kv_heads=4,
            head_size=128
        )
        self.assertEqual(shape, (10, 16, 512))  # 4 * 128 = 512

        raw = torch.randn(2 * 10 * 16 * 512, dtype=torch.bfloat16)
        kv_cache_spec = AttentionSpec(
            block_size=16,
            num_kv_heads=4,
            head_size=128,
            dtype=torch.bfloat16,
        )
        k_cache, v_cache = self.impl_interface_cls.reshape_kv_cache(
            raw, num_blocks=10, kv_cache_spec=kv_cache_spec,
        )
        self.assertEqual(k_cache.shape, (10, 16, 512))
        self.assertEqual(v_cache.shape, (10, 16, 512))
        self.assertTrue(torch.equal(raw[:10*16*512].view(10,16,512), k_cache))


@pytest.mark.unit
class TestNPUAttentionBackendDefaultMetadataBuilder(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def setup_classes(self, npu_attention_classes):
        """Inject the attention classes from the module-scoped fixture."""
        self.metadata_builder_cls = npu_attention_classes['NPUAttentionMetadataBuilder']
        self.npu_attention_classes = npu_attention_classes

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
        vllm_config.compilation_config = None
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

        with patch(
                'vllm.v1.attention.backends.utils.split_decodes_and_prefills',
                return_value=(0, 2, 0, 20)
        ), patch(
                'omni_npu.attention.backends.attention.split_decodes_and_prefills',
                return_value=(0, 2, 0, 20)):
            meta = builder.build(common_prefix_len=0,
                                 common_attn_metadata=common_meta)

        self.assertIsInstance(meta, self.npu_attention_classes['NPUMetadata'])
        self.assertEqual(meta.num_actual_tokens, 20)
        self.assertEqual(meta.num_prefills, 2)
        self.assertEqual(meta.query_cumlens, [10, 20])
        self.assertEqual(meta.seq_lens, [10, 10])
        self.assertEqual(meta.max_query_len, 10)


@pytest.mark.unit
class TestNPUAttentionBackendDefaultImpl(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def setup_classes(self, npu_attention_classes):
        """Inject the attention classes from the module-scoped fixture."""
        self.impl_cls = npu_attention_classes['NPUAttentionBackendImpl']
        self.metadata_cls = npu_attention_classes['NPUMetadata']
        self.AttentionType = npu_attention_classes['AttentionType']

    def test_init_success(self):
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=1.0,
            num_kv_heads=4,
            attn_type=self.AttentionType.DECODER,
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
                attn_type=self.AttentionType.DECODER,
            )

    def test_forward_calls_npu_fused_infer_attention_score_v2(self):
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=0.125,
            num_kv_heads=4,
            attn_type=self.AttentionType.DECODER,
        )

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        batch_size = 10
        query = torch.randn(batch_size, 8 , 128).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device)
        key = torch.randn(batch_size, 4 , 128).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device)
        value = torch.randn(batch_size, 4 , 128).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device)
        kv_cache = (torch.zeros(batch_size ** 2, 16, 4 * 128).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device), torch.zeros(100, 16, 4 * 128).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device))

        metadata = self.metadata_cls(
            num_actual_tokens=10,
            block_tables=torch.randint(0, 100, (2, 10)).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device),
            query_start_loc=[0, 10],
            seq_lens=[10],
            max_query_len=1,
            slot_mapping=torch.arange(10).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device),
            num_prefills=0,
            num_decode_tokens=8,
            num_decodes=2,
        )

        attn_output = torch.randn(batch_size, 8, 128).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device)
        output = torch.empty_like(attn_output).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device)
        prefill_output=output.clone()

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

            # self.assertEqual(mock_scatter.call_count, 2)
            mock_decode.assert_called_once()
            args, kwargs = mock_decode.call_args
            self.assertEqual(kwargs['num_query_heads'], 8)
            self.assertEqual(kwargs['num_key_value_heads'], 4)
            self.assertEqual(kwargs['input_layout'], "TND")
            self.assertAlmostEqual(kwargs['softmax_scale'], 0.125)
            self.assertIs(result, output)

    def test_forward_calls_npu_fused_infer_attention_sink(self):
        sink = torch.randn(8)
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=0.125,
            num_kv_heads=4,
            sliding_window=256,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name="mock_layer",
            sinks=sink,
            head_size_v=128,
            sink_len=128,
        )

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        batch_size = 10
        device = self.impl_cls.SHARE_MASK_TRIL_SPARSE.device
        query = torch.randn(batch_size, 8, 128, device=device)
        key = torch.randn(batch_size, 4, 128, device=device)
        value = torch.randn(batch_size, 4, 128, device=device)
        kv_cache = (
            torch.zeros(batch_size ** 2, 16, 4 * 128, device=device),
            torch.zeros(batch_size ** 2, 16, 4 * 128, device=device),
        )
        output = torch.empty_like(query)
        prefill_output = output.clone()
        metadata = self.metadata_cls(
            num_actual_tokens=10,
            block_tables=torch.randint(0, 100, (2, 10), device=device),
            query_start_loc=[0, 10],
            seq_lens=[10 + 128],
            max_query_len=1,
            slot_mapping=torch.arange(10, device=device),
            num_prefills=0,
            num_decode_tokens=8,
            num_decodes=2,
        )

        with patch('torch.ops.custom.npu_fused_infer_attention_sink', return_value=(prefill_output,)) as mock_decode:
            impl.forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )
            kwargs = mock_decode.call_args.kwargs
            self.assertEqual(kwargs["sparse_mode"], 4)
            self.assertEqual(kwargs["pre_tokens"], 256)
            self.assertEqual(kwargs["next_tokens"], 0)
            self.assertEqual(kwargs["sink_number"], 128)
            self.assertEqual(kwargs["actual_seq_qlen"], [10])

    def test_forward_passes_sink_and_sliding_kwargs(self):
        sink = torch.randn(8)
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=0.125,
            num_kv_heads=4,
            sliding_window=256,
            attn_type=self.AttentionType.DECODER,
            sinks=sink,
        )

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        batch_size = 10
        device = self.impl_cls.SHARE_MASK_TRIL_SPARSE.device
        query = torch.randn(batch_size, 8, 128, device=device)
        key = torch.randn(batch_size, 4, 128, device=device)
        value = torch.randn(batch_size, 4, 128, device=device)
        kv_cache = (
            torch.zeros(batch_size ** 2, 16, 4 * 128, device=device),
            torch.zeros(batch_size ** 2, 16, 4 * 128, device=device),
        )
        output = torch.empty_like(query)
        prefill_output = output.clone()
        metadata = self.metadata_cls(
            num_actual_tokens=10,
            block_tables=torch.randint(0, 100, (2, 10), device=device),
            query_start_loc=[0, 10],
            seq_lens=[10],
            max_query_len=1,
            slot_mapping=torch.arange(10, device=device),
            num_prefills=0,
            num_decode_tokens=8,
            num_decodes=2,
        )

        def fake_scatter_nd_update_(tensor, indices, updates):
            if indices.ndim == 2 and indices.shape[1] == 1:
                indices = indices.squeeze(1)
            tensor[indices] = updates[:indices.shape[0]]
            return tensor

        with patch('torch_npu.npu_scatter_nd_update_', side_effect=fake_scatter_nd_update_), \
         patch('torch_npu.npu_fused_infer_attention_score_v2', return_value=(prefill_output,)) as mock_decode:
            impl.forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )
            kwargs = mock_decode.call_args.kwargs
            self.assertEqual(kwargs["sparse_mode"], 4)
            self.assertEqual(kwargs["pre_tokens"], 256)
            self.assertEqual(kwargs["next_tokens"], 0)
            self.assertTrue(torch.equal(kwargs["learnable_sink"], sink.view(8)))

    def test_forward_default_sink_none_and_non_sliding_kwargs(self):
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=0.125,
            num_kv_heads=4,
            attn_type=self.AttentionType.DECODER,
        )

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        batch_size = 10
        device = self.impl_cls.SHARE_MASK_TRIL_SPARSE.device
        query = torch.randn(batch_size, 8, 128, device=device)
        key = torch.randn(batch_size, 4, 128, device=device)
        value = torch.randn(batch_size, 4, 128, device=device)
        kv_cache = (
            torch.zeros(batch_size ** 2, 16, 4 * 128, device=device),
            torch.zeros(batch_size ** 2, 16, 4 * 128, device=device),
        )
        output = torch.empty_like(query)
        prefill_output = output.clone()
        metadata = self.metadata_cls(
            num_actual_tokens=10,
            block_tables=torch.randint(0, 100, (2, 10), device=device),
            query_start_loc=[0, 10],
            seq_lens=[10],
            max_query_len=1,
            slot_mapping=torch.arange(10, device=device),
            num_prefills=0,
            num_decode_tokens=8,
            num_decodes=2,
        )

        def fake_scatter_nd_update_(tensor, indices, updates):
            if indices.ndim == 2 and indices.shape[1] == 1:
                indices = indices.squeeze(1)
            tensor[indices] = updates[:indices.shape[0]]
            return tensor

        with patch('torch_npu.npu_scatter_nd_update_', side_effect=fake_scatter_nd_update_), \
         patch('torch_npu.npu_fused_infer_attention_score_v2', return_value=(prefill_output,)) as mock_decode:
            impl.forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )
            kwargs = mock_decode.call_args.kwargs
            self.assertEqual(kwargs["sparse_mode"], 3)
            self.assertNotIn("pre_tokens", kwargs)
            self.assertNotIn("next_tokens", kwargs)
            self.assertNotIn("learnable_sink", kwargs)

    def test_forward_calls_npu_fused_infer_attention_score_bsnd(self):
        head_size = 256
        impl = self.impl_cls(
            num_heads=8,
            head_size=head_size,
            scale=0.125,
            num_kv_heads=4,
            attn_type=self.AttentionType.DECODER,
        )

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        query = torch.randn(20, 8, head_size)
        key = torch.randn(20, 4, head_size)
        value = torch.randn(20, 4, head_size)
        kv_cache = (torch.zeros(100, 16, 4 * head_size), torch.zeros(100, 16, 4 * head_size))
        output = torch.empty_like(query)

        metadata = self.metadata_cls(
            num_actual_tokens=20,
            block_tables=torch.randint(0, 100, (2, 10)),
            query_start_loc=[0, 10, 20],
            seq_lens=[10, 10],
            max_query_len=10,
            slot_mapping=torch.arange(20),
            num_prefills=2,
        )

        prefill_output = output.clone()
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
         patch('torch_npu.npu_fused_infer_attention_score', return_value=(prefill_output,)) as mock_decode:
            result = impl.forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )
            mock_decode.assert_called_once()
            self.assertIs(result, output)

    def test_forward_requires_output_tensor(self):
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=1.0,
            attn_type=self.AttentionType.DECODER,
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
            query_start_loc=[0, 1],
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
            attn_type=self.AttentionType.DECODER,
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
            query_start_loc=[0, 1],
            seq_lens=[1],
            slot_mapping=torch.tensor([0], dtype=torch.int64),
            num_prefills=0,
        )

        with self.assertRaises(RuntimeError):
            impl.forward(layer, query, key, value, kv_cache, metadata, output=output)
