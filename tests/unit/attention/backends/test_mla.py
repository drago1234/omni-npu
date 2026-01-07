# SPDX-License-Identifier: Apache-2.0
import sys
import types
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import ClassVar, Optional, Any, Type, Generic, TypeVar, List
import pytest
import torch

# ==============================
# 1. Mock basic vLLM modules
# ==============================
VLLM_MODULES_TO_MOCK = {
    'vllm.config',
    'vllm.logger',
    'vllm.platforms',
    'vllm.model_executor',
    'vllm.model_executor.models',
    'vllm.model_executor.models.interfaces',
    'vllm.distributed',
    'vllm.distributed.eplb',
    'vllm.distributed.eplb.eplb_state',
    'vllm.distributed.device_communicators',
    'vllm.distributed.device_communicators.cuda_communicator',
    'vllm.distributed.device_communicators.base_device_communicator',
}

if 'vllm' not in sys.modules:
    sys.modules['vllm'] = MagicMock()

for mod in VLLM_MODULES_TO_MOCK:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# ==============================
# 2. Define minimal stubs for vllm.v1 interfaces
# ==============================

# --- vllm.v1.kv_cache_interface ---
kv_iface = types.ModuleType('vllm.v1.kv_cache_interface')
@dataclass
class AttentionSpec:
    pass
kv_iface.AttentionSpec = AttentionSpec
sys.modules['vllm.v1.kv_cache_interface'] = kv_iface

# --- vllm.v1.attention.backends.utils ---
utils_mod = types.ModuleType('vllm.v1.attention.backends.utils')
def split_decodes_and_prefills(seq_lens, query_lens=None):
    if query_lens is None:
        prefill = [i for i, s in enumerate(seq_lens) if s > 1]
        decode = [i for i, s in enumerate(seq_lens) if s == 1]
    else:
        prefill = [i for i, ql in enumerate(query_lens) if ql > 1]
        decode = [i for i, ql in enumerate(query_lens) if ql == 1]
    return prefill, decode

utils_mod.split_decodes_and_prefills = split_decodes_and_prefills
class AttentionCGSupport:
    UNIFORM_BATCH = "uniform_batch"
utils_mod.AttentionCGSupport = AttentionCGSupport

@dataclass
class CommonAttentionMetadata:
    pass
utils_mod.CommonAttentionMetadata = CommonAttentionMetadata
sys.modules['vllm.v1.attention.backends.utils'] = utils_mod

# --- vllm.attention.backends.abstract (fallback) ---
class AttentionType:
    DECODER = "decoder"
    ENCODER = "encoder"
    ENCODER_ONLY = "encoder_only"
    ENCODER_DECODER = "encoder_decoder"

class AttentionMetadata:
    pass

class AttentionLayer:
    def __init__(self):
        self._q_scale = None
        self._k_scale = None
        self._v_scale = None
        self._q_scale_float = 0.0
        self._k_scale_float = 0.0
        self._v_scale_float = 0.0
        self._prob_scale = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

abstract_mod = types.ModuleType('vllm.attention.backends.abstract')
abstract_mod.AttentionLayer = AttentionLayer
abstract_mod.AttentionType = AttentionType
sys.modules['vllm.attention.backends.abstract'] = abstract_mod

# ==============================
# 3. DEFINE STUB BASE CLASSES for vllm.v1.attention.backends.mla.common
# ==============================

TDecodeMeta = TypeVar('TDecodeMeta')
TMetadata = TypeVar('TMetadata')

@dataclass
class MLACommonDecodeMetadata:
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    dcp_tot_seq_lens: torch.Tensor | None

@dataclass
class MLACommonMetadata(Generic[TDecodeMeta]):
    prefill: Any = None
    decode: TDecodeMeta = None
    num_actual_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 4
    num_decode_tokens: int = 8
    slot_mapping: Optional[torch.Tensor] = None

class MLACommonMetadataBuilder(Generic[TMetadata]):
    _cudagraph_support: ClassVar[Any] = None
    supports_uniform_spec_as_decode: ClassVar[bool] = False

    def __init__(
        self,
        kv_cache_spec: Any,
        layer_names: List[str],
        vllm_config: Any,
        device: torch.device,
        metadata_cls: Type[TMetadata],
    ):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device
        self.metadata_cls = metadata_cls

        compilation_config = getattr(vllm_config, 'compilation_config', None)
        parallel_config = getattr(vllm_config, 'parallel_config', None)

        self._use_fi_prefill = bool(
            getattr(compilation_config, 'enable_flashinfer', False)
        )
        self._use_cudnn_prefill = bool(
            getattr(compilation_config, 'enable_cudnn', False)
        )
        self.dcp_world_size = getattr(parallel_config, 'dcp_world_size', 1) if parallel_config else 1
        self.aot_schedule = bool(
            getattr(compilation_config, 'aot_compile', False)
        )

    def _build_decode(self, *args, **kwargs):
        raise NotImplementedError("_build_decode must be implemented by subclass")

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: Any,
        fast_build: bool = False,
    ) -> TMetadata:
        from vllm.v1.attention.backends.utils import split_decodes_and_prefills

        seq_lens = getattr(common_attn_metadata, 'seq_lens', [])
        query_lens = getattr(common_attn_metadata, 'query_lens', [s for s in seq_lens])

        prefill_indices, decode_indices = split_decodes_and_prefills(seq_lens, query_lens)

        num_prefills = len(prefill_indices)
        num_decodes = len(decode_indices)
        num_decode_tokens = num_decodes
        num_actual_tokens = sum(seq_lens)

        # Prefill meta
        prefill_meta = None
        if num_prefills > 0:
            cumsum = [0]
            for ql in query_lens:
                cumsum.append(cumsum[-1] + ql)
            query_start_loc = torch.tensor(cumsum, dtype=torch.int32, device=self.device)
            prefill_meta = type('PrefillMeta', (), {'query_start_loc': query_start_loc})()

        # Decode meta
        decode_meta = None
        if num_decodes > 0:
            block_table = getattr(common_attn_metadata, 'block_tables', None)
            if block_table is None:
                max_blocks = 10
                block_table = torch.arange(
                    len(seq_lens) * max_blocks, dtype=torch.int32, device=self.device
                ).view(len(seq_lens), max_blocks)

            seq_lens_tensor = torch.tensor(seq_lens, device=self.device)
            query_start_loc_cpu = torch.tensor([0] + [sum(query_lens[:i+1]) for i in range(len(query_lens))], dtype=torch.int32)
            query_start_loc_device = query_start_loc_cpu.to(self.device)

            decode_seq_lens = seq_lens_tensor[decode_indices]
            decode_block_table = block_table[decode_indices]

            decode_meta = self._build_decode(
                block_table_tensor=decode_block_table,
                seq_lens_cpu=torch.tensor(seq_lens)[decode_indices],
                seq_lens_device=decode_seq_lens,
                query_start_loc_cpu=query_start_loc_cpu,
                query_start_loc_device=query_start_loc_device,
                num_decode_tokens=num_decode_tokens,
                dcp_tot_seq_lens_device=None,
            )

        return self.metadata_cls(
            prefill=prefill_meta,
            decode=decode_meta,
            num_actual_tokens=num_actual_tokens,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            slot_mapping=getattr(common_attn_metadata, 'slot_mapping', None),
        )

    @staticmethod
    def determine_chunked_prefill_workspace_size(vllm_config):
        return 1024

class MLACommonBaseImpl(Generic[TMetadata]):
    def __init__(self, *args, **mla_args):
        for k, v in mla_args.items():
            setattr(self, k, v)
        self.num_heads = args[0] if len(args) > 0 else 32
        self.head_size = args[1] if len(args) > 1 else 128
        self.scale = args[2] if len(args) > 2 else 1.0
        self.num_kv_heads = args[3] if len(args) > 3 else 8
        self.attn_type = args[8] if len(args) > 8 else "decoder"

class MLACommonBackend:
    @staticmethod
    def get_name() -> str:
        raise NotImplementedError

    @staticmethod
    def get_metadata_cls() -> Type:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> Type:
        raise NotImplementedError

    @staticmethod
    def get_impl_cls() -> Type:
        raise NotImplementedError

    @staticmethod
    def reshape_kv_cache(*args, **kwargs):
        raise NotImplementedError
class QueryLenSupport:
    SUPPORTED = "supported"
    VARLEN = "varlen"  
    NOT_SUPPORTED = "not_supported"
    
common_mod = types.ModuleType('vllm.v1.attention.backends.mla.common')
common_mod.MLACommonBackend = MLACommonBackend
common_mod.MLACommonDecodeMetadata = MLACommonDecodeMetadata
common_mod.MLACommonMetadata = MLACommonMetadata
common_mod.MLACommonMetadataBuilder = MLACommonMetadataBuilder
common_mod.MLACommonBaseImpl = MLACommonBaseImpl
common_mod.QueryLenSupport = QueryLenSupport
sys.modules['vllm.v1.attention.backends.mla.common'] = common_mod

# ==============================
# 4. DEFINE YOUR ACTUAL IMPLEMENTATION (or mock if not available)
# ==============================
from omni_npu.attention.backends.mla import (
    NPUMLAImpl,
    NPUMLAMetadata,
    NPUMLADecodeMetadata,
    NPUMLAMetadataBuilder,
    NPUMLABackend,
)

# ==============================
# 5. Helper & Test
# ==============================
def make_fake_metadata(
    num_prefills: int,
    num_decode_tokens: int,
    seq_lens: list[int],
    block_size: int = 128,
    device: torch.device = torch.device("cpu"),
):
    total_tokens = sum(seq_lens)
    query_start_loc = [0]
    cumsum = 0
    for i, slen in enumerate(seq_lens):
        if i < len(seq_lens) - num_decode_tokens:
            cumsum += slen
        else:
            cumsum += 1
        query_start_loc.append(cumsum)

    query_start_loc = torch.tensor(query_start_loc, dtype=torch.int32, device=device)
    max_blocks_per_seq = max((s + block_size - 1) // block_size for s in seq_lens)
    block_table = torch.arange(
        len(seq_lens) * max_blocks_per_seq, dtype=torch.int32, device=device
    ).view(len(seq_lens), max_blocks_per_seq)
    slot_mapping = torch.arange(total_tokens, dtype=torch.int64, device=device)

    class FakePrefillMeta:
        def __init__(self):
            self.query_start_loc = query_start_loc

    prefill_meta = FakePrefillMeta() if num_prefills > 0 else None
    decode_meta = None
    if num_decode_tokens > 0:
        decode_meta = NPUMLADecodeMetadata(
            block_table=block_table[-num_decode_tokens:],
            seq_lens=seq_lens[-num_decode_tokens:],
            query_cumlens=query_start_loc[1:].tolist(),
            dcp_tot_seq_lens=None,
        )

    metadata = NPUMLAMetadata(
        prefill=prefill_meta,
        decode=decode_meta,
        num_actual_tokens=total_tokens,
        num_prefills=num_prefills,
        num_decodes=len(seq_lens) - (num_prefills > 0),
        num_decode_tokens=num_decode_tokens,
        slot_mapping=slot_mapping,
    )
    return metadata

# Mock torch_npu
torch_npu_mock = MagicMock()
sys.modules['torch_npu'] = torch_npu_mock


@pytest.mark.unit
class TestNPUAttentionBackendMLAUtilsFunc(unittest.TestCase):

    def test_npu_mla_reshape_kv_cache(self):
        backend = NPUMLABackend

        self.assertEqual(backend.get_name(), "NPUMLA")
        self.assertEqual(backend.get_metadata_cls(), NPUMLAMetadata)
        self.assertEqual(backend.get_builder_cls(), NPUMLAMetadataBuilder)
        self.assertEqual(backend.get_impl_cls(), NPUMLAImpl)

        num_blocks = 10
        block_size = 128
        kv_lora_rank = 512
        qk_rope_head_dim = 64
        dtype = torch.bfloat16

        total_bf16_elements = num_blocks * block_size * (kv_lora_rank + qk_rope_head_dim)
        total_bytes = total_bf16_elements * 2
        raw_tensor = torch.empty(total_bytes, dtype=torch.uint8)

        result = backend.reshape_kv_cache(
            raw_tensor=raw_tensor,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=8,
            head_size=128,
            dtype=dtype,
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        nope_cache, rope_cache = result
        self.assertEqual(nope_cache.shape, (num_blocks, block_size, kv_lora_rank))
        self.assertEqual(rope_cache.shape, (num_blocks, block_size, qk_rope_head_dim))
        self.assertEqual(nope_cache.dtype, dtype)
        self.assertEqual(rope_cache.dtype, dtype)

        print(" Backend contract test passed!")
    
    def test_v_up_proj(self):
        device = torch.device("npu:0")
        dtype = torch.bfloat16

        num_heads = 32
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        hidden_size = num_heads * v_head_dim
        kv_lora_rank = 512

        impl = NPUMLAImpl(
            num_heads=num_heads,
            head_size=128,
            scale=1.0 / (128 ** 0.5),
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            logits_soft_cap=None,
            kv_sharing_target_layer_name=None,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            hidden_size=hidden_size,
            kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER,
        )

        B = 2
        x = torch.randn(num_heads * B, kv_lora_rank, dtype=dtype, device=device)
        out = torch.empty(B, hidden_size, dtype=dtype, device=device)

        impl.W_UV = torch.zeros(num_heads, kv_lora_rank, v_head_dim, dtype=dtype, device=device)
        for h in range(num_heads):
            impl.W_UV[h, :v_head_dim, :] = torch.eye(v_head_dim, dtype=dtype, device=device)

        impl._v_up_proj(x, out)

        expected = torch.zeros(B, hidden_size, dtype=dtype, device=device)
        for b in range(B):
            for h in range(num_heads):
                token_idx = h * B + b
                expected[b, h * v_head_dim:(h + 1) * v_head_dim] = x[token_idx, :v_head_dim]

        self.assertTrue(torch.allclose(out, expected, atol=1e-3))
        print("_v_up_proj test passed!")

@pytest.mark.unit
class TestNPUAttentionBackendMLANpuMlaImpl(unittest.TestCase):
    def test_npu_mla_impl(self):
        device = torch.device("cpu")
        dtype = torch.bfloat16

        num_heads = 32
        head_size = 128
        num_kv_heads = 8
        scale = 1.0 / (head_size ** 0.5)
        kv_cache_dtype = "auto"
        attn_type = AttentionType.DECODER

        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        kv_lora_rank = 512
        hidden_size = num_heads * head_size

        impl = NPUMLAImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            logits_soft_cap=None,
            kv_sharing_target_layer_name=None,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            hidden_size=hidden_size,
            kv_cache_dtype=kv_cache_dtype,
            attn_type=attn_type,
        )

        impl.W_UK_T = torch.randn(num_heads, qk_nope_head_dim, kv_lora_rank, device=device, dtype=dtype)
        impl.W_UV = torch.randn(num_heads, kv_lora_rank, v_head_dim, device=device, dtype=dtype)
        impl.kv_b_proj = lambda x: (
            torch.empty(x.shape[0], num_heads * (qk_nope_head_dim + v_head_dim), dtype=x.dtype, device=x.device),
        )

        num_prefills = 1
        num_decode_tokens = 2
        seq_lens = [8, 1, 1]

        metadata = make_fake_metadata(
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            device=device
        )

        total_tokens = sum(seq_lens)
        output = torch.empty(total_tokens, hidden_size, dtype=dtype, device=device)

        q = torch.randn(total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim, dtype=dtype, device=device)
        k_c_normed = torch.randn(total_tokens, kv_lora_rank, dtype=dtype, device=device)
        k_pe = torch.randn(total_tokens, qk_rope_head_dim, dtype=dtype, device=device)

        num_blocks = 10
        block_size = 128
        nope_cache = torch.empty(num_blocks * block_size * 512, dtype=torch.uint8, device=device)
        rope_cache = torch.empty(num_blocks * block_size * 64, dtype=torch.uint8, device=device)
        kv_cache = (nope_cache, rope_cache)

        layer = AttentionLayer()

        def mock_forward_prefill(*args, **kwargs):
            q_tensor = args[1]
            return torch.zeros(q_tensor.shape[0], hidden_size, dtype=dtype, device=device)

        def mock_forward_decode(*args, **kwargs):
            q_tensor = args[1]
            return torch.zeros(q_tensor.shape[0], hidden_size, dtype=dtype, device=device)

        def mock_v_up_proj(x, **kwargs):
            return torch.zeros(x.shape[0], num_heads, v_head_dim, dtype=x.dtype, device=x.device)

        with patch.object(NPUMLAImpl, '_forward_prefill', side_effect=mock_forward_prefill), \
             patch.object(NPUMLAImpl, '_forward_decode', side_effect=mock_forward_decode), \
             patch.object(NPUMLAImpl, '_v_up_proj', side_effect=mock_v_up_proj):

            out = impl.forward(
                layer=layer,
                q=q,
                k_c_normed=k_c_normed,
                k_pe=k_pe,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )

        self.assertEqual(out.shape, (total_tokens, hidden_size))
        print(" Impl test passed!")
    
    def test_forward_prefill(self):
        device = torch.device("cpu")
        dtype = torch.bfloat16

        impl = NPUMLAImpl(
            num_heads=32,
            head_size=128,
            scale=1.0 / (128 ** 0.5),
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            logits_soft_cap=None,
            kv_sharing_target_layer_name=None,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            kv_lora_rank=512,
            hidden_size=4096,
            kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER,
        )

        impl.kv_b_proj = lambda x: (torch.randn(x.shape[0], 32 * (128 + 128), dtype=dtype, device=device),)

        num_prefill_tokens = 8
        q = torch.randn(num_prefill_tokens, 32, 128 + 64, dtype=dtype, device=device)
        kv_c_normed = torch.randn(num_prefill_tokens, 512, dtype=dtype, device=device)
        k_pe = torch.randn(num_prefill_tokens, 64, dtype=dtype, device=device)
        k_scale = torch.tensor(1.0, dtype=dtype, device=device)

        metadata = MagicMock(
            prefill=type('PrefillMeta', (), {'query_start_loc': [0, 8]})(),
            decode=None,
            num_actual_tokens=8,
            num_prefills=1,
            num_decodes=0,
            num_decode_tokens=0,
            slot_mapping=None,
        )
        metadata.prefill.chunked_context = None
        kv_cache = (torch.empty(0), torch.empty(0))

        mock_output = torch.randn(num_prefill_tokens, 32, 128, dtype=dtype, device=device)
        mock_lse = torch.randn(8, 32).npu()
        with patch("torch.ops.npu.npu_fused_infer_attention_score", return_value=(mock_output, mock_lse)) as mock_op:
            result = impl._forward_prefill(q, kv_c_normed, k_pe, kv_cache, metadata, k_scale)

        mock_op.assert_called_once()
        args, kwargs = mock_op.call_args
        self.assertEqual(args[0].shape, (8, 32, 128))
        self.assertEqual(args[1].shape, (8, 32, 128))
        self.assertEqual(args[2].shape, (8, 32, 128))
        self.assertEqual(kwargs["actual_seq_lengths"], [8])
        self.assertEqual(kwargs["scale"], impl.scale)

        self.assertEqual(result.shape, (8, 4096))
        print(" _forward_prefill test passed!")
    
    def test_forward_decode(self):
        mock_context = MagicMock()
        mock_context.batch_descriptor = MagicMock()
        with patch('omni_npu.attention.backends.mla.get_forward_context', return_value=mock_context):
            device = torch.device("npu:0")
            dtype = torch.bfloat16

            num_heads = 32
            qk_nope_head_dim = 128
            qk_rope_head_dim = 64
            v_head_dim = 128
            hidden_size = num_heads * v_head_dim  # 4096

            impl = NPUMLAImpl(
                num_heads=num_heads,
                head_size=128,
                scale=1.0 / (128 ** 0.5),
                num_kv_heads=8,
                alibi_slopes=None,
                sliding_window=None,
                logits_soft_cap=None,
                kv_sharing_target_layer_name=None,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                kv_lora_rank=512,
                hidden_size=hidden_size,
                kv_cache_dtype="auto",
                attn_type=AttentionType.DECODER,
            )

            T = 2  # number of decode tokens (batch size for decode)

            #  FIX: decode_ql_nope is (T, num_heads, qk_nope_head_dim)
            decode_ql_nope = torch.randn(T, num_heads, qk_nope_head_dim, dtype=dtype, device=device)

            #  FIX: decode_q_pe MUST have enough elements to be reshaped to (T, 1, num_heads, qk_rope_head_dim)
            # So its shape should be (T, num_heads, qk_rope_head_dim) — i.e., per-head RoPE
            decode_q_pe = torch.randn(T, num_heads, qk_rope_head_dim, dtype=dtype, device=device)

            num_blocks, block_size = 10, 128
            nope_cache = torch.randn(num_blocks, block_size, 512, dtype=dtype, device=device)
            rope_cache = torch.randn(num_blocks, block_size, qk_rope_head_dim, dtype=dtype, device=device)
            kv_cache = (nope_cache, rope_cache)

            decode_meta = NPUMLADecodeMetadata(
                block_table=torch.randint(0, num_blocks, (T, 10), dtype=torch.int32, device=device),
                seq_lens=[5, 3],
                query_cumlens=[5, 8],
                dcp_tot_seq_lens=None,
            )
            metadata = NPUMLAMetadata(
                prefill=None,
                decode=decode_meta,
                num_actual_tokens=8,
                num_prefills=0,
                num_decodes=T,
                num_decode_tokens=T,
                slot_mapping=None,
            )

            layer = AttentionLayer()

            # Mock the NPU op
            mock_attn_out = torch.randn(T, 1, num_heads, v_head_dim, dtype=dtype, device=device)
            with patch("torch.ops.npu.npu_fused_infer_attention_score", return_value=(mock_attn_out,)) as mock_op:
                o = impl._forward_decode(decode_ql_nope, decode_q_pe, kv_cache, metadata, layer)

            mock_op.assert_called_once()
            kwargs = mock_op.call_args.kwargs
            self.assertEqual(kwargs["block_table"].shape, (2, 10))
            self.assertEqual(kwargs["actual_seq_lengths_kv"], [5, 3])
            self.assertEqual(kwargs["input_layout"], "BSND")
            self.assertEqual(kwargs["num_key_value_heads"], 1)

            # Output from _forward_decode is (N, T, D) after transpose
            self.assertEqual(o.shape, (num_heads, T, v_head_dim))  # e.g., (32, 2, 128)
            # self.assertIsNone(extra)
            print(" _forward_decode test passed!")
    
@pytest.mark.unit
class TestNPUAttentionBackendMLAMetadataBuilder(unittest.TestCase):
    def test_npu_mla_metadata_builder(self):
        device = torch.device("cpu")

        vllm_config = MagicMock()
        vllm_config.compilation_config = MagicMock(
            enable_flashinfer=False,
            enable_cudnn=False,
            aot_compile=False,
        )
        vllm_config.parallel_config = MagicMock(
            dcp_world_size=1,
        )

        kv_cache_spec = AttentionSpec()
        layer_names = ["mla_attn"]

        builder = NPUMLAMetadataBuilder(
            kv_cache_spec=kv_cache_spec,
            layer_names=layer_names,
            vllm_config=vllm_config,
            device=device,
        )

        original_build = MLACommonMetadataBuilder.build

        def patched_build(self_obj, common_prefix_len, common_attn_metadata, fast_build=False):
            metadata = original_build(self_obj, common_prefix_len, common_attn_metadata, fast_build)
            if metadata.prefill is not None and not hasattr(metadata.prefill, 'chunked_context'):
                metadata.prefill.chunked_context = None
            return metadata

        with patch.object(MLACommonMetadataBuilder, 'build', patched_build):
            class MockCommonAttnMeta:
                def __init__(self):
                    self.seq_lens = [8, 5, 3]
                    self.query_lens = [8, 1, 1]
                    self.slot_mapping = torch.arange(16, dtype=torch.int64, device=device)
                    self.block_tables = torch.arange(3 * 10, dtype=torch.int32, device=device).view(3, 10)
                    self.max_query_len = 8
                    self.num_prefills = 1
                    self.num_decode_tokens = 2
                    self.context_lens = None
                    self.dcp_tot_seq_lens = None

            common_meta = MockCommonAttnMeta()

            metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_meta,
                fast_build=False,
            )

        self.assertIsInstance(metadata, NPUMLAMetadata)
        self.assertEqual(metadata.num_prefills, 1)
        self.assertEqual(metadata.num_decodes, 2)
        self.assertEqual(metadata.num_decode_tokens, 2)
        self.assertEqual(metadata.num_actual_tokens, 16)

        if metadata.decode is not None:
            self.assertEqual(metadata.decode.seq_lens, [5, 3])

        print(" MetadataBuilder test passed!")

if __name__ == "__main__":
    unittest.main()