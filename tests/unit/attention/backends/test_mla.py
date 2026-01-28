# SPDX-License-Identifier: Apache-2.0
import sys
import types
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import ClassVar, Optional, Any, Type, Generic, TypeVar, List
import pytest
import torch

from vllm.attention.backends.abstract import (
    AttentionLayer,
    AttentionType,
    AttentionBackend,
    AttentionImpl
) 

from vllm.v1.kv_cache_interface import AttentionSpec

utils_mod_patcher = None
NPUAttentionBackendImpl = None
NPUAttentionMetadata = None
NPUMLABackend = None
NPUMLAMetadataBuilder = None
NPUMLADecodeMetadata= None
NPUMLAImpl = None
make_fake_metadata = None

def setup_module():
    global NPUMLAImpl, NPUMLAMetadata, NPUMLABackend, NPUMLAMetadataBuilder, NPUMLADecodeMetadata, make_fake_metadata, utils_mod_patcher
    try:
        # Define a TypeVar for the metadata type
        MetadataT = TypeVar('MetadataT')

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
            
            def _init_reorder_batch_threshold(self, *args, **kwargs):
                self.reorder_batch_threshold = 0

        utils_mod = types.ModuleType('vllm.v1.attention.backends.utils')
        utils_mod.split_decodes_and_prefills = MagicMock(return_value=(1, 0, 1, 0))
        utils_mod.AttentionMetadataBuilder = AttentionMetadataBuilder
        utils_mod.CommonAttentionMetadata = MagicMock()
        utils_mod.AttentionCGSupport = MagicMock()
        utils_mod.dcp_local_seq_lens = MagicMock()
        utils_mod.get_dcp_local_seq_lens = MagicMock()
        utils_mod.get_per_layer_parameters= MagicMock()
        utils_mod.infer_global_hyperparameters = MagicMock()
        utils_mod_patcher = patch.dict('sys.modules', {
                'vllm.v1.attention.backends.utils': utils_mod
        })
        utils_mod_patcher.start()

        from omni_npu.attention.backends.mla import (
            NPUMLAImpl as __impl,
            NPUMLAMetadata,
            NPUMLADecodeMetadata,
            NPUMLAMetadataBuilder,
            NPUMLABackend,
        )
        
        global NPUMLAImpl
        NPUMLAImpl = __impl
        global make_fake_metadata
        def make_fake_metadata(
            num_prefills: int,
            num_decode_tokens: int,
            seq_lens: list[int],
            num_reqs: int,
            max_query_len: int,
            max_seq_len: int,
            query_start_loc: torch.Tensor,
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
            batch_size = 1
            prompt_len = 1
            total_prefill_tokens = batch_size * prompt_len
            query_start_loc = torch.tensor([0, total_prefill_tokens], dtype=torch.int32, device=device)
            
            metadata = NPUMLAMetadata(
                prefill=prefill_meta,
                decode=decode_meta,
                num_actual_tokens=total_tokens,
                num_prefills=num_prefills,
                num_decodes=len(seq_lens) - (num_prefills > 0),
                num_decode_tokens=num_decode_tokens,
                slot_mapping=slot_mapping,
                num_reqs=batch_size,
                max_query_len=prompt_len,
                max_seq_len=prompt_len,
                query_start_loc=query_start_loc,
            )
            return metadata    
    except Exception as e:
        print(f"❌ FAILED to import omni_npu classes: {e}")
        import traceback
        traceback.print_exc()
        raise

def teardown_module():
    if utils_mod_patcher is not None:
        utils_mod_patcher.stop()


@pytest.mark.unit
class TestNPUAttentionBackendMLAUtilsFunc(unittest.TestCase):

    def test_npu_mla_reshape_kv_cache(self):
        global NPUMLABackend, NPUMLAMetadata, NPUMLAMetadataBuilder, NPUMLAImpl
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
        num_kv_heads = 8
        q_lora_rank = 256
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        kv_b_proj = torch.nn.Linear(
            kv_lora_rank,
            num_kv_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        ).to(device).to(torch.bfloat16)
        
        batch_size = 1
        prompt_len = 1
        mock_ctx = MagicMock()
        mock_ctx.batch_descriptor = MagicMock(
            num_reqs=batch_size,
            max_q_len=prompt_len,
            max_seq_len=prompt_len,
            uniform=True,
        )

        with patch('vllm.v1.attention.backends.mla.common.MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size', return_value=64), \
            patch('omni_npu.attention.backends.mla.get_current_vllm_config', return_value=None), \
            patch('omni_npu.attention.backends.mla.get_forward_context', return_value=mock_ctx):
            global NPUMLAImpl
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
                q_lora_rank=q_lora_rank,
                qk_head_dim=qk_head_dim,
                kv_b_proj=kv_b_proj,
                kv_cache_dtype="auto",
                attn_type=AttentionType.DECODER,
            )

            B = 2
            x = torch.randn(num_heads * B, kv_lora_rank, dtype=dtype, device=device)

            impl.W_UV = torch.zeros(num_heads, kv_lora_rank, v_head_dim, dtype=dtype, device=device)
            for h in range(num_heads):
                impl.W_UV[h, :v_head_dim, :] = torch.eye(v_head_dim, dtype=dtype, device=device)

            out = impl._v_up_proj(x)

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
        q_lora_rank = 256
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        kv_b_proj = torch.nn.Linear(
            kv_lora_rank,
            num_kv_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        ).to(device).to(torch.bfloat16)

        batch_size = 1
        prompt_len = 1
        mock_ctx = MagicMock()
        mock_ctx.batch_descriptor = MagicMock(
            num_reqs=batch_size,
            max_q_len=prompt_len,
            max_seq_len=prompt_len,
            uniform=True,
        )
        with patch('vllm.v1.attention.backends.mla.common.MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size', return_value=64), \
            patch('omni_npu.attention.backends.mla.get_current_vllm_config', return_value=None), \
            patch('omni_npu.attention.backends.mla.get_forward_context', return_value=mock_ctx):
            global NPUMLAImpl
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
                q_lora_rank=q_lora_rank,
                qk_head_dim=qk_head_dim,
                kv_b_proj=kv_b_proj,
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

            batch_size = 1
            prompt_len = 1
            total_prefill_tokens = batch_size * prompt_len
            query_start_loc = torch.tensor([0, total_prefill_tokens], dtype=torch.int32, device=device)
            
            metadata = make_fake_metadata(
                num_prefills=num_prefills,
                num_decode_tokens=num_decode_tokens,
                seq_lens=seq_lens,
                device=device,
                num_reqs=batch_size,
                max_query_len=prompt_len,
                max_seq_len=prompt_len,
                query_start_loc=query_start_loc,
            )

            total_tokens = sum(seq_lens)
            output = torch.empty(total_tokens, hidden_size, dtype=dtype, device=device)

            q = torch.randn(total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim, dtype=dtype, device=device)
            k_c_normed = torch.randn(total_tokens, kv_lora_rank, dtype=dtype, device=device)
            k_pe = torch.randn(total_tokens, qk_rope_head_dim, dtype=dtype, device=device)

            num_blocks = 10
            block_size = 128
            nope_cache = torch.empty(num_blocks, block_size, 512, dtype=torch.uint8, device=device)
            rope_cache = torch.empty(num_blocks, block_size, 64, dtype=torch.uint8, device=device)
            kv_cache = (nope_cache, rope_cache)

            layer = MagicMock()

            def mock_forward_prefill(*args, **kwargs):
                q_tensor = args[1]
                return torch.zeros(q_tensor.shape[0], hidden_size, dtype=dtype, device=device)

            def mock_forward_decode(*args, **kwargs):
                q_tensor = args[1]
                return torch.zeros(q_tensor.shape[0], hidden_size, dtype=dtype, device=device)

            def mock_v_up_proj(x, **kwargs):
                return torch.zeros(x.shape[0], num_heads * v_head_dim, dtype=x.dtype, device=x.device)

            def fake_scatter_nd_update_(tensor, indices, updates):
                idx = indices.squeeze(-1)  # [N]
                max_idx = idx.max().item()
                current_size = tensor.size(0)

                if max_idx >= current_size:
                    new_size = max_idx + 1
                    new_tensor = torch.zeros(
                        (new_size,) + tensor.shape[1:],
                        dtype=tensor.dtype,
                        device=tensor.device
                    )
                    new_tensor[:current_size] = tensor
                    tensor.resize_(new_tensor.shape)
                    tensor.copy_(new_tensor)

                update_dim = updates.shape[-1]
                tensor[idx, :update_dim] = updates.to(tensor.dtype)
                return tensor

            with patch('torch_npu.npu_scatter_nd_update_', side_effect=fake_scatter_nd_update_), \
                patch.object(NPUMLAImpl, '_forward_prefill', side_effect=mock_forward_prefill), \
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
        num_kv_heads = 8
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        q_lora_rank = 256
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        kv_lora_rank = 512
        kv_b_proj = torch.nn.Linear(
            kv_lora_rank,
            num_kv_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        ).to(device).to(torch.bfloat16)

        batch_size = 1
        prompt_len = 1
        mock_ctx = MagicMock()
        mock_ctx.batch_descriptor = MagicMock(
            num_reqs=batch_size,
            max_q_len=prompt_len,
            max_seq_len=prompt_len,
            uniform=True,
        )
        with patch('vllm.v1.attention.backends.mla.common.MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size', return_value=64), \
            patch('omni_npu.attention.backends.mla.get_current_vllm_config', return_value=None), \
            patch('omni_npu.attention.backends.mla.get_forward_context', return_value=mock_ctx):
            global NPUMLAImpl
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
                q_lora_rank=q_lora_rank,
                qk_head_dim=qk_head_dim,
                kv_b_proj=kv_b_proj,
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
            q_lora_rank = 256
            qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
            kv_lora_rank = 128
            num_kv_heads = 8
            kv_b_proj = torch.nn.Linear(
                kv_lora_rank,
                num_kv_heads * (qk_nope_head_dim + v_head_dim),
                bias=False,
            ).to(device).to(torch.bfloat16)

            batch_size = 1
            prompt_len = 1
            mock_ctx = MagicMock()
            mock_ctx.batch_descriptor = MagicMock(
                num_reqs=batch_size,
                max_q_len=prompt_len,
                max_seq_len=prompt_len,
                uniform=True,
            )
            with patch('vllm.v1.attention.backends.mla.common.MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size', return_value=64), \
                patch('omni_npu.attention.backends.mla.get_current_vllm_config', return_value=None), \
                patch('omni_npu.attention.backends.mla.get_forward_context', return_value=mock_ctx):
                global NPUMLAImpl
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
                    q_lora_rank=q_lora_rank,
                    qk_head_dim=qk_head_dim,
                    kv_b_proj=kv_b_proj,
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
                batch_size = 1
                prompt_len = 1
                total_prefill_tokens = batch_size * prompt_len
                query_start_loc = torch.tensor([0, total_prefill_tokens], dtype=torch.int32, device=device)
                
                metadata = NPUMLAMetadata(
                    prefill=None,
                    decode=decode_meta,
                    num_actual_tokens=8,
                    num_prefills=0,
                    num_decodes=T,
                    num_decode_tokens=T,
                    slot_mapping=None,
                    num_reqs=batch_size,
                    max_query_len=prompt_len,
                    max_seq_len=prompt_len,
                    query_start_loc=query_start_loc,
                )

                layer = MagicMock()

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