# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for NPUAttentionBackendImpl that require NPU hardware.
"""

import unittest
import torch
import pytest
from omni_npu.attention.backends.attention import (
    NPUAttentionBackendImpl,
    NPUMetadata,
)
from vllm.attention.backends.abstract import AttentionType

# Skip all tests if NPU is not available
from integration.utils.common_utils import (
    skipif_no_npu,
    NPU_AVAILABLE,
)


@pytest.mark.integration
@skipif_no_npu
class TestNPUAttentionBackendDefaultImplIntegration(unittest.TestCase):

    def setUp(self):
        if not NPU_AVAILABLE:
            self.skipTest("NPU not available")
        self.impl_cls = NPUAttentionBackendImpl
        self.metadata_cls = NPUMetadata
        self.device = "npu"

    def test_forward_decode_path_calls_npu_fused_infer_attention_score(self):
        # Initialize the attention implementation
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=0.125,
            num_kv_heads=4,
            attn_type=AttentionType.DECODER,
        )

        # Mock the attention layer with required attributes
        layer = unittest.mock.MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        # Define tensor dimensions
        num_tokens = 10
        hidden_size = 8 * 128      # = 1024 (num_heads * head_size)
        kv_hidden_size = 4 * 128   # = 512  (num_kv_heads * head_size)
        block_size = 16
        block_id = 0               # Use block 0 for simplicity

        # Create input tensors on NPU
        query = torch.randn(num_tokens, hidden_size, device=self.device, dtype=torch.float16)
        key = torch.randn(num_tokens, kv_hidden_size, device=self.device, dtype=torch.float16)
        value = torch.randn(num_tokens, kv_hidden_size, device=self.device, dtype=torch.float16)

        # Allocate KV cache: shape [num_blocks, block_size, kv_hidden_size]
        num_blocks = 100
        k_cache = torch.zeros(num_blocks, block_size, kv_hidden_size, device=self.device, dtype=torch.float16)
        v_cache = torch.zeros(num_blocks, block_size, kv_hidden_size, device=self.device, dtype=torch.float16)
        kv_cache = (k_cache, v_cache)

        # Compute max number of blocks needed per sequence (ceil(seq_len / block_size))
        max_blocks_per_seq = (num_tokens + block_size - 1) // block_size  # = 1
        # Block table: [num_seqs=1, max_blocks_per_seq=1], filled with block_id=0
        block_tables = torch.full(
            (1, max_blocks_per_seq),
            block_id,
            dtype=torch.int32,
            device=self.device
        )

        # Output buffer: must match expected output shape [num_tokens, num_heads, head_size]
        output = torch.empty(num_tokens, 8, 128, device=self.device, dtype=torch.float16)

        # Construct metadata for decode-only path (num_prefills=0)
        metadata = self.metadata_cls(
            num_actual_tokens=num_tokens,
            block_tables=block_tables,
            query_cumlens=[0, num_tokens],     # Cumulative lengths: [0, 10]
            seq_lens=[num_tokens],             # Each sequence length = 10
            max_query_len=1,                   # Decode: one token per sequence
            slot_mapping=torch.arange(num_tokens, device=self.device),  # Physical slot indices [0..9]
            num_prefills=0,
        )

        # Run forward pass
        result = impl.forward(
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=metadata,
            output=output,
        )

        # Verify output is written in-place
        self.assertIs(result, output)

        # Synchronize to catch any asynchronous NPU kernel errors
        torch.npu.synchronize()

        # Optional: sanity check for invalid values
        self.assertFalse(torch.isnan(result).any(), "Output contains NaN")
        self.assertFalse(torch.isinf(result).any(), "Output contains Inf")

if __name__ == '__main__':
    unittest.main()