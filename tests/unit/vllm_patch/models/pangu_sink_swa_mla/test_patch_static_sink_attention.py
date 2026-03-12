import types
import importlib
import sys

import pytest
import torch
from unittest.mock import MagicMock

# Import the patch module containing reinit_block_table_with_sink
sink_patch_mod = importlib.import_module(
    "omni_npu.vllm_patches.patches.models.pangu_sink_swa_mla.patch_static_sink_attention"
)
create_static_sink_attention_backendPatch = sink_patch_mod.create_static_sink_attention_backendPatch


def test_reinit_block_table_with_sink_basic_functionality(monkeypatch):
    """Test reinit_block_table_with_sink basic functionality.

    Verifies that the method correctly resets the block_table_with_sink to zeros
    and then fills the sink block positions with values from 1 to num_sink_blocks
    """
    create_static_sink_attention_backendPatch.create_static_sink_attention_backend.cache_clear()

    class _FakeUnderlyingBuilder:
        def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
            self.vllm_config = vllm_config
            self.device = device

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            return common_attn_metadata
    
    class _FakeUnderlyingBackend:
        @staticmethod
        def get_builder_cls():
            return _FakeUnderlyingBuilder
        
    monkeypatch.setattr(
        sink_patch_mod,
        "subclass_attention_backend",
        lambda name_prefix, attention_backend_cls, builder_cls: types.SimpleNamespace(
            builder_cls=builder_cls
        ),
    )

    # Test with sink_len=8 and block_size=2, so num_sink_blocks=4
    backend = create_static_sink_attention_backendPatch.create_static_sink_attention_backend(
        _FakeUnderlyingBackend,
        sink_len=8,
    )

    # Create builder with necessary configuration
    vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(max_model_len=1024),
        scheduler_config=types.SimpleNamespace(max_num_seqs=8),
        cache_config=types.SimpleNamespace(block_size=2),
    )

    builder = backend.builder_cls(
        kv_cache_spec=None,
        layer_names=["test_layer"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )

    # Verify initial setup
    assert builder.num_sink_blocks == 4
    assert builder.max_num_blocks == 512  # ceil(1024 / 2)
    # (max_num_seqs, max_num_blocks + num_sink_blocks)
    assert builder.block_table_with_sink.shape == (8, 516) 

    # Modify the block_table_with_sink to simulate non-zero state
    builder.block_table_with_sink[:, :4] = 555

    # Call reinit_block_table_with_sink
    builder.reinit_block_table_with_sink()

    # Verigy sink blocks are filled with values from 1 to num_sink_blocks
    expected_sink_values = torch.tensor([[1, 2, 3, 4] for _ in range(8)], dtype=torch.int32, device=torch.device("cpu"))
    assert torch.equal(builder.block_table_with_sink[:, :4], expected_sink_values)

    