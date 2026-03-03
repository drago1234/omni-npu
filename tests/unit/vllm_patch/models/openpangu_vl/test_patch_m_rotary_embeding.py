import numpy as np
import pytest

from omni_npu.vllm_patches.patches.models.openpangu_vl import patch_m_rotary_embedding as rope_patch_mod
from omni_npu.vllm_patches.patches.models.openpangu_vl.patch_m_rotary_embedding import (
    rotary_embeddingPatch,
)


def test_get_np_position_slice_cache_grow_and_dtype_isolation(monkeypatch):
    monkeypatch.setattr(rotary_embeddingPatch, "_position_cache", {})

    int64 = np.dtype(np.int64)
    first = rotary_embeddingPatch._get_np_position_slice(0, 4, int64)
    cache64 = rotary_embeddingPatch._position_cache[int64]
    assert np.array_equal(first, np.array([0, 1, 2, 3], dtype=np.int64))
    assert np.shares_memory(first, cache64)

    _ = rotary_embeddingPatch._get_np_position_slice(0, 9, int64)
    grown_cache64 = rotary_embeddingPatch._position_cache[int64]
    assert grown_cache64.shape[0] >= 9
    assert grown_cache64.shape[0] >= cache64.shape[0]

    int32 = np.dtype(np.int32)
    second = rotary_embeddingPatch._get_np_position_slice(2, 5, int32)
    cache32 = rotary_embeddingPatch._position_cache[int32]
    assert np.array_equal(second, np.array([2, 3, 4], dtype=np.int32))
    assert cache32.dtype == np.int32
    assert cache32 is not grown_cache64


def test_get_next_input_positions_tensor_handles_zero_single_and_multi_tokens(monkeypatch):
    monkeypatch.setattr(rotary_embeddingPatch, "_position_cache", {})
    out = np.full((3, 8), -1, dtype=np.int64)

    rotary_embeddingPatch.get_next_input_positions_tensor(
        out=out,
        out_offset=1,
        mrope_position_delta=3,
        context_len=5,
        num_new_tokens=0,
    )
    assert np.all(out == -1)

    rotary_embeddingPatch.get_next_input_positions_tensor(
        out=out,
        out_offset=2,
        mrope_position_delta=3,
        context_len=5,
        num_new_tokens=1,
    )
    assert np.array_equal(out[:, 2], np.array([8, 8, 8], dtype=np.int64))

    rotary_embeddingPatch.get_next_input_positions_tensor(
        out=out,
        out_offset=4,
        mrope_position_delta=3,
        context_len=5,
        num_new_tokens=3,
    )
    expected = np.array([8, 9, 10], dtype=np.int64)
    assert np.array_equal(out[:, 4:7], np.tile(expected, (3, 1)))


def test_get_next_input_positions_tensor_single_token_skips_slice_helper(monkeypatch):
    out = np.zeros((2, 4), dtype=np.int64)

    def _should_not_call(*args, **kwargs):
        raise AssertionError("_get_np_position_slice should not be used for single token")

    monkeypatch.setattr(
        rotary_embeddingPatch,
        "_get_np_position_slice",
        classmethod(_should_not_call),
    )

    rotary_embeddingPatch.get_next_input_positions_tensor(
        out=out,
        out_offset=1,
        mrope_position_delta=4,
        context_len=2,
        num_new_tokens=1,
    )
    assert np.array_equal(out[:, 1], np.array([6, 6], dtype=np.int64))


def test_get_rope_wrapper_fallback_sets_default_rope_parameters(monkeypatch):
    captured = {}
    sentinel = object()

    def _fake_orig_get_rope(
        head_size,
        rotary_dim,
        max_position,
        is_neox_style,
        rope_parameters,
        dtype,
        partial_rotary_factor,
        dual_chunk_attention_config,
    ):
        captured["args"] = (
            head_size,
            rotary_dim,
            max_position,
            is_neox_style,
            rope_parameters,
            dtype,
            partial_rotary_factor,
            dual_chunk_attention_config,
        )
        return sentinel

    monkeypatch.setattr(rope_patch_mod, "_orig_get_rope", _fake_orig_get_rope)

    got = rotary_embeddingPatch.get_rope_wrapper(
        head_size=64,
        rotary_dim=32,
        max_position=1024,
        base=12345.0,
        is_neox_style=True,
        rope_scaling=None,
        dtype=None,
        partial_rotary_factor=0.5,
        dual_chunk_attention_config={"chunk": 16},
    )
    assert got is sentinel
    rope_parameters = captured["args"][4]
    assert rope_parameters["rope_theta"] == 12345.0
    assert rope_parameters["rope_type"] == "default"


def test_get_rope_wrapper_mrope_branch_uses_cache_and_pp_group(monkeypatch):
    calls = []

    class _FakePPGroup:
        world_size = 2

    class _FakeMRotaryEmbeddingInterleaved:
        def __init__(
            self,
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            dtype,
            mrope_section,
            mrope_interleaved,
            rotary_mode,
            num_hidden_layers_cache,
        ):
            calls.append(
                {
                    "head_size": head_size,
                    "rotary_dim": rotary_dim,
                    "max_position": max_position,
                    "base": base,
                    "is_neox_style": is_neox_style,
                    "dtype": dtype,
                    "mrope_section": mrope_section,
                    "mrope_interleaved": mrope_interleaved,
                    "rotary_mode": rotary_mode,
                    "num_hidden_layers_cache": num_hidden_layers_cache,
                }
            )

    monkeypatch.setattr(rope_patch_mod, "_ROPE_DICT", {})
    monkeypatch.setattr(rope_patch_mod, "get_pp_group", lambda: _FakePPGroup())
    monkeypatch.setattr(
        rope_patch_mod.rotary_embedding,
        "MRotaryEmbeddingInterleaved",
        _FakeMRotaryEmbeddingInterleaved,
    )

    rope_scaling = {
        "mrope_interleaved": True,
        "mrope_section": [4, 4],
        "rotary_mode": "half",
    }
    first = rotary_embeddingPatch.get_rope_wrapper(
        head_size=64,
        rotary_dim=16,
        max_position=128,
        base=10000.0,
        is_neox_style=True,
        rope_scaling=rope_scaling,
        num_hidden_layers_cache=8,
    )
    second = rotary_embeddingPatch.get_rope_wrapper(
        head_size=64,
        rotary_dim=16,
        max_position=128,
        base=10000.0,
        is_neox_style=True,
        rope_scaling=rope_scaling,
        num_hidden_layers_cache=8,
    )

    assert first is second
    assert len(calls) == 1
    assert calls[0]["num_hidden_layers_cache"] == 1
    assert calls[0]["mrope_section"] == [4, 4]


@pytest.mark.parametrize(
    ("a", "b", "c", "force_last"),
    [
        (2, 1, 1, False),
        (2, 1, 1, True),
        (3, 2, 0, False),
    ],
)
def test_get_mrope_interleaved_id_list_keeps_counts(a, b, c, force_last):
    seq = rotary_embeddingPatch.MRotaryEmbeddingInterleaved.get_mrope_interleaved_id_list(
        a, b, c, force_last=force_last
    )
    expected_len = a + b + c
    assert len(seq) == expected_len
    assert seq.count(0) == a
    assert seq.count(1) == b
    assert seq.count(2) == c
    if force_last:
        assert seq[-1] == 0
