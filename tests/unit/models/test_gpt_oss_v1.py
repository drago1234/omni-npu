# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import importlib.machinery
import sys
from pathlib import Path
from types import SimpleNamespace
import types

import pytest
import torch


@pytest.mark.unit
def test_gpt_oss_model_load_weights_remap_and_permute(monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(repo_root / "src"))
    if not hasattr(torch, "npu"):
        monkeypatch.setattr(
            torch, "npu", SimpleNamespace(config=SimpleNamespace()), raising=False
        )
    if not hasattr(torch.npu, "config"):
        torch.npu.config = SimpleNamespace()
    if not hasattr(torch.npu, "is_available"):
        torch.npu.is_available = lambda: False
    torch_npu_mod = types.ModuleType("torch_npu")
    torch_npu_mod.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu_mod)
    omni_pkg = types.ModuleType("omni_npu")
    omni_pkg.__path__ = [str(repo_root / "src" / "omni_npu")]
    monkeypatch.setitem(sys.modules, "omni_npu", omni_pkg)
    omni_v1_pkg = types.ModuleType("omni_npu.v1")
    omni_v1_pkg.__path__ = [str(repo_root / "src" / "omni_npu" / "v1")]
    monkeypatch.setitem(sys.modules, "omni_npu.v1", omni_v1_pkg)
    omni_models_pkg = types.ModuleType("omni_npu.v1.models")
    omni_models_pkg.__path__ = [str(repo_root / "src" / "omni_npu" / "v1" / "models")]
    monkeypatch.setitem(sys.modules, "omni_npu.v1.models", omni_models_pkg)
    gpt_oss_mod = importlib.import_module("omni_npu.v1.models.gpt_oss.gpt_oss")
    GptOssModel = gpt_oss_mod.GptOssModel

    model = GptOssModel.__new__(GptOssModel)
    model.config = SimpleNamespace(
        num_attention_heads=4,
        intermediate_size=3,
        num_local_experts=4,
    )
    model.parallel_config = SimpleNamespace(enable_expert_parallel=True)

    qkv_w = torch.nn.Parameter(torch.zeros(8, 2, dtype=torch.bfloat16), requires_grad=False)
    qkv_b = torch.nn.Parameter(torch.zeros(8, dtype=torch.bfloat16), requires_grad=False)
    sinks = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16), requires_grad=False)
    w13_weight = torch.nn.Parameter(torch.zeros(2, 6, 2, dtype=torch.int8), requires_grad=False)
    w2_weight = torch.nn.Parameter(torch.zeros(2, 2, 3, dtype=torch.int8), requires_grad=False)
    w13_scale = torch.nn.Parameter(torch.zeros(2, 6, 1, dtype=torch.float32), requires_grad=False)
    w2_scale = torch.nn.Parameter(torch.zeros(2, 2, 1, dtype=torch.float32), requires_grad=False)
    w13_bias = torch.nn.Parameter(torch.zeros(2, 6, dtype=torch.bfloat16), requires_grad=False)
    w2_bias = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16), requires_grad=False)

    def qkv_loader(param, loaded_weight, shard_id):
        if shard_id == "q":
            start, size = 0, 4
        elif shard_id == "k":
            start, size = 4, 2
        else:
            start, size = 6, 2
        param.data[start : start + size].copy_(loaded_weight)

    qkv_w.weight_loader = qkv_loader
    qkv_b.weight_loader = qkv_loader

    params = {
        "layers.0.self_attn.qkv_proj.weight": qkv_w,
        "layers.0.self_attn.qkv_proj.bias": qkv_b,
        "layers.0.self_attn.sinks": sinks,
        "layers.0.mlp.experts.w13_weight": w13_weight,
        "layers.0.mlp.experts.w2_weight": w2_weight,
        "layers.0.mlp.experts.w13_weight_scale": w13_scale,
        "layers.0.mlp.experts.w2_weight_scale": w2_scale,
        "layers.0.mlp.experts.w13_bias": w13_bias,
        "layers.0.mlp.experts.w2_bias": w2_bias,
    }
    model.named_parameters = lambda remove_duplicate=False: params.items()

    ep_size = 2
    ep_rank = 1
    monkeypatch.setattr(gpt_oss_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(gpt_oss_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        gpt_oss_mod,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=ep_size, rank_in_group=ep_rank),
    )
    monkeypatch.setattr(gpt_oss_mod, "is_pp_missing_parameter", lambda name, _: False)

    q_weight = torch.arange(8, dtype=torch.bfloat16).view(4, 2)
    k_weight = torch.arange(100, 104, dtype=torch.bfloat16).view(2, 2)
    v_weight = torch.arange(200, 204, dtype=torch.bfloat16).view(2, 2)
    q_bias = torch.arange(4, dtype=torch.bfloat16)
    k_bias = torch.arange(10, 12, dtype=torch.bfloat16)
    v_bias = torch.arange(20, 22, dtype=torch.bfloat16)
    sinks_weight = torch.arange(30, 34, dtype=torch.bfloat16)
    w13_ckpt = torch.arange(4 * 2 * 6, dtype=torch.int32).to(torch.int8).view(4, 2, 6)
    w2_ckpt = torch.arange(4 * 3 * 2, dtype=torch.int32).to(torch.int8).view(4, 3, 2)
    w13_scale_ckpt = torch.arange(24, dtype=torch.float32).view(4, 1, 6)
    w2_scale_ckpt = torch.arange(8, dtype=torch.float32).view(4, 1, 2)
    w13_bias_ckpt = torch.arange(24, dtype=torch.bfloat16).view(4, 6)
    w2_bias_ckpt = torch.arange(8, dtype=torch.bfloat16).view(4, 2)

    loaded = model.load_weights(
        [
            ("model.layers.0.self_attn.q_proj.weight", q_weight),
            ("model.layers.0.self_attn.k_proj.weight", k_weight),
            ("model.layers.0.self_attn.v_proj.weight", v_weight),
            ("model.layers.0.self_attn.q_proj.bias", q_bias),
            ("model.layers.0.self_attn.k_proj.bias", k_bias),
            ("model.layers.0.self_attn.v_proj.bias", v_bias),
            ("model.layers.0.self_attn.sinks", sinks_weight),
            ("model.layers.0.mlp.experts.gate_up_proj.weight", w13_ckpt),
            ("model.layers.0.mlp.experts.down_proj.weight", w2_ckpt),
            ("model.layers.0.mlp.experts.gate_up_proj.scale", w13_scale_ckpt),
            ("model.layers.0.mlp.experts.down_proj.scale", w2_scale_ckpt),
            ("model.layers.0.mlp.experts.gate_up_proj.bias", w13_bias_ckpt),
            ("model.layers.0.mlp.experts.down_proj.bias", w2_bias_ckpt),
        ]
    )

    assert "layers.0.self_attn.qkv_proj.weight" in loaded
    assert "layers.0.self_attn.qkv_proj.bias" in loaded
    assert "layers.0.self_attn.sinks" in loaded
    assert "layers.0.mlp.experts.w13_weight" in loaded
    assert "layers.0.mlp.experts.w2_weight" in loaded
    assert "layers.0.mlp.experts.w13_weight_scale" in loaded
    assert "layers.0.mlp.experts.w2_weight_scale" in loaded
    assert "layers.0.mlp.experts.w13_bias" in loaded
    assert "layers.0.mlp.experts.w2_bias" in loaded

    assert torch.equal(qkv_w, torch.cat([q_weight, k_weight, v_weight], dim=0))
    assert torch.equal(qkv_b, torch.cat([q_bias, k_bias, v_bias], dim=0))
    assert torch.equal(sinks, sinks_weight)

    experts_per_rank = model.config.num_local_experts // ep_size
    ep_rank_start = ep_rank * experts_per_rank
    ep_rank_end = (ep_rank + 1) * experts_per_rank
    assert ep_rank_end <= w13_ckpt.shape[0]

    expected_w13 = w13_ckpt[ep_rank_start:ep_rank_end].permute(0, 2, 1).contiguous()
    expected_w2 = w2_ckpt[ep_rank_start:ep_rank_end].permute(0, 2, 1).contiguous()
    expected_w13_scale = w13_scale_ckpt[ep_rank_start:ep_rank_end].permute(0, 2, 1).contiguous()
    expected_w2_scale = w2_scale_ckpt[ep_rank_start:ep_rank_end].permute(0, 2, 1).contiguous()
    expected_w13_bias = w13_bias_ckpt[ep_rank_start:ep_rank_end]
    expected_w2_bias = w2_bias_ckpt[ep_rank_start:ep_rank_end]

    assert torch.equal(w13_weight, expected_w13)
    assert torch.equal(w2_weight, expected_w2)
    assert torch.equal(w13_scale, expected_w13_scale)
    assert torch.equal(w2_scale, expected_w2_scale)
    assert torch.equal(w13_bias, expected_w13_bias)
    assert torch.equal(w2_bias, expected_w2_bias)


@pytest.mark.unit
def test_gpt_oss_attention_flashcomm_transform_overrides(monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(repo_root / "src"))
    if not hasattr(torch, "npu"):
        monkeypatch.setattr(
            torch, "npu", SimpleNamespace(config=SimpleNamespace()), raising=False
        )
    if not hasattr(torch.npu, "config"):
        torch.npu.config = SimpleNamespace()
    if not hasattr(torch.npu, "is_available"):
        torch.npu.is_available = lambda: False

    torch_npu_mod = types.ModuleType("torch_npu")
    torch_npu_mod.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu_mod)

    omni_pkg = types.ModuleType("omni_npu")
    omni_pkg.__path__ = [str(repo_root / "src" / "omni_npu")]
    monkeypatch.setitem(sys.modules, "omni_npu", omni_pkg)
    omni_v1_pkg = types.ModuleType("omni_npu.v1")
    omni_v1_pkg.__path__ = [str(repo_root / "src" / "omni_npu" / "v1")]
    monkeypatch.setitem(sys.modules, "omni_npu.v1", omni_v1_pkg)
    omni_models_pkg = types.ModuleType("omni_npu.v1.models")
    omni_models_pkg.__path__ = [str(repo_root / "src" / "omni_npu" / "v1" / "models")]
    monkeypatch.setitem(sys.modules, "omni_npu.v1.models", omni_models_pkg)

    gpt_oss_mod = importlib.import_module("omni_npu.v1.models.gpt_oss.gpt_oss")

    class DummyFlashLinear:
        def __init__(self, *args, **kwargs):
            self.x_transform = "INIT_X"
            self.y_transform = "INIT_Y"

    class DummyRope:
        def __call__(self, positions, q, k):
            return q, k

    class DummyAttention:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, q, k, v):
            return q

    rope_call_kwargs = {}

    def fake_get_rope(*args, **kwargs):
        rope_call_kwargs.update(kwargs)
        return DummyRope()

    monkeypatch.setattr(gpt_oss_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(gpt_oss_mod, "QKVParallelFlashCommLinear", DummyFlashLinear)
    monkeypatch.setattr(gpt_oss_mod, "RowParallelFlashCommLinear", DummyFlashLinear)
    monkeypatch.setattr(gpt_oss_mod, "get_rope", fake_get_rope)
    monkeypatch.setattr(gpt_oss_mod, "Attention", DummyAttention)

    cfg = SimpleNamespace(
        head_dim=64,
        rope_parameters={
            "rope_theta": 150000.0,
            "factor": 8.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "attn_factor": 1.1,
            "extrapolation_factor": 1.0,
            "apply_yarn_scaling": True,
            "truncate": True,
        },
        max_position_embeddings=131072,
        sliding_window=128,
        layer_types=None,
    )

    attn = gpt_oss_mod.GptOssAttention(
        config=cfg,
        hidden_size=2880,
        num_heads=64,
        num_kv_heads=8,
        model_config=None,
        cache_config=None,
        quant_config=None,
        prefix="layers.0.self_attn",
    )

    assert attn.qkv_proj.x_transform == "NoOp"
    assert attn.qkv_proj.y_transform == "NoOp"
    assert attn.o_proj.x_transform == "NoOp"
    assert attn.o_proj.y_transform == "AllReduce"
    assert rope_call_kwargs["rope_parameters"]["rope_type"] == "yarn"
    assert rope_call_kwargs["rope_parameters"]["attn_factor"] == 1.1
    assert rope_call_kwargs["rope_parameters"]["apply_yarn_scaling"] is True
