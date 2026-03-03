# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import importlib.machinery
import sys
from pathlib import Path
from types import SimpleNamespace
import types
from unittest.mock import MagicMock

import pytest
import torch


def _import_moe_module(monkeypatch):
    repo_root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(repo_root / "src"))
    if not hasattr(torch, "npu"):
        monkeypatch.setattr(
            torch, "npu", SimpleNamespace(config=SimpleNamespace()), raising=False
        )
    if not hasattr(torch.npu, "config"):
        torch.npu.config = SimpleNamespace()
    torch_npu_mod = types.ModuleType("torch_npu")
    torch_npu_mod.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu_mod)
    omni_pkg = types.ModuleType("omni_npu")
    omni_pkg.__path__ = [str(repo_root / "src" / "omni_npu")]
    monkeypatch.setitem(sys.modules, "omni_npu", omni_pkg)
    omni_v1_pkg = types.ModuleType("omni_npu.v1")
    omni_v1_pkg.__path__ = [str(repo_root / "src" / "omni_npu" / "v1")]
    monkeypatch.setitem(sys.modules, "omni_npu.v1", omni_v1_pkg)
    omni_layers_pkg = types.ModuleType("omni_npu.v1.layers")
    omni_layers_pkg.__path__ = [str(repo_root / "src" / "omni_npu" / "v1" / "layers")]
    monkeypatch.setitem(sys.modules, "omni_npu.v1.layers", omni_layers_pkg)
    return importlib.import_module(
        "omni_npu.v1.layers.quantization.compressed_tensors.npu_compressed_tensors_moe"
    )


def _build_prepare_permute_result():
    return SimpleNamespace(
        hidden_states_sorted_by_experts=torch.ones(3, 2, dtype=torch.int8),
        expert_tokens=torch.tensor([1, 2], dtype=torch.int64),
        avg_tokens_per_expert=[1, 2],
        dynamic_scale=torch.ones(3, dtype=torch.float32),
    )


def _build_layer(activation: str, with_w2_bias: bool = True):
    layer = SimpleNamespace(
        w13_weight=torch.ones(2, 2, 6, dtype=torch.int8),
        w2_weight=torch.ones(2, 4, 3, dtype=torch.int8),
        w13_weight_scale=torch.ones(2, 6, dtype=torch.float32),
        w2_weight_scale=torch.ones(2, 4, dtype=torch.float32),
        w13_bias=torch.ones(2, 6, dtype=torch.bfloat16),
        intermediate_size_per_partition=3,
        swiglu_limit=8.0,
        glu_alpha=1.702,
        glu_bias=1.0,
        activation=activation,
    )
    if with_w2_bias:
        layer.w2_bias = torch.ones(2, 4, dtype=torch.bfloat16)
    return layer


@pytest.mark.unit
def test_create_weights_uses_parent_bias_registration(monkeypatch):
    moe_mod = _import_moe_module(monkeypatch)
    def fake_parent_create_weights(
        _self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        layer.register_parameter(
            "w13_bias",
            torch.nn.Parameter(
                torch.zeros(num_experts, 2 * intermediate_size_per_partition, dtype=torch.bfloat16),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w2_bias",
            torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=torch.bfloat16),
                requires_grad=False,
            ),
        )
    monkeypatch.setattr(moe_mod.NPUCompressedTensorsW8A8Int8MoEMethod, "create_weights", fake_parent_create_weights)

    method = moe_mod.NPUCompressedTensorsW8A8Int8MoEMethodV1.__new__(
        moe_mod.NPUCompressedTensorsW8A8Int8MoEMethodV1
    )
    method.moe = SimpleNamespace(has_bias=True)
    layer = torch.nn.Module()
    layer.activation = "swigluoai"
    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=4,
        intermediate_size_per_partition=3,
        params_dtype=torch.bfloat16,
    )

    assert isinstance(layer.w13_bias, torch.nn.Parameter)
    assert isinstance(layer.w2_bias, torch.nn.Parameter)
    assert layer.w13_bias.shape == (2, 6)
    assert layer.w2_bias.shape == (2, 4)
    assert not hasattr(layer, "swiglu_limit")
    assert not hasattr(layer, "glu_alpha")
    assert not hasattr(layer, "glu_bias")


@pytest.mark.unit
def test_apply_experts_gpt_oss_swigluoai_passes_kernel_kwargs(monkeypatch):
    moe_mod = _import_moe_module(monkeypatch)
    method = moe_mod.NPUCompressedTensorsW8A8Int8MoEMethodV1.__new__(
        moe_mod.NPUCompressedTensorsW8A8Int8MoEMethodV1
    )
    method.moe = SimpleNamespace(has_bias=True)

    grouped_calls = []

    def fake_grouped_matmul(inputs, weights, **kwargs):
        grouped_calls.append(kwargs)
        if kwargs["output_dtype"] == torch.int32:
            hidden = inputs[0]
            return [torch.ones(hidden.shape[0], 6, dtype=torch.int32)]
        return [torch.ones(3, 4, dtype=torch.bfloat16)]

    dequant_mock = MagicMock(
        return_value=(
            torch.ones(3, 3, dtype=torch.int8),
            torch.ones(3, dtype=torch.float32),
        )
    )
    monkeypatch.setattr(moe_mod.torch_npu, "npu_grouped_matmul", fake_grouped_matmul, raising=False)
    monkeypatch.setattr(moe_mod.torch_npu, "npu_dequant_swiglu_quant", dequant_mock, raising=False)

    out = method.apply_experts(_build_layer("swigluoai"), _build_prepare_permute_result())
    assert out.dtype == torch.bfloat16

    dequant_kwargs = dequant_mock.call_args.kwargs
    assert dequant_kwargs["bias"] is not None
    assert dequant_kwargs["swiglu_mode"] == 1
    assert dequant_kwargs["clamp_limit"] == 8.0
    assert dequant_kwargs["glu_alpha"] == 1.702
    assert dequant_kwargs["glu_bias"] == 1.0
    assert grouped_calls[1]["bias"] is not None


@pytest.mark.unit
def test_apply_experts_non_gpt_oss_does_not_pass_swigluoai_kwargs(monkeypatch):
    moe_mod = _import_moe_module(monkeypatch)
    method = moe_mod.NPUCompressedTensorsW8A8Int8MoEMethodV1.__new__(
        moe_mod.NPUCompressedTensorsW8A8Int8MoEMethodV1
    )
    method.moe = SimpleNamespace(has_bias=False)

    def fake_grouped_matmul(inputs, weights, **kwargs):
        if kwargs["output_dtype"] == torch.int32:
            hidden = inputs[0]
            return [torch.ones(hidden.shape[0], 6, dtype=torch.int32)]
        return [torch.ones(3, 4, dtype=torch.bfloat16)]

    dequant_mock = MagicMock(
        return_value=(
            torch.ones(3, 3, dtype=torch.int8),
            torch.ones(3, dtype=torch.float32),
        )
    )
    monkeypatch.setattr(moe_mod.torch_npu, "npu_grouped_matmul", fake_grouped_matmul, raising=False)
    monkeypatch.setattr(moe_mod.torch_npu, "npu_dequant_swiglu_quant", dequant_mock, raising=False)

    method.apply_experts(_build_layer("silu"), _build_prepare_permute_result())
    dequant_kwargs = dequant_mock.call_args.kwargs
    assert dequant_kwargs["bias"] is None
    assert "swiglu_mode" not in dequant_kwargs
    assert "clamp_limit" not in dequant_kwargs
    assert "glu_alpha" not in dequant_kwargs
    assert "glu_bias" not in dequant_kwargs


@pytest.mark.unit
def test_apply_experts_handles_missing_w2_bias(monkeypatch):
    moe_mod = _import_moe_module(monkeypatch)
    method = moe_mod.NPUCompressedTensorsW8A8Int8MoEMethodV1.__new__(
        moe_mod.NPUCompressedTensorsW8A8Int8MoEMethodV1
    )
    method.moe = SimpleNamespace(has_bias=True)

    grouped_calls = []

    def fake_grouped_matmul(inputs, weights, **kwargs):
        grouped_calls.append(kwargs)
        if kwargs["output_dtype"] == torch.int32:
            hidden = inputs[0]
            return [torch.ones(hidden.shape[0], 6, dtype=torch.int32)]
        return [torch.ones(3, 4, dtype=torch.bfloat16)]

    monkeypatch.setattr(moe_mod.torch_npu, "npu_grouped_matmul", fake_grouped_matmul, raising=False)
    monkeypatch.setattr(
        moe_mod.torch_npu,
        "npu_dequant_swiglu_quant",
        lambda **kwargs: (
            torch.ones(3, 3, dtype=torch.int8),
            torch.ones(3, dtype=torch.float32),
        ),
        raising=False,
    )

    method.apply_experts(_build_layer("silu", with_w2_bias=False), _build_prepare_permute_result())
    assert grouped_calls[1]["bias"] is None


@pytest.mark.unit
def test_apply_experts_non_swigluoai_can_still_use_bias(monkeypatch):
    moe_mod = _import_moe_module(monkeypatch)
    method = moe_mod.NPUCompressedTensorsW8A8Int8MoEMethodV1.__new__(
        moe_mod.NPUCompressedTensorsW8A8Int8MoEMethodV1
    )
    method.moe = SimpleNamespace(has_bias=True)

    grouped_calls = []

    def fake_grouped_matmul(inputs, weights, **kwargs):
        grouped_calls.append(kwargs)
        if kwargs["output_dtype"] == torch.int32:
            hidden = inputs[0]
            return [torch.ones(hidden.shape[0], 6, dtype=torch.int32)]
        return [torch.ones(3, 4, dtype=torch.bfloat16)]

    dequant_mock = MagicMock(
        return_value=(
            torch.ones(3, 3, dtype=torch.int8),
            torch.ones(3, dtype=torch.float32),
        )
    )
    monkeypatch.setattr(moe_mod.torch_npu, "npu_grouped_matmul", fake_grouped_matmul, raising=False)
    monkeypatch.setattr(moe_mod.torch_npu, "npu_dequant_swiglu_quant", dequant_mock, raising=False)

    method.apply_experts(_build_layer("silu"), _build_prepare_permute_result())
    dequant_kwargs = dequant_mock.call_args.kwargs
    assert dequant_kwargs["bias"] is not None
    assert "swiglu_mode" not in dequant_kwargs
    assert "clamp_limit" not in dequant_kwargs
    assert "glu_alpha" not in dequant_kwargs
    assert "glu_bias" not in dequant_kwargs
    assert grouped_calls[1]["bias"] is not None
