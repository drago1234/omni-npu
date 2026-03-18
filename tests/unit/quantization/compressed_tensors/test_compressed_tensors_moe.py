import importlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


def _make_package(monkeypatch: pytest.MonkeyPatch, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _make_module(monkeypatch: pytest.MonkeyPatch, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class DummyCompressedTensorsW8A8Int8MoEMethod:
    def __init__(self, weight_quant, parent, moe_config):
        self.weight_quant = weight_quant
        self.moe = parent
        self.moe_config = moe_config
        self.static_input_scales = False


class DummyNPUFusedMoEMethodBase:
    def __init__(self):
        self.communication_strategy_selector = None

    def select_communication_strategy(self, num_tokens: int):
        return self.communication_strategy_selector.select_communication_strategy(
            num_tokens
        )

    def apply_prepare_permute(self, strategy_impl, layer, x, topk_ids):
        return strategy_impl.prepare_permute(layer, x, topk_ids)

    def apply_unpermute_finalize(
        self, strategy_impl, layer, hidden_states, topk_ids, topk_weights, result
    ):
        return strategy_impl.unpermute_finalize(
            layer, hidden_states, topk_ids, topk_weights, result
        )


class DummyPreparePermuteResult:
    def __init__(
        self,
        hidden_states_sorted_by_experts,
        expert_tokens,
        dynamic_scale,
        avg_tokens_per_expert=None,
        row_idx_type=0,
    ):
        self.hidden_states_sorted_by_experts = hidden_states_sorted_by_experts
        self.expert_tokens = expert_tokens
        self.dynamic_scale = dynamic_scale
        self.avg_tokens_per_expert = avg_tokens_per_expert
        self.row_idx_type = row_idx_type


class DummyStrategyImpl:
    def __init__(self, prepare_result, final_output=None):
        self._prepare_result = prepare_result
        self._final_output = final_output

    def prepare_permute(self, layer, x, topk_ids):
        return self._prepare_result

    def unpermute_finalize(self, layer, hidden_states, topk_ids, topk_weights, result):
        if self._final_output is None:
            return hidden_states
        return self._final_output


class DummyNPUFusedMoE:
    def select_experts(
        self,
        hidden_states,
        router_logits
    ):
        num_tokens = router_logits.shape[0]
        topk_weights = torch.ones(num_tokens, top_k, dtype=torch.float32)
        topk_ids = torch.zeros(num_tokens, top_k, dtype=torch.int64)
        return topk_weights, topk_ids


class DummyNPUStream:
    def wait_stream(self, stream):
        return None


class DummyStreamContext:
    def __init__(self, stream):
        self.stream = stream

    def __enter__(self):
        return self.stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class MockMoELayer(torch.nn.Module):
    def __init__(self, use_ep=True, num_experts=2):
        super().__init__()
        self.layer_name = "test_layer"
        self.moe_config = SimpleNamespace(num_experts=num_experts)
        self.local_num_experts = num_experts
        self.enable_eplb = False
        self.moe_parallel_config = SimpleNamespace(use_ep=use_ep)
        self.quant_config = object()
        self.gate = None
        self.shared_experts = None
        self.activation = "silu"
        self.global_num_experts = -1
        self.apply_router_weight_on_input = False
        self.expert_map = None

    def select_experts(
        self,
        hidden_states,
        router_logits
    ):
        batch = router_logits.shape[0]
        weights = torch.ones(batch, self.top_k, dtype=torch.float32)
        ids = torch.zeros(batch, self.top_k, dtype=torch.int32)
        return weights, ids


@pytest.fixture
def mock_dependencies(monkeypatch: pytest.MonkeyPatch):
    if not hasattr(torch, "npu"):
        monkeypatch.setattr(torch, "npu", SimpleNamespace(), raising=False)
    torch.npu.config = SimpleNamespace(allow_internal_format=False)
    torch.npu.current_stream = lambda: DummyNPUStream()
    torch.npu.stream = lambda s: DummyStreamContext(s)

    torch_npu_module = _make_module(monkeypatch, "torch_npu")
    torch_npu_module.npu_format_cast = lambda t, fmt: t
    torch_npu_module.npu_grouped_matmul = (
        lambda inputs, weights, **kwargs: [inputs[0].to(kwargs.get("output_dtype", inputs[0].dtype))]
    )
    torch_npu_module.npu_swiglu = lambda x: x
    torch_npu_module.npu_dequant_swiglu_quant = (
        lambda **kwargs: (kwargs["x"].to(torch.int8), torch.ones(kwargs["x"].shape[0]))
    )

    _make_package(monkeypatch, "vllm")
    logger_module = _make_module(monkeypatch, "vllm.logger")
    logger_module.init_logger = lambda _: SimpleNamespace(
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )

    distributed_module = _make_module(monkeypatch, "vllm.distributed")
    distributed_module.tensor_model_parallel_all_gather = lambda x, dim=0: x
    distributed_module.tensor_model_parallel_all_reduce = lambda x: x
    distributed_module.get_tensor_model_parallel_world_size = lambda: 1
    distributed_module.get_tensor_model_parallel_rank = lambda: 0
    distributed_module.get_dp_group = lambda: SimpleNamespace(world_size=1, rank=0)
    distributed_module.get_world_group = lambda: SimpleNamespace(
        rank_in_group=0, world_size=1
    )

    platforms_module = _make_module(monkeypatch, "vllm.platforms")
    platforms_module.current_platform = SimpleNamespace(device_type="cpu")

    utils_module = _make_module(monkeypatch, "vllm.model_executor.utils")
    utils_module.set_weight_attrs = (
        lambda param, attrs: [setattr(param, k, v) for k, v in attrs.items()]
    )

    ct_moe_module = _make_module(
        monkeypatch,
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe",
    )
    ct_moe_module.CompressedTensorsW8A8Int8MoEMethod = (
        DummyCompressedTensorsW8A8Int8MoEMethod
    )

    fused_moe_layer_module = _make_module(
        monkeypatch, "vllm.model_executor.layers.fused_moe.layer"
    )

    class FusedMoeWeightScaleSupported:
        CHANNEL = SimpleNamespace(value="channel")

    fused_moe_layer_module.FusedMoeWeightScaleSupported = FusedMoeWeightScaleSupported

    vllm_config_module = _make_module(monkeypatch, "vllm.config")
    vllm_config_module.get_current_vllm_config = lambda: SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(first_k_dense_replace=0))
    )

    npu_layer_module = _make_module(monkeypatch, "omni_npu.layers.fused_moe.layer")
    npu_layer_module.NPUFusedMoE = DummyNPUFusedMoE

    npu_fused_moe_module = _make_module(
        monkeypatch, "omni_npu.layers.fused_moe.fused_moe"
    )
    npu_fused_moe_module.fused_experts_tp = MagicMock(
        return_value=torch.tensor([2.0], dtype=torch.float32)
    )

    npu_base_module = _make_module(
        monkeypatch, "omni_npu.layers.fused_moe.fused_moe_method_base"
    )
    npu_base_module.NPUFusedMoEMethodBase = DummyNPUFusedMoEMethodBase

    prepare_module = _make_module(
        monkeypatch, "omni_npu.layers.fused_moe.prepare_permute_unpermute_finalize"
    )
    prepare_module.PreparePermuteResult = DummyPreparePermuteResult

    utils_layer_module = _make_module(monkeypatch, "omni_npu.layers.utils")
    utils_layer_module.named_stream = lambda name: DummyNPUStream()

    base_path = Path(__file__).resolve().parents[4]
    omni_pkg = types.ModuleType("omni_npu")
    omni_pkg.__path__ = [str(base_path / "src" / "omni_npu")]
    monkeypatch.setitem(sys.modules, "omni_npu", omni_pkg)
    layers_pkg = types.ModuleType("omni_npu.layers")
    layers_pkg.__path__ = [str(base_path / "src" / "omni_npu" / "layers")]
    monkeypatch.setitem(sys.modules, "omni_npu.layers", layers_pkg)
    quant_pkg = types.ModuleType("omni_npu.layers.quantization")
    quant_pkg.__path__ = [str(base_path / "src" / "omni_npu" / "layers" / "quantization")]
    monkeypatch.setitem(sys.modules, "omni_npu.layers.quantization", quant_pkg)
    ct_pkg = types.ModuleType("omni_npu.layers.quantization.compressed_tensors")
    ct_pkg.__path__ = [
        str(base_path / "src" / "omni_npu" / "layers" / "quantization" / "compressed_tensors")
    ]
    monkeypatch.setitem(
        sys.modules, "omni_npu.layers.quantization.compressed_tensors", ct_pkg
    )

    sys.modules.pop(
        "omni_npu.layers.quantization.compressed_tensors.compressed_tensors_moe", None
    )
    yield


@pytest.fixture
def compressed_moe_module(mock_dependencies):
    module = importlib.import_module(
        "omni_npu.layers.quantization.compressed_tensors.compressed_tensors_moe"
    )
    importlib.reload(module)
    return module


def _make_method(module, layer, has_bias=False):
    parent = SimpleNamespace(moe_parallel_config=layer.moe_parallel_config, has_bias=has_bias)
    return module.NPUCompressedTensorsW8A8Int8MoEMethod(parent, layer)


def _make_prepare_result(dynamic_scale=None, row_idx_type=0):
    return DummyPreparePermuteResult(
        hidden_states_sorted_by_experts=torch.ones(3, 4, dtype=torch.int8),
        expert_tokens=torch.tensor([1, 2], dtype=torch.int64),
        dynamic_scale=dynamic_scale,
        avg_tokens_per_expert=[1, 2],
        row_idx_type=row_idx_type,
    )


def test_create_weights_registers_expected_params(compressed_moe_module):
    layer = MockMoELayer(use_ep=True)
    method = _make_method(compressed_moe_module, layer, has_bias=False)

    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=4,
        intermediate_size_per_partition=3,
        params_dtype=torch.float16,
        weight_loader="mock_loader",
    )

    assert layer.w13_weight.dtype == torch.int8
    assert layer.w13_weight.shape == (2, 6, 4)
    assert layer.w2_weight.shape == (2, 4, 3)
    assert layer.w13_weight_scale.shape == (2, 6, 1)
    assert layer.w2_weight_scale.shape == (2, 4, 1)
    assert layer.w13_weight_scale.quant_method == "channel"
    assert layer.w2_weight_scale.quant_method == "channel"
    assert layer.w13_input_scale is None
    assert layer.w2_input_scale is None
    assert not hasattr(layer, "w13_bias")
    assert not hasattr(layer, "w2_bias")


def test_create_weights_registers_bias_when_enabled(compressed_moe_module):
    layer = MockMoELayer(use_ep=True)
    method = _make_method(compressed_moe_module, layer, has_bias=True)

    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=4,
        intermediate_size_per_partition=3,
        params_dtype=torch.float16,
        weight_loader="mock_loader",
    )

    assert layer.w13_bias.shape == (2, 6)
    assert layer.w2_bias.shape == (2, 4)


def test_process_weights_after_loading_transpose_and_cast(compressed_moe_module):
    layer = MockMoELayer(use_ep=True)
    method = _make_method(compressed_moe_module, layer, has_bias=False)
    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=4,
        intermediate_size_per_partition=3,
        params_dtype=torch.float16,
        weight_loader="mock_loader",
    )

    method.process_weights_after_loading(layer)

    assert layer.w13_weight.shape == (2, 4, 6)
    assert layer.w2_weight.shape == (2, 3, 4)
    assert layer.w13_weight_scale.dtype == torch.float32
    assert layer.w13_weight_scale.shape == (2, 6)
    assert layer.w2_weight_scale.dtype == torch.bfloat16
    assert layer.w2_weight_scale.shape == (2, 4)


def test_apply_experts_quant_requires_dynamic_scale(compressed_moe_module):
    layer = MockMoELayer(use_ep=True)
    layer.quant_config = object()
    method = _make_method(compressed_moe_module, layer)
    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=4,
        intermediate_size_per_partition=3,
        params_dtype=torch.float16,
        weight_loader="mock_loader",
    )
    method.process_weights_after_loading(layer)

    with pytest.raises(ValueError, match="dynamic per-token scale"):
        method.apply_experts(layer, _make_prepare_result(dynamic_scale=None))


def test_apply_experts_swigluoai_passes_kernel_kwargs(
    compressed_moe_module, monkeypatch
):
    layer = MockMoELayer(use_ep=True)
    layer.quant_config = object()
    layer.swiglu_limit = 8.0
    layer.glu_alpha = 1.7
    layer.glu_bias = 1.0
    method = _make_method(compressed_moe_module, layer, has_bias=True)
    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=4,
        intermediate_size_per_partition=3,
        params_dtype=torch.float16,
        weight_loader="mock_loader",
    )
    method.process_weights_after_loading(layer)

    grouped_calls = []

    def _fake_grouped_matmul(inputs, weights, **kwargs):
        grouped_calls.append(kwargs)
        if kwargs.get("output_dtype") == torch.int32:
            return [torch.ones(inputs[0].shape[0], 6, dtype=torch.int32)]
        return [torch.ones(inputs[0].shape[0], 4, dtype=torch.bfloat16)]

    dequant_mock = MagicMock(
        return_value=(
            torch.ones(3, 3, dtype=torch.int8),
            torch.ones(3, dtype=torch.float32),
        )
    )
    monkeypatch.setattr(
        compressed_moe_module.torch_npu, "npu_grouped_matmul", _fake_grouped_matmul
    )
    monkeypatch.setattr(
        compressed_moe_module.torch_npu, "npu_dequant_swiglu_quant", dequant_mock
    )

    prepare_result = _make_prepare_result(dynamic_scale=torch.ones(1, 3))
    output = method.apply_experts(layer, prepare_result, activation="swigluoai")

    assert output.dtype == torch.bfloat16
    kwargs = dequant_mock.call_args.kwargs
    assert kwargs["bias"] is not None
    assert kwargs["swiglu_mode"] == 1
    assert kwargs["clamp_limit"] == 8.0
    assert kwargs["glu_alpha"] == 1.7
    assert kwargs["glu_bias"] == 1.0
    assert grouped_calls[1]["output_dtype"] == torch.bfloat16


def test_apply_experts_grouped_finalize_returns_intermediate_and_scale(
    compressed_moe_module, monkeypatch
):
    layer = MockMoELayer(use_ep=True)
    layer.quant_config = object()
    method = _make_method(compressed_moe_module, layer)
    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=4,
        intermediate_size_per_partition=3,
        params_dtype=torch.float16,
        weight_loader="mock_loader",
    )
    method.process_weights_after_loading(layer)

    grouped_calls = []

    def _fake_grouped_matmul(inputs, weights, **kwargs):
        grouped_calls.append(kwargs)
        return [torch.ones(inputs[0].shape[0], 6, dtype=torch.int32)]

    intermediate_h = torch.ones(3, 3, dtype=torch.int8)
    pertoken_scale = torch.ones(3, dtype=torch.float32)
    dequant_mock = MagicMock(return_value=(intermediate_h, pertoken_scale))
    monkeypatch.setattr(
        compressed_moe_module.torch_npu, "npu_grouped_matmul", _fake_grouped_matmul
    )
    monkeypatch.setattr(
        compressed_moe_module.torch_npu, "npu_dequant_swiglu_quant", dequant_mock
    )

    output = method.apply_experts(
        layer,
        _make_prepare_result(dynamic_scale=torch.ones(1, 3)),
        use_grouped_matmul_finalize_routing=True,
    )

    assert output[0] is intermediate_h
    assert output[1] is pertoken_scale
    assert len(grouped_calls) == 1

def test_apply_with_ep_returns_tuple_when_shared_experts_enabled(
    compressed_moe_module, monkeypatch
):
    layer = MockMoELayer(use_ep=True)
    layer.quant_config = object()
    layer.shared_experts = lambda x: torch.full_like(x, 2.0)
    layer.shared_experts.gate_up_proj = SimpleNamespace(tp_size=1)

    method = _make_method(compressed_moe_module, layer)
    method.select_communication_strategy = lambda n: (
        "agrs",
        DummyStrategyImpl(_make_prepare_result(dynamic_scale=torch.ones(3))),
    )
    method.apply_experts = MagicMock(return_value=torch.full((3, 4), 3.0))

    monkeypatch.setenv("VLLM_PLUGINS", "omni_custom_models")
    hidden_states = torch.randn(3, 4, dtype=torch.bfloat16)
    router_logits = torch.randn(3, 2, dtype=torch.float32)
    output = method.apply(
        layer=layer,
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=1,
        renormalize=False,
    )
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)

    assert isinstance(output, tuple)
    shared_output, routed_output = output
    assert torch.equal(shared_output, torch.full((3, 4), 2.0))
    assert torch.equal(routed_output, torch.full((3, 4), 5.0))


def test_apply_passes_grouped_finalize_flag_for_agrs_decode(compressed_moe_module):
    layer = MockMoELayer(use_ep=True)
    layer.quant_config = object()
    method = _make_method(compressed_moe_module, layer)
    method.select_communication_strategy = lambda n: (
        "agrs",
        DummyStrategyImpl(
            _make_prepare_result(dynamic_scale=torch.ones(3), row_idx_type=1),
            final_output=torch.full((3, 4), 4.0),
        ),
    )
    method.apply_experts = MagicMock(
        return_value=(torch.ones(3, 3, dtype=torch.int8), torch.ones(3))
    )

    output = method.apply(
        layer=layer,
        hidden_states=torch.randn(3, 4, dtype=torch.bfloat16),
        router_logits=torch.randn(3, 2, dtype=torch.float32),
        top_k=1,
        renormalize=False,
    )

    assert method.apply_experts.call_args.kwargs["use_grouped_matmul_finalize_routing"] is True
    assert torch.equal(output, torch.full((3, 4), 4.0))

def test_apply_shared_experts_tp_gt_1_uses_full_hidden_states(
    compressed_moe_module, monkeypatch
):
    layer = MockMoELayer(use_ep=True)
    layer.quant_config = object()
    method = _make_method(compressed_moe_module, layer)
    method.tp_size = 2
    method.tp_rank = 1
    method.select_communication_strategy = lambda n: (
        "all2all",
        DummyStrategyImpl(_make_prepare_result(dynamic_scale=torch.ones(3))),
    )
    method.apply_experts = MagicMock(return_value=torch.full((2, 4), 3.0))
    monkeypatch.setattr(
        compressed_moe_module,
        "tensor_model_parallel_all_gather",
        lambda x, dim=0: torch.cat([x, x], dim=dim),
    )
    all_reduce_mock = MagicMock(return_value=torch.full((3, 4), 9.0))
    monkeypatch.setattr(
        compressed_moe_module, "tensor_model_parallel_all_reduce", all_reduce_mock
    )

    shared = MagicMock(return_value=torch.full((3, 4), 6.0))
    shared.gate_up_proj = SimpleNamespace(tp_size=2)
    layer.shared_experts = shared

    hidden_states = torch.randn(3, 4, dtype=torch.bfloat16)
    router_logits = torch.randn(3, 2, dtype=torch.float32)
    output = method.apply(
        layer=layer,
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=1,
        renormalize=False,
    )

    assert isinstance(output, tuple)
    shared_output, routed_output = output
    assert shared.call_count == 1
    assert torch.equal(shared.call_args.args[0], hidden_states)
    all_reduce_mock.assert_called_once()
    assert torch.equal(all_reduce_mock.call_args.args[0], torch.full((3, 4), 6.0))
    assert torch.equal(shared_output, torch.full((3, 4), 9.0))
    assert routed_output.shape == (3, 4)


def test_apply_enable_eplb_calls_planner(compressed_moe_module):
    layer = MockMoELayer(use_ep=True)
    layer.quant_config = object()
    method = _make_method(compressed_moe_module, layer)
    method.select_communication_strategy = lambda n: (
        "agrs",
        DummyStrategyImpl(_make_prepare_result(dynamic_scale=torch.ones(3))),
    )
    method.apply_experts = MagicMock(return_value=torch.full((3, 4), 1.0))
    method.planner = MagicMock(
        plan=MagicMock(
            return_value=(
                None,
                torch.zeros(3, 1, dtype=torch.int64),
                None,
            )
        )
    )
    method.moe_layer_idx = 1
    method.expert_mapping = {"x": 1}

    output = method.apply(
        layer=layer,
        hidden_states=torch.randn(3, 4, dtype=torch.bfloat16),
        router_logits=torch.randn(3, 2, dtype=torch.float32),
        top_k=1,
        renormalize=False,
        enable_eplb=True,
    )

    assert method.planner.plan.call_count == 1
    assert layer.planner is method.planner
    assert layer.moe_layer_idx == 1
    assert output.shape == (3, 4)


def test_supports_eplb_property(compressed_moe_module):
    layer = MockMoELayer(use_ep=True)
    method = _make_method(compressed_moe_module, layer)
    assert method.supports_eplb is True


def test_init_eplb_sets_redundant_experts(compressed_moe_module, monkeypatch):
    class DummyPlanner:
        def __init__(self, **kwargs):
            pass

        @staticmethod
        def get_deepseek_v3_moe_layer_idx(prefix, first_k_dense_replace):
            return 3

        def expert_mapping_on_current_layer(self, moe_layer_idx):
            return {"idx": moe_layer_idx}

        def get_num_of_redundant_experts(
            self, moe_layer_idx, num_expert_per_device_origin, rank_device
        ):
            return 1

    omni_pkg = types.ModuleType("omni_placement")
    omni_planner_module = types.ModuleType("omni_placement.omni_planner")
    omni_planner_module.OmniPlanner = DummyPlanner
    monkeypatch.setitem(sys.modules, "omni_placement", omni_pkg)
    monkeypatch.setitem(sys.modules, "omni_placement.omni_planner", omni_planner_module)

    layer = MockMoELayer(use_ep=True, num_experts=2)
    layer.enable_eplb = True
    method = _make_method(compressed_moe_module, layer)
    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=4,
        intermediate_size_per_partition=3,
        params_dtype=torch.float16,
        weight_loader="mock_loader",
    )

    assert method.num_of_redundant_experts == 1
    assert layer.w13_weight.shape[0] == 3
