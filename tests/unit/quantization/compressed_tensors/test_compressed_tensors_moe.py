import importlib
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
    def __init__(self, parent, moe_config):
        self.moe = parent
        self.moe_config = moe_config
        self.moe_quant_config = SimpleNamespace()
        self.fused_experts = MagicMock(return_value=torch.tensor([7.0]))
        self.static_input_scales = False


class DummyNpuPrepareAndFinalize:
    def __init__(self, moe):
        self.moe = moe


class DummyNPUFusedMoEPermuteExpertsUnpermute:
    def __init__(self, moe_quant_config, layer):
        self.moe_quant_config = moe_quant_config
        self.layer = layer


class DummyNPUFusedMoE:
    @staticmethod
    def select_experts(
        router_logits,
        top_k,
        use_grouped_topk=False,
        renormalize=False,
        topk_group=None,
        num_expert_group=None,
        custom_routing_function=None,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
    ):
        batch = router_logits.shape[0]
        weights = torch.ones(batch, top_k, dtype=torch.float32)
        ids = torch.zeros(batch, top_k, dtype=torch.int32)
        return weights, ids


class MockMoELayer(torch.nn.Module):
    def __init__(self, use_ep: bool = False, num_experts: int = 2):
        super().__init__()
        self.layer_name = "test_layer"
        self.moe_config = SimpleNamespace(num_experts=num_experts)
        self.local_num_experts = num_experts
        self.enable_eplb = False
        self.moe_parallel_config = SimpleNamespace(use_ep=use_ep)
        self.shared_experts = None


@pytest.fixture
def mock_dependencies(monkeypatch: pytest.MonkeyPatch):
    if not hasattr(torch, "npu"):
        monkeypatch.setattr(
            torch, "npu", SimpleNamespace(config=SimpleNamespace()), raising=False
        )
    if not hasattr(torch.npu, "config"):
        torch.npu.config = SimpleNamespace()
    torch.npu.config.allow_internal_format = False

    torch_npu = SimpleNamespace(
        npu=SimpleNamespace(get_device_name=lambda idx: "MockDevice"),
        npu_format_cast=lambda tensor, fmt: tensor,
        npu_grouped_matmul=lambda inputs, weights, **kwargs: [inputs[0]],
        npu_swiglu=lambda tensor: tensor,
        npu_dynamic_quant=lambda tensor: (
            tensor.to(torch.int8),
            torch.ones(tensor.shape[0], dtype=torch.float32),
        ),
        npu_dequant_swiglu_quant=lambda *args, **kwargs: (
            args[0],
            torch.ones(args[0].shape[0], dtype=torch.float32),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)

    vllm_module = _make_package(monkeypatch, "vllm")
    logger_module = _make_module(monkeypatch, "vllm.logger")
    logger_module.init_logger = lambda name: SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )

    distributed_module = _make_module(monkeypatch, "vllm.distributed")
    distributed_module.tensor_model_parallel_all_gather = lambda x, dim=0: x
    distributed_module.tensor_model_parallel_all_reduce = lambda x: x
    distributed_module.get_tensor_model_parallel_world_size = lambda: 1
    distributed_module.get_tensor_model_parallel_rank = lambda: 0
    distributed_module.get_world_group = lambda: SimpleNamespace(
        rank_in_group=0, world_size=1
    )

    platforms_module = _make_module(monkeypatch, "vllm.platforms")
    platforms_module.current_platform = SimpleNamespace(device_type="cpu")

    forward_context_module = _make_module(monkeypatch, "vllm.forward_context")
    forward_context_module.get_forward_context = lambda: SimpleNamespace(
        attn_metadata=None
    )

    utils_module = _make_module(monkeypatch, "vllm.model_executor.utils")

    def _set_weight_attrs(param, attrs):
        for key, value in attrs.items():
            setattr(param, key, value)

    utils_module.set_weight_attrs = _set_weight_attrs

    model_executor_module = _make_package(monkeypatch, "vllm.model_executor")
    layers_module = _make_package(monkeypatch, "vllm.model_executor.layers")
    quant_module = _make_package(monkeypatch, "vllm.model_executor.layers.quantization")
    ct_module = _make_package(
        monkeypatch, "vllm.model_executor.layers.quantization.compressed_tensors"
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

    modular_kernel_module = _make_module(
        monkeypatch, "vllm.model_executor.layers.fused_moe.modular_kernel"
    )

    class FusedMoEPermuteExpertsUnpermute:
        def __init__(self, moe_quant_config, layer):
            self.moe_quant_config = moe_quant_config
            self.layer = layer

    class FusedMoEPrepareAndFinalize:
        def __init__(self, moe):
            self.moe = moe

    modular_kernel_module.FusedMoEPermuteExpertsUnpermute = (
        FusedMoEPermuteExpertsUnpermute
    )
    modular_kernel_module.FusedMoEPrepareAndFinalize = FusedMoEPrepareAndFinalize

    vllm_config_module = _make_module(monkeypatch, "vllm.config")
    vllm_config_module.get_current_vllm_config = lambda: SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(first_k_dense_replace=0))
    )

    vllm_module.logger = logger_module
    vllm_module.distributed = distributed_module
    vllm_module.platforms = platforms_module
    vllm_module.forward_context = forward_context_module
    vllm_module.model_executor = model_executor_module
    model_executor_module.layers = layers_module
    layers_module.quantization = quant_module
    quant_module.compressed_tensors = ct_module

    npu_prepare_module = _make_module(
        monkeypatch, "omni_npu.layers.fused_moe.npu_moe_prepare_finalize"
    )
    npu_prepare_module.NpuMoEPrepareAndFinalize = DummyNpuPrepareAndFinalize
    npu_permute_module = _make_module(
        monkeypatch, "omni_npu.layers.fused_moe.npu_moe_permute_unpermute"
    )
    npu_permute_module.NPUFusedMoEPermuteExpertsUnpermute = (
        DummyNPUFusedMoEPermuteExpertsUnpermute
    )
    npu_layer_module = _make_module(monkeypatch, "omni_npu.layers.fused_moe.layer")
    npu_layer_module.NPUFusedMoE = DummyNPUFusedMoE
    npu_fused_moe_module = _make_module(
        monkeypatch, "omni_npu.layers.fused_moe.fused_moe"
    )
    npu_fused_moe_module.moe_infer_fusion = MagicMock(
        return_value=torch.tensor([1.0])
    )
    npu_fused_moe_module.fused_experts_tp = MagicMock(
        return_value=torch.tensor([2.0])
    )
    npu_fused_moe_module.fused_experts_allgather_ep = MagicMock(
        return_value=torch.tensor([4.0])
    )

    base_path = Path(__file__).resolve().parents[4]
    omni_pkg = types.ModuleType("omni_npu")
    omni_pkg.__path__ = [str(base_path / "src" / "omni_npu")]
    monkeypatch.setitem(sys.modules, "omni_npu", omni_pkg)
    layers_pkg = types.ModuleType("omni_npu.layers")
    layers_pkg.__path__ = [str(base_path / "src" / "omni_npu" / "layers")]
    monkeypatch.setitem(sys.modules, "omni_npu.layers", layers_pkg)
    quant_pkg = types.ModuleType("omni_npu.layers.quantization")
    quant_pkg.__path__ = [
        str(base_path / "src" / "omni_npu" / "layers" / "quantization")
    ]
    monkeypatch.setitem(sys.modules, "omni_npu.layers.quantization", quant_pkg)
    ct_pkg = types.ModuleType("omni_npu.layers.quantization.compressed_tensors")
    ct_pkg.__path__ = [
        str(
            base_path
            / "src"
            / "omni_npu"
            / "layers"
            / "quantization"
            / "compressed_tensors"
        )
    ]
    monkeypatch.setitem(
        sys.modules, "omni_npu.layers.quantization.compressed_tensors", ct_pkg
    )

    sys.modules.pop(
        "omni_npu.layers.quantization.compressed_tensors.compressed_tensors_moe",
        None,
    )
    yield


@pytest.fixture
def compressed_moe_module(mock_dependencies):
    module = importlib.import_module(
        "omni_npu.layers.quantization.compressed_tensors.compressed_tensors_moe"
    )
    importlib.reload(module)
    return module


def _make_w8a8_method(module, layer):
    parent = SimpleNamespace(
        moe_parallel_config=layer.moe_parallel_config,
        has_bias=False,
    )
    return module.NPUCompressedTensorsW8A8Int8MoEMethod(parent, layer)


def _make_w4a8_method(module, layer):
    parent = SimpleNamespace(
        moe_parallel_config=layer.moe_parallel_config,
        has_bias=False,
    )
    return module.NPUCompressedTensorsW4A8Int4MoEMethod(parent, layer)


class TestNPUCompressedTensorsW8A8Int8MoEMethod:
    def test_create_weights_registers_params(self, compressed_moe_module):
        layer = MockMoELayer()
        method = _make_w8a8_method(compressed_moe_module, layer)

        method.create_weights(
            layer=layer,
            num_experts=2,
            hidden_size=4,
            intermediate_size_per_partition=3,
            params_dtype=torch.float16,
            weight_loader="mock",
        )

        assert layer.w13_weight.dtype == torch.int8
        assert layer.w13_weight.shape == (2, 6, 4)
        assert layer.w2_weight.shape == (2, 4, 3)
        assert layer.w13_weight_scale.shape == (2, 6, 1)
        assert layer.w13_weight_scale.quant_method == "channel"
        assert layer.w2_weight_scale.quant_method == "channel"
        assert layer.w13_weight_offset.dtype == torch.bfloat16
        assert layer.w2_weight_offset.dtype == torch.bfloat16

    def test_process_weights_after_loading_transposes_and_casts(
        self, compressed_moe_module
    ):
        layer = MockMoELayer()
        method = _make_w8a8_method(compressed_moe_module, layer)
        method.create_weights(
            layer=layer,
            num_experts=2,
            hidden_size=4,
            intermediate_size_per_partition=3,
            params_dtype=torch.float16,
            weight_loader="mock",
        )

        method.process_weights_after_loading(layer)

        assert layer.w13_weight.shape == (2, 4, 6)
        assert layer.w2_weight.shape == (2, 3, 4)
        assert layer.w13_weight_scale.dtype == torch.float32
        assert layer.w13_weight_scale.shape == (2, 6)
        assert layer.w2_weight_scale.dtype == torch.bfloat16
        assert layer.w2_weight_scale.shape == (2, 4)

    def test_prepare_finalize_and_select_gemm_impl(self, compressed_moe_module):
        layer = MockMoELayer()
        method = _make_w8a8_method(compressed_moe_module, layer)

        prepare_finalize = method.maybe_make_prepare_finalize("routing")
        assert prepare_finalize.moe is method.moe

        permute = method.select_gemm_impl(prepare_finalize, layer)
        assert permute.moe_quant_config is method.moe_quant_config
        assert permute.layer is layer

    def test_apply_use_ep_false_calls_fused_experts_tp(
        self, compressed_moe_module, monkeypatch
    ):
        layer = MockMoELayer(use_ep=False)
        method = _make_w8a8_method(compressed_moe_module, layer)

        fused_experts_tp = MagicMock(return_value=torch.tensor([9.0]))
        monkeypatch.setattr(compressed_moe_module, "fused_experts_tp", fused_experts_tp)

        x = torch.randn(2, 4)
        router_logits = torch.randn(2, 2)
        output = method.apply(
            layer=layer,
            x=x,
            router_logits=router_logits,
            top_k=1,
            renormalize=False,
        )

        fused_experts_tp.assert_called_once()
        assert torch.equal(output, torch.tensor([9.0]))

    def test_apply_use_ep_all2all_true_calls_moe_infer_fusion(
        self, compressed_moe_module, monkeypatch
    ):
        layer = MockMoELayer(use_ep=True)
        method = _make_w8a8_method(compressed_moe_module, layer)

        moe_infer_fusion = MagicMock(return_value=torch.tensor([3.0]))
        monkeypatch.setattr(compressed_moe_module, "moe_infer_fusion", moe_infer_fusion)
        monkeypatch.setattr(
            compressed_moe_module, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None)
        )

        x = torch.randn(2, 4)
        router_logits = torch.randn(2, 2)
        output = method.apply(
            layer=layer,
            x=x,
            router_logits=router_logits,
            top_k=1,
            renormalize=False,
        )

        moe_infer_fusion.assert_called_once()
        assert torch.equal(output, torch.tensor([3.0]))

    def test_apply_use_ep_all2all_false_returns_shared_output(
        self, compressed_moe_module, monkeypatch
    ):
        layer = MockMoELayer(use_ep=True)
        layer.shared_experts = lambda x: torch.tensor([5.0])
        method = _make_w8a8_method(compressed_moe_module, layer)
        method.create_weights(
            layer=layer,
            num_experts=2,
            hidden_size=4,
            intermediate_size_per_partition=3,
            params_dtype=torch.float16,
            weight_loader="mock",
        )
        method.fused_experts = MagicMock(return_value=torch.tensor([7.0]))

        monkeypatch.setattr(
            compressed_moe_module,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={"a": SimpleNamespace(num_prefills=0)}),
        )

        x = torch.randn(2, 4)
        router_logits = torch.randn(2, 2)
        output = method.apply(
            layer=layer,
            x=x,
            router_logits=router_logits,
            top_k=1,
            renormalize=False,
        )

        assert isinstance(output, tuple)
        assert torch.equal(output[0], torch.tensor([5.0]))
        assert torch.equal(output[1], torch.tensor([7.0]))

    def test_supports_eplb_property(self, compressed_moe_module):
        layer = MockMoELayer()
        method = _make_w8a8_method(compressed_moe_module, layer)
        assert method.supports_eplb is True

    def test_gmm_expert_no_ep_path(self, compressed_moe_module):
        layer = MockMoELayer(use_ep=False)
        method = _make_w8a8_method(compressed_moe_module, layer)
        method.create_weights(
            layer=layer,
            num_experts=2,
            hidden_size=4,
            intermediate_size_per_partition=3,
            params_dtype=torch.float16,
            weight_loader="mock",
        )

        h = torch.randn(2, 4)
        dynamic_scale = torch.ones(1, 2)
        expert_tokens = torch.tensor([2], dtype=torch.int32)
        output = method.gmm_expert(layer, h, expert_tokens, dynamic_scale=dynamic_scale)

        assert output.shape == h.shape

    def test_gmm_expert_ep_path(self, compressed_moe_module):
        layer = MockMoELayer(use_ep=True)
        method = _make_w8a8_method(compressed_moe_module, layer)
        method.create_weights(
            layer=layer,
            num_experts=2,
            hidden_size=4,
            intermediate_size_per_partition=3,
            params_dtype=torch.float16,
            weight_loader="mock",
        )

        h = torch.randn(2, 4)
        dynamic_scale = torch.ones(1, 2)
        expert_tokens = torch.tensor([2], dtype=torch.int32)
        output = method.gmm_expert(layer, h, expert_tokens, dynamic_scale=dynamic_scale)

        assert output.shape == h.shape

    def test_init_eplb_adds_redundant_experts(
        self, compressed_moe_module, monkeypatch
    ):
        class DummyPlanner:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            @staticmethod
            def get_deepseek_v3_moe_layer_idx(prefix, first_k_dense_replace):
                return 1

            def expert_mapping_on_current_layer(self, moe_layer_idx):
                return {"idx": moe_layer_idx}

            def get_num_of_redundant_experts(
                self, moe_layer_idx, num_expert_per_device_origin, rank_device
            ):
                return 1

            def plan(self, **kwargs):
                return None, kwargs["token_expert_ids"], None

            def record_activation(self, *args, **kwargs):
                return None

        omni_pkg = types.ModuleType("omni_placement")
        omni_planner_module = types.ModuleType("omni_placement.omni_planner")
        omni_planner_module.OmniPlanner = DummyPlanner
        monkeypatch.setitem(sys.modules, "omni_placement", omni_pkg)
        monkeypatch.setitem(sys.modules, "omni_placement.omni_planner", omni_planner_module)

        layer = MockMoELayer(use_ep=False, num_experts=2)
        layer.enable_eplb = True
        method = _make_w8a8_method(compressed_moe_module, layer)
        method.create_weights(
            layer=layer,
            num_experts=2,
            hidden_size=4,
            intermediate_size_per_partition=3,
            params_dtype=torch.float16,
            weight_loader="mock",
        )

        assert method.num_of_redundant_experts == 1
        assert layer.w13_weight.shape[0] == 3


class TestNPUCompressedTensorsW4A8Int4MoEMethod:
    def test_create_weights_registers_params(self, compressed_moe_module):
        layer = MockMoELayer()
        method = _make_w4a8_method(compressed_moe_module, layer)

        method.create_weights(
            layer=layer,
            num_experts=2,
            hidden_size=4,
            intermediate_size_per_partition=6,
            params_dtype=torch.float16,
            weight_loader="mock",
        )

        assert layer.w13_weight.dtype == torch.int8
        assert layer.w13_weight.shape == (2, 6, 4)
        assert layer.w2_weight.shape == (2, 2, 6)
        assert layer.w13_weight_int4_scale.dtype == torch.int64
        assert layer.w13_weight_int4_scale.shape == (2, 1, 12)
        assert layer.w13_weight_int4_scale.quant_method == "channel"
        assert layer.w2_weight_int4_scale.quant_method == "channel"
        assert layer.w13_weight_bias.dtype == torch.float32
        assert layer.w13_weight_bias.shape == (2, 12)
        assert layer.w2_weight_bias.dtype == torch.float32

    def test_process_weights_after_loading_transposes_and_casts(
        self, compressed_moe_module
    ):
        layer = MockMoELayer()
        method = _make_w4a8_method(compressed_moe_module, layer)
        method.create_weights(
            layer=layer,
            num_experts=2,
            hidden_size=8,
            intermediate_size_per_partition=8,
            params_dtype=torch.float16,
            weight_loader="mock",
        )

        method.process_weights_after_loading(layer)

        assert layer.w13_weight.shape == (2, 8, 2)
        assert layer.w2_weight.shape == (2, 8, 1)
        assert layer.w13_weight.dtype == torch.int32

    def test_gmm_expert_ep_path(self, compressed_moe_module):
        layer = MockMoELayer(use_ep=True)
        method = _make_w4a8_method(compressed_moe_module, layer)
        method.create_weights(
            layer=layer,
            num_experts=2,
            hidden_size=4,
            intermediate_size_per_partition=6,
            params_dtype=torch.float16,
            weight_loader="mock",
        )

        h = torch.randn(2, 4)
        dynamic_scale = torch.ones(1, 2)
        expert_tokens = torch.tensor([2], dtype=torch.int32)
        output = method.gmm_expert(layer, h, expert_tokens, dynamic_scale=dynamic_scale)

        assert output.shape == h.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
