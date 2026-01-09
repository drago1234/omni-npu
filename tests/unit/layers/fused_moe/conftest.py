# SPDX-License-Identifier: MIT
import sys
import types
from types import SimpleNamespace

import pytest
import torch


def _ensure_module(monkeypatch: pytest.MonkeyPatch, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _register_oot(cls, subcls):
    return subcls


class _Registerable:
    register_oot = classmethod(_register_oot)


@pytest.fixture(autouse=True)
def _stub_fused_moe_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    if not hasattr(torch, "npu"):
        monkeypatch.setattr(torch, "npu", SimpleNamespace(config=SimpleNamespace()), raising=False)
    if not hasattr(torch.npu, "config"):
        torch.npu.config = SimpleNamespace()
    if not hasattr(torch.npu.config, "allow_internal_format"):
        try:
            torch.npu.config.allow_internal_format = False
        except Exception:
            pass

    _ensure_module(monkeypatch, "torch_npu")

    vllm_module = _ensure_module(monkeypatch, "vllm")
    logger_module = _ensure_module(monkeypatch, "vllm.logger")
    logger_module.init_logger = lambda name: SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )

    distributed_module = _ensure_module(monkeypatch, "vllm.distributed")
    distributed_module.get_tp_group = lambda: SimpleNamespace(all_gather=lambda x, dim=0: x)
    distributed_module.get_ep_group = lambda: SimpleNamespace(rank=0, world_size=1)
    distributed_module.tensor_model_parallel_all_reduce = lambda x: x
    distributed_module.tensor_model_parallel_all_gather = lambda x, dim=0: x
    distributed_module.get_tensor_model_parallel_world_size = lambda: 1
    distributed_module.get_tensor_model_parallel_rank = lambda: 0

    platforms_module = _ensure_module(monkeypatch, "vllm.platforms")
    platforms_module.current_platform = SimpleNamespace(device_type="cpu", dispatch_key="cpu")

    forward_context_module = _ensure_module(monkeypatch, "vllm.forward_context")
    forward_context_module.get_forward_context = lambda: SimpleNamespace(attn_metadata=None, virtual_engine=0)

    model_executor_module = _ensure_module(monkeypatch, "vllm.model_executor")
    model_executor_module.__path__ = []
    layers_module = _ensure_module(monkeypatch, "vllm.model_executor.layers")
    layers_module.__path__ = []

    activation_module = _ensure_module(monkeypatch, "vllm.model_executor.layers.activation")

    class SiluAndMul(_Registerable):
        pass

    activation_module.SiluAndMul = SiluAndMul

    layernorm_module = _ensure_module(monkeypatch, "vllm.model_executor.layers.layernorm")

    class RMSNorm(_Registerable):
        variance_epsilon = 1e-5
        weight = SimpleNamespace(data=None)

    layernorm_module.RMSNorm = RMSNorm

    mla_module = _ensure_module(monkeypatch, "vllm.model_executor.layers.mla")

    class MultiHeadLatentAttentionWrapper(_Registerable):
        pass

    mla_module.MultiHeadLatentAttentionWrapper = MultiHeadLatentAttentionWrapper

    fused_moe_pkg = _ensure_module(monkeypatch, "vllm.model_executor.layers.fused_moe")
    fused_moe_pkg.__path__ = []

    class FusedMoEConfig:
        def __init__(self, ep_size=1, ep_rank=0, num_experts=1):
            self.ep_size = ep_size
            self.ep_rank = ep_rank
            self.num_experts = num_experts

    fused_moe_pkg.FusedMoEConfig = FusedMoEConfig

    fused_moe_layer_module = _ensure_module(monkeypatch, "vllm.model_executor.layers.fused_moe.layer")

    class FusedMoE(_Registerable):
        pass

    class UnquantizedFusedMoEMethod(_Registerable):
        pass

    fused_moe_layer_module.FusedMoE = FusedMoE
    fused_moe_layer_module.UnquantizedFusedMoEMethod = UnquantizedFusedMoEMethod

    fused_moe_config_module = _ensure_module(monkeypatch, "vllm.model_executor.layers.fused_moe.config")

    class FusedMoEQuantConfig:
        def __init__(self, use_int8_w8a8: bool = False):
            self.use_int8_w8a8 = use_int8_w8a8

    fused_moe_config_module.FusedMoEQuantConfig = FusedMoEQuantConfig

    modular_kernel_module = _ensure_module(monkeypatch, "vllm.model_executor.layers.fused_moe.modular_kernel")

    class FusedMoEPermuteExpertsUnpermute(_Registerable):
        pass

    class FusedMoEPrepareAndFinalize(_Registerable):
        pass

    modular_kernel_module.FusedMoEPermuteExpertsUnpermute = FusedMoEPermuteExpertsUnpermute
    modular_kernel_module.FusedMoEPrepareAndFinalize = FusedMoEPrepareAndFinalize
    if not hasattr(modular_kernel_module, "ExpertTokensMetadata"):
        class ExpertTokensMetadata:
            def __init__(self, expert_num_tokens, expert_num_tokens_cpu=None):
                self.expert_num_tokens = expert_num_tokens
                self.expert_num_tokens_cpu = expert_num_tokens_cpu
        modular_kernel_module.ExpertTokensMetadata = ExpertTokensMetadata
    if not hasattr(modular_kernel_module, "FusedMoEActivationFormat"):
        class FusedMoEActivationFormat:
            Standard = "standard"
        modular_kernel_module.FusedMoEActivationFormat = FusedMoEActivationFormat
    if not hasattr(modular_kernel_module, "PrepareResultType"):
        modular_kernel_module.PrepareResultType = tuple
    if not hasattr(modular_kernel_module, "TopKWeightAndReduce"):
        modular_kernel_module.TopKWeightAndReduce = object

    fused_moe_modular_method_module = _ensure_module(
        monkeypatch,
        "vllm.model_executor.layers.fused_moe.fused_moe_modular_method",
    )

    class FusedMoEModularMethod(_Registerable):
        @classmethod
        def make(cls, *args, **kwargs):
            return cls()

    fused_moe_modular_method_module.FusedMoEModularMethod = FusedMoEModularMethod

    shared_fused_moe_module = _ensure_module(monkeypatch, "vllm.model_executor.layers.fused_moe.shared_fused_moe")

    class SharedFusedMoE(_Registerable):
        pass

    shared_fused_moe_module.SharedFusedMoE = SharedFusedMoE

    compressed_tensors_module = _ensure_module(
        monkeypatch,
        "omni_npu.layers.quantization.compressed_tensors.compressed_tensors",
    )

    class NPUCompressedTensorsConfig:
        pass

    compressed_tensors_module.NPUCompressedTensorsConfig = NPUCompressedTensorsConfig

    vllm_module.logger = logger_module
    vllm_module.distributed = distributed_module
    vllm_module.platforms = platforms_module
    vllm_module.forward_context = forward_context_module
    vllm_module.model_executor = model_executor_module
    model_executor_module.layers = layers_module
    layers_module.activation = activation_module
    layers_module.layernorm = layernorm_module
    layers_module.mla = mla_module
    layers_module.fused_moe = fused_moe_pkg
    fused_moe_pkg.layer = fused_moe_layer_module
    fused_moe_pkg.modular_kernel = modular_kernel_module
    fused_moe_pkg.fused_moe_modular_method = fused_moe_modular_method_module
    fused_moe_pkg.shared_fused_moe = shared_fused_moe_module


@pytest.fixture(autouse=True)
def _force_cpu_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    orig_arange = torch.arange
    orig_tensor = torch.tensor
    orig_ones = torch.ones
    orig_zeros = torch.zeros
    orig_full = torch.full
    orig_empty = torch.empty
    orig_rand = torch.rand
    orig_randn = torch.randn

    def _coerce_device(kwargs):
        dev = kwargs.get("device")
        if dev is None:
            try:
                if orig_tensor([]).device.type == "npu":
                    kwargs["device"] = "cpu"
            except Exception:
                pass
            return
        if isinstance(dev, torch.device):
            if dev.type == "npu":
                kwargs["device"] = "cpu"
        elif dev == "npu":
            kwargs["device"] = "cpu"

    def _wrap(fn):
        def _inner(*args, **kwargs):
            _coerce_device(kwargs)
            return fn(*args, **kwargs)
        return _inner

    monkeypatch.setattr(torch, "arange", _wrap(orig_arange))
    monkeypatch.setattr(torch, "tensor", _wrap(orig_tensor))
    monkeypatch.setattr(torch, "ones", _wrap(orig_ones))
    monkeypatch.setattr(torch, "zeros", _wrap(orig_zeros))
    monkeypatch.setattr(torch, "full", _wrap(orig_full))
    monkeypatch.setattr(torch, "empty", _wrap(orig_empty))
    monkeypatch.setattr(torch, "rand", _wrap(orig_rand))
    monkeypatch.setattr(torch, "randn", _wrap(orig_randn))
