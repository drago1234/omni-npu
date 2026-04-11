import os
import re
import time
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch_npu

import vllm.model_executor.models.openpangu as openpangu_mod
from vllm.config import set_current_vllm_config
from vllm.config.load import LoadConfig
import vllm.distributed.parallel_state as parallel_state
import vllm.model_executor.model_loader.base_loader as base_loader
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader import utils as loader_utils

from omni_npu.worker.npu_mem_pool import NpuMemAllocator, npu_mem_available
from omni_npu.worker.npu_worker import NPUWorker
from omni_npu.model_config.config_loader.loader import model_extra_config
from tests.unit.platform.utils import DeviceConfig, create_vllm_config

#use pangu718b model in base layer
# (shape, dtype, device_str, npu_format) per ``named_parameters()`` entry.
ParamLayoutSnapshot = tuple[tuple[int, ...], torch.dtype, str, int]

def _snapshot_parameters_from_model(
    model: openpangu_mod.OpenPanguModel,
) -> dict[str, ParamLayoutSnapshot]:
    return {
        name: (
            tuple(param.shape),
            param.dtype,
            str(param.device),
            torch_npu.get_npu_format(param),
        )
        for name, param in model.named_parameters()
    }


def _assert_parameter_layout_equal(
    first: dict[str, ParamLayoutSnapshot],
    second: dict[str, ParamLayoutSnapshot],
) -> None:
    """Every weight: shape/dtype/device/npu_format must match across the two snapshots."""
    first_keys = set(first.keys())
    second_keys = set(second.keys())
    if first_keys != second_keys:
        raise AssertionError(
            "Parameter key set mismatch between layout snapshots. "
            f"missing_in_second={sorted(first_keys - second_keys)}, "
            f"missing_in_first={sorted(second_keys - first_keys)}"
        )
    mismatches: list[tuple[str, ParamLayoutSnapshot, ParamLayoutSnapshot]] = []
    for key in sorted(first_keys):
        if first[key] != second[key]:
            mismatches.append((key, first[key], second[key]))
    if mismatches:
        lines = []
        for name, a, b in mismatches:
            ash, adt, adev, afmt = a
            bsh, bdt, bdev, bfmt = b
            lines.append(
                f"  {name}: first shape={ash} dtype={adt} {adev=} {afmt=} | "
                f"second shape={bsh} dtype={bdt} {bdev=} {bfmt=}"
            )
        raise AssertionError(
            "Parameter layout mismatch (shape/dtype/device/npu_format):\n" + "\n".join(lines)
        )


class _OpenPanguModelRegistry:
    def resolve_model_cls(self, _architectures, model_config=None):
        del model_config
        return openpangu_mod.OpenPanguModel, "OpenPanguModel"


def _patch_single_rank_parallel_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeBackend:
        def get_hccl_comm_name(self, _rank: int) -> str:
            return "ut_fake_hccl_comm"

    class _FakeDeviceGroup:
        def rank(self) -> int:
            return 0

        def size(self) -> int:
            return 1

        def _get_backend(self, _device: torch.device) -> _FakeBackend:
            return _FakeBackend()

    fake_device_group = _FakeDeviceGroup()
    fake_tp = SimpleNamespace(
        rank_in_group=0,
        world_size=1,
        device_group=fake_device_group,
    )
    fake_pp = SimpleNamespace(
        is_first_rank=True,
        is_last_rank=True,
        rank_in_group=0,
        world_size=1,
    )
    fake_ep = SimpleNamespace(
        rank_in_group=0,
        world_size=1,
        device_group=fake_device_group,
    )
    fake_dp = SimpleNamespace(
        rank_in_group=0,
        world_size=1,
        device_group=fake_device_group,
    )
    fake_pcp = SimpleNamespace(
        rank_in_group=0,
        world_size=1,
        device_group=fake_device_group,
    )
    monkeypatch.setattr(parallel_state, "_TP", fake_tp, raising=False)
    monkeypatch.setattr(parallel_state, "_PP", fake_pp, raising=False)
    monkeypatch.setattr(parallel_state, "_EP", fake_ep, raising=False)
    monkeypatch.setattr(parallel_state, "_DP", fake_dp, raising=False)
    monkeypatch.setattr(parallel_state, "_PCP", fake_pcp, raising=False)
    monkeypatch.setattr(
        parallel_state,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        parallel_state,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "omni_npu.layers.fused_moe.prepare_permute_unpermute_finalize.current_platform",
        SimpleNamespace(device_type="npu"),
    )


def _init_model_via_gpu_model_runner_path(
    worker: NPUWorker,
    monkeypatch: pytest.MonkeyPatch,
) -> openpangu_mod.OpenPanguModel:
    _patch_single_rank_parallel_groups(monkeypatch)

    mc = worker.vllm_config.model_config
    mc.dtype = torch.bfloat16
    mc.quantization = None
    mc.model = "ut-openpangu"
    mc.device = "npu"
    mc.convert_type = "none"
    mc.runner_type = "generate"
    mc.trust_remote_code = False
    mc.model_impl = "vllm"
    mc.registry = _OpenPanguModelRegistry()
    mc._get_transformers_backend_cls = lambda: "__none__"
    mc.hf_config = SimpleNamespace(
        architectures=["OpenPanguModel"],
        pad_token_id=0,
        vocab_size=8192,
        hidden_size=256,
        num_hidden_layers=4,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        max_position_embeddings=4096,
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_act="silu",
        intermediate_size=1024,
        n_routed_experts=8,
        first_k_dense_replace=1,
        moe_intermediate_size=512,
        num_experts_per_tok=2,
        norm_topk_prob=False,
        routed_scaling_factor=1.0,
        n_shared_experts=1,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        num_nextn_predict_layers=0,
        q_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        kv_lora_rank=16,
    )
    # Keep UT path-1 responsible for post-load processing.
    monkeypatch.setattr(
        base_loader,
        "process_weights_after_loading",
        lambda *args, **kwargs: None,
    )
    worker.vllm_config.load_config = LoadConfig(load_format="dummy")
    model_loader = get_model_loader(worker.vllm_config.load_config)
    return model_loader.load_model(
        vllm_config=worker.vllm_config,
        model_config=worker.vllm_config.model_config,
    )

_EXPERT_W13_RE = re.compile(r"^(.*\.mlp\.experts)\.w13_weight$")
_EXPERT_W2_RE = re.compile(r"^(.*\.mlp\.experts)\.w2_weight$")

def _build_ckpt_schema_from_model(
    model: openpangu_mod.OpenPanguModel,
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    schema: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
    for name, param in model.named_parameters():
        shape = tuple(param.shape)
        dtype = param.dtype
        m_w13 = _EXPERT_W13_RE.match(name)
        if m_w13:
            if len(shape) != 3:
                raise ValueError(f"{name} must be rank-3, got shape={shape}")
            num_experts, double_intermediate, hidden = shape
            if double_intermediate % 2 != 0:
                raise ValueError(
                    f"{name} shape={shape} has non-even middle dim for w13 split"
                )
            intermediate = double_intermediate // 2
            prefix = m_w13.group(1)
            for expert_id in range(num_experts):
                schema[f"{prefix}.{expert_id}.gate_proj.weight"] = (
                    (intermediate, hidden),
                    dtype,
                )
                schema[f"{prefix}.{expert_id}.up_proj.weight"] = (
                    (intermediate, hidden),
                    dtype,
                )
            continue

        m_w2 = _EXPERT_W2_RE.match(name)
        if m_w2:
            if len(shape) != 3:
                raise ValueError(f"{name} must be rank-3, got shape={shape}")
            num_experts, hidden, intermediate = shape
            prefix = m_w2.group(1)
            for expert_id in range(num_experts):
                schema[f"{prefix}.{expert_id}.down_proj.weight"] = (
                    (hidden, intermediate),
                    dtype,
                )
            continue

        schema[name] = (shape, dtype)
    return schema


def _build_random_weights(
    schema: dict[str, tuple[tuple[int, ...], torch.dtype]], seed: int
) -> list[tuple[str, torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    weights: list[tuple[str, torch.Tensor]] = []
    for name, (shape, dtype) in schema.items():
        weights.append((name, torch.randn(shape, dtype=dtype, device="cpu", generator=generator)))
    return weights


def _snapshot_values_from_model(
    model: openpangu_mod.OpenPanguModel,
) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
    }


def _assert_value_snapshots_equal(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
) -> None:
    """Compare on each tensor's device (e.g. NPU); first/second must use the same device."""
    first_keys = set(first.keys())
    second_keys = set(second.keys())
    if first_keys != second_keys:
        raise AssertionError(
            "Value snapshot key mismatch. "
            f"missing_in_second={sorted(first_keys - second_keys)}, "
            f"missing_in_first={sorted(second_keys - first_keys)}"
        )
    for name in sorted(first_keys):
        a, b = first[name].detach(), second[name].detach()
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(
                f"{name}: shape/dtype mismatch first={a.shape}/{a.dtype} "
                f"second={b.shape}/{b.dtype}"
            )
        if a.device != b.device:
            raise AssertionError(
                f"{name}: device mismatch first={a.device} second={b.device}"
            )
        if not torch.allclose(a, b, rtol=1e-5, atol=1e-8):
            raise AssertionError(f"{name}: tensors not close enough on device={a.device}")


def _create_worker(monkeypatch: pytest.MonkeyPatch) -> NPUWorker:
    vllm_cfg = create_vllm_config()
    vllm_cfg.device_config = DeviceConfig("npu")

    mock_platform = SimpleNamespace(
        device_type="npu",
        pre_register_and_update=lambda: None,
        set_device=lambda device: None,
        dist_backend="hccl",
        is_sleep_mode_available=lambda: True,
    )
    monkeypatch.setattr("omni_npu.worker.npu_worker.current_platform", mock_platform)
    monkeypatch.setattr(
        "omni_npu.worker.npu_worker.init_worker_distributed_environment",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("omni_npu.worker.npu_worker.set_random_seed", lambda seed: None)

    mock_model_runner = MagicMock()
    monkeypatch.setattr(
        "omni_npu.worker.npu_worker.NPUModelRunner",
        lambda *args, **kwargs: mock_model_runner,
    )
    monkeypatch.setattr("vllm.v1.utils.report_usage_stats", lambda cfg: None)

    worker = NPUWorker(
        vllm_config=vllm_cfg,
        local_rank=0,
        rank=0,
        distributed_init_method="tcp://localhost:12345",
        is_driver_worker=True,
    )
    worker.model_runner = mock_model_runner
    return worker


def _sleep_wake_weights_cycle(worker: NPUWorker) -> None:
    """Align with production RL: pause NPU, then wake weights (+ kv) pool on same allocator."""
    time.sleep(1)
    worker.sleep(level=1)
    worker.wake_up(tags=["weights", "kv_cache"])
    time.sleep(1)


@pytest.mark.skipif(
    not npu_mem_available,
    reason="Requires libnpu_mem_allocator and NPU sleep pool (npu_mem_available).",
)
def test_pangu_rl_weight_reload_schema_consistent_across_sleep_wakeup(monkeypatch):

    monkeypatch.setattr(
        "omni_npu.layers.fused_moe.layer.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "omni_npu.layers.fused_moe.layer.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        model_extra_config.operator_opt_config,
        "gmm_nz",
        True,
    )
    worker = _create_worker(monkeypatch)
    # Some UT containers lack TBE (compile backend for real format cast).
    if find_spec("tbe") is None:
        monkeypatch.setattr(
            torch_npu,
            "npu_format_cast",
            lambda tensor, _format: tensor,
        )

    NpuMemAllocator.instance = None
    allocator = NpuMemAllocator.get_instance()
    # CustomOp / model init requires active vLLM config.
    with set_current_vllm_config(worker.vllm_config):
        with allocator.use_memory_pool(tag="weights"):
            model = _init_model_via_gpu_model_runner_path(worker, monkeypatch)
            schema = _build_ckpt_schema_from_model(model)
            weights = _build_random_weights(schema, seed=7)
    _sleep_wake_weights_cycle(worker)
    with set_current_vllm_config(worker.vllm_config):
        with allocator.use_memory_pool(tag="weights"):
            # Path-1: initial load via model script (OpenPanguModel.load_weights) + vLLM post process.
            model.load_weights(iter(weights))
            loader_utils.process_weights_after_loading(
                model,
                SimpleNamespace(dtype=torch.bfloat16, quantization=None),
                torch.device("npu"),
            )
            first_param_layout_snapshot = _snapshot_parameters_from_model(model)
            first_value_snapshot = _snapshot_values_from_model(model)

    _sleep_wake_weights_cycle(worker)

    # Path-2: RL reload + sleep semantics.
    with set_current_vllm_config(worker.vllm_config):
        with allocator.use_memory_pool(tag="weights"):
            model.load_weights(iter(weights))
    second_param_layout_snapshot = _snapshot_parameters_from_model(model)
    second_value_snapshot = _snapshot_values_from_model(model)

    _assert_parameter_layout_equal(first_param_layout_snapshot, second_param_layout_snapshot)
    _assert_value_snapshots_equal(first_value_snapshot, second_value_snapshot)
