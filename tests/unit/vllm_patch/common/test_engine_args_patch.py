import argparse
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from vllm.utils.argparse_utils import FlexibleArgumentParser

from omni_npu.vllm_patches.patch_manager import PatchManager
from omni_npu.vllm_patches.patches.common import patch_args_utils
from omni_npu.vllm_patches.patches.common import patch_eplb_parallel


@dataclass
class _DummyEngineArgs:
    model: str = "dummy-model"
    reasoning_config: object | None = None
    routed_experts_serialization_mode: str = (
        patch_args_utils.DEFAULT_ROUTED_EXPERTS_SERIALIZATION_MODE
    )


@pytest.mark.unit
def test_engine_args_patch_add_cli_args_registers_serialization_mode(monkeypatch):
    monkeypatch.setattr(
        patch_args_utils,
        "_original_add_cli_args",
        lambda parser: parser,
    )
    parser = FlexibleArgumentParser(prog="test")

    parser = patch_args_utils.EngineArgsPatch.add_cli_args(parser)

    action = parser._option_string_actions["--routed-experts-serialization-mode"]
    assert action.default == (
        patch_args_utils.DEFAULT_ROUTED_EXPERTS_SERIALIZATION_MODE
    )
    assert (
        tuple(action.choices)
        == patch_args_utils.ROUTED_EXPERTS_SERIALIZATION_MODES
    )

    args = parser.parse_args(
        ["--routed-experts-serialization-mode", "base64"]
    )
    assert args.routed_experts_serialization_mode == "base64"


@pytest.mark.unit
def test_engine_args_patch_from_cli_args_delegates_and_parses_overrides(
    monkeypatch,
):
    captured = {}

    def _fake_original_from_cli_args(cls, parsed_args):
        captured["cls"] = cls
        captured["args"] = parsed_args
        return cls(
            model="from-original",
            reasoning_config="from-original",
            routed_experts_serialization_mode="zip_base64",
        )

    monkeypatch.setattr(
        patch_args_utils,
        "_original_from_cli_args",
        _fake_original_from_cli_args,
    )

    args = argparse.Namespace(
        model="dummy-model",
        reasoning_config=(
            '{"think_start_str":"<think>","think_end_str":"</think>"}'
        ),
        routed_experts_serialization_mode="base64",
    )

    instance = patch_args_utils.EngineArgsPatch.from_cli_args.__func__(
        _DummyEngineArgs,
        args,
    )

    assert captured["cls"] is _DummyEngineArgs
    assert captured["args"] is args
    assert instance.model == "from-original"
    assert instance.routed_experts_serialization_mode == "base64"
    assert instance.reasoning_config is not None
    assert instance.reasoning_config.think_start_str == "<think>"
    assert instance.reasoning_config.think_end_str == "</think>"


@pytest.mark.unit
def test_engine_args_patch_create_engine_config_applies_bridge_attrs(monkeypatch):
    captured = {}
    model_config = SimpleNamespace()
    vllm_config = SimpleNamespace(model_config=model_config)

    def _fake_create_engine_config(self, usage_context, headless):
        captured["usage_context"] = usage_context
        captured["headless"] = headless
        return vllm_config

    monkeypatch.setattr(
        patch_args_utils,
        "_original_create_engine_config",
        _fake_create_engine_config,
    )

    reasoning_config = SimpleNamespace(
        initialize_token_ids=lambda config: captured.update(
            {"initialized_model_config": config}
        )
    )
    engine_args = SimpleNamespace(
        reasoning_config=reasoning_config,
        routed_experts_serialization_mode="base64",
        _omni_wrap_create_engine_config=lambda builder: builder(),
    )

    result = patch_args_utils.EngineArgsPatch.create_engine_config(
        engine_args,
        usage_context="usage",
        headless=True,
    )

    assert result is vllm_config
    assert result.reasoning_config is reasoning_config
    assert result.routed_experts_serialization_mode == "base64"
    assert captured["usage_context"] == "usage"
    assert captured["headless"] is True
    assert captured["initialized_model_config"] is model_config


@pytest.mark.unit
def test_eplb_engine_config_wraps_bridge_builder_without_extra_attrs(monkeypatch):
    import vllm.platforms as vllm_platforms

    captured = {}
    original_is_cuda_alike = lambda: False
    fake_platform = SimpleNamespace(
        device_type="npu",
        is_cuda_alike=original_is_cuda_alike,
    )
    monkeypatch.setattr(vllm_platforms, "current_platform", fake_platform)

    def _fake_create_engine_config(self, usage_context, headless):
        captured["cuda_alike_during_builder"] = fake_platform.is_cuda_alike()
        return SimpleNamespace(model_config=None)

    monkeypatch.setattr(
        patch_args_utils,
        "_original_create_engine_config",
        _fake_create_engine_config,
    )

    engine_args = SimpleNamespace(
        enable_eplb=True,
        _omni_wrap_create_engine_config=lambda builder: (
            patch_args_utils.EngineArgsPatch._omni_wrap_create_engine_config(
                engine_args,
                builder,
            )
        ),
    )

    result = patch_args_utils.EngineArgsPatch.create_engine_config(
        engine_args
    )

    assert captured["cuda_alike_during_builder"] is True
    assert fake_platform.is_cuda_alike is original_is_cuda_alike
    assert result.routed_experts_serialization_mode == (
        patch_args_utils.DEFAULT_ROUTED_EXPERTS_SERIALIZATION_MODE
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("enable_eplb", "device_type"),
    [
        (False, "npu"),
        (True, "cuda"),
    ],
)
def test_bridge_engine_config_wrap_leaves_non_eplb_or_non_npu_unchanged(
    monkeypatch,
    enable_eplb,
    device_type,
):
    import vllm.platforms as vllm_platforms

    captured = {}
    original_is_cuda_alike = lambda: False
    fake_platform = SimpleNamespace(
        device_type=device_type,
        is_cuda_alike=original_is_cuda_alike,
    )
    monkeypatch.setattr(vllm_platforms, "current_platform", fake_platform)

    def _builder():
        captured["cuda_alike_during_builder"] = fake_platform.is_cuda_alike()
        return "ok"

    engine_args = SimpleNamespace(enable_eplb=enable_eplb)

    result = patch_args_utils.EngineArgsPatch._omni_wrap_create_engine_config(
        engine_args,
        _builder,
    )

    assert result == "ok"
    assert captured["cuda_alike_during_builder"] is False
    assert fake_platform.is_cuda_alike is original_is_cuda_alike


@pytest.mark.unit
def test_engine_args_and_eplb_shared_fused_moe_patches_apply_without_conflict(
    monkeypatch,
):
    class _DummyEngineArgsTarget:
        pass

    class _DummySharedFusedMoETarget:
        pass

    monkeypatch.setattr(
        patch_args_utils.EngineArgsPatch,
        "_target",
        _DummyEngineArgsTarget,
    )
    monkeypatch.setattr(
        patch_eplb_parallel.SharedFusedMoEPatch,
        "_target",
        _DummySharedFusedMoETarget,
    )

    patch_args_utils.EngineArgsPatch.apply()
    patch_eplb_parallel.SharedFusedMoEPatch.apply()

    assert _DummyEngineArgsTarget._omni_npu_applied_patches[
        "_omni_wrap_create_engine_config"
    ] == "EngineArgsPatch"
    assert _DummySharedFusedMoETarget._omni_npu_applied_patches["__init__"] == (
        "SharedFusedMoEPatch"
    )
    assert "EPLBEngineConfig" not in PatchManager.registered_patches
