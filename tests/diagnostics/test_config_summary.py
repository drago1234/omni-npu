# SPDX-License-Identifier: MIT
"""Unit tests for the OMNI-CONF runtime collector (synthetic configs, no NPU).

Class names below intentionally mirror real vLLM config class names so the
three-state init=False projection tables key correctly.
"""

import dataclasses
import logging
from dataclasses import dataclass, field

import pytest

from omni_npu.diagnostics import classification as cls
from omni_npu.diagnostics import config_summary as summ


_CONFIG_SUMMARY_LOGGER = "omni_npu.diagnostics.config_summary"


class _Poison:
    """Sentinel: ANY interaction must crash the test if the walker touches it."""

    def __getattr__(self, name):  # pragma: no cover - must never run
        raise AssertionError("walker touched runtime state!")

    def __repr__(self):  # pragma: no cover - must never run
        raise AssertionError("walker repr'd runtime state!")


@dataclass
class CompilationConfig:
    mode: int = 3
    backend: str = "eager"
    cudagraph_capture_sizes: list = field(default_factory=lambda: [16])
    # runtime fields (RUNTIME_FIELD_EXCLUDE)
    static_forward_context: dict = field(default_factory=dict, init=False)
    compilation_time: float = field(default=0.0, init=False)
    # deployment vLLM 0.14.0+empty field (real-NPU finding): post_init-derived
    # cudagraph padding table, RUNTIME_FIELD_EXCLUDE
    bs_to_padded_graph_size: list = field(default_factory=list, init=False)


@dataclass
class ParallelConfig:
    tensor_parallel_size: int = 1
    data_parallel_size: int = 32
    data_parallel_rank: int = 7
    enable_expert_parallel: bool = True
    # per-worker / per-process / per-node seat numbers (init=True, IDENTITY_KEYS)
    rank: int = 0
    _api_process_rank: int = 0
    node_rank: int = 0
    world_size: int = field(default=32, init=False)        # DERIVED_FIELD_INCLUDE
    totally_new_runtime_thing: int = field(default=0, init=False)  # unclassified!


@dataclass
class SchedulerConfig:
    enable_chunked_prefill: bool = True
    max_num_batched_tokens: int = 128


class _FakeHF:
    """PretrainedConfig-like: not a dataclass, exposes to_dict()."""

    def __init__(self):
        self.model_type = "openpangu_v2"
        self.num_key_value_heads = 8
        self.hidden_size = 7168

    def to_dict(self):
        return dict(vars(self))


@dataclass
class ModelConfig:
    model: str = "/mnt/model/path"
    tokenizer_mode: str = "auto"
    hf_config: object = field(default_factory=_FakeHF)


@dataclass
class VllmConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    compilation_config: CompilationConfig = field(default_factory=CompilationConfig)
    speculative_config: object = None


@pytest.fixture(autouse=True)
def _reset_guard():
    summ.reset_once_guard()
    yield
    summ.reset_once_guard()


@pytest.fixture()
def caplog_config_summary(caplog):
    """Capture config_summary logs even when omni_npu propagation is disabled."""
    logger = logging.getLogger(_CONFIG_SUMMARY_LOGGER)
    added_handler = caplog.handler not in logger.handlers
    if added_handler:
        logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=_CONFIG_SUMMARY_LOGGER):
            yield caplog
    finally:
        if added_handler:
            logger.removeHandler(caplog.handler)


@pytest.fixture()
def cfg():
    c = VllmConfig()
    c.compilation_config.static_forward_context["layer.0"] = _Poison()
    return c


class TestProjection:
    def test_runtime_fields_never_walked(self, cfg):
        entries = summ.build_entries(cfg, scope="engine")
        assert not any("static_forward_context" in k for k in entries)
        assert not any("compilation_time" in k for k in entries)

    def test_derived_config_included(self, cfg):
        entries = summ.build_entries(cfg, scope="engine")
        assert entries["vllm.parallel.world_size"] == 32

    def test_bs_to_padded_graph_size_excluded_no_warning(
            self, cfg, caplog_config_summary):
        """Real-NPU finding: deployment vLLM exposes
        CompilationConfig.bs_to_padded_graph_size (init=False, post_init-derived
        cudagraph padding table). Must be excluded WITHOUT an unclassified
        warning (17 such warnings appeared in the real prefill log)."""
        entries = summ.build_entries(cfg, scope="engine")
        assert not any("bs_to_padded_graph_size" in k for k in entries)
        assert not any("bs_to_padded_graph_size" in r.message
                       for r in caplog_config_summary.records)

    def test_rank_seat_numbers_do_not_enter_hash(self, cfg):
        """Real-NPU finding: w0-w15 (same prefill role) had 16 distinct hashes
        solely because vllm.parallel.rank 0..15 leaked into the shared hash.
        rank / _api_process_rank / node_rank are seat numbers, not config."""
        base = summ.build_entries(cfg, scope="worker", rank=0, local_rank=0)
        base["meta.ts"] = "T"
        hashes = set()
        for r in range(16):
            e = dict(base)
            e["vllm.parallel.rank"] = r
            e["vllm.parallel._api_process_rank"] = r
            e["vllm.parallel.node_rank"] = r // 8
            e["meta.ts"] = "T"
            hashes.add(summ.compute_hash(e))
        assert len(hashes) == 1, "rank seat numbers must not change the hash"
        # but a genuine config change still must
        drift = dict(base)
        drift["vllm.scheduler.max_num_batched_tokens"] = 99999
        assert summ.compute_hash(drift) != summ.compute_hash(base)

    def test_rank_fields_are_identity(self):
        for k in ("vllm.parallel.rank", "vllm.parallel._api_process_rank",
                  "vllm.parallel.node_rank"):
            assert cls.classify_key(k) == cls.CLASS_IDENTITY, k

    def test_unclassified_init_false_warned_and_skipped(
            self, cfg, caplog_config_summary):
        entries = summ.build_entries(cfg, scope="engine")
        assert not any("totally_new_runtime_thing" in k for k in entries)
        assert any("unclassified init=False" in r.message
                   for r in caplog_config_summary.records)

    def test_no_underscore_wildcard_exemption(self, caplog_config_summary):
        """impl-review round-2 P2 minimal repro: a future `_foo` init=False
        field must trip the unclassified warning - never a silent skip via
        name-based wildcard."""
        @dataclass
        class FutureConfig:
            normal: int = 1
            _future_config_toggle: bool = field(default=False, init=False)

        @dataclass
        class Cfg:
            future_config: FutureConfig = field(default_factory=FutureConfig)

        summ.reset_once_guard()
        entries = summ.build_entries(Cfg(), scope="engine")
        assert not any("_future_config_toggle" in k for k in entries)
        assert any("FutureConfig._future_config_toggle" in r.message
                   for r in caplog_config_summary.records)

    def test_hf_expanded_not_repr(self, cfg):
        entries = summ.build_entries(cfg, scope="engine")
        assert entries["model.hf.num_key_value_heads"] == 8
        assert entries["model.hf.model_type"] == "openpangu_v2"
        # never a repr-degraded blob
        assert "model.hf" not in entries

    def test_named_configs_present(self, cfg):
        entries = summ.build_entries(cfg, scope="engine")
        assert entries["vllm.scheduler.enable_chunked_prefill"] is True
        assert entries["vllm.scheduler.max_num_batched_tokens"] == 128
        assert entries["vllm.parallel.tensor_parallel_size"] == 1
        assert entries["vllm.speculative"] is None

    def test_audit_all_init_false_classified_in_fakes(self):
        """Mini schema audit over the synthetic classes (real one runs on CI
        with vllm importable; see test_schema_audit below)."""
        unclassified = []
        for klass in (CompilationConfig, ParallelConfig):
            for f in dataclasses.fields(klass):
                if f.init:
                    continue
                fq = f"{klass.__name__}.{f.name}"
                if fq not in cls.DERIVED_FIELD_INCLUDE and fq not in cls.RUNTIME_FIELD_EXCLUDE:
                    unclassified.append(fq)
        assert unclassified == ["ParallelConfig.totally_new_runtime_thing"]


class TestMaskAndHash:
    def test_env_credentials_masked(self, cfg, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
        monkeypatch.setenv("ENABLE_OMNI_CACHE", "1")
        entries = summ.build_entries(cfg, scope="engine")
        joined = "\n".join(summ.canonical_lines(entries))
        assert "sk-secret" not in joined
        assert 'env.OPENAI_API_KEY="***"' in joined
        assert 'env.ENABLE_OMNI_CACHE="1"' in joined  # real config survives

    def test_render_masks_credentials_at_value_layer(self):
        # codex round-5 leak pair: masked regardless of collection source
        # (locally vllm.envs is unavailable; on NPU/CI these ARE collected)
        assert summ._render_value("env.S3_SECRET_ACCESS_KEY", "sssh") == '"***"'
        assert summ._render_value("env.S3_ACCESS_KEY_ID", "AKIA") == '"***"'
        assert summ._render_value("env.VLLM_API_KEY", "k") == '"***"'

    def test_requirement_targets_never_masked(self, cfg):
        lines = "\n".join(summ.canonical_lines(summ.build_entries(cfg, scope="engine")))
        assert "model.hf.num_key_value_heads=8" in lines
        assert "vllm.scheduler.max_num_batched_tokens=128" in lines
        assert 'vllm.model.tokenizer_mode="auto"' in lines

    def test_hash_ignores_identity_but_not_shared(self, cfg):
        e1 = summ.build_entries(cfg, scope="worker", rank=0, local_rank=0)
        e2 = summ.build_entries(cfg, scope="worker", rank=7, local_rank=7)
        # strip volatile meta noise that differs run-to-run (ts/pid identical here)
        for e in (e1, e2):
            e["meta.ts"] = "T"
        assert summ.compute_hash(e1) == summ.compute_hash(e2)
        cfg2 = VllmConfig()
        cfg2.scheduler_config.max_num_batched_tokens = 256
        e3 = summ.build_entries(cfg2, scope="worker", rank=0, local_rank=0)
        e3["meta.ts"] = "T"
        assert summ.compute_hash(e3) != summ.compute_hash(e1)

    def test_data_parallel_rank_is_identity(self):
        assert cls.classify_key("vllm.parallel.data_parallel_rank") == cls.CLASS_IDENTITY


class _ListLogger:
    def __init__(self):
        self.lines = []

    def info(self, fmt, *args):
        self.lines.append(fmt % args if args else fmt)

    def warning(self, *a, **k):
        self.lines.append(f"WARNING:{a}")


class TestEmit:
    def test_emit_format_and_once_guard(self, cfg):
        log = _ListLogger()
        assert summ.emit_config_summary(cfg, scope="worker", rank=0, local_rank=0,
                                        log=log) is True
        assert log.lines[0].startswith("[OMNI-CONF:w0] #begin scope=worker")
        assert log.lines[-1].startswith("[OMNI-CONF:w0] #end n=")
        n = int(log.lines[-1].split("n=")[1].split()[0])
        assert n == len(log.lines) - 2
        assert "config_hash=sha256:" in log.lines[-1]
        # every payload line is single, marker-prefixed, k=v
        for ln in log.lines[1:-1]:
            assert ln.startswith("[OMNI-CONF:w0] ") and "=" in ln
        # once-guard
        assert summ.emit_config_summary(cfg, scope="worker", rank=0, log=log) is False

    def test_hash_only_single_line(self, cfg):
        log = _ListLogger()
        assert summ.emit_config_summary(cfg, scope="worker", rank=3, local_rank=1,
                                        hash_only=True, log=log) is True
        assert len(log.lines) == 1
        assert "#hashonly" in log.lines[0] and "config_hash=sha256:" in log.lines[0]

    def test_disabled_by_env(self, cfg, monkeypatch):
        monkeypatch.setenv("OMNI_CONFIG_SUMMARY", "0")
        log = _ListLogger()
        assert summ.emit_config_summary(cfg, scope="engine", log=log) is False
        assert log.lines == []

    def test_never_crashes_on_poisoned_config(self):
        class Exploding:
            @property
            def model_config(self):
                raise RuntimeError("boom")

        log = _ListLogger()
        # must not raise, degrades to warning
        summ.emit_config_summary(Exploding(), scope="engine", log=log)


def test_ast_audit_omni_custom_configs():
    """AST-level init=False audit over omni-npu's OWN config sources
    (no imports needed -> always runs, even without torch). Catches e.g.
    the custom ReasoningConfig._tool_call_start_token_ids (impl-review r3)."""
    import ast
    import pathlib

    src_root = pathlib.Path(__file__).parent.parent.parent / "src" / "omni_npu"
    unclassified = []
    for py in (src_root / "v1" / "config").glob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if not (isinstance(stmt, ast.AnnAssign) and stmt.value is not None
                        and isinstance(stmt.target, ast.Name)):
                    continue
                if "init=False" not in ast.unparse(stmt.value):
                    continue
                fq = f"{node.name}.{stmt.target.id}"
                if (fq not in cls.DERIVED_FIELD_INCLUDE
                        and fq not in cls.RUNTIME_FIELD_EXCLUDE
                        and fq not in cls.SPECIAL_FIELD_HANDLED):
                    unclassified.append(f"{fq} ({py.name})")
    assert not unclassified, (
        f"unclassified init=False fields in omni-npu configs: {unclassified}")


def test_schema_audit_real_vllm():
    """Full init=False audit over real vLLM configs (runs where vllm imports;
    auto-skips on boxes without vllm/torch - covered on NPU CI)."""
    pytest.importorskip("vllm.config")
    import vllm.config as vc

    unclassified = []
    for name in dir(vc):
        obj = getattr(vc, name)
        if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
            continue
        for f in dataclasses.fields(obj):
            if f.init:
                continue
            fq = f"{obj.__name__}.{f.name}"
            if (fq not in cls.DERIVED_FIELD_INCLUDE
                    and fq not in cls.RUNTIME_FIELD_EXCLUDE
                    and fq not in cls.SPECIAL_FIELD_HANDLED):
                unclassified.append(fq)
    assert not unclassified, (
        f"unclassified init=False fields (add to DERIVED_FIELD_INCLUDE or "
        f"RUNTIME_FIELD_EXCLUDE): {sorted(set(unclassified))}")
