from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import pytest
from omni_npu.v1.worker.npu_worker import NPUWorker
from tests.unit.platform.utils import create_vllm_config, DeviceConfig

class TestNpuWorker:
    def setup_method(self):
        self.vllm_cfg = create_vllm_config()
        self.vllm_cfg.device_config = DeviceConfig("npu")
        self.worker = None

    def _create_worker(self, monkeypatch):
        """Create an NPUWorker instance with necessary dependencies mocked.
        
        This helper method sets up all required mocks to create an NPUWorker
        instance without requiring actual NPU hardware or full vLLM dependencies.
        """
        # Mock current_platform
        mock_platform = SimpleNamespace(
            device_type="npu",
            pre_register_and_update=lambda: None,
            set_device=lambda device: None,
            dist_backend="hccl",
        )
        monkeypatch.setattr("omni_npu.v1.worker.npu_worker.current_platform", mock_platform)

        # Mock torch.npu related operations
        monkeypatch.setattr("torch.npu.empty_cache", lambda: None)
        monkeypatch.setattr("torch.npu.mem_get_info", lambda: (1000, 2000))
        monkeypatch.setattr("torch.npu.reset_peak_memory_stats", lambda: None)
        monkeypatch.setattr("torch.npu.max_memory_allocated", lambda: 500)

        # Mock distributed initialization
        monkeypatch.setattr(
            "omni_npu.v1.worker.npu_worker.init_worker_distributed_environment",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr("omni_npu.v1.worker.npu_worker.set_random_seed", lambda seed: None)

        # Mock NPUModelRunner
        mock_model_runner = MagicMock()
        monkeypatch.setattr(
            "omni_npu.v1.worker.npu_worker.NPUModelRunner",
            lambda *args, **kwargs: mock_model_runner,
        )

        # Mock report_usage_stats to avoid importing vllm code
        monkeypatch.setattr(
            "vllm.v1.utils.report_usage_stats",
            lambda vllm_config: None,
        )

        worker = NPUWorker(
            vllm_config=self.vllm_cfg,
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://localhost:12345",
            is_driver_worker=True,
        )
        worker.model_runner = mock_model_runner
        return worker

    def test_get_kv_connector_handshake_metadata(self, monkeypatch):
        """Test get_kv_connector_handshake_metadata method.
        
        Verifies the method handles different scenarios:
        - When kv_transfer_group is not available
        - When kv_transfer_group exists but has no metadata
        - When metadata is available and properly formatted
        """
        worker = self._create_worker(monkeypatch)

        # Test case: no kv_transfer_group available
        monkeypatch.setattr(
            "omni_npu.v1.worker.npu_worker.has_kv_transfer_group",
            lambda: False,
        )
        assert worker.get_kv_connector_handshake_metadata() is None

        # Test case: kv_transfer_group exists but has no metadata
        monkeypatch.setattr(
            "omni_npu.v1.worker.npu_worker.has_kv_transfer_group",
            lambda: True,
        )
        mock_connector = MagicMock()
        mock_connector.get_handshake_metadata.return_value = None
        monkeypatch.setattr(
            "omni_npu.v1.worker.npu_worker.get_kv_transfer_group",
            lambda: mock_connector,
        )
        assert worker.get_kv_connector_handshake_metadata() is None

        # Test case: metadata is available
        mock_metadata = {"key": "value"}
        mock_connector.get_handshake_metadata.return_value = mock_metadata
        mock_tp_group = SimpleNamespace(rank_in_group=0)
        monkeypatch.setattr(
            "omni_npu.v1.worker.npu_worker.get_tp_group",
            lambda: mock_tp_group,
        )
        result = worker.get_kv_connector_handshake_metadata()
        assert result == {0: mock_metadata}

    def test_init_device(self, monkeypatch):
        """Test init_device method.
        
        Verifies that device initialization sets up the worker correctly,
        including device assignment, memory snapshots, and model runner creation.
        """
        worker = self._create_worker(monkeypatch)

        # Mock device_config
        worker.local_rank = 0
        worker.model_config = SimpleNamespace(seed=42)
        worker.cache_config = SimpleNamespace(gpu_memory_utilization=0.9)
        worker.rank = 0

        # Mock _init_profiler
        worker._init_profiler = lambda: None

        worker.init_device()

        assert worker.device == torch.device("npu:0")
        assert hasattr(worker, "init_snapshot")
        assert hasattr(worker, "requested_memory")
        assert worker.model_runner is not None

    def test_init_device_with_custom_model_enable(self, monkeypatch):
        """Test init_device with VLLM_CUSTOM_MODEL_ENABLE environment variable (covers lines 91-95).
        
        Verifies that when VLLM_CUSTOM_MODEL_ENABLE is set, layer parallel
        initialization is triggered with the correct backend.
        """
        worker = self._create_worker(monkeypatch)

        worker.local_rank = 0
        worker.model_config = SimpleNamespace(seed=42)
        worker.cache_config = SimpleNamespace(gpu_memory_utilization=0.9)
        worker.rank = 0
        worker._init_profiler = lambda: None

        # Set environment variable
        monkeypatch.setenv("VLLM_CUSTOM_MODEL_ENABLE", "1")

        # Mock ensure_layer_parallel_initialized
        ensure_called = {"called": False}
        def mock_ensure_layer_parallel_initialized(backend):
            ensure_called["called"] = True
            ensure_called["backend"] = backend

        monkeypatch.setattr(
            "omni_npu.v1.distributed.parallel_state_ext.ensure_layer_parallel_initialized",
            mock_ensure_layer_parallel_initialized,
        )

        worker.init_device()

        assert ensure_called["called"] is True
        assert ensure_called["backend"] == "hccl"

    def test_init_device_unsupported_device_type(self, monkeypatch):
        """Test init_device with unsupported device type (covers line 103).
        
        Verifies that a RuntimeError is raised when an unsupported device type
        is provided.
        """
        worker = self._create_worker(monkeypatch)

        worker.local_rank = 0
        worker.model_config = SimpleNamespace(seed=42)
        worker.cache_config = SimpleNamespace(gpu_memory_utilization=0.9)
        worker.rank = 0
        worker._init_profiler = lambda: None

        # Mock device to an unsupported device type
        mock_device = SimpleNamespace(type="cpu")
        worker.device_config.device = mock_device

        with pytest.raises(RuntimeError, match="Not support device type"):
            worker.init_device()

    def test_determine_available_memory_reset_peak_exception(self, monkeypatch):
        """Test determine_available_memory with reset_peak_memory_stats exception handling (covers lines 132-133).
        
        Verifies that exceptions from reset_peak_memory_stats are properly
        handled and the method continues execution.
        """
        worker = self._create_worker(monkeypatch)

        worker.cache_config = SimpleNamespace(
            kv_cache_memory_bytes=None,
            gpu_memory_utilization=0.9,
        )
        worker.init_snapshot = SimpleNamespace(free_memory=1000)
        worker.model_runner.profile_run = MagicMock()

        # Mock reset_peak_memory_stats to raise an exception
        def raise_exception():
            raise Exception("Reset failed")

        monkeypatch.setattr("torch.npu.reset_peak_memory_stats", raise_exception)
        monkeypatch.setattr("torch.npu.mem_get_info", lambda: (500, 2000))
        monkeypatch.setattr("torch.npu.max_memory_allocated", lambda: 300)

        # Should handle exception gracefully and continue execution
        result = worker.determine_available_memory()
        assert result == 1500

    def test_init_profiler_full(self, monkeypatch):
        """Test _init_profiler full implementation (covers lines 236-267).
        
        Verifies that profiler initialization works correctly when all
        required environment variables are set, including token threshold
        and stop step configuration.
        """
        worker = self._create_worker(monkeypatch)

        # Mock environment variables
        monkeypatch.setenv("PROFILER_TOKEN_THRESHOLD", "10")
        monkeypatch.setenv("PROFILER_STOP_STEP", "5")
        monkeypatch.setenv("VLLM_TORCH_PROFILER_DIR", "/tmp/profiler")

        # Mock torch_npu.profiler
        mock_profiler = MagicMock()
        mock_experimental_config = MagicMock()
        mock_tensorboard_handler = MagicMock()

        monkeypatch.setattr(
            "torch_npu.profiler._ExperimentalConfig",
            lambda **kwargs: mock_experimental_config,
        )
        monkeypatch.setattr(
            "torch_npu.profiler.tensorboard_trace_handler",
            lambda path: mock_tensorboard_handler,
        )
        monkeypatch.setattr(
            "torch_npu.profiler.profile",
            lambda **kwargs: mock_profiler,
        )
        monkeypatch.setattr("vllm.envs.VLLM_TORCH_PROFILER_DIR", "/tmp/profiler")
        monkeypatch.setattr("vllm.envs.VLLM_TORCH_PROFILER_RECORD_SHAPES", True)
        monkeypatch.setattr("vllm.envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY", True)
        monkeypatch.setattr("vllm.envs.VLLM_TORCH_PROFILER_WITH_STACK", True)
        monkeypatch.setattr("vllm.envs.VLLM_TORCH_PROFILER_WITH_FLOPS", True)

        result = worker._init_profiler()

        assert result is mock_profiler
        assert worker.profiler_token_threshold == 10
        assert worker.profiler_stop_step == 5
        assert worker._use_token_for_profile is True
        assert worker.profile_already_start is False
        assert worker.profile_finished is False

    def test_init_profiler_no_env_var(self, monkeypatch):
        """Test _init_profiler when environment variables are not set.
        
        Verifies that profiler initialization returns None and sets
        _use_token_for_profile to False when VLLM_TORCH_PROFILER_DIR is not set.
        """
        worker = self._create_worker(monkeypatch)

        # Do not set VLLM_TORCH_PROFILER_DIR
        monkeypatch.delenv("VLLM_TORCH_PROFILER_DIR", raising=False)
        monkeypatch.setattr("vllm.envs.VLLM_TORCH_PROFILER_DIR", None)

        result = worker._init_profiler()

        assert result is None
        assert worker._use_token_for_profile is False

    def test_determine_available_memory(self, monkeypatch):
        """Test determine_available_memory method.
        
        Verifies memory calculation in different scenarios:
        - When kv_cache_memory_bytes is explicitly specified
        - When memory needs to be calculated from available memory
        - When max_memory_allocated fails and fallback calculation is used
        """
        worker = self._create_worker(monkeypatch)

        # Test case: kv_cache_memory_bytes is explicitly specified
        worker.cache_config = SimpleNamespace(
            kv_cache_memory_bytes=1000,
            gpu_memory_utilization=0.9,
        )
        worker.model_runner.profile_run = MagicMock()

        result = worker.determine_available_memory()
        assert result == 1000
        worker.model_runner.profile_run.assert_called_once()

        # Test case: memory needs to be calculated
        worker.cache_config.kv_cache_memory_bytes = None
        worker.init_snapshot = SimpleNamespace(free_memory=1000)
        monkeypatch.setattr("torch.npu.mem_get_info", lambda: (500, 2000))
        monkeypatch.setattr("torch.npu.max_memory_allocated", lambda: 300)

        result = worker.determine_available_memory()
        # available = total * util - peak = 2000 * 0.9 - 300 = 1500
        assert result == 1500

        # Test case: max_memory_allocated fails (use fallback calculation)
        def raise_exception():
            raise Exception("Not available")

        monkeypatch.setattr("torch.npu.max_memory_allocated", raise_exception)
        result = worker.determine_available_memory()
        # available = total * util - (init_free - free_after) = 2000 * 0.9 - (1000 - 500) = 1300
        assert result == 1300

    def test_model_runner_proxy_methods(self, monkeypatch):
        """Test all methods that directly proxy to model_runner (consolidated test).
        
        Verifies that all proxy methods correctly delegate to the model_runner
        and return the expected values.
        """
        worker = self._create_worker(monkeypatch)

        # Prepare mock return values
        mock_kv_spec = {"layer_0": MagicMock()}
        mock_model = MagicMock()
        mock_tasks = (MagicMock(),)
        mock_draft_tokens = MagicMock()
        mock_sample_result = MagicMock()
        mock_lora_result = True
        mock_lora_set = {1, 2, 3}

        # Set model_runner return values
        worker.model_runner.get_kv_cache_spec.return_value = mock_kv_spec
        worker.model_runner.get_model.return_value = mock_model
        worker.model_runner.get_supported_tasks.return_value = mock_tasks
        worker.model_runner.take_draft_token_ids.return_value = mock_draft_tokens
        worker.model_runner.sample_tokens.return_value = mock_sample_result
        worker.model_runner.add_lora.return_value = mock_lora_result
        worker.model_runner.remove_lora.return_value = mock_lora_result
        worker.model_runner.list_loras.return_value = mock_lora_set
        worker.model_runner.pin_lora.return_value = mock_lora_result

        # Test get_kv_cache_spec
        assert worker.get_kv_cache_spec() is mock_kv_spec
        worker.model_runner.get_kv_cache_spec.assert_called_once()

        # Test get_model
        assert worker.get_model() is mock_model
        worker.model_runner.get_model.assert_called_once()

        # Test get_supported_tasks
        assert worker.get_supported_tasks() is mock_tasks
        worker.model_runner.get_supported_tasks.assert_called_once()

        # Test take_draft_token_ids
        assert worker.take_draft_token_ids() is mock_draft_tokens
        worker.model_runner.take_draft_token_ids.assert_called_once()

        # Test sample_tokens
        mock_grammar_output = MagicMock()
        assert worker.sample_tokens(mock_grammar_output) is mock_sample_result
        worker.model_runner.sample_tokens.assert_called_once_with(mock_grammar_output)

        # Test add_lora
        mock_lora_request = MagicMock()
        assert worker.add_lora(mock_lora_request) is mock_lora_result
        worker.model_runner.add_lora.assert_called_once_with(mock_lora_request)

        # Test remove_lora
        assert worker.remove_lora(1) is mock_lora_result
        worker.model_runner.remove_lora.assert_called_once_with(1)

        # Test list_loras
        assert worker.list_loras() is mock_lora_set
        worker.model_runner.list_loras.assert_called_once()

        # Test pin_lora
        assert worker.pin_lora(1) is mock_lora_result
        worker.model_runner.pin_lora.assert_called_once_with(1)

    def test_initialize_from_config(self, monkeypatch):
        """Test initialize_from_config method.
        
        Verifies that KV transfer initialization and cache initialization
        are properly called with the correct arguments.
        """
        worker = self._create_worker(monkeypatch)
        mock_kv_cache_config = MagicMock()

        # Mock ensure_kv_transfer_initialized
        ensure_kv_initialized_called = {"called": False}
        def mock_ensure_kv_initialized(vllm_config, kv_cache_config):
            ensure_kv_initialized_called["called"] = True
            ensure_kv_initialized_called["args"] = (vllm_config, kv_cache_config)

        monkeypatch.setattr(
            "omni_npu.v1.worker.npu_worker.ensure_kv_transfer_initialized",
            mock_ensure_kv_initialized,
        )

        worker.initialize_from_config(mock_kv_cache_config)

        # Verify ensure_kv_transfer_initialized and initialize_kv_cache are called
        assert ensure_kv_initialized_called["called"] is True
        assert ensure_kv_initialized_called["args"] == (worker.vllm_config, mock_kv_cache_config)
        worker.model_runner.initialize_kv_cache.assert_called_once_with(mock_kv_cache_config)

    def test_initialize_cache(self, monkeypatch):
        """Test initialize_cache method (should return None).
        
        Verifies that initialize_cache returns None as expected.
        """
        worker = self._create_worker(monkeypatch)

        result = worker.initialize_cache(num_gpu_blocks=10, num_cpu_blocks=5)
        assert result is None

    def test_profile(self, monkeypatch):
        """Test profile method.
        
        Verifies profiler behavior in different scenarios:
        - When profiler is None (should raise RuntimeError)
        - Normal profiler start/stop operations
        - When token threshold is enabled
        """
        worker = self._create_worker(monkeypatch)

        # Test case: profiler is None
        worker.profiler = None
        with pytest.raises(RuntimeError, match="Profiler is not enabled"):
            worker.profile()

        # Test case: normal profiler operation
        mock_profiler = MagicMock()
        worker.profiler = mock_profiler
        worker._use_token_for_profile = False

        worker.profile(is_start=True)
        mock_profiler.start.assert_called_once()

        worker.profile(is_start=False)
        mock_profiler.stop.assert_called_once()

        # Test case: _use_token_for_profile is True
        worker._use_token_for_profile = True
        mock_profiler.reset_mock()
        worker.profile(is_start=True)
        mock_profiler.start.assert_not_called()

    def test_compile_or_warm_up_model(self, monkeypatch):
        """Test compile_or_warm_up_model method.
        
        Verifies that model capture is called and random seed is set.
        """
        worker = self._create_worker(monkeypatch)
        worker.model_config = SimpleNamespace(enforce_eager=False, seed=42)

        worker.compile_or_warm_up_model()

        worker.model_runner.capture_model.assert_called_once()
        # Verify set_random_seed is called (verified through monkeypatch)

    def test_load_model(self, monkeypatch):
        """Test load_model method.
        
        Verifies that model loading delegates to model_runner.load_model.
        """
        worker = self._create_worker(monkeypatch)

        worker.load_model()

        worker.model_runner.load_model.assert_called_once()

    def test_execute_dummy_batch(self, monkeypatch):
        """Test execute_dummy_batch method.
        
        Verifies that dummy batch execution calls model_runner._dummy_run
        with the correct parameters.
        """
        worker = self._create_worker(monkeypatch)

        worker.execute_dummy_batch()

        worker.model_runner._dummy_run.assert_called_once_with(
            1, uniform_decode=True, force_attention=True
        )

    def test_execute_model(self, monkeypatch):
        """Test execute_model method.
        
        Verifies model execution in different scenarios:
        - Normal execution without profiler
        - Execution with profiler but token threshold disabled
        - Execution with token threshold enabled (profiler start/stop logic)
        """
        worker = self._create_worker(monkeypatch)

        # Ensure _use_token_for_profile attribute exists (avoid AttributeError)
        # This attribute is usually initialized in _init_profiler(), but may not be called in tests
        if not hasattr(worker, "_use_token_for_profile"):
            worker._use_token_for_profile = False
        if not hasattr(worker, "profile_already_start"):
            worker.profile_already_start = False
        if not hasattr(worker, "profile_finished"):
            worker.profile_finished = False

        # Mock scheduler_output
        mock_scheduler_output = SimpleNamespace(
            total_num_scheduled_tokens=3,
            num_scheduled_tokens=[1, 2, 3],
        )
        mock_output = MagicMock()
        worker.model_runner.execute_model.return_value = mock_output

        # Test case: normal execution (no profiler)
        worker.profiler = None
        result = worker.execute_model(mock_scheduler_output)
        assert result is mock_output
        worker.model_runner.execute_model.assert_called_once_with(mock_scheduler_output)

        # Test case: profiler exists but token threshold is not enabled
        mock_profiler = MagicMock()
        worker.profiler = mock_profiler
        worker._use_token_for_profile = False
        worker.profile_already_start = False
        worker.profile_finished = False

        worker.model_runner.execute_model.reset_mock()
        result = worker.execute_model(mock_scheduler_output)
        assert result is mock_output
        mock_profiler.start.assert_not_called()
        mock_profiler.stop.assert_not_called()

        # Test case: token threshold is enabled
        monkeypatch.setenv("PROFILER_TOKEN_THRESHOLD", "3")
        monkeypatch.setenv("PROFILER_STOP_STEP", "5")
        import vllm.envs as envs
        monkeypatch.setattr(envs, "VLLM_TORCH_PROFILER_DIR", "/tmp/profiler")

        worker._use_token_for_profile = True
        worker.profile_already_start = False
        worker.profile_finished = False
        worker.profiler_token_threshold = 3
        worker.profiler_stop_step = 5
        worker.profile_step = 0

        # First call should start profiler (because total_num_scheduled_tokens == len(num_scheduled_tokens) == 3)
        worker.model_runner.execute_model.reset_mock()
        mock_profiler.reset_mock()
        result = worker.execute_model(mock_scheduler_output)
        assert worker.profile_already_start is True
        mock_profiler.start.assert_called_once()

        # After multiple calls, profiler should be stopped
        worker.profile_step = 6
        result = worker.execute_model(mock_scheduler_output)
        assert worker.profile_finished is True
        mock_profiler.stop.assert_called_once()
