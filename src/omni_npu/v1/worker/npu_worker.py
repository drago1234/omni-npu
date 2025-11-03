# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional, Union

import torch
import torch.distributed

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.tasks import SupportedTask
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ModelRunnerOutput,
)
from vllm.v1.worker.worker_base import WorkerBase
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment

from .npu_model_runner import NPUModelRunner

logger = init_logger(__name__)


class NPUWorker(WorkerBase):
    """An NPU worker class using torch_npu and HCCL backend."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        device_config = self.device_config
        assert device_config.device_type == "npu"
        assert current_platform.device_type == "npu"

        # Torch profiler (optional) using NPU activities if enabled via env
        if envs.VLLM_TORCH_PROFILER_DIR:
            trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s", trace_dir)
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    getattr(torch.profiler.ProfilerActivity, "NPU", torch.profiler.ProfilerActivity.CPU),
                ],
                record_shapes=envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                profile_memory=envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                with_stack=envs.VLLM_TORCH_PROFILER_WITH_STACK,
                with_flops=envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir, use_gzip=True),
            )
        else:
            self.profiler = None

    def init_device(self):
        if self.device_config.device.type == "npu" and current_platform.device_type == "npu":
            self.device = torch.device(f"npu:{self.local_rank}")
            current_platform.set_device(self.device)
            torch.npu.empty_cache()
            # Initialize distributed before measuring memory
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                getattr(current_platform, "dist_backend", "hccl"),
            )
            # Set random seed
            set_random_seed(self.model_config.seed)
            # Snapshot available memory
            free, total = torch.npu.mem_get_info()
            self.init_snapshot = type("_Snap", (), {"free_memory": free, "total_memory": total})()
            self.requested_memory = total * self.cache_config.gpu_memory_utilization
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Construct the model runner
        self.model_runner = NPUModelRunner(self.vllm_config, self.device)  # type: ignore

        if self.rank == 0:
            from vllm.v1.utils import report_usage_stats
            report_usage_stats(self.vllm_config)

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profile to determine memory available for KV cache on NPU."""
        def GiB(b):
            return b / (1 << 30)

        if self.cache_config.kv_cache_memory_bytes:
            # still do compile/profile run to initialize kernels
            self.model_runner.profile_run()
            logger.info(
                "Reserved %.2f GiB for KV cache as specified; skipping profiling.",
                GiB(self.cache_config.kv_cache_memory_bytes),
            )
            return self.cache_config.kv_cache_memory_bytes

        torch.npu.empty_cache()
        try:
            torch.npu.reset_peak_memory_stats()
        except Exception:
            pass

        # Profile run compiles and warms kernels
        self.model_runner.profile_run()

        free_after, total = torch.npu.mem_get_info()
        try:
            peak = torch.npu.max_memory_allocated()
        except Exception:
            # Fallback: estimate by delta from init snapshot
            peak = max(0, self.init_snapshot.free_memory - free_after)

        available = int(total * self.cache_config.gpu_memory_utilization - peak)
        logger.info(
            "Available KV cache memory: %.2f GiB (total=%.2f, util=%.2f, peak=%.2f)",
            GiB(available), GiB(total), self.cache_config.gpu_memory_utilization, GiB(peak)
        )
        return max(available, 0)

    def get_kv_cache_spec(self):
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config) -> None:
        # Allocate KV cache on NPU according to provided config
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        # NOP: KV caches are fully initialized in initialize_from_config.
        # vLLM calls this with (num_gpu_blocks, num_cpu_blocks);
        # for NPU we don't need additional allocation here.
        return None

    def compile_or_warm_up_model(self) -> None:
        # Optional: call capture/compile path if supported in shim
        try:
            self.model_runner.capture_model()
        except Exception:
            pass
        set_random_seed(self.model_config.seed)

    def get_model(self):
        return self.model_runner.get_model()

    def load_model(self) -> None:
        self.model_runner.load_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",  # type: ignore[name-defined]
    ) -> Optional[Union[ModelRunnerOutput, AsyncModelRunnerOutput]]:
        return self.model_runner.execute_model(scheduler_output)

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(1)

    def add_lora(self, lora_request) -> bool:  # type: ignore[no-untyped-def]
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)
