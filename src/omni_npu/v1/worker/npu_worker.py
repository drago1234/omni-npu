# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
from typing import Optional, Union

import torch
import torch_npu
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.tasks import SupportedTask
from vllm.v1.outputs import (
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
        local_rank = (local_rank + vllm_config.parallel_config.data_parallel_rank * vllm_config.parallel_config.tensor_parallel_size) % 16
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
        self.profiler = None
        current_platform.pre_register_and_update()


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
        self.profiler = self._init_profiler()

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

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if getattr(self, '_use_token_for_profile', False):
            logger.info("origin profiler is disabled because PROFILER_TOKEN_THRESHOLD is set.")
            return
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()

    def compile_or_warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()
        set_random_seed(self.model_config.seed)

    def get_model(self):
        return self.model_runner.get_model()

    def load_model(self) -> None:
        self.model_runner.load_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(1, uniform_decode=True, force_attention=True)

    def add_lora(self, lora_request) -> bool:  # type: ignore[no-untyped-def]
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:

        return self.model_runner.pin_lora(lora_id)

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",  # type: ignore[name-defined]
    ) -> Optional[Union[ModelRunnerOutput, AsyncModelRunnerOutput]]:
        if envs.VLLM_TORCH_PROFILER_DIR and self._use_token_for_profile:
            if not self.profile_already_start and scheduler_output.total_num_scheduled_tokens==len(scheduler_output.num_scheduled_tokens)==self.profiler_token_threshold:
                self.profiler.start()
                self.profile_already_start = True
                self.profile_step = 0

        output = self.model_runner.execute_model(scheduler_output)
        if envs.VLLM_TORCH_PROFILER_DIR and self._use_token_for_profile:
            if self.profile_already_start and not self.profile_finished:
                self.profile_step += 1
            if not self.profile_finished and self.profile_step > self.profiler_stop_step:
                self.profiler.stop()
                self.profile_finished = True
        return output

    def _init_profiler(self):
        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        # PROFILER_TOKEN_THRESHOLD=1
        # PROFILER_STOP_STEP=5
        self.profile_already_start = False
        self.profile_step = 0
        self.profile_finished = False
        self._use_token_for_profile = os.getenv("PROFILER_TOKEN_THRESHOLD") is not None

        if envs.VLLM_TORCH_PROFILER_DIR:
            self.profiler_token_threshold = int(os.environ.get('PROFILER_TOKEN_THRESHOLD',"1"))
            self.profiler_stop_step = int(os.environ.get('PROFILER_STOP_STEP',"5"))
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)

            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            )
            self.profile_already_start = False
            self.profile_finished = False
            return torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                record_shapes=envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                profile_memory=envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                with_stack=envs.VLLM_TORCH_PROFILER_WITH_STACK,
                with_flops=envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
                experimental_config=experimental_config,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir))
        else:
            return None

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        return self.model_runner.take_draft_token_ids()