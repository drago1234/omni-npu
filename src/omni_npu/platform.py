# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import os
import torch
from vllm.logger import init_logger
from vllm.utils import DEFAULT_MAX_NUM_BATCHED_TOKENS
from vllm.platforms.interface import Platform, PlatformEnum

logger = init_logger(__name__)


def ensure_v1_engine() -> None:
    return


class NPUPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "npu"
    device_type: str = "npu"
    dispatch_key: str = "PrivateUse1"
    ray_device_key: str = "NPU"
    dist_backend: str = "hccl"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        torch.npu.set_device(device)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.npu.get_device_name(device_id)

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def get_current_memory_usage(cls, device: Optional[torch.types.Device] = None) -> float:
        torch.npu.reset_peak_memory_stats(device)
        return torch.npu.max_memory_allocated(device)

    @classmethod
    def device_count(cls) -> int:
        return torch.npu.device_count()

    @classmethod
    def mem_get_info(cls) -> tuple[int, int]:
        return torch.npu.mem_get_info()

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:  # type: ignore[name-defined]
        # Minimal defaults to match vLLM expectations.
        parallel_config = vllm_config.parallel_config
        parallel_config.worker_cls = "omni_npu.v1.worker.npu_worker.NPUWorker"

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 128

        # If MLA is enabled on non-GPU, disable chunked prefill/prefix caching to be safe.
        model_config = vllm_config.model_config
        if model_config and model_config.use_mla:
            logger.info(
                "MLA enabled on NPU; disabling chunked prefill and setting max_num_batched_tokens"
            )
            vllm_config.scheduler_config.enable_chunked_prefill = False
            vllm_config.scheduler_config.chunked_prefill_enabled = False
            vllm_config.scheduler_config.max_num_batched_tokens = max(
                vllm_config.scheduler_config.max_model_len, DEFAULT_MAX_NUM_BATCHED_TOKENS
            )

    @classmethod
    def get_punica_wrapper(cls) -> str:
        # Use CPU punica wrapper by default
        return "vllm.lora.punica_wrapper.punica_cpu.PunicaWrapperCPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        # Point vLLM to our HCCL-based communicator implementation
        return "omni_npu.distributed.communicator.NPUCommunicator"

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: str,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: torch.dtype,
        block_size: int,
        use_v1: bool,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
    ) -> str:
        ensure_v1_engine()
        return (
            "omni_npu.attention.backends.mla.AscendMLABackend"
            if use_mla
            else "omni_npu.attention.backends.attention.AscendAttentionBackend"
        )
