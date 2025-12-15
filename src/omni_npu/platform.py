# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch
import torch_npu
import torchair
from vllm.logger import init_logger
from vllm.platforms.interface import Platform, PlatformEnum
from omni_npu.logger import update_configure_vllm_root_logger

update_configure_vllm_root_logger()
logger = init_logger(__name__)


class NPUPlatform(Platform):
    try:
        # In case vllm already defined HUAWEI_NPU platform
        _enum = PlatformEnum.HUAWEI_NPU
    except AttributeError:
        # fallback to OOT
        _enum = PlatformEnum.OOT
    device_name: str = "npu"
    device_type: str = "npu"
    dispatch_key: str = "PrivateUse1"
    ray_device_key: str = "NPU"
    dist_backend: str = "hccl"
    #device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"

    def __init__(self):
        """Initialize the NPU platform and configure environment."""
        super().__init__()

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
    def import_core_kernels(cls):
        from omni_npu.compilation.decorators import patch_compile_decorators
        patch_compile_decorators()

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
    def pre_register_and_update(
        cls, parser: Optional["FlexibleArgumentParser"] = None
    ) -> None:
        """
        Do some pre-registration or update action for the current platform.

        This function is called before global VllmConfig is initialized or cli
        arguments are parsed. It's used for out-of-tree platforms to register or
        update the configuration.

        For example, the out-of-tree quantization config can be imported and
        registered here dynamically.
        """
        from omni_npu.layers.quantization.compressed_tensors.compressed_tensors import NPUCompressedTensorsConfig
        import omni_npu.layers.fused_moe.layer
        from omni_npu.connector import register_connectors
        register_connectors()
        from omni_npu.distributed.eplb_state import EplbState

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
        selected_backend: "_Backend",
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: Optional[str],
        block_size: int,
        use_v1: bool,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
    ) -> str:
        return (
            "omni_npu.attention.backends.mla.NPUMLABackend"
            if use_mla
            else "omni_npu.attention.backends.attention.NPUAttentionBackend"
        )

    @property
    def simple_compile_backend(self):
        return "eager"

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        """
        Returns if the graph mode is supported by the current platform.
        """
        return True

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        """
        Get piecewise backend class for piecewise graph.
        """
        return "omni_npu.compilation.acl_graph.ACLGraphWrapper"
