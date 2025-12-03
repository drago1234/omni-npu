# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List, Callable, Tuple
import os
import torch
import torch_npu
import torchair
from torch.library import Library
from vllm.logger import init_logger
import vllm.utils
from vllm.utils import DEFAULT_MAX_NUM_BATCHED_TOKENS, vllm_lib
from vllm.platforms.interface import Platform, PlatformEnum

logger = init_logger(__name__)


def ensure_v1_engine() -> None:
    return


def ascend_direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: list[str] = None,
    fake_impl: Optional[Callable] = None,
    target_lib: Optional[Library] = None,
    dispatch_key: Optional[str] = None,
    tags: Tuple[torch.Tag, ...] = (),
):
    # In pytorch 2.5.1, torch.library.infer_schema require the input function to
    # have annotations supported by typing library. But in pytorch 2.7.0 which
    # vllm using, torch.library.infer_schema require the python builtin type. In
    # this case, we should revert built type to typing type for 2.5.1 backward
    # compatibility.
    for k, v in op_func.__annotations__.items():
        if v == list[int]:
            op_func.__annotations__[k] = List[int]
        if v == Optional[list[int]]:
            op_func.__annotations__[k] = Optional[List[int]]

    if mutates_args is None:
        mutates_args = []
    if dispatch_key is None:
        dispatch_key = NPUPlatform.dispatch_key
    import torch.library
    schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    my_lib = target_lib or vllm_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)


def update_utils_custom_op():
    vllm.utils.direct_register_custom_op = ascend_direct_register_custom_op


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
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"

    def __init__(self):
        """Initialize the NPU platform and configure environment."""
        update_utils_custom_op()
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
        from omni_npu.layers.quantization.compressed_tensors.compressed_tensors import AscendCompressedTensorsConfig

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