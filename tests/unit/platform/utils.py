from itertools import count
from unittest.mock import MagicMock

import torch
from vllm.config import VllmConfig, SchedulerConfig, ModelConfig, CacheConfig, KVTransferConfig, DeviceConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, FullAttentionSpec
from vllm.v1.outputs import ModelRunnerOutput, KVConnectorOutput
from vllm.v1.request import Request
from vllm import SamplingParams


def create_vllm_config(
    kv_role: str="kv_producer",
    kv_connector: str = "LLMDataDistConnector",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 64,
    block_size: int = 16,
    max_model_len: int = 10000,
    enable_chunked_prefill: bool = True,
    enable_permute_local_kv: bool = False,
) -> VllmConfig:
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        enable_chunked_prefill=enable_chunked_prefill,
        is_encoder_decoder=False,
    )
    model_config = MagicMock()
    model_config.max_model_len = max_model_len
    model_config.is_encoder_decoder = False
    model_config.is_multimodal_model = False
    model_config.inputs_embeds_size = 16
    model_config.num_query_heads = 16
    model_config.attention_chunk_size = 16
    model_config.dtype = torch.float16
    model_config.uses_xdrope_dim = 0
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector=kv_connector,
        kv_role=kv_role,
        enable_permute_local_kv=enable_permute_local_kv,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        device_config=DeviceConfig("cpu"),
    )
    vllm_config.model_config = model_config
    return vllm_config