import argparse
from typing import Any

from vllm.logger import init_logger
logger = init_logger(__name__)

import vllm.engine.arg_utils as arg_utils
from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from vllm.config.cache import CacheConfig
from vllm.utils.argparse_utils import FlexibleArgumentParser


# Different vLLM versions use different method names for building configs.
_orig_add_cli_args = arg_utils.EngineArgs.add_cli_args
# _orig_create_engine_config = getattr(arg_utils.EngineArgs, "create_engine_config", None)

@register_patch("EnableKVRMSNormRoPECacheEngineArgsPatch", arg_utils.EngineArgs)
class EnableKVRMSNormRoPECacheEngineArgsPatch(VLLMPatch):
    """
    Runtime patch (no vLLM source modifications):
    1) Add CLI flag: --enable-kv-rmsnorm-rope-cache
    2) Inject the flag into CacheConfig instance via setattr, so downstream can read:
       cache_config.enable_kv_rmsnorm_rope_cache
    """
    _attr_names_to_apply = ["add_cli_args", "enable_kv_rmsnorm_rope_cache"]
    enable_kv_rmsnorm_rope_cache: bool = False

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        parser = _orig_add_cli_args(parser)
        # 检查是否重复添加（防御性编程）
        if any("--enable-kv-rmsnorm-rope-cache" in a.option_strings for a in parser._actions):
            return parser

        # 找到 CacheConfig 分组
        cache_group = next((g for g in parser._action_groups if g.title == "CacheConfig"), parser)
        
        cache_group.add_argument(
            "--enable-kv-rmsnorm-rope-cache",
            action="store_true",
            default=False, # 明确默认值
            help="Enable NPU fused operator npu_kv_rmsnorm_rope_cache."
        )
        return parser



# import vllm.config.cache as cache_mod
from vllm.config.cache import CacheConfig
@register_patch("EnableKVRMSNormRoPECacheHashFactorsPatch", CacheConfig)
class EnableKVRMSNormRoPECacheHashFactorsPatch(VLLMPatch):
    """
    Ensure cache_config.enable_kv_rmsnorm_rope_cache participates in config hashing
    (so compiled graph / cache keys remain stable).
    """
    _attr_names_to_apply = ["compute_hash", "enable_kv_rmsnorm_rope_cache"]
    enable_kv_rmsnorm_rope_cache: bool = False

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        ignored_factors = {
            # Runtime/derived knobs that don't affect compiled graph shape
            "gpu_memory_utilization",
            "swap_space",
            "is_attention_free",
            "num_gpu_blocks_override",
            "enable_prefix_caching",
            "prefix_caching_hash_algo",
            "cpu_kvcache_space_bytes",
            "mamba_page_size_padded",
            # Post-init/derived counters
            "num_gpu_blocks",
            "num_cpu_blocks",
            # WIP feature toggle not impacting compiled graph shape
            "kv_sharing_fast_prefill",
            "enable_kv_rmsnorm_rope_cache",
        }

        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors)
        return hash_factors(factors)



_orig_from_cli_args = arg_utils.AsyncEngineArgs.from_cli_args

@register_patch("AsyncEngineArgsInjectFlagPatch", arg_utils.AsyncEngineArgs)
class AsyncEngineArgsInjectFlagPatch(VLLMPatch):
    _attr_names_to_apply = ["from_cli_args"]

    def _from_cli_args(cls, args, *a, **kw):
        engine_args = _orig_from_cli_args(args, *a, **kw)
        setattr(engine_args, "enable_kv_rmsnorm_rope_cache",
                bool(getattr(args, "enable_kv_rmsnorm_rope_cache", False)))
        return engine_args

    from_cli_args = classmethod(_from_cli_args)
