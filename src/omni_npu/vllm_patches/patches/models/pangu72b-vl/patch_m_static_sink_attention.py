from vllm.attention.selector import get_attn_backend
import torch
import torch_npu
from vllm.forward_context import ForwardContext, get_forward_context
import functools
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionMetadata,
    AttentionType,
)
from vllm.attention.layer import Attention
from vllm.attention.selector import get_attn_backend
from vllm.config import CacheConfig, VllmConfig
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    subclass_attention_backend,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    SinkFullAttentionSpec,
)

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from vllm.attention import layers
from vllm.platforms import current_platform
import types
import sys

logger = init_logger(__name__)

dynamic_module = types.ModuleType("static_sink_attention")
sys.modules[layers.__name__+".static_sink_attention"] = dynamic_module
layers.static_sink_attention = dynamic_module


 
@register_patch("create_static_sink_attention_backendPatch", layers)
class create_static_sink_attention_backendPatch(VLLMPatch):
    _attr_names_to_apply = ['create_static_sink_attention_backend']


    @functools.lru_cache
    def create_static_sink_attention_backend(
        underlying_attn_backend: type[AttentionBackend],
        sink_len: int = 0,
    ) -> type[AttentionBackend]:
        prefix = "StaticSink_"
        underlying_builder = underlying_attn_backend.get_builder_cls()

        class StaticSinkAttentionBuilder(underlying_builder):  # type: ignore
            def __init__(
                self,
                kv_cache_spec: AttentionSpec,
                layer_names: list[str],
                vllm_config: VllmConfig,
                device: torch.device,
            ):
                super().__init__(kv_cache_spec, layer_names, vllm_config, device)
                self.sink_len = sink_len
                self.num_sink_blocks = self.sink_len // vllm_config.cache_config.block_size

                self.sink_block_table = torch.arange(
                    1,
                    self.num_sink_blocks + 1,
                    device=device,
                    dtype=torch.int32,
                )

            def build(
                self,
                common_prefix_len: int,
                common_attn_metadata: CommonAttentionMetadata,
                fast_build: bool = False,
            ) -> AttentionMetadata:
                common_attn_metadata.seq_lens[:] = (
                    common_attn_metadata.seq_lens + self.sink_len
                )
                common_attn_metadata.seq_lens[
                    common_attn_metadata.seq_lens == self.sink_len
                ] = 0
                common_attn_metadata.seq_lens_cpu[:] = (
                    common_attn_metadata.seq_lens_cpu + self.sink_len
                )
                common_attn_metadata.seq_lens_cpu[
                    common_attn_metadata.seq_lens_cpu == self.sink_len
                ] = 0
                common_attn_metadata.max_seq_len = (
                    common_attn_metadata.max_seq_len + self.sink_len
                )


                return super().build(common_prefix_len, common_attn_metadata, fast_build)

        attn_backend = subclass_attention_backend(
            name_prefix=prefix,
            attention_backend_cls=underlying_attn_backend,
            builder_cls=StaticSinkAttentionBuilder,
        )

        return attn_backend
    
    layers.static_sink_attention.create_static_sink_attention_backend = create_static_sink_attention_backend


@register_patch("StaticSinkAttentionPatch", layers)
class StaticSinkAttentionPatch(VLLMPatch):
    _attr_names_to_apply = ['StaticSinkAttention']


    class StaticSinkAttention(Attention):
        """
        Attention with static sink tokens
        """

        def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            sink_len: int,
            attn_backend: type[AttentionBackend] | None = None,
            cache_config: CacheConfig | None = None,
            **kwargs,
        ):
            dtype = torch.get_default_dtype()

            if cache_config is not None:
                kv_cache_dtype = cache_config.cache_dtype
                block_size = cache_config.block_size
            else:
                kv_cache_dtype = "auto"
                block_size = 16

            if attn_backend is not None:
                underlying_attn_backend = attn_backend
            else:
                underlying_attn_backend = get_attn_backend(
                    head_size, dtype, kv_cache_dtype, block_size
                )
            attn_backend = layers.static_sink_attention.create_static_sink_attention_backend(
                underlying_attn_backend,
                sink_len=sink_len,
            )
            super().__init__(
                num_heads=num_heads,
                head_size=head_size,
                scale=scale,
                cache_config=cache_config,
                attn_backend=attn_backend,
                **kwargs,
            )

            self.sink_len = sink_len
            self.block_size = block_size
            self.sink_populated = False
            self.sink_key = None
            self.sink_value = None

        def update_sink_kv(self, sink_key, sink_value) -> None:
            self.sink_key = sink_key
            self.sink_value = sink_value

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            output_shape: torch.Size | None = None,
        ) -> torch.Tensor:
            assert self.sink_key is not None and self.sink_value is not None, (
                "sink_key and sink_value have not been prepared"
            )
            if not self.sink_populated:
                forward_context: ForwardContext = get_forward_context()
                self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                if self_kv_cache is not None and len(self_kv_cache) > 0:
                    torch.ops.vllm.maybe_populate_sink(self_kv_cache[0], self_kv_cache[1], self.layer_name)

            return super().forward(query, key, value, output_shape)



        def populate_sink_kv(self, self_k_cache, self_v_cache):
            sink_kv_slot_mapping = torch.arange(
                self.block_size,
                self.sink_len + self.block_size,
                device=current_platform.current_device(),
                dtype=torch.long,
            ).unsqueeze(dim=1)

            torch_npu.npu_scatter_nd_update_(self_k_cache.view(-1, self_k_cache.shape[-1]), sink_kv_slot_mapping, self.sink_key.squeeze(dim=1))
            torch_npu.npu_scatter_nd_update_(self_v_cache.view(-1, self_v_cache.shape[-1]), sink_kv_slot_mapping, self.sink_value.squeeze(dim=1))

            # We only populate the sink_key and sink_value once
            self.sink_populated = True

        def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
            # Block size may get updated after model loading, refresh it
            block_size = vllm_config.cache_config.block_size
            # Should not be called for enc-dec or encoder-only attention.
            assert self.attn_type == AttentionType.DECODER

            attentionSpec = SinkFullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                sink_len=self.sink_len,
                dtype=self.kv_cache_torch_dtype,
            )
            attentionSpec.set_head_size_v(self.head_size_v)

            return attentionSpec
    
    layers.static_sink_attention.StaticSinkAttention = StaticSinkAttention



@register_patch("maybe_populate_sinkPatch", layers)
class maybe_populate_sinkPatch(VLLMPatch):
    _attr_names_to_apply = ['maybe_populate_sink', 'maybe_populate_sink_fake']

    def maybe_populate_sink(
        self_k_cache: torch.Tensor,
        self_v_cache: torch.Tensor,
        layer_name: str,
    ) -> None:
        forward_context: ForwardContext = get_forward_context()
        self = forward_context.no_compile_layers[layer_name]
        if self.sink_populated or self_k_cache.numel() == 0:
            return
        self.populate_sink_kv(self_k_cache, self_v_cache)


    def maybe_populate_sink_fake(
        self_k_cache: torch.Tensor,
        self_v_cache: torch.Tensor,
        layer_name: str,
    ) -> None:
        return


    direct_register_custom_op(
        op_name="maybe_populate_sink",
        op_func=maybe_populate_sink,
        mutates_args=["self_k_cache", "self_v_cache"],
        fake_impl=maybe_populate_sink_fake,
    )

    layers.static_sink_attention.maybe_populate_sink = maybe_populate_sink
    layers.static_sink_attention.maybe_populate_sink_fake = maybe_populate_sink_fake