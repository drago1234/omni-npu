# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import Any, Optional, Literal

import torch
import torch_npu
from torch import nn
from transformers import PretrainedConfig

import vllm.envs as envs
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionMetadata,
    AttentionType,
)
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.attention.layer import _init_kv_cache_quant
from vllm.attention.selector import get_attn_backend
from vllm.attention.utils.kv_sharing_utils import validate_kv_sharing_target
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mla import (
    MLAModules,
    MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding,
    _ROPE_DICT,
    get_rope,
)
import vllm.model_executor.layers.rotary_embedding as _rope_mod
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.openpangu import (
    OpenPanguEmbeddedAttention as VllmOpenPanguEmbeddedAttention,
    OpenPanguMLAAttention as VllmOpenPanguMLAAttention,
    OpenPanguMLP as VllmOpenPanguMLP,
    OpenPanguMoE as VllmOpenPanguMoE,
    OpenPanguMoEModel as VllmOpenPanguMoEModel,
    OpenPanguModel as VllmOpenPanguModel,
    OpenPanguDecoderLayer as VllmOpenPanguDecoderLayer,
    check_ffn_act_fn,
)
from vllm.model_executor.models.utils import (
    extract_layer_index,
    is_pp_missing_parameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    subclass_attention_backend,
)
from vllm.v1 import kv_cache_interface

logger = init_logger(__name__)
AttentionSpec = kv_cache_interface.AttentionSpec
KVCacheSpec = kv_cache_interface.KVCacheSpec
SinkFullAttentionSpec = getattr(
    kv_cache_interface,
    "SinkFullAttentionSpec",
    kv_cache_interface.FullAttentionSpec,
)


def _apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


def _rotary_forward_native(
    module: MRotaryEmbedding,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
    output_cos_sin: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if offsets is not None:
        positions = positions + offsets
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = module.cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)
    if output_cos_sin:
        cos_cache = torch.cat((cos, cos), dim=1)
        sin_cache = torch.cat((sin, sin), dim=1)
    else:
        cos_cache = None
        sin_cache = None

    query_shape = query.shape
    query = query.view(num_tokens, -1, module.head_size)
    query_rot = query[..., :module.rotary_dim]
    query_pass = query[..., module.rotary_dim:]
    if output_cos_sin:
        query_rot = torch_npu.npu_interleave_rope(
            query_rot.unsqueeze(2),
            cos_cache.unsqueeze(1).unsqueeze(1),
            sin_cache.unsqueeze(1).unsqueeze(1),
        ).squeeze(2)
    else:
        query_rot = _apply_rotary_emb_torch(
            query_rot, cos, sin, module.is_neox_style
        )
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    if key is not None:
        key_shape = key.shape
        key = key.view(num_tokens, -1, module.head_size)
        key_rot = key[..., :module.rotary_dim]
        key_pass = key[..., module.rotary_dim:]
        if output_cos_sin:
            key_rot = torch_npu.npu_interleave_rope(
                key_rot.unsqueeze(2),
                cos_cache.unsqueeze(1).unsqueeze(1),
                sin_cache.unsqueeze(1).unsqueeze(1),
            ).squeeze(2)
        else:
            key_rot = _apply_rotary_emb_torch(
                key_rot, cos, sin, module.is_neox_style
            )
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

    return query, key, cos_cache, sin_cache


class MRotaryEmbeddingInterleaved(MRotaryEmbedding):
    """Rotary Embedding with Multimodal Sections and Interleaved Support."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: Optional[list[int]] = None,
        mrope_interleaved: bool = True,
        rotary_mode: Literal["half", "interleaved"] = "half",
        num_hidden_layers_cache: int = 1,
    ) -> None:
        self.cache_max_position_num = max_position_embeddings
        super().__init__(
            head_size,
            rotary_dim,
            self.cache_max_position_num,
            base,
            is_neox_style,
            dtype,
        )
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved
        self.rotary_mode = rotary_mode
        self.num_hidden_layers_cache = num_hidden_layers_cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        output_cos_sin: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        query, key, cos, sin = _rotary_forward_native(
            self,
            positions,
            query,
            key=key,
            offsets=offsets,
            output_cos_sin=output_cos_sin,
        )
        return query, key, cos, sin


_orig_get_rope = _rope_mod.get_rope


def get_rope_wrapper(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    is_neox_style: bool = True,
    rope_scaling: Optional[dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
    dual_chunk_attention_config: Optional[dict[str, Any]] = None,
    num_hidden_layers_cache: int = 1,
) -> _rope_mod.RotaryEmbedding:
    if rope_scaling is not None and rope_scaling.get("mrope_interleaved") is True:
        if dtype is None:
            dtype = torch.get_default_dtype()
        if rope_scaling is not None:
            rope_scaling_tuple = {
                k: tuple(v) if isinstance(v, list) else v
                for k, v in rope_scaling.items()
            }
            rope_scaling_args = tuple(rope_scaling_tuple.items())
        else:
            rope_scaling_args = None

        if partial_rotary_factor < 1.0:
            rotary_dim = int(rotary_dim * partial_rotary_factor)
        key = (
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            rope_scaling_args,
            None,
            dtype,
            num_hidden_layers_cache,
        )
        if key in _ROPE_DICT:
            return _ROPE_DICT[key]

        mrope_section = rope_scaling.get("mrope_section")
        rotary_mode = rope_scaling.get("rotary_mode", "half")
        num_hidden_layers_cache = (
            1 if get_pp_group().world_size > 1 else num_hidden_layers_cache
        )

        rotary_emb = MRotaryEmbeddingInterleaved(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            dtype,
            mrope_section=mrope_section,
            mrope_interleaved=True,
            rotary_mode=rotary_mode,
            num_hidden_layers_cache=num_hidden_layers_cache,
        )

        _ROPE_DICT[key] = rotary_emb
        return rotary_emb

    if rope_scaling is None:
        rope_parameters = {"rope_theta": base, "rope_type": "default"}
    else:
        rope_parameters = rope_scaling.copy()
        if "rope_theta" not in rope_parameters:
            rope_parameters["rope_theta"] = base
        if "rope_type" not in rope_parameters:
            rope_parameters["rope_type"] = "default"

    return _orig_get_rope(
        head_size,
        rotary_dim,
        max_position,
        is_neox_style,
        rope_parameters,
        dtype,
        partial_rotary_factor,
        dual_chunk_attention_config,
    )


_rope_mod.MRotaryEmbedding.forward_native = _rotary_forward_native
_rope_mod.get_rope_wrapper = get_rope_wrapper
_rope_mod.MRotaryEmbeddingInterleaved = MRotaryEmbeddingInterleaved


class PanguAttention(AttentionLayerBase, nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        logits_soft_cap: float | None = None,
        per_layer_sliding_window: int | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        attn_backend: type[AttentionBackend] | None = None,
        head_size_v: int | None = None,
        **extra_impl_args,
    ) -> None:
        nn.Module.__init__(self)
        AttentionLayerBase.__init__(self)

        if per_layer_sliding_window is not None:
            sliding_window = per_layer_sliding_window
        elif cache_config is not None:
            sliding_window = cache_config.sliding_window
        else:
            sliding_window = None

        vllm_config = get_current_vllm_config()
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
            calculate_kv_scales = cache_config.calculate_kv_scales
        else:
            kv_cache_dtype = "auto"
            block_size = 16
            calculate_kv_scales = False
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            kv_cache_dtype, vllm_config.model_config
        )
        if num_kv_heads is None:
            num_kv_heads = num_heads
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) is not divisible by num_kv_heads ({num_kv_heads})"
        )

        _init_kv_cache_quant(
            self, quant_config, prefix, kv_cache_dtype, calculate_kv_scales
        )

        self.num_heads = num_heads
        self.head_size = head_size
        self.head_size_v = self.head_size if head_size_v is None else head_size_v
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.has_sink = extra_impl_args.get("sinks") is not None

        dtype = torch.get_default_dtype()
        if attn_backend is None:
            self.attn_backend = get_attn_backend(
                head_size,
                dtype,
                kv_cache_dtype,
                block_size,
                use_mla=False,
                has_sink=self.has_sink,
                attn_type=attn_type,
            )
        else:
            self.attn_backend = attn_backend

        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **extra_impl_args,
        )
        backend_name = self.attn_backend.get_name()
        self.backend = AttentionBackendEnum.__members__.get(backend_name)
        self.dtype = dtype

        self.use_direct_call = not current_platform.opaque_attention_op()

        self.use_output = self.attn_backend.accept_output_buffer
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix
        self.attn_type = attn_type

        if kv_sharing_target_layer_name is not None:
            validate_kv_sharing_target(
                prefix,
                kv_sharing_target_layer_name,
                compilation_config.static_forward_context,
            )
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.kv_cache = [
            torch.tensor([])
            for _ in range(vllm_config.parallel_config.pipeline_parallel_size)
        ]

        self.q_range = torch.tensor(envs.Q_SCALE_CONSTANT, dtype=torch.float32)
        self.k_range = torch.tensor(envs.K_SCALE_CONSTANT, dtype=torch.float32)
        self.v_range = torch.tensor(envs.V_SCALE_CONSTANT, dtype=torch.float32)

        self.query_quant = None
        if (
            self.kv_cache_dtype.startswith("fp8")
            and self.impl.supports_quant_query_input()
        ):
            self.query_quant = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        if self.calculate_kv_scales:
            torch.ops.vllm.maybe_calc_kv_scales(
                query, key, value, self.layer_name
            )

        forward_context: ForwardContext = get_forward_context()
        if self.use_direct_call:
            metadata: AttentionMetadata = forward_context.attn_metadata
            output = self.attn_backend.forward(
                query,
                key,
                value,
                kv_cache=self.kv_cache,
                attn_metadata=metadata,
                kv_scale=self.kv_scale,
                output=forward_context.output if self.use_output else None,
                output_shape=output_shape,
                layer_name=self.layer_name,
                layer_idx=self.layer_idx,
                q_range=self.q_range,
                k_range=self.k_range,
                v_range=self.v_range,
                query_quant=self.query_quant,
            )
        else:
            output = self.impl.forward(
                query,
                key,
                value,
                self.kv_cache,
                forward_context.attn_metadata,
                self.kv_scale,
                output=forward_context.output if self.use_output else None,
                output_shape=output_shape,
                layer_name=self.layer_name,
                layer_idx=self.layer_idx,
                q_range=self.q_range,
                k_range=self.k_range,
                v_range=self.v_range,
                query_quant=self.query_quant,
            )
        return output


@lru_cache
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

    return subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=StaticSinkAttentionBuilder,
    )


class StaticSinkAttention(PanguAttention):
    """Attention with static sink tokens."""

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
        attn_backend = create_static_sink_attention_backend(
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

    def update_sink_kv(self, sink_key: torch.Tensor, sink_value: torch.Tensor) -> None:
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
                torch.ops.vllm.maybe_populate_sink(
                    self_kv_cache[0], self_kv_cache[1], self.layer_name
                )

        return super().forward(query, key, value, output_shape)

    def populate_sink_kv(self, self_k_cache: torch.Tensor, self_v_cache: torch.Tensor):
        sink_kv_slot_mapping = torch.arange(
            self.block_size,
            self.sink_len + self.block_size,
            device=current_platform.current_device(),
            dtype=torch.long,
        ).unsqueeze(dim=1)

        torch_npu.npu_scatter_nd_update_(
            self_k_cache.view(-1, self_k_cache.shape[-1]),
            sink_kv_slot_mapping,
            self.sink_key.squeeze(dim=1),
        )
        torch_npu.npu_scatter_nd_update_(
            self_v_cache.view(-1, self_v_cache.shape[-1]),
            sink_kv_slot_mapping,
            self.sink_value.squeeze(dim=1),
        )

        self.sink_populated = True

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        block_size = vllm_config.cache_config.block_size
        assert self.attn_type == AttentionType.DECODER

        attention_spec = SinkFullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            sink_len=self.sink_len,
            dtype=self.kv_cache_torch_dtype,
        )
        attention_spec.set_head_size_v(self.head_size_v)

        return attention_spec


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


class OpenPanguMoE(VllmOpenPanguMoE):
    def __init__(
        self,
        config: PretrainedConfig,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tp_group().rank_in_group

        self.routed_scaling_factor = config.routed_scaling_factor
        self.ep_group = get_ep_group().device_group
        self.ep_rank = self.ep_group.rank()
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe
        check_ffn_act_fn(config.hidden_act)

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        if (
            hasattr(config, "router_enable_expert_bias")
            and config.router_enable_expert_bias
        ):
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32)
            )
        else:
            self.gate.e_score_correction_bias = None

        # Load balancing settings.
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = VllmOpenPanguMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_experts,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            prefix=f"{prefix}.experts",
            scoring_func="sigmoid",
            # we do scaling outside, set factor to 1.0 to avoid double mul
            routed_scaling_factor=1.0,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
        )


class OpenPanguMLAAttention(VllmOpenPanguMLAAttention):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.tp_size = get_tensor_model_parallel_world_size()
        if num_heads % self.tp_size != 0:
            raise ValueError(
                f"num_heads {num_heads} is not divisible by tp_size {self.tp_size}."
            )
        self.num_local_heads = num_heads // self.tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.prefix = prefix

        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True,
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
            self.q_proj = None
            self.kv_a_proj_with_mqa = None
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
            self.fused_qkv_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None

        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )

        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        set_default_rope_theta(config, default_theta=10000)
        rope_parameters = {
            "rope_theta": config.rope_parameters["rope_theta"],
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": max_position_embeddings,
            "type": "yarn",
            "rope_type": "deepseek_yarn",
        }

        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=False,
        )

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj
            if self.q_lora_rank is not None
            else None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa
            if self.q_lora_rank is None
            else None,
            q_a_layernorm=self.q_a_layernorm if self.q_lora_rank is not None else None,
            q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else None,
            indexer=None,
            indexer_rotary_emb=None,
            is_sparse=False,
            topk_indices_buffer=None,
        )

        self.mla_attn = MultiHeadLatentAttentionWrapper(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
        )


class OpenPanguSinkAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.total_num_heads = num_heads
        self.enable_kv_rmsnorm_rope_cache = False

        vllm_config = get_current_vllm_config()
        self.enable_kv_rmsnorm_rope_cache = getattr(
            vllm_config.cache_config, "enable_kv_rmsnorm_rope_cache", False
        )

        if self.total_num_heads % self.tp_size != 0:
            raise ValueError(
                f"total_num_heads {self.total_num_heads} "
                f"is not divisible by tp_size {self.tp_size}."
            )
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = num_kv_heads
        if (
            self.total_num_kv_heads > self.tp_size
            and self.total_num_kv_heads % self.tp_size != 0
        ):
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel ranks.
            raise ValueError(
                "Number of KV heads is greater than TP size, "
                f"but total_num_kv_heads {self.total_num_kv_heads} "
                f"is not divisible by tp_size {self.tp_size}."
            )
        elif self.total_num_kv_heads < self.tp_size:
            # TODO: Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel ranks.
            raise ValueError(
                f"Number of KV heads {self.total_num_kv_heads} is less than "
                f"TP size {self.tp_size}, KV heads replication is not support yet."
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.qk_nope_dim = getattr(config, "qk_nope_dim", None)
        self.qk_rope_dim = getattr(config, "qk_rope_dim", None)
        self.v_channels = getattr(config, "v_channels", None)
        self.head_dim = self.qk_rope_dim + self.qk_nope_dim
        self.q_size = self.num_heads * self.head_dim
        self.k_size = self.num_kv_heads * self.head_dim
        self.v_size = self.num_kv_heads * self.v_channels
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.param_sink_number = getattr(config, "param_sink_number", 0)
        self.param_sink_with_value = getattr(config, "param_sink_with_value", False)
        self.param_sink_scalar = getattr(config, "param_sink_scalar", None)
        self.param_sink_of_head_num = getattr(config, "param_sink_of_head_dim", False)

        self.qkv_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[
                self.q_size * self.tp_size,
                self.k_size * self.tp_size,
                self.v_size * self.tp_size,
            ],
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.v_channels,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.k_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # cache_layer = config.num_hidden_layers
        # is_mtp_layer = getattr(config, "is_mtp_layer", False)
        # if is_mtp_layer:
        #     cache_layer = config.num_nextn_predict_layers
        # self._init_rotary_emb(
        #     config, cache_layer, rope_parameters=rope_parameters, quant_config=quant_config
        # )
        self._init_rotary_emb(
            config, rope_parameters=rope_parameters, quant_config=quant_config
        )

        if hasattr(config, "interleaved_sliding_window"):
            interleaved_sliding_window = config.interleaved_sliding_window
            if isinstance(interleaved_sliding_window, int):
                sliding_window = interleaved_sliding_window
            elif isinstance(interleaved_sliding_window, list):
                sw_idx = layer_idx % len(interleaved_sliding_window)
                sliding_window = interleaved_sliding_window[sw_idx]
            else:
                raise ValueError(
                    f"{type(interleaved_sliding_window)} "
                    "for interleaved_sliding_window is not supported."
                )
        else:
            sliding_window = None

        self.attn = StaticSinkAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            sink_len=self.param_sink_number,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
            attn_backend=None,
            head_size_v=self.v_channels,
        )

        if self.param_sink_number > 0:
            self.param_sink_key = torch.nn.Parameter(
                torch.empty(
                    (
                        self.param_sink_number,
                        self.num_kv_heads,
                        self.head_dim,
                    ),
                    device=current_platform.current_device(),
                    dtype=config.torch_dtype,
                )
            )
            set_weight_attrs(
                self.param_sink_key,
                {
                    "output_dim": 1,
                    "weight_loader": self.weight_loader,
                },
            )

            if self.param_sink_with_value:
                self.param_sink_value = torch.nn.Parameter(
                    torch.empty(
                        (
                            self.param_sink_number,
                            self.num_kv_heads,
                            self.v_channels,
                        ),
                        device=current_platform.current_device(),
                        dtype=config.torch_dtype,
                    )
                )
                set_weight_attrs(
                    self.param_sink_value,
                    {
                        "output_dim": 1,
                        "weight_loader": self.weight_loader,
                    },
                )
            else:
                self.param_sink_value = torch.zeros(
                    (
                        self.param_sink_number,
                        self.num_kv_heads,
                        self.v_channels,
                    ),
                    device=current_platform.current_device(),
                    dtype=config.torch_dtype,
                )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        output_dim = getattr(param, "output_dim", None)

        is_sharded_weight = getattr(param, "is_sharded_weight", False)
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow
        is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

        # Special case for GGUF
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, nn.UninitializedParameter):
            final_shape = list(loaded_weight.shape)
            if output_dim is not None:
                assert final_shape[output_dim] % self.tp_size == 0
                final_shape[output_dim] = final_shape[output_dim] // self.tp_size
            param.materialize(final_shape, dtype=loaded_weight.dtype)

        param_data = param.data
        if output_dim is not None and not is_sharded_weight:
            shard_size = param_data.shape[output_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        if self.enable_kv_rmsnorm_rope_cache:
            q, _, cos, sin = self.rotary_emb(
                positions, q.contiguous(), output_cos_sin=True
            )
        else:
            k = self.k_layernorm(k.view(-1, self.num_kv_heads, self.head_dim))
            q, k, _, _ = self.rotary_emb(positions, q.contiguous(), key=k)

        q = q.view(-1, self.q_size)
        k = k.view(-1, self.k_size)

        if self.enable_kv_rmsnorm_rope_cache:
            self.attn.impl.set_kv_rmsnorm_rope_params(
                self.k_layernorm.weight,
                self.k_layernorm.variance_epsilon,
                cos,
                sin,
                self.enable_kv_rmsnorm_rope_cache,
                self.v_channels,
            )

        attn_output = self.attn(
            q,
            k,
            v,
            output_shape=torch.Size(
                [q.shape[0], q.shape[1] // self.head_dim * self.v_channels]
            ),
        )
        output, _ = self.o_proj(attn_output)
        return output

    def _init_rotary_emb(
        self,
        config: PretrainedConfig,
        cache_layer: int,
        rope_parameters: dict[str, Any] | None,
        quant_config: QuantizationConfig | None,
    ) -> None:
        is_neox_style = False
        rope_parameters = {"partial_rotary_factor": self.qk_rope_dim / self.head_dim}
        from vllm.model_executor.layers.rotary_embedding import get_rope_wrapper
        self.rotary_emb = get_rope_wrapper(
            self.head_dim,
            self.qk_rope_dim,
            max_position=self.max_position_embeddings,
            rotary_dim=self.qk_rope_dim,
            base=config.rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
            is_neox_style=is_neox_style,
            num_hidden_layers_cache=config.num_hidden_layers,
        )

    def sink_key_interleave(self, weight: torch.Tensor) -> torch.Tensor:
        rope_dim = 64
        key_rot = weight[..., :rope_dim]
        key_pass = weight[..., rope_dim:]
        o1 = key_rot[..., ::2]
        o2 = key_rot[..., 1::2]

        key_rot = torch.cat((o1, o2), dim=-1)
        return torch.cat((key_rot, key_pass), dim=-1)

    def post_weight_load(self) -> None:
        if hasattr(self, "k_layernorm") and self.k_layernorm is not None:
            param_sink_key = self.k_layernorm(self.param_sink_key)
        else:
            param_sink_key = self.param_sink_key

        param_sink_key = self.sink_key_interleave(param_sink_key) # K & Q 保持一致，都做interleave(奇偶交叉处理)
        self.attn.update_sink_kv(param_sink_key, self.param_sink_value)


class OpenPanguDecoderLayer(VllmOpenPanguDecoderLayer):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        vllm_config: VllmConfig,
    ) -> None:
        nn.Module.__init__(self)

        if config is None:
            config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        self.use_mla = (
            hasattr(config, "qk_nope_head_dim")
            and hasattr(config, "qk_rope_head_dim")
            and hasattr(config, "v_head_dim")
            and hasattr(config, "kv_lora_rank")
        )

        self.use_sink_attention = (
            hasattr(config, "param_sink_number") and config.param_sink_number > 0
        )

        if self.use_mla:
            self.self_attn = OpenPanguMLAAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=(
                    config.q_lora_rank if hasattr(config, "q_lora_rank") else None
                ),
                kv_lora_rank=config.kv_lora_rank,
                max_position_embeddings=max_position_embeddings,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        elif self.use_sink_attention:
            attention_bias = getattr(config, "attention_bias", False) or getattr(
                config, "bias", False
            )
            bias_o_proj = attention_bias
            if hasattr(config, "qkv_bias"):
                attention_bias = config.qkv_bias
            if getattr(config, "is_causal", True):
                attn_type = AttentionType.DECODER
            else:
                raise ValueError(
                    f"is_causal={config.is_causal} is not support "
                    "for attention with sink"
                )
            rope_parameters = getattr(config, "rope_scaling", None)
            if rope_parameters is None:
                rope_parameters = {
                    "rope_type": "default",
                    "rope_theta": config.rope_theta,
                }
            self.self_attn = OpenPanguSinkAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=getattr(
                    config, "num_key_value_heads", config.num_attention_heads
                ),
                rope_parameters=rope_parameters,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                bias=attention_bias,
                bias_o_proj=bias_o_proj,
                cache_config=cache_config,
                prefix=f"{prefix}.self_attn",
                attn_type=attn_type,
            )
        else:
            attention_bias = getattr(config, "attention_bias", False) or getattr(
                config, "bias", False
            )
            bias_o_proj = attention_bias
            if hasattr(config, "qkv_bias"):
                attention_bias = config.qkv_bias
            if getattr(config, "is_causal", True):
                attn_type = AttentionType.DECODER
            else:
                attn_type = AttentionType.ENCODER_ONLY
            self.self_attn = VllmOpenPanguEmbeddedAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=getattr(
                    config, "num_key_value_heads", config.num_attention_heads
                ),
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                bias=attention_bias,
                bias_o_proj=bias_o_proj,
                cache_config=cache_config,
                prefix=f"{prefix}.self_attn",
                attn_type=attn_type,
            )

        if (
            getattr(config, "n_routed_experts", None) is not None
            and layer_idx >= config.first_k_dense_replace
        ):
            self.mlp = OpenPanguMoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = VllmOpenPanguMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                bias=getattr(config, "mlp_bias", False),
                prefix=f"{prefix}.mlp",
            )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.num_hidden_layers = config.num_hidden_layers
        self.first_k_dense_replace = getattr(
            config, "first_k_dense_replace", self.num_hidden_layers
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.tp_group = get_tp_group().device_group
        self.sandwich_norm = getattr(config, "sandwich_norm", False)
        if self.sandwich_norm:
            self.pre_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_mlp_layernorm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )


class OpenPanguModel(VllmOpenPanguModel):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        attn_mlp_replace_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        has_experts = hasattr(self.config, "n_routed_experts")
        if has_experts:
            expert_merge_mapping = SharedFusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.n_routed_experts,
                num_redundant_experts=self.num_redundant_experts,
            )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            if (
                "layers" in name
                and hasattr(self.config, "num_nextn_predict_layers")
                and (self.config.num_nextn_predict_layers > 0)
            ):
                layer_idx = int(name.split("layers.")[-1].split(".")[0])
                mtp_idx = layer_idx - self.config.num_hidden_layers
                if mtp_idx >= 0 and mtp_idx < self.config.num_nextn_predict_layers:
                    continue

            flag_dict = {"is_expert_weight": False}
            if (
                self.load_attn_mlp_weight(
                    attn_mlp_replace_mapping,
                    params_dict,
                    name,
                    loaded_weight,
                    loaded_params,
                )
                or has_experts
                and self.load_expert_weight(
                    expert_merge_mapping,
                    params_dict,
                    name,
                    loaded_weight,
                    loaded_params,
                    flag_dict,
                )
            ):
                continue
            else:
                if flag_dict["is_expert_weight"]:
                    continue
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)

                if name and name.endswith("e_score_correction_bias"):
                    name = name.replace(
                        "e_score_correction_bias", "gate.e_score_correction_bias"
                    )

                if name is None:
                    continue

                if name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        self.post_weight_load()

        return loaded_params

    def post_weight_load(self) -> None:
        for _, module in self.named_modules():
            if module is self:
                continue
            if hasattr(module, "post_weight_load"):
                module.post_weight_load()


class OpenPanguMoEModel(VllmOpenPanguMoEModel):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def _remap_weights():
            for name, loaded_weight in weights:
                if name.endswith("e_score_correction_bias") and (
                    "gate.e_score_correction_bias" not in name
                ):
                    name = name.replace(
                        "e_score_correction_bias", "gate.e_score_correction_bias"
                    )
                yield name, loaded_weight

        return super().load_weights(_remap_weights())

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        for layer_idx, layer in enumerate(self.moe_layers):
            self.expert_weights.append(layer.get_expert_weights())
            layer.set_eplb_state(
                moe_layer_idx=layer_idx,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )


class PanguProMoEV2ForCausalLM(OpenPanguMoEModel):
    pass
