# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn
from transformers import GptOssConfig, PretrainedConfig

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.sequence import IntermediateTensors

from omni_npu.attention.backends.attention import NPUAttentionBackend
from omni_npu.v1.layers.fused_moe.layer import NPUFusedMoEV1
from omni_npu.v1.layers.linear import (
    QKVParallelFlashCommLinear,
    RowParallelFlashCommLinear,
)


class GptOssAttention(nn.Module):
    def __init__(
        self,
        config: GptOssConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.head_dim = config.head_dim

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = self.total_num_heads // tp_size
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
        else:
            self.num_kv_heads = self.total_num_kv_heads // tp_size

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelFlashCommLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.qkv_proj.x_transform = "NoOp"
        self.qkv_proj.y_transform = "NoOp"
        self.o_proj = RowParallelFlashCommLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.o_proj.x_transform = "NoOp"
        self.o_proj.y_transform = "AllReduce"

        rope_params = dict(config.rope_parameters)
        rope_params["rope_type"] = "yarn"
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            dtype=torch.float32,
            rope_parameters=rope_params,
            is_neox_style=True,
        )

        self.sinks = nn.Parameter(
            torch.empty(self.num_heads, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            sliding_window = (
                config.sliding_window if self.layer_idx % 2 == 0 else None
            )
        else:
            layer_type = layer_types[self.layer_idx]
            sliding_window = None if layer_type == "full_attention" else config.sliding_window

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
            attn_backend=NPUAttentionBackend,
            sinks=self.sinks,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class GptOssMoE(nn.Module):
    def __init__(
        self,
        config: GptOssConfig,
        quant_config: QuantizationConfig | None = None,
        parallel_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        self.router = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=True,
            quant_config=None,
            prefix=f"{prefix}.router",
        )

        self.experts = NPUFusedMoEV1(
            shared_experts=None,
            gate=self.router,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            reduce_results=True,
            renormalize=True,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            apply_router_weight_on_input=False,
            has_bias=True,
            activation="swigluoai",
            is_sequence_parallel=self.is_sequence_parallel,
        )
        self.experts.swiglu_limit = config.swiglu_limit
        self.experts.glu_alpha = 1.702
        self.experts.glu_bias = 1.0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        _, out = self.experts(hidden_states=hidden_states, router_logits=None)

        if self.is_sequence_parallel:
            out = tensor_model_parallel_all_gather(out.contiguous(), 0)
            out = out[:num_tokens]
        return out


class GptOssDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GptOssConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = GptOssAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = GptOssMoE(
            config=config,
            quant_config=quant_config,
            parallel_config=parallel_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=1e-5)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=1e-5)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    },
)
class GptOssModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.parallel_config = vllm_config.parallel_config
        self.quant_config = vllm_config.quant_config

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.num_hidden_layers,
            lambda prefix: GptOssDecoderLayer(
                config=self.config,
                model_config=vllm_config.model_config,
                cache_config=vllm_config.cache_config,
                quant_config=self.quant_config,
                parallel_config=vllm_config.parallel_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(self.config.hidden_size, eps=1e-5)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"],
            self.config.hidden_size,
        )

    @staticmethod
    def _remap_name(name: str) -> str:
        if name.startswith("model."):
            name = name[len("model.") :]
        if name.endswith(".mlp.experts.down_proj.scale"):
            return name.replace(".mlp.experts.down_proj.scale", ".mlp.experts.w2_weight_scale")
        if name.endswith(".mlp.experts.gate_up_proj.scale"):
            return name.replace(".mlp.experts.gate_up_proj.scale", ".mlp.experts.w13_weight_scale")
        if name.endswith(".mlp.experts.down_proj.bias"):
            return name.replace(".mlp.experts.down_proj.bias", ".mlp.experts.w2_bias")
        if name.endswith(".mlp.experts.gate_up_proj.bias"):
            return name.replace(".mlp.experts.gate_up_proj.bias", ".mlp.experts.w13_bias")
        if name.endswith(".mlp.experts.down_proj.weight"):
            return name.replace(".mlp.experts.down_proj.weight", ".mlp.experts.w2_weight")
        if name.endswith(".mlp.experts.gate_up_proj.weight"):
            return name.replace(".mlp.experts.gate_up_proj.weight", ".mlp.experts.w13_weight")
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        ep_group = get_ep_group()
        ep_size = ep_group.world_size
        ep_rank = ep_group.rank_in_group
        experts_per_rank = self.config.num_local_experts // ep_size
        ep_rank_start = ep_rank * experts_per_rank
        ep_rank_end = (ep_rank + 1) * experts_per_rank

        use_ep = self.parallel_config.enable_expert_parallel
        intermediate_size = self.config.intermediate_size
        per_rank_intermediate_size = (intermediate_size + tp_size - 1) // tp_size
        tp_rank_start = tp_rank * per_rank_intermediate_size
        tp_rank_end = min((tp_rank + 1) * per_rank_intermediate_size, intermediate_size)

        for raw_name, loaded_weight in weights:
            mapped_name = self._remap_name(raw_name)
            if "rotary_emb.inv_freq" in mapped_name:
                continue
            if is_pp_missing_parameter(mapped_name, self):
                continue

            if mapped_name.endswith(".mlp.experts.w13_weight"):
                if use_ep:
                    local_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
                else:
                    local_weight = loaded_weight[:, :, 2 * tp_rank_start : 2 * tp_rank_end]
                params_dict[mapped_name].copy_(local_weight.permute(0, 2, 1).contiguous())
                loaded_params.add(mapped_name)
                continue

            if mapped_name.endswith(".mlp.experts.w2_weight"):
                if use_ep:
                    local_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
                else:
                    local_weight = loaded_weight[:, tp_rank_start:tp_rank_end, :]
                params_dict[mapped_name].copy_(local_weight.permute(0, 2, 1).contiguous())
                loaded_params.add(mapped_name)
                continue

            if mapped_name.endswith(".mlp.experts.w13_weight_scale"):
                if use_ep:
                    local_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
                else:
                    local_weight = loaded_weight[:, :, 2 * tp_rank_start : 2 * tp_rank_end]
                params_dict[mapped_name].copy_(local_weight.permute(0, 2, 1).contiguous())
                loaded_params.add(mapped_name)
                continue

            if mapped_name.endswith(".mlp.experts.w2_weight_scale"):
                if use_ep:
                    local_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
                else:
                    local_weight = loaded_weight
                params_dict[mapped_name].copy_(local_weight.permute(0, 2, 1).contiguous())
                loaded_params.add(mapped_name)
                continue

            if mapped_name.endswith(".mlp.experts.w13_bias"):
                if use_ep:
                    local_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
                else:
                    local_weight = loaded_weight[:, 2 * tp_rank_start : 2 * tp_rank_end]
                params_dict[mapped_name].copy_(local_weight)
                loaded_params.add(mapped_name)
                continue

            if mapped_name.endswith(".mlp.experts.w2_bias"):
                if use_ep:
                    local_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
                elif tp_rank == 0:
                    local_weight = loaded_weight
                else:
                    local_weight = torch.zeros_like(loaded_weight)
                params_dict[mapped_name].copy_(local_weight)
                loaded_params.add(mapped_name)
                continue

            if mapped_name.endswith(".self_attn.sinks"):
                params_dict[mapped_name].copy_(
                    loaded_weight.narrow(0, head_start, heads_per_rank)
                )
                loaded_params.add(mapped_name)
                continue

            found_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in mapped_name:
                    continue
                stacked_name = mapped_name.replace(weight_name, param_name)
                if stacked_name.endswith(".bias") and stacked_name not in params_dict:
                    found_stacked = True
                    break
                param = params_dict[stacked_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(stacked_name)
                found_stacked = True
                break
            if found_stacked:
                continue

            if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                continue
            if mapped_name not in params_dict:
                continue

            param = params_dict[mapped_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(mapped_name)

        return loaded_params

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.get_input_embeddings(input_ids)
            )
            residual = None
        else:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for idx in range(self.start_layer, self.end_layer):
            layer = self.layers[idx]
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class GptOssForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = GptOssModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)

    def process_weights_after_loading(self) -> None:
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            layer.mlp.experts.process_weights_after_loading()
