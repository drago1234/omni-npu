# SPDX-License-Identifier: Apache-2.0

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3MoE model compatible with HuggingFace weights."""
from collections.abc import Iterable
from typing import Any, Optional, Union, List, Tuple
import os

import torch
import torch_npu
import torch.distributed as dist
from torch import nn
from transformers import PretrainedConfig
from vllm.attention.layer import Attention
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import get_pp_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
                              ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (AutoWeightsLoader, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers)
from vllm.platforms import current_platform
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.distributed import get_pp_group

from omni_npu.v0.models.qwen.fused_moe import FusedMoE
from omni_npu.v0.layers.layernorm import RMSNormFlashComm, NPURMSNorm
from omni_npu.v0.layers.linear import (RowParallelFlashCommLinear, QKVParallelFlashCommLinear)
from omni_npu.v0.layers.rotary_embedding import get_rope
from omni_npu.v0.layers.utils import ConditionalTNGScope
from omni_npu.v0.models.config_loader.loader import model_extra_config

SEQ_SPLIT_LENGTH = 4096

class Qwen3MoeSparseMoeBlock(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.n_routed_experts = config.num_experts
        self.experts = FusedMoE(
                num_experts=self.n_routed_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                renormalize=config.norm_topk_prob,
                quant_config=quant_config,
                prefix=f"{prefix}.experts",
            )

        self.gate = RowParallelFlashCommLinear(
                config.hidden_size,
                self.n_routed_experts,
                tp_size=1,
                tp_rank=0,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate"
            )

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool = False) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states,
                                           router_logits=router_logits,
                                           is_prefill=is_prefill)
        final_hidden_states = final_hidden_states

        if (is_prefill and model_extra_config.operator_opt_config.prefill_moe_all_to_all) or \
            model_extra_config.operator_opt_config.decode_moe_dispatch_combine or \
            model_extra_config.operator_opt_config.decode_flash_comm_1:
            return final_hidden_states.view(orig_shape)
        return final_hidden_states


class Qwen3MoeAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tp_group().world_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        tp_rank = get_tp_group().rank_in_group

        self.qkv_proj = QKVParallelFlashCommLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj"
        )
        if model_extra_config.operator_opt_config.prefill_enable_mla_alltoall:
            self.o_proj = ReplicatedLinear(self.total_num_heads * self.head_dim,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.o_proj")
        else:
            self.o_proj = RowParallelFlashCommLinear(
                self.total_num_heads * self.head_dim,
                hidden_size,
                tp_size=tp_size,
                tp_rank=tp_rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj"
            )

        if rope_scaling is None:
            rope_scaling = {'factor': '0'}
        rope_scaling["rope_type"] = 'qwen'
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn"
        )
        
        self.q_norm = NPURMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = NPURMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        is_prefill = attn_metadata is None or attn_metadata.num_prefills != 0
        if (is_prefill and model_extra_config.operator_opt_config.prefill_moe_all_to_all) or \
            model_extra_config.operator_opt_config.decode_moe_dispatch_combine or \
            model_extra_config.operator_opt_config.decode_flash_comm_1:
            qkv, _ = self.qkv_proj(hidden_states, x_transform='AG', is_prefill = is_prefill)
        else:
            qkv, _ = self.qkv_proj(hidden_states, is_prefill = is_prefill)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                           self.head_dim)
        with ConditionalTNGScope(super_kernel=model_extra_config.operator_opt_config.use_super_kernel,
                                        scope="superkernel_Qwen_attn1"):
            # 这里开始做 rmsnorm
            q_by_head = self.q_norm(q_by_head)
            q = q_by_head.view(q.shape)

            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                                self.head_dim)
            # 第二个 rmsnorm
            k_by_head = self.k_norm(k_by_head)
            k = k_by_head.view(k.shape)

        # 这里做ApplyRotaryPosEmb 和 自注意力
        q, k = self.rotary_emb(positions, q, k, cos, sin)
        attn_output = self.attn(q, k, v)
        if model_extra_config.operator_opt_config.prefill_enable_mla_alltoall:
            attn_output = attn_output.reshape(-1)
            local_s = attn_output.shape[0] // (self.total_num_heads * self.head_dim)
            all_to_all_attn_output = torch.empty(
                [attn_output.shape[0]],
                dtype=attn_output.dtype,
                device=current_platform.device_type
            )
            device_group = get_tp_group().device_group
            dist.all_to_all_single(all_to_all_attn_output, attn_output, group=device_group)
            attn_output = all_to_all_attn_output.view(
                get_tensor_model_parallel_world_size(),
                local_s,
                self.num_heads * self.head_dim
            ).transpose(0, 1).contiguous().view(local_s, -1)
            output,_ = self.o_proj.forward(attn_output)
        else:
            if (is_prefill and model_extra_config.operator_opt_config.prefill_moe_all_to_all) or \
                model_extra_config.operator_opt_config.decode_moe_dispatch_combine or \
                model_extra_config.operator_opt_config.decode_flash_comm_1:
                output, _ = self.o_proj(attn_output, reduce_type="RS")
            else:
                output, _ = self.o_proj(attn_output, reduce_type="AR")
        return output


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_name = f"{prefix}.self_attn.attn"
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.self_attn = Qwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        # 'mlp_only_layers' in the config.
        layer_idx = int(prefix.split(sep='.')[-1])
        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)
        if (layer_idx not in mlp_only_layers) and (
                config.num_experts > 0 and
            (layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3MoeSparseMoeBlock(config=config,
                                              quant_config=quant_config,
                                              prefix=f"{prefix}.mlp")
        else:
            raise NotImplementedError("Qwen3MoeMLP not implemented")
        self.input_layernorm = RMSNormFlashComm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFlashComm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        is_prefill = attn_metadata is None or attn_metadata.num_prefills != 0
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        if model_extra_config.operator_opt_config.use_prefetch and not is_prefill:
            torch_npu.npu_prefetch(self.self_attn.qkv_proj.weight, hidden_states, model_extra_config.operator_opt_config.attn_prefetch * 1024 * 1024)
            torch_npu.npu_prefetch(self.self_attn.o_proj.weight, hidden_states, model_extra_config.operator_opt_config.attn_prefetch * 1024 * 1024)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            cos=cos,
            sin=sin,
            attn_metadata=attn_metadata
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
    
        if is_prefill and model_extra_config.operator_opt_config.enable_mlp_seq_split:
            local_length = hidden_states.shape[0]
            reduce_length = torch.tensor(local_length, dtype=torch.int64, device="npu")
            dist.all_reduce(reduce_length, op=dist.ReduceOp.MAX, async_op=False)
            global_max_length = reduce_length.item()
            pad_size = global_max_length - local_length
            hidden_states = torch.nn.functional.pad(
                hidden_states, (0, 0, 0, pad_size)
            )
            hidden_states_list = hidden_states.split(SEQ_SPLIT_LENGTH)
            hidden_states_out = []
            for i in range(len(hidden_states_list)):
                hidden_states = self.mlp(hidden_states_list[i], is_prefill=is_prefill)
                hidden_states_out.append(hidden_states)
            hidden_states = torch.cat(hidden_states_out)[:local_length]
        else:
            hidden_states = self.mlp(hidden_states, is_prefill=is_prefill)
        return hidden_states, residual


class Qwen3MoeModel(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()

        config = config
        cache_config = cache_config

        self.quant_config = quant_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.quant_config = quant_config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens")

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen3MoeDecoderLayer(config=config,
                                                cache_config=cache_config,
                                                quant_config=quant_config,
                                                prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNormFlashComm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        self.aux_hidden_state_layers: tuple[int] = tuple()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
    
    def get_tp_slice(self, x: torch.Tensor):
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group

        assert x.shape[0] % tp_size == 0, f"x {x.shape[0]} can't be divided along tp_size {tp_size}!"
        slice_size = x.shape[0] // tp_size
        x_slice = x[tp_rank * slice_size: (tp_rank + 1) * slice_size]

        return x_slice

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = []
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[next(iter(attn_metadata))]
        is_prefill = attn_metadata is None or attn_metadata.num_prefills != 0
        cos, sin = self.layers[self.start_layer].self_attn.rotary_emb.get_cos_sin(positions)
        if (is_prefill and model_extra_config.operator_opt_config.prefill_moe_all_to_all) or \
            model_extra_config.operator_opt_config.decode_moe_dispatch_combine or \
            model_extra_config.operator_opt_config.decode_flash_comm_1:
            # 采用FlashComm1.0, 通过slice使用hidden_states转为DP
            hidden_states = self.get_tp_slice(hidden_states)
        for i in range(self.start_layer, self.end_layer):
            if i in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)
            layer = self.layers[i]
            hidden_states, residual = layer(positions,
                                            hidden_states,
                                            residual,
                                            cos,
                                            sin,
                                            attn_metadata
                                            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        if (is_prefill and model_extra_config.operator_opt_config.prefill_moe_all_to_all) or \
            model_extra_config.operator_opt_config.decode_moe_dispatch_combine or \
            model_extra_config.operator_opt_config.decode_flash_comm_1:
            hidden_states, _ = self.norm(hidden_states, residual, y_transform='AG')
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
        if len(aux_hidden_states) > 0:
            return aux_hidden_states, hidden_states
        return hidden_states
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
    
        moe_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts)

        kv_scale_mapping = {
            "k_scale",
            "v_scale",
            "kv_cache_scale",
        }

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # Pre-compute MTP layer prefixex to skip
        num_mtp_layers = getattr(self.config, "num_nextn_predict_layers", 0)
        mtp_prefix = []
        if self.config.architectures[0] == "Qwen3MoeForCausalLM" and num_mtp_layers > 0:
            mtp_prefix = [f"layers.{self.config.num_hidden_layers+layer_idx}" 
                            for layer_idx in range(num_mtp_layers)]

        for name, loaded_weight in weights:
            # Skip MTP layers
            if mtp_prefix and name.startswith(tuple(mtp_prefix)):
                continue

            # The weight_scale often has shape (n,1)
            if 'weight_scale' in name:
                loaded_weight = loaded_weight.view(-1)
            if (self.quant_config is not None and 
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                if scale_name not in params_dict:
                    continue
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = loaded_weight.view(-1)
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:

                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in moe_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                loaded_weight,
                                name,
                                shard_id=shard_id,
                                expert_id=expert_id)
                    break
                else:
                    if any(key in name for key in kv_scale_mapping):
                        name = self.quant_config.get_cache_scale(name)
                        loaded_weight = loaded_weight.view(-1)
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

@support_torch_compile
class Qwen3MoeForCausalLM(nn.Module, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3MoeModel(config, 
                                   vllm_config.cache_config, 
                                   quant_config,
                                   prefix=f"model")
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config,
                                      parallel_lmhead=False)

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds: torch.Tensor | None = None,
    ) -> Optional[torch.Tensor]:
        attn_metadata = get_forward_context().attn_metadata
        hidden_states = self.model(input_ids, positions, attn_metadata, intermediate_tensors, inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
    
    def should_use_eager_mode(self, *args, **kwargs):
        attn_metadata = get_forward_context().attn_metadata
        if not attn_metadata:
            return True

        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.model.layers[self.model.start_layer].layer_name]

        return attn_metadata.num_prefills != 0
    
    def set_aux_hidden_state_layers(self, layers: tuple[int]) -> None:
        self.model.aux_hidden_state_layers = layers
    
    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)