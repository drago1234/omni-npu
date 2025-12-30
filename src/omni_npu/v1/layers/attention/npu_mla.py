# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Optional
import torch
import torch_npu
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.config import VllmConfig, CacheConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
    ColumnParallelLinear,
    RowParallelLinear
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.attention.layer import MLAAttention

from omni_npu.attention.backends.mla import NPUMLAMetadata
from omni_npu.v1.layers.utils import yarn_get_mscale

KVCACHE_NZ_DIM = 16


class NPUDeepseekMLAAttention(torch.nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
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
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.quant_symbol = quant_config is not None
        self.prefix = prefix

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_a_proj",
            )
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )

        if self.q_lora_rank is not None:
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
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

        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )

        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )

        if (
            config.rope_parameters["rope_type"] != "default"
            and config.rope_parameters["rope_type"] == "deepseek_yarn"
        ):
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.mla_attn = MLAAttention(
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=False,
            indexer=None,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[f"{self.prefix}.attn"]

        if attn_metadata is None or attn_metadata.prefill is not None:
            return self._forward_prefill(hidden_states, cos, sin, attn_metadata)
        else:
            return self._forward_decode(hidden_states, cos, sin, attn_metadata)

    def _forward_decode(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional['NPUMLAMetadata'] = None,
    ) -> torch.Tensor:
        kv_cache = self.mla_attn.kv_cache[get_forward_context().virtual_engine]
        nz_block_size = 16
        if self.q_lora_rank is not None:
            q_lora = self.q_a_proj(hidden_states)[0]
        else:
            q_lora = self.q_proj(hidden_states)[0]
        kv = self.kv_a_proj_with_mqa(hidden_states)[0]

        if self.q_lora_rank is not None:
            q_norm = self.q_a_layernorm(q_lora)
            q = self.q_b_proj(q_norm)[0]
        else:
            q = q_lora

        bsz, _ = q.shape
        q = q.view(bsz, self.num_local_heads, 1, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1) # b,n,s,d
        q_nope = q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim).transpose(0, 1) # n, bs, d
        q_nope = (
            torch.matmul(q_nope, self.mla_attn.impl.W_UK_T)
            .transpose(1, 0)
            .view(bsz, 1, self.num_local_heads, -1)
        )

        block_num, block_size, head_size, _ = kv_cache[0].shape
        k_rope, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
            kv.unsqueeze(1).unsqueeze(1),
            self.kv_a_layernorm.weight,
            cos,
            sin,
            attn_metadata.slot_mapping,
            kv_cache[1],
            kv_cache[0],
            epsilon=self.kv_a_layernorm.variance_epsilon,
            cache_mode="PA_NZ"
        )

        k_nope = k_nope.view(block_num, 1, self.kv_lora_rank // nz_block_size, block_size, nz_block_size)
        k_rope = k_rope.view(block_num, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin) # BNSD
        q_nope = q_nope.view(bsz, self.num_local_heads, self.kv_lora_rank)
        q_pe = q_pe.view(bsz, self.num_local_heads, -1)
        kwargs = {
            "query": q_nope,
            "key": k_nope,
            "value": k_nope,
            "query_rope": q_pe,
            "key_rope": k_rope,
            "num_heads": self.num_local_heads,
            "num_key_value_heads": 1,
            "input_layout": "TND_NTD",
            "atten_mask": None,
            "sparse_mode": 0,
            "scale": self.scaling,
            "antiquant_mode": 0,
            "antiquant_scale": None,
            "block_table": attn_metadata.decode.block_table,
            "block_size": 128,
            "actual_seq_lengths": attn_metadata.decode.query_cumlens,
            "actual_seq_lengths_kv": attn_metadata.decode.seq_lens,
        }
        attn_output = torch.ops.npu.npu_fused_infer_attention_score(**kwargs)[0]

        # Apply UV, (N, B, L) @ W_UV (N, L, V) -> (N, B, V)
        attn_output = attn_output.view(self.num_local_heads, bsz, self.kv_lora_rank) # adapter BSND_NBSD
        attn_output = (
            torch.matmul(attn_output, self.mla_attn.impl.W_UV)
            .transpose(1, 0)
            .reshape(bsz, 1, -1)
        )
        attn_output = attn_output.view(-1, self.num_local_heads * self.v_head_dim)
        return self.o_proj.forward(attn_output)[0]

    def _forward_prefill(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional['NPUMLAMetadata'] = None,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            if self.quant_symbol:
                q_quant, q_scale = torch_npu.npu_dynamic_quant(q)
                q = {'x_int8':q_quant, 'pertoken_scale':q_scale}
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(-1, self.num_local_heads, self.qk_head_dim)
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim],  dim=-1)
        q_pe = q_pe.unsqueeze(2)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin) # BNSD
        q_pe = q_pe.squeeze(2) # BSH
        q[..., self.qk_nope_head_dim:] = q_pe

        latent_cache = latent_cache.view(-1, latent_cache.size(-1))
        attn_output = torch.empty(
            q.shape[0],
            self.num_local_heads,
            self.v_head_dim,
            device=q_nope.device,
            dtype=q_nope.dtype)

        if attn_metadata is None:
            latent_cache = latent_cache.view(-1, latent_cache.size(-1))
            kv_a, _ = torch.split(latent_cache, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            latent_cache = latent_cache.unsqueeze(1)
            kv_a = self.kv_a_layernorm(kv_a)
            k_pe = latent_cache[:, :, self.kv_lora_rank:]
            k_pe = k_pe.unsqueeze(2)
            k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin)
            k_pe = k_pe.squeeze(2)
            attn_output.fill_(0)
        else:
            kv_cache = self.mla_attn.kv_cache[get_forward_context().virtual_engine]
            _, _, k_pe, kv_a = torch_npu.npu_kv_rmsnorm_rope_cache(
                latent_cache.view(-1, 1, 1, 576), # bnsd
                self.kv_a_layernorm.weight,
                cos.view(-1, 1, 1, self.qk_rope_head_dim),
                sin.view(-1, 1, 1, self.qk_rope_head_dim),
                attn_metadata.slot_mapping,
                kv_cache[1],
                kv_cache[0],
                k_rope_scale=None,
                k_rope_offset=None,
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode="PA_NZ",
                is_output_kv=True
            )

            prefill_metadata = attn_metadata.prefill
            actual_seq_kvlen = prefill_metadata.seq_lens
            actual_seq_qlen = prefill_metadata.query_cumlens
            prefill_kv_a = kv_a[:actual_seq_kvlen[-1]]
            prefill_k_pe = k_pe[:actual_seq_kvlen[-1]]
            kv = self.kv_b_proj.forward(prefill_kv_a)[0]
            kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            if prefill_metadata.max_query_len > 1:
                attn_mask = self.mla_attn.impl.SHARE_MASK_TRIL_SPARSE
                sparse_mode = 3
            else:
                attn_mask = None
                sparse_mode = 0
            prefill_k_rope = prefill_k_pe.view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_local_heads, 1)

            attn_output[:actual_seq_qlen[-1]] = torch.ops.npu.npu_fused_infer_attention_score(
                q_nope[:actual_seq_qlen[-1]],
                k_nope,
                v,
                query_rope=q_pe[:actual_seq_qlen[-1]],
                key_rope=prefill_k_rope,
                num_heads=self.num_local_heads,
                num_key_value_heads=self.num_local_heads,
                input_layout="TND",
                atten_mask=attn_mask,
                sparse_mode=sparse_mode,
                actual_seq_lengths=actual_seq_qlen,
                actual_seq_lengths_kv=actual_seq_kvlen,
                scale=self.scaling,
                next_tokens=0
            )[0]

        attn_output = attn_output.view(-1, self.num_local_heads * self.v_head_dim)
        output = self.o_proj.forward(attn_output)[0]
        return output