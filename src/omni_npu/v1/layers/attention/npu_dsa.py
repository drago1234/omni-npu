# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Optional
import torch
import torch_npu
from transformers import DeepseekV2Config, DeepseekV3Config

try:
    import custom_ops
except:
    print("custom_ops failed to import!!!")

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.config import VllmConfig, CacheConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.attention.layer import MLAAttention

from omni_npu.attention.backends.dsa import NPUDSAMetadata
from omni_npu.v1.layers.utils import yarn_get_mscale
from omni_npu.v1.layers.linear import (
    RowParallelFlashCommLinear,
    ColumnParallelFlashCommLinear
)


class Indexer(torch.nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wk = ReplicatedLinear(
            hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wk",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = ReplicatedLinear(
            hidden_size, self.n_head, quant_config=None, prefix=f"{prefix}.weights_proj"
        )

    def _apply_lightning_indexer(
        self,
        q,
        weights,
        attn_metadata,
        kv_cache,
    ):
        if attn_metadata.prefill is not None:
            metadata = attn_metadata.prefill
        else:
            metadata = attn_metadata.decode
        actual_seq_lens_query = metadata.query_cumlens.to(torch.int32)
        actual_seq_lens_key = metadata.seq_lens.to(torch.int32)
        block_table = metadata.block_table

        return torch.ops.custom.npu_lightning_indexer(
            query=q,
            key=kv_cache[2],
            weights=weights,
            actual_seq_lengths_query=actual_seq_lens_query,
            actual_seq_lengths_key=actual_seq_lens_key,
            block_table=block_table,
            layout_key="PA_BSND",
            layout_query="TND",
            sparse_count=self.topk_tokens,
            sparse_mode=3
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional['NPUDSAMetadata'] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        q_pe = q_pe.unsqueeze(2)
        q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin)
        q_pe = q_pe.squeeze(2)
        q = torch.cat([q_pe, q_nope], dim=-1)

        k, _ = self.wk(hidden_states)
        k = self.k_norm(k).unsqueeze(1)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        k_pe = k_pe.unsqueeze(2)
        k_pe = torch_npu.npu_rotary_mul(k_pe, cos, sin)
        k_pe = k_pe.squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)

        if kv_cache[2] is not None:
            torch_npu.npu_scatter_nd_update_(
                kv_cache[2].view(-1, 1, k.shape[-1]),
                attn_metadata.slot_mapping.view(-1, 1),
                k.view(-1, 1, k.shape[-1])
            )

        weights, _ = self.weights_proj(hidden_states)
        return self._apply_lightning_indexer(q, weights, attn_metadata, kv_cache)


class NPUDeepseekSparseAttention(torch.nn.Module):
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
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.prefix = prefix
        self.quant_symbol = quant_config is not None

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
            self.q_b_proj = ColumnParallelFlashCommLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelFlashCommLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelFlashCommLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelFlashCommLinear(
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
        
        self.indexer = Indexer(
            vllm_config,
            config,
            hidden_size,
            q_lora_rank,
            quant_config,
            cache_config,
            f"{prefix}.indexer",
        )

        self.attn = MLAAttention(
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
            use_sparse=True,
            indexer=self.indexer,
        )
        self.use_mlaprolog = False

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

    def _apply_attention(
        self,
        topk_indices: torch.Tensor,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: Optional['NPUDSAMetadata'] = None
    ) -> torch.Tensor:
        if attn_metadata is None:
            return torch.zeros(
                q_nope.shape[0],
                self.num_local_heads,
                self.kv_lora_rank,
                device=q_nope.device,
                dtype=q_nope.dtype,
            )

        if attn_metadata.prefill is not None:
            metadata = attn_metadata.prefill
        else:
            metadata = attn_metadata.decode
        actual_seq_lens_query = metadata.query_cumlens.to(torch.int32)
        actual_seq_lens_kv = metadata.seq_lens.to(torch.int32)
        block_table = metadata.block_table

        return torch.ops.custom.npu_sparse_flash_attention(
            query=q_nope,
            key=k_nope,
            value=k_nope,
            sparse_indices=topk_indices,
            scale_value=self.scaling,
            sparse_block_size=1,
            block_table=block_table,
            actual_seq_lengths_query=actual_seq_lens_query,
            actual_seq_lengths_kv=actual_seq_lens_kv,
            query_rope=q_pe,
            key_rope=k_pe,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
        )
            
    def _mla_epilog(
        self,
        attn_output: torch.Tensor,
    ):
        attn_output = attn_output.transpose(0, 1)
        attn_output = (
            torch.matmul(attn_output, self.attn.impl.W_UV)
            .transpose(1, 0)
            .reshape(-1, self.num_local_heads * self.v_head_dim)
        )
        return self.o_proj(attn_output)[0]

    def _forward_prefill(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional['NPUDSAMetadata'] = None,
    ) -> torch.Tensor:
        q_lora = self.q_a_proj(hidden_states)[0]
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

        q_lora = self.q_a_layernorm(q_lora)
        q = self.q_b_proj(q_lora)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_nope = q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim).transpose(0, 1)
        q_nope = (
            torch.matmul(q_nope, self.attn.impl.W_UK_T)
            .transpose(1, 0)
            .view(-1, self.num_local_heads, self.kv_lora_rank)
        )

        q_pe = q_pe.unsqueeze(2)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin)
        q_pe = q_pe.squeeze(2)

        if attn_metadata is None:
            kv_cache = None
            latent_cache = latent_cache.view(-1, latent_cache.shape[-1])
            kv_a, _ = torch.split(latent_cache, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            latent_cache = latent_cache.unsqueeze(1)
            kv_a = self.kv_a_layernorm(kv_a)
            k_pe = latent_cache[:, :, self.kv_lora_rank:]
            k_pe = k_pe.unsqueeze(2)
            k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin)
            k_pe = k_pe.squeeze(2)
            k_nope = None
        else:
            kv_cache = self.attn.kv_cache[get_forward_context().virtual_engine]
            k_pe, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
                latent_cache.view(-1, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim), # bnsd
                self.kv_a_layernorm.weight,
                cos.view(-1, 1, 1, self.qk_rope_head_dim),
                sin.view(-1, 1, 1, self.qk_rope_head_dim),
                attn_metadata.slot_mapping,
                kv_cache[1],
                kv_cache[0],
                k_rope_scale=None,
                c_kv_scale=None,
                k_rope_offset=None,
                c_kv_offset=None,
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode="PA")

        topk_indices = None
        if attn_metadata is not None:
            topk_indices = self.indexer(hidden_states, q_lora, cos, sin, attn_metadata, kv_cache)

        attn_output = self._apply_attention(
            topk_indices, q_nope, q_pe, k_nope, k_pe, attn_metadata
        )

        return self._mla_epilog(attn_output)

    def _forward_mlaprolog(
        self,
        hidden_states,
        cos,
        sin,
        kv_cache,
        attn_metadata
    ):
        bs, _ = hidden_states.view(-1, hidden_states.shape[-1]).shape
        q_nope, q_pe, dequant_scale_q_nope, q_norm, dequant_scale_q_norm = torch.ops.custom.npu_mla_prolog_v3(
            token_x=hidden_states.view(bs, 1, -1),
            weight_dq=self.q_a_proj.weight,
            weight_uq_qr=self.q_b_proj.weight,
            weight_uk=self.attn.impl.W_UK_T,
            weight_dkv_kr=self.kv_a_proj_with_mqa.weight,
            rmsnorm_gamma_cq=self.q_a_layernorm.weight,
            rmsnorm_gamma_ckv=self.kv_a_layernorm.weight,
            rope_sin=sin.squeeze(1),
            rope_cos=cos.squeeze(1),
            kv_cache=kv_cache[1],
            kr_cache=kv_cache[0],
            cache_index=attn_metadata.slot_mapping.view(bs, -1),
            dequant_scale_x=None,
            dequant_scale_w_dq=None,
            dequant_scale_w_uq_qr=self.q_b_proj.weight_scale.view(1, -1) if self.quant_symbol else None,
            dequant_scale_w_dkv_kr=None,
            rmsnorm_epsilon_cq=self.q_a_layernorm.variance_epsilon,
            rmsnorm_epsilon_ckv=self.kv_a_layernorm.variance_epsilon,
            cache_mode="PA_BSND",
            query_norm_flag=True,
            weight_quant_mode=1 if self.quant_symbol else 0
        )
        k_nope = kv_cache[0]
        k_pe = kv_cache[1]
        q_nope = q_nope.view(bs, self.num_local_heads, self.kv_lora_rank)
        q_pe = q_pe.view(bs, self.num_local_heads, -1)
        if self.quant_symbol:
            q_norm = q_norm.view(-1, q_norm.shape[-1])
            dequant_scale_q_norm = dequant_scale_q_norm.view(-1)
            q_norm = {'x_int8': q_norm, 'pertoken_scale': dequant_scale_q_norm}
        return q_nope, q_pe, q_norm, k_nope, k_pe, dequant_scale_q_nope,dequant_scale_q_norm

    def _forward_decode(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional['NPUDSAMetadata'] = None,
    ) -> torch.Tensor:
        kv_cache = self.attn.kv_cache[get_forward_context().virtual_engine]
        if self.use_mlaprolog:
            q_nope, q_pe, q_norm, k_nope, k_pe, dequant_scale_q_nope, dequant_scale_q_norm = \
                self._forward_mlaprolog(hidden_states, cos, sin, attn_metadata, kv_cache)
        else:
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
                torch.matmul(q_nope, self.attn.impl.W_UK_T)
                .transpose(1, 0)
                .view(bsz, 1, self.num_local_heads, -1)
            )
            k_pe, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
                kv.unsqueeze(1).unsqueeze(1),
                self.kv_a_layernorm.weight,
                cos,
                sin,
                attn_metadata.slot_mapping,
                kv_cache[1],
                kv_cache[0],
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode="PA"
            )
            q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin) # BNSD
            q_nope = q_nope.view(bsz, self.num_local_heads, self.kv_lora_rank)
            q_pe = q_pe.view(bsz, self.num_local_heads, -1)
        tok_indices = self.indexer(hidden_states, q_norm, cos, sin, attn_metadata, kv_cache)
        attn_output = self._apply_attention(
            tok_indices, q_nope, q_pe, k_nope, k_pe, attn_metadata
        )
        return self._mla_epilog(attn_output)
