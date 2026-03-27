# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Optional, Union, Tuple

import torch
import torch_npu
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm.model_executor.models.utils import extract_layer_index
from vllm.distributed import get_tp_group
from vllm.config import VllmConfig, CacheConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.attention.layer import MLAAttention
from vllm.model_executor.layers.linear import (
    RowParallelLinear,
    ColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.logger import init_logger

from omni_npu.v1.utils import current_stream, on_ascend950
from omni_npu.v1.layers.utils import yarn_get_mscale
from omni_npu.v1.models.config_loader.loader import model_extra_config

from omni_npu.layers.mome.npu_mome import ColumnParallelMOME
from omni_npu.layers.attention.npu_sparse_attentions import (
    MLASWAAttention,
    DSAAttention,
    MomeAttention,
)

logger = init_logger(__name__)
try:
    import omni_custom_ops
except ImportError as e:
    logger.warning(f"Failed to import omni_custom_ops: {e}")


class NPUPanguIndexer(torch.nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.index_topk = config.index_topk
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.hidden_size = config.hidden_size
        self.quant_config = quant_config
        self.cache_config = cache_config
        self.layer_name = prefix
        self.block_size = cache_config.block_size
        self.enable_non_contiguous_kv = model_extra_config.operator_opt_config.use_noncontiguous_kv
        self.on_ascend950 = on_ascend950()
        self._init_indexer_weights()

    def _init_indexer_weights(self):
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.index_head_dim * self.index_n_heads,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{self.layer_name}.wq_b",
            return_bias=False,
        )
        if self.cache_config.cache_dtype in ["hif8_ds_mla", "fp8_ds_mla"]:
            self.wq_b.weight.data = torch.ones_like(self.wq_b.weight.data)
            self.wq_b.weight_scale.data = torch.ones_like(self.wq_b.weight_scale.data)

        self.wk = ReplicatedLinear(
            self.hidden_size,
            self.index_head_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{self.layer_name}.wk",
            return_bias=False,
        )
        self.k_norm = RMSNorm(
            self.index_head_dim,
            eps=self.config.rms_norm_eps,
        )
        self.weights_proj = ReplicatedLinear(
            self.hidden_size,
            self.index_n_heads,
            quant_config=None,
            bias=False,
            prefix=f"{self.layer_name}.weights_proj",
            return_bias=False,
        )

    def _apply_lightning_indexer(self, *args, **kwargs):
        if quant_output := self._apply_lightning_indexer_quant(*args, **kwargs):
            return quant_output
        else:
            unquant_output = self._apply_lightning_indexer_unquant(*args, **kwargs)
            return unquant_output

    def _update_indexer_cache(self, *args, **kwargs):
        if quant_output := self._update_indexer_cache_quant(*args, **kwargs):
            return quant_output
        else:
            unquant_output = self._update_indexer_cache_unquant(*args, **kwargs)
            return unquant_output

    def _apply_lightning_indexer_unquant(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if attn_metadata.prefill is not None:
            metadata = attn_metadata.prefill
        else:
            metadata = attn_metadata.decode

        return torch.ops.custom.npu_lightning_indexer_enhance(
            query=q,
            key=kv_cache[1].unsqueeze(2),
            weights=weights,
            actual_seq_lengths_query=metadata.query_cumlens.to(torch.int32),
            actual_seq_lengths_key=metadata.seq_lens.to(torch.int32),
            block_table=metadata.block_table,
            layout_key="PA_BSND",
            layout_query="TND",
            sparse_count=self.index_topk,
            sparse_mode=3,
            sparse_block_size=1,
            sparse_block_mode=False,
        )[0]

    def _apply_lightning_indexer_quant(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> Union[torch.Tensor, bool]:
        if attn_metadata.prefill is not None:
            metadata = attn_metadata.prefill
        else:
            metadata = attn_metadata.decode

        if self.on_ascend950 and self.cache_config.cache_dtype in ["hif8_ds_mla"]:
            q_scale = torch.ones(
                (q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device,
            )
            q_hif8 = torch_npu.npu_dtype_cast(q, torch_npu.hifloat8)

            return torch_npu.npu_quant_lightning_indexer(
                query=q_hif8,
                key=kv_cache[1],
                weights=weights,
                query_dequant_scale=q_scale,
                key_dequant_scale=kv_cache[2].squeeze(-1),
                actual_seq_lengths_query=metadata.query_cumlens.to(torch.int32),
                actual_seq_lengths_key=metadata.seq_lens.to(torch.int32),
                block_table=metadata.block_table,
                query_quant_mode=0,
                key_quant_mode=0,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
                query_dtype=torch_npu.hifloat8,
                key_dtype=torch_npu.hifloat8
            )
        else:
            return False

    def _update_indexer_cache_unquant(
        self,
        k: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> bool:

        slot_indices = torch.stack([
            attn_metadata.slot_mapping // self.block_size,
            attn_metadata.slot_mapping % self.block_size,
            ], dim=1,
        )
        torch.ops.custom.npu_ai_infra_scatter_block_update_(
            kv_cache[1],
            slot_indices,
            k.view(-1, k.shape[-1]),
        )
        return True

    def _update_indexer_cache_quant(
        self,
        k: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> bool:

        if self.on_ascend950 and self.cache_config.cache_dtype in ["hif8_ds_mla"]:
            k_scale = torch.ones(
                (k.shape[0], 1), dtype=torch.float32, device=k.device,
            )
            k_hif8 = torch_npu.npu_dtype_cast(k, torch_npu.hifloat8)
            torch_npu.npu_scatter_nd_update_(
                kv_cache[1].view(-1, k_hif8.shape[-1]).view(torch.int8),
                attn_metadata.slot_mapping.view(-1, 1),
                k_hif8.view(-1, k_hif8.shape[-1]).view(torch.int8),
            )
            torch_npu.npu_scatter_nd_update_(
                kv_cache[2].view(-1, k_scale.shape[-1]),
                attn_metadata.slot_mapping.view(-1, 1),
                k_scale.view(-1, k_scale.shape[-1]),
            )
            return True

        else:
            return False

    def _indexer_prolog(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        q = self.wq_b(qr)
        q = q.view(-1, self.index_n_heads, self.index_head_dim)
        q_pe, q_nope = torch.split(
            q,
            [self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim],
            dim=-1,
        )
        q_pe = torch_npu.npu_rotary_mul(
            q_pe.view(-1, 1, self.index_n_heads, self.qk_rope_head_dim),
            cos.view(-1, 1, 1, self.qk_rope_head_dim),
            sin.view(-1, 1, 1, self.qk_rope_head_dim),
        ).squeeze(1)
        q = torch.cat([q_pe, q_nope], dim=-1)

        k = self.wk(hidden_states)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, 
            [self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim],
            dim=-1,
        )
        k_pe = torch_npu.npu_rotary_mul(
            k_pe.view(-1, 1, 1, self.qk_rope_head_dim),
            cos.view(-1, 1, 1, self.qk_rope_head_dim),
            sin.view(-1, 1, 1, self.qk_rope_head_dim),
        ).squeeze(1).squeeze(1)

        k = torch.cat([k_pe, k_nope], dim=-1)

        weights = self.weights_proj(hidden_states)

        return q, k, weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:

        q, k, weights = self._indexer_prolog(
            hidden_states,
            qr,
            cos,
            sin,
        )

        self._update_indexer_cache(
            k,
            attn_metadata,
            kv_cache,
        )

        return self._apply_lightning_indexer(
            q,
            weights,
            attn_metadata,
            kv_cache,
        )


class NPUPanguSparseAttention(torch.nn.Module):
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
        rope_theta: int,
        swa_layers: list[int],
        param_sink_number: int,
        sliding_window_list: list[int],
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.prefix = prefix
        self.layer_idx = extract_layer_index(self.prefix)
        assert len(swa_layers) == len(sliding_window_list)
        self.swa_layers = swa_layers if swa_layers else []
        self.sliding_window_list = sliding_window_list if sliding_window_list else []
        self.aligned_window_size = max(self.sliding_window_list)
        if self.layer_idx in self.swa_layers:
            pos_in_swa = self.swa_layers.index(self.layer_idx)
            self.sliding_window = self.sliding_window_list[pos_in_swa]
            self.is_dsa_layer = False
        elif self.layer_idx >= config.num_hidden_layers:
            self.sliding_window = self.sliding_window_list[-1]
            self.is_dsa_layer = False
        else:
            self.sliding_window = None
            self.is_dsa_layer = hasattr(config, "index_topk")
            self.index_topk = getattr(config, "index_topk", None)
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.rope_theta = rope_theta
        self.num_heads = num_heads
        self.tp_size = get_tp_group().world_size
        assert num_heads % self.tp_size == 0
        self.num_local_heads = num_heads // self.tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.quant_symbol = quant_config is not None
        self.rope_interleaved = getattr(config, "rope_interleaved", True)
        self.all2all_backend = vllm_config.parallel_config.all2all_backend
        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self.cache_config = cache_config
        self.hf_config = config
        self.layer_name = prefix
        self.param_sink_number = param_sink_number
        self.on_ascend950 = on_ascend950()
        self.enable_non_contiguous_kv = model_extra_config.operator_opt_config.use_noncontiguous_kv
        self.dummy_value_cache = torch.zeros(
            (1, cache_config.block_size, 1, self.kv_lora_rank),
            device='npu',
            dtype=torch.bfloat16,
        )

        self._init_MLA_weights()
        self._init_rotary_emb()
        self._init_param_sinks()
        self._align_pagesize()
        self._init_attention_layers()
        self._init_mome_layer()

    def _init_MLA_weights(self):
        self.q_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{self.layer_name}.q_a_proj",
            return_bias=False,
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{self.layer_name}.kv_a_proj_with_mqa",
            return_bias=False,
        )
        self.q_a_layernorm = RMSNorm(
            self.q_lora_rank,
            eps=self.hf_config.rms_norm_eps,
        )
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{self.layer_name}.q_b_proj",
            return_bias=False,
        )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            eps=self.hf_config.rms_norm_eps,
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{self.layer_name}.kv_b_proj",
            return_bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            reduce_results=False,
            prefix=f"{self.layer_name}.o_proj",
            return_bias=False,
        )

    def _init_rotary_emb(self):
        rope_parameters = {
            "rope_theta": self.hf_config.rope_parameters["rope_theta"],
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": self.max_position_embeddings,
            "type": "yarn",
            "rope_type": "deepseek_yarn",
        }
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=(not self.rope_interleaved),
        )
        if (
            self.hf_config.rope_parameters["rope_type"] != "default"
            and self.hf_config.rope_parameters["rope_type"] == "deepseek_yarn"
        ):
            mscale_all_dim = self.hf_config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = self.hf_config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

    def _init_param_sinks(self):
        self.param_sink_compressed_kv = torch.nn.Parameter(
            torch.empty(
                (self.param_sink_number, self.kv_lora_rank), 
                device='npu', 
                dtype=torch.bfloat16,
            )
        )
        self.param_sink_k_pe = torch.nn.Parameter(
            torch.empty(
                (self.param_sink_number, self.qk_rope_head_dim), 
                device='npu', 
                dtype=torch.bfloat16,
            )
        )
        assert self.cache_config.block_size == self.param_sink_number
        self.block_size = self.cache_config.block_size
        self.sink_slot_mapping = torch.arange(
            self.param_sink_number, device='npu', dtype=torch.int32,
        )
        self.sink_slot_indices = torch.stack(
            [self.sink_slot_mapping // self.block_size, self.sink_slot_mapping % self.block_size],
            dim=1,
        )

    def _align_pagesize(self):
        self.use_mome = getattr(self.hf_config, "use_mome", False)
        self.mome_kernel_width = getattr(self.hf_config, "router_sliding_window", 0)
        if self.use_mome:
            self.mome_state_shapes = (
                (self.q_lora_rank,),
                (self.kv_lora_rank,),
                (self.num_heads * self.v_head_dim,),
            )
            self.mome_state_dtypes = (
                torch.bfloat16,
                torch.bfloat16,
                torch.bfloat16,
            )

        if not self.vllm_config.speculative_config:
            self.num_spec_tokens = 0
        else:
            self.num_spec_tokens = self.vllm_config.speculative_config.num_speculative_tokens

        # TODO: support more cache_dtype_str
        self.cache_dtype_str = None
        self.page_size_padded = self._calculate_page_size_padded(
            cache_config=self.cache_config,
            cache_dtype_str=self.cache_dtype_str,
            config=self.hf_config,
        )

    def _init_attention_layers(self):
        if self.is_dsa_layer:
            self.indexer = NPUPanguIndexer(
                self.vllm_config,
                self.hf_config,
                self.quant_config,
                self.cache_config,
                f"{self.layer_name}.indexer",
            )
        else:
            self.indexer = None

        attn_kwargs = {
            "num_heads": self.num_local_heads,
            "scale":  self.scaling,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            "q_lora_rank": self.q_lora_rank,
            "kv_lora_rank": self.kv_lora_rank,
            "kv_b_proj": self.kv_b_proj,
            "quant_config": self.quant_config,
            "cache_config": self.cache_config,
            "prefix": f"{self.layer_name}.attn",
        }

        if not self.enable_non_contiguous_kv:
            attn_kwargs.update({
                "use_sparse": self.is_dsa_layer,
                "indexer": self.indexer,
            })
            self.attn = MLAAttention(**attn_kwargs)
        elif self.is_dsa_layer:
            attn_kwargs.update({
                "indexer": self.indexer,
                "indexer_head_dim": self.indexer.index_head_dim,
                "cache_dtype_str": self.cache_dtype_str,
                "page_size_padded": self.page_size_padded,
            })
            self.attn = DSAAttention(**attn_kwargs)
        else:
            attn_kwargs.update({
                "cache_dtype_str": self.cache_dtype_str,
                "page_size_padded": self.page_size_padded,
                "sliding_window": self.aligned_window_size if self.sliding_window is not None else None,
            })
            self.attn = MLASWAAttention(**attn_kwargs)

    def _init_mome_layer(self):
        if not self.use_mome:
            return

        if self.enable_non_contiguous_kv:
            mome_kwargs = {
                "kernel_size": self.mome_kernel_width,
                "num_spec_tokens": self.num_spec_tokens,
                "state_dtypes": self.mome_state_dtypes,
                "state_shapes": self.mome_state_shapes,
                "quant_config": self.quant_config,
                "cache_config": self.cache_config,
                "prefix": f"{self.layer_name}.mome",
                "page_size_padded": self.page_size_padded,
            }
            self.mome_attn = MomeAttention(**mome_kwargs)

        self.qa_conv = ColumnParallelMOME(
            dim=self.q_lora_rank,
            kernel_width=self.mome_kernel_width,
            prefix=f"{self.layer_name}.qa_conv",
            disable_tp=True,
        )
        self.compresskv_conv = ColumnParallelMOME(
            dim=self.kv_lora_rank,
            kernel_width=self.mome_kernel_width,
            prefix=f"{self.layer_name}.compresskv_conv",
            disable_tp=True,
        )
        self.o_conv = ColumnParallelMOME(
            dim=self.num_heads * self.v_head_dim,
            kernel_width=self.mome_kernel_width,
            prefix=f"{self.layer_name}.o_conv",
            disable_tp=False,
        )

        # prepare the fake cache, the fake cache indices
        # this should be removed in a functional version
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        self.q_mome_cache = torch.zeros(
            (max_num_seqs,
            self.mome_kernel_width - 1 + self.num_spec_tokens,
            self.q_lora_rank),
            device='npu',
        )
        self.kv_mome_cache = torch.zeros(
            (max_num_seqs,
            self.mome_kernel_width - 1 + self.num_spec_tokens,
            self.kv_lora_rank),
            device='npu',
        )
        self.o_mome_cache = torch.zeros(
            (max_num_seqs,
            self.mome_kernel_width - 1 + self.num_spec_tokens,
            self.o_conv.output_size_per_partition),
            device='npu',
        )
        self.cache_indices = torch.arange(
            0, max_num_seqs, dtype=torch.int32,
            device='npu',
        )

    def _calculate_page_size_padded(
        self,
        cache_config: CacheConfig,
        cache_dtype_str: str | None,
        config: DeepseekV2Config | DeepseekV3Config,
    ) -> int | None:
        """
        Calculate page_size_padded for alignment across different attention mechanisms.

        Alignment priority:
        1. If DSA exists: align to DSA page size
        2. Otherwise: align to max(MOME page size, MLA/SWA page size)

        Args:
            cache_config: Cache configuration
            cache_dtype_str: Quantization dtype string (e.g., "fp8_ds_mla", "hif8_ds_mla", "int8_ds_mla")
            config: Model configuration

        Returns:
            page_size_padded in bytes, or None if no padding needed
        """
        from vllm.utils.torch_utils import get_dtype_size
        from math import prod

        block_size = cache_config.block_size
        dtype = torch.bfloat16  # Default dtype
        dtype_size = get_dtype_size(dtype)

        # Calculate MLA/SWA page size
        mla_head_size = self.kv_lora_rank + self.qk_rope_head_dim
        mla_page_size = block_size * mla_head_size * dtype_size

        # Calculate DSA page size if DSA layer exists
        dsa_page_size = None
        if hasattr(config, "index_topk") and config.index_topk > 0:
            index_head_dim = getattr(config, "index_head_dim", 0)
            if cache_dtype_str in ["fp8_ds_mla", "hif8_ds_mla"]:
                # Quant case: 512 fp8 + 64 bf16 + 4 fp32 + 128 int8 + 1 fp32
                # See DeepseekV3 quantized DSA format
                dsa_page_size = self.block_size * (656 + 128 + 4)
            elif cache_dtype_str == "int8_ds_mla":
                # Quant case: 512 int8 + 64 bf16 + 4 fp32 + 128 int8 + 1 bf16
                dsa_page_size = self.block_size * (656 + 128 + 2)
            else:
                # Non-quant case: standard attention format
                dsa_page_size = block_size * (mla_head_size + index_head_dim) * dtype_size

        # Calculate MOME page size if MOME is enabled
        mome_page_size = None
        if self.use_mome:
            num_total_tokens = self.mome_kernel_width - 1 + self.num_spec_tokens
            mome_page_size = sum(
                prod(shape) * get_dtype_size(dtype)
                for (shape, dtype) in zip(self.mome_state_shapes, self.mome_state_dtypes)
            ) * num_total_tokens

        # Determine alignment priority
        if dsa_page_size is not None:
            target_page_size = dsa_page_size
        elif mome_page_size is not None:
            target_page_size = max(mome_page_size, mla_page_size)
        else:
            target_page_size = mla_page_size

        return target_page_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[f"{self.prefix}.attn"]

        if attn_metadata is None:
            return self._forward_dummy(
                hidden_states,
            )
        elif attn_metadata.prefill is not None:
            return self._forward_prefill(
                hidden_states,
                cos,
                sin,
                attn_metadata,
            )
        else:
            return self._forward_decode(
                hidden_states,
                cos,
                sin,
                attn_metadata,
            )

    def _forward_dummy(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        if self.tp_size > 1:
            if not self.all2all_backend == "naive":
                hidden_states = get_tp_group().all_gather(hidden_states, dim=0)

        attn_output = torch.zeros(
            hidden_states.shape[0],
            self.num_local_heads * self.v_head_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        hidden_states = self.o_proj(attn_output)
        if self.tp_size > 1:
            if self.all2all_backend == "naive":
                hidden_states = get_tp_group().all_reduce(hidden_states)
            else:
                hidden_states = get_tp_group().reduce_scatter(hidden_states, dim=0)
        return hidden_states

    def _forward_decode(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
    ) -> torch.Tensor:

        q_nope, q_pe, kv_cache, topk_indices = self._mla_prolog(
            hidden_states,
            cos,
            sin,
            attn_metadata,
        )

        if self.is_dsa_layer:
            attn_output = self._apply_DSA_attention(
                q_nope,
                q_pe,
                kv_cache,
                topk_indices,
                attn_metadata,
            )
        else:
            if self.layer_idx in self.swa_layers:
                attn_output = self._apply_SWA_attention_decode(
                    q_nope,
                    q_pe,
                    kv_cache,
                    attn_metadata,
                )
            else:
                attn_output = self._apply_MLA_attention_decode(
                    q_nope,
                    q_pe,
                    kv_cache,
                    attn_metadata,
                )

        return self._mla_epilog(attn_output, attn_metadata)

    def _rescale_attention(
        self,
        lse: torch.Tensor,
        lse_sink: torch.Tensor,
        attn_output: torch.Tensor,
        attn_output_sink: torch.Tensor,
    ):
        lse = lse.to(torch.float32)
        lse_sink = lse_sink.to(torch.float32)

        lse_max = torch.maximum(
            lse_sink,
            lse,
        )
        w_sink = torch.exp(
            lse_sink - lse_max,
        )
        w = torch.exp(
            lse - lse_max,
        )
        attn_output = (attn_output_sink * w_sink + attn_output * w) / (w_sink + w)
        return attn_output

    def _apply_MOME(
        self,
        x: torch.Tensor, 
        layer: ColumnParallelMOME, 
        cache: torch.Tensor,           # should be obtained in attn_metadata when MoME cache is handled by vLLM
        cache_indices: torch.Tensor,   # should be obtained in attn_metadata when MoME cache is handled by vLLM
        attn_metadata: Optional[MLACommonMetadata],
    ):
        if attn_metadata is None:
            # warm up run
            return x

        x, x_padded = torch.split(
            x, [attn_metadata.num_actual_tokens, x.shape[0] - attn_metadata.num_actual_tokens], dim=0
        )

        
        width = self.mome_kernel_width
        num_reqs = attn_metadata.num_reqs
        
        if attn_metadata.prefill is not None:
            assert attn_metadata.num_decodes == 0
            x = layer.forward_prefill(
                x, 
                cache[:, :width-1], 
                cache_indices[:num_reqs], 
                attn_metadata.query_start_loc, 
            )
            x = torch.cat([x, x_padded], dim=0)
            return x

        # decode and speculative decoding branch
        assert attn_metadata.decode is not None
        assert attn_metadata.num_prefills == 0

        num_accepted_tokens = getattr(attn_metadata.decode, "num_accepted_tokens", None)
        x = layer.forward_decode(
            x, 
            cache, 
            cache_indices[:num_reqs], 
            attn_metadata.query_start_loc, 
            num_accepted_tokens=num_accepted_tokens,
        )
        x = torch.cat([x, x_padded], dim=0)
        return x

    def _apply_MLA_attention_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: Optional[MLACommonMetadata] = None,
    ) -> torch.Tensor:

        sink_kv_nope = self.kv_a_layernorm(self.param_sink_compressed_kv) \
                                           .view(self.param_sink_number, self.kv_lora_rank)
        sink_kv_rope = self.param_sink_k_pe.view(self.param_sink_number, self.qk_rope_head_dim)

        torch_npu.npu_scatter_nd_update_(
            kv_cache[0],
            self.sink_slot_indices, 
            sink_kv_nope,
        )

        torch_npu.npu_scatter_nd_update_(
            kv_cache[1],
            self.sink_slot_indices,
            sink_kv_rope,
        )

        block_table = torch.nn.functional.pad(attn_metadata.decode.block_table, pad=(1,0))
        query_cumlens = attn_metadata.decode.query_cumlens
        seq_lens = [seq + self.param_sink_number for seq in attn_metadata.decode.seq_lens]

        kwargs = {
            "query": q_nope,
            "key": kv_cache[0],
            "value": kv_cache[0],
            "query_rope": q_pe,
            "key_rope": kv_cache[1],
            "num_query_heads": self.num_local_heads,
            "num_key_value_heads": 1,
            "input_layout": "TND_NTD",
            "atten_mask": self.attn.impl.SHARE_MASK_TRIL_SPARSE,
            "sparse_mode": 4,
            "pre_tokens": 2**31 - 1,
            "next_tokens": 0,
            "softmax_scale": self.scaling,
            "block_table": block_table,
            "block_size": self.block_size,
            "sink_number": self.param_sink_number,
            "actual_seq_qlen": query_cumlens,
            "actual_seq_kvlen": seq_lens,
        }
        if self.on_ascend950:
            attn_output = torch_npu.npu_fused_infer_attention_score_sink(**kwargs)[0]
        else:
            attn_output = torch.ops.custom.npu_fused_infer_attention_sink(**kwargs)[0]

        attn_output = attn_output.view(self.num_local_heads, -1, self.kv_lora_rank) 
        attn_output = (
            torch.matmul(attn_output, self.attn.impl.W_UV)
                .transpose(1, 0)
                .reshape(-1, self.num_local_heads * self.v_head_dim)
        )

        return attn_output

    def _apply_SWA_attention_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: Optional[MLACommonMetadata] = None,
    ) -> torch.Tensor:

        sink_k_nope = self.kv_a_layernorm(self.param_sink_compressed_kv).unsqueeze(1)
        sink_k_pe = self.param_sink_k_pe.unsqueeze(1)

        kwargs = {
            "query": q_nope,
            "key": kv_cache[0],
            "value": kv_cache[0],
            "query_rope": q_pe,
            "key_rope": kv_cache[1],
            "num_key_value_heads": 1,
            "input_layout": "TND_NTD",
            "atten_mask": self.attn.impl.SHARE_MASK_TRIL_SPARSE,
            "sparse_mode": 4,
            "pre_tokens": self.sliding_window-1,
            "next_tokens": 0,
            "block_table": attn_metadata.decode.block_table,
            "block_size": self.block_size,
            "key_sink": sink_k_nope,
            "value_sink": sink_k_nope,
            "key_rope_sink": sink_k_pe,
        }
        if self.on_ascend950:
            kwargs.update({
                "num_heads": self.num_local_heads,
                "actual_seq_lengths": attn_metadata.decode.query_cumlens,
                "actual_seq_lengths_kv": attn_metadata.decode.seq_lens,
                "scale": self.scaling,
            })
            attn_output = torch_npu._npu_attention_pioneer(**kwargs)[0]
        else:
            kwargs.update({
                "num_query_heads": self.num_local_heads,
                "actual_seq_qlen": attn_metadata.decode.query_cumlens,
                "actual_seq_kvlen": attn_metadata.decode.seq_lens,
                "softmax_scale": self.scaling,
            })
            attn_output = torch.ops.custom.npu_fused_infer_attention_sink(**kwargs)[0]

        attn_output = attn_output.view(self.num_local_heads, -1, self.kv_lora_rank) 
        attn_output = (
            torch.matmul(attn_output, self.attn.impl.W_UV)
                .transpose(1, 0)
                .reshape(-1, self.num_local_heads * self.v_head_dim)
        )

        return attn_output

    def _forward_prefill(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
    ) -> torch.Tensor:

        if self.is_dsa_layer:
            q_nope, q_pe, kv_cache, topk_indices = self._mla_prolog(
                hidden_states,
                cos,
                sin,
                attn_metadata,
            )
            attn_output = self._apply_DSA_attention(
                q_nope,
                q_pe,
                kv_cache,
                topk_indices,
                attn_metadata,
            )
        else:
            q_nope, q_pe, k_up_nope, k_pe, v_up = self._mla_prolog(
                hidden_states,
                cos,
                sin,
                attn_metadata,
            )
            if self.layer_idx in self.swa_layers:
                attn_output = self._apply_SWA_attention_prefill(
                    q_nope,
                    q_pe,
                    k_up_nope,
                    k_pe,
                    v_up,
                    attn_metadata,
                )
            else:
                attn_output = self._apply_MLA_attention_prefill(
                    q_nope,
                    q_pe,
                    k_up_nope,
                    k_pe,
                    v_up,
                    attn_metadata,
                )

        return self._mla_epilog(attn_output, attn_metadata)

    def _apply_MLA_attention_prefill(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
    ) -> torch.Tensor:

        kwargs = {
            "query": q_nope,
            "key": k_nope,
            "value": v,
            "query_rope": q_pe,
            "key_rope": k_pe,
            "num_query_heads": self.num_local_heads,
            "num_key_value_heads": self.num_local_heads,
            "input_layout": "TND",
            "atten_mask": self.attn.impl.SHARE_MASK_TRIL_SPARSE,
            "sparse_mode": 3,
            "softmax_scale": self.scaling,
            "return_softmax_lse": True,
            "actual_seq_qlen": attn_metadata.prefill.query_cumlens,
            "actual_seq_kvlen": attn_metadata.prefill.seq_lens,
        }
        if self.on_ascend950:
            attn_output, lse = torch_npu.npu_fused_infer_attention_score_sink(**kwargs)
        else:
            attn_output, lse = torch.ops.custom.npu_fused_infer_attention_sink(**kwargs)

        attn_output_sink, lse_sink = self._apply_attention_prefill_sink(
            q_nope,
            q_pe,
        )

        attn_output = self._rescale_attention(
            lse,
            lse_sink,
            attn_output,
            attn_output_sink,
        )

        return attn_output.view(-1, self.num_local_heads * self.v_head_dim).to(torch.bfloat16)

    def _apply_SWA_attention_prefill(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
    ) -> torch.Tensor:

        sink_kv = self.kv_b_proj(
            self.kv_a_layernorm(self.param_sink_compressed_kv)
        )
        sink_k_nope, sink_v = torch.split(
            sink_kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim),
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        sink_k_pe = self.param_sink_k_pe.view(-1, 1, self.qk_rope_head_dim) \
                                        .repeat(1, self.num_local_heads, 1)

        if self.on_ascend950:
            query = torch.cat([q_nope, q_pe], dim=-1)
            key = torch.cat([k_nope, k_pe], dim=-1)
            sink_key = torch.cat([sink_k_nope, sink_k_pe], dim=-1)
            kwargs = {
                "query": query,
                "key": key,
                "value": v,
                "actual_seq_lengths": attn_metadata.decode.query_cumlens,
                "actual_seq_lengths_kv": attn_metadata.decode.seq_lens,
                "num_heads": self.num_local_heads,
                "num_key_value_heads": self.num_local_heads,
                "input_layout": "TND",
                "scale": self.scaling,
                "sparse_mode": 4,
                "pre_tokens": self.sliding_window-1,
                "next_tokens": 0,
                "atten_mask": self.attn.impl.SHARE_MASK_TRIL_SPARSE,
                "softmax_lse_flag": False,
                "key_sink": sink_key,
                "value_sink": sink_v,
            }
            attn_output = torch_npu._npu_attention_pioneer(**kwargs)[0]
        else:
            kwargs = {
                "query": q_nope,
                "key": k_nope,
                "value": v,
                "query_rope": q_pe,
                "key_rope": k_pe,
                "num_query_heads": self.num_local_heads,
                "num_key_value_heads": self.num_local_heads,
                "input_layout": "TND",
                "atten_mask": self.attn.impl.SHARE_MASK_TRIL_SPARSE,
                "sparse_mode": 4,
                "softmax_scale": self.scaling,
                "pre_tokens": self.sliding_window-1,
                "next_tokens": 0,
                "actual_seq_qlen": attn_metadata.prefill.query_cumlens,
                "actual_seq_kvlen": attn_metadata.prefill.seq_lens,
                "key_sink": sink_k_nope,
                "value_sink": sink_v,
                "key_rope_sink": sink_k_pe,
            }
            attn_output = torch.ops.custom.npu_fused_infer_attention_sink(**kwargs)[0]

        return attn_output.view(-1, self.num_local_heads * self.v_head_dim)

    def _apply_DSA_attention(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        topk_indices: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
    ) -> torch.Tensor:

        if attn_metadata.prefill is not None:
            metadata = attn_metadata.prefill
        else:
            metadata = attn_metadata.decode

        q = torch.cat([q_nope, q_pe], dim=-1)

        sink_k_nope = self.kv_a_layernorm(self.param_sink_compressed_kv).unsqueeze(1)
        sink_k_pe = self.param_sink_k_pe.unsqueeze(1)
        sink_kv = torch.cat([sink_k_nope, sink_k_pe], dim=-1)

        if self.on_ascend950 and self.cache_config.cache_dtype in ["hif8_ds_mla"]:
            attn_output = torch_npu.npu_kv_quant_sparse_flash_attention(
                query=q,
                key=kv_cache[0],
                value=kv_cache[1],
                sparse_indices=topk_indices,
                scale_value=self.scaling,
                key_quant_mode=2,
                value_quant_mode=2,
                block_table=metadata.block_table,
                actual_seq_lengths_query=metadata.query_cumlens.to(torch.int32),
                actual_seq_lengths_kv=metadata.seq_lens.to(torch.int32),
                sparse_block_size=1,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                attention_mode=2,
                quant_scale_repo_mode=1,
                tile_size=128,
                rope_head_dim=64,
                key_dtype=torch_npu.hifloat8,
                value_dtype=torch_npu.hifloat8,
            )
        else:
            attn_output = torch.ops.custom.npu_ai_infra_sparse_flash_attention_pioneer(
                query=q,
                key=kv_cache[0].unsqueeze(2),
                value=self.dummy_value_cache,
                sparse_indices=topk_indices,
                scale_value=self.scaling,
                sparse_block_size=1,
                block_table=metadata.block_table,
                actual_seq_lengths_query=metadata.query_cumlens.to(torch.int32),
                actual_seq_lengths_kv=metadata.seq_lens.to(torch.int32),
                pre_tokens=(1<<63)-1,
                next_tokens=(1<<63)-1,
                attention_mode=2,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                key_sink=sink_kv,
                value_sink=sink_k_nope,
            )[0]

        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.view(self.num_local_heads, -1, self.kv_lora_rank)
        attn_output = (
            torch.matmul(attn_output, self.attn.impl.W_UV)
                .transpose(1, 0)
                .reshape(-1, self.num_local_heads * self.v_head_dim)
        )

        return attn_output

    def _apply_attention_prefill_sink(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
    ) -> torch.Tensor:

        sink_kv = self.kv_b_proj(
            self.kv_a_layernorm(self.param_sink_compressed_kv)
        )
        sink_k_nope, sink_v = torch.split(
            sink_kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim),
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        sink_k_pe = self.param_sink_k_pe.view(-1, 1, self.qk_rope_head_dim) \
                                        .repeat(1, self.num_local_heads, 1)

        kwargs = {
            "query": q_nope,
            "key": sink_k_nope,
            "value": sink_v,
            "query_rope": q_pe,
            "key_rope": sink_k_pe,
            "num_query_heads": self.num_local_heads,
            "num_key_value_heads": self.num_local_heads,
            "input_layout": "TND",
            "atten_mask": None,
            "sparse_mode": 0,
            "softmax_scale": self.scaling,
            "return_softmax_lse": True,
            "actual_seq_qlen": [q_nope.shape[0]],
            "actual_seq_kvlen": [self.param_sink_number],
        }        
        if self.on_ascend950:
            return torch_npu.npu_fused_infer_attention_score_sink(**kwargs)
        else:
            return torch.ops.custom.npu_fused_infer_attention_sink(**kwargs)

    def _mla_prolog(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor], # DSA/MLA/SWA absorb
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]: # MLA/SWA non-absorb

        # get KV cache for this layer
        kv_cache = self.attn.kv_cache[get_forward_context().virtual_engine]

        if self.tp_size > 1:
            if not self.all2all_backend == "naive":
                hidden_states = get_tp_group().all_gather(hidden_states, dim=0)

        ### Q stream begins ###
        q_lora = self.q_a_proj(hidden_states)
        if self.use_mome:
            q_lora = self._apply_MOME(
                q_lora,
                self.qa_conv,
                self.q_mome_cache,
                self.cache_indices,
                attn_metadata,
            )
        q_lora = self.q_a_layernorm(q_lora)
        q = self.q_b_proj(q_lora)
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        if attn_metadata.decode is not None or self.is_dsa_layer:
            q_nope = q_nope.transpose(0, 1) \
                        .reshape(self.num_local_heads, -1, self.qk_nope_head_dim)
            q_nope = (
                torch.matmul(q_nope, self.attn.impl.W_UK_T)
                    .transpose(1, 0)
                    .reshape(-1, self.num_local_heads, self.kv_lora_rank)
            )

        q_pe = torch_npu.npu_rotary_mul(
            q_pe.view(-1, 1, self.num_local_heads, self.qk_rope_head_dim),
            cos.view(-1, 1, 1, self.qk_rope_head_dim),
            sin.view(-1, 1, 1, self.qk_rope_head_dim),
            rotary_mode="half" if not self.rope_interleaved else "interleave",
        ).squeeze(1)
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()
        ### Q stream ends ###


        ### Indexer stream begins ###
        if self.is_dsa_layer:
            topk_indices = self.indexer(
                hidden_states,
                q_lora,
                cos,
                sin,
                attn_metadata,
                kv_cache,
            )
        else:
            topk_indices = None
        ### Indexer stream ends ###


        ### KV stream begins ###
        kv = self.kv_a_proj_with_mqa(hidden_states)
        k_nope, k_pe = torch.split(
            kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        if self.use_mome:
            k_nope = self._apply_MOME(
                k_nope,
                self.compresskv_conv,
                self.kv_mome_cache,
                self.cache_indices,
                attn_metadata,
            )

        ret = self._npu_kvrmsnorm_rope_cache(
            k_nope,
            k_pe,
            kv_cache,
            cos,
            sin,
            attn_metadata,
            topk_indices,
        )
        ### KV stream ends ###

        output = (q_nope, q_pe, *ret)
        return output

    def _npu_kvrmsnorm_rope_cache(self, *args, **kwargs):
        if quant_output := self._npu_kvrmsnorm_rope_cache_quant(*args, **kwargs):
            return quant_output
        else:
            unquant_output = self._npu_kvrmsnorm_rope_cache_unquant(*args, **kwargs)
            return unquant_output

    def _npu_kvrmsnorm_rope_cache_quant(
        self,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata],
        topk_indices: torch.Tensor,
    ):
        # DSA layer c8 shape
        if self.cache_config.cache_dtype in ["hif8_ds_mla"] and self.is_dsa_layer:
            k_nope = self.kv_a_layernorm(k_nope)
            k_nope_scale = torch.ones(
                (k_nope.shape[0], k_nope.shape[1] // 128), dtype=torch.float32, device=k_nope.device,
            )
            k_nope_hif8 = torch_npu.npu_dtype_cast(k_nope, torch_npu.hifloat8)
            k_pe = torch_npu.npu_rotary_mul(
                k_pe.view(-1, 1, 1, self.qk_rope_head_dim),
                cos.view(-1, 1, 1, self.qk_rope_head_dim),
                sin.view(-1, 1, 1, self.qk_rope_head_dim),
            ).squeeze(1).squeeze(1)

            kv = torch.cat(
                [k_nope_hif8, k_pe.view(torch.uint8), k_nope_scale.view(torch.uint8)],
                dim=-1,
            )
            torch_npu.npu_scatter_nd_update_(
                kv_cache[0].view(-1, kv_cache[0].shape[-1]).view(torch.int8),
                attn_metadata.slot_mapping.view(-1, 1),
                kv.view(-1, kv.shape[-1]).view(torch.int8)
            )

            return kv_cache, topk_indices
        else:
            return False

    def _npu_kvrmsnorm_rope_cache_unquant(
        self,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata],
        topk_indices: torch.Tensor,
    ):
        # the rest cases are unquantized
        kv = torch.cat([k_nope, k_pe], dim=-1)

        kwargs = {
            "kv": kv.view(-1, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim),
            "gamma": self.kv_a_layernorm.weight,
            "cos": cos.view(-1, 1, 1, self.qk_rope_head_dim),
            "sin": sin.view(-1, 1, 1, self.qk_rope_head_dim),
            "index": attn_metadata.slot_mapping,
            "epsilon": self.kv_a_layernorm.variance_epsilon,
            "cache_mode": "PA",
            "rotary_mode": "half" if not self.rope_interleaved else "interleave-half",
            "quant_mode": "none",
            "is_output_kv": True,
        }

        if self.is_dsa_layer:
            # DSA shape
            kwargs.update({
                "k_cache": None,
                "ckv_cache": kv_cache[0].unsqueeze(2),
            })
            k_pe, k_nope = torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2(**kwargs)

            return kv_cache, topk_indices

        elif attn_metadata.decode is not None:
            # MLA/SWA absorb shape
            kwargs.update({
                "k_cache": kv_cache[1].unsqueeze(2),
                "ckv_cache": kv_cache[0].unsqueeze(2),
            })
            k_pe, k_nope = torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2(**kwargs)

            return kv_cache, topk_indices
    
        else:
            # MLA/SWA non-absorb shape
            kwargs.update({
                "k_cache": kv_cache[1].unsqueeze(2),
                "ckv_cache": kv_cache[0].unsqueeze(2),
            })
            k_pe, k_nope = torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2(**kwargs)

            kv_up = self.kv_b_proj(k_nope)
            kv_up = kv_up.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_up_nope, v_up = torch.split(
                kv_up,
                [self.qk_nope_head_dim, self.v_head_dim],
                dim=-1,
            )
            k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim) \
                       .repeat(1, self.num_local_heads, 1)

            return k_up_nope.contiguous(), k_pe.contiguous(), v_up.contiguous()

    def _mla_epilog(
        self,
        attn_output: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
    ) -> torch.Tensor:

        if self.use_mome:
            attn_output = self._apply_MOME(
                attn_output,
                self.o_conv,
                self.o_mome_cache,
                self.cache_indices,
                attn_metadata,
            )

        hidden_states = self.o_proj(attn_output)

        if self.tp_size > 1:
            if self.all2all_backend == "naive":
                hidden_states = get_tp_group().all_reduce(hidden_states)
            else:
                hidden_states = get_tp_group().reduce_scatter(hidden_states, dim=0)

        return hidden_states

