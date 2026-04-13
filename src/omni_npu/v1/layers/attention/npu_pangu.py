# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Optional, Union, Tuple
from itertools import accumulate

import torch
import torch_npu
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm.model_executor.models.utils import extract_layer_index
from vllm.distributed import get_tp_group
from vllm.config import VllmConfig, CacheConfig, get_current_vllm_config
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
from omni_npu.attention.backends.mome import NPUMomeAttentionMetadata
from vllm.logger import init_logger

from omni_npu.v1.utils import current_stream, on_ascend950
from omni_npu.v1.layers.utils import yarn_get_mscale
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.attention.backends.utils import SPManager, DummySPManager, lazy_init_cos_sin
from omni_npu.layers.mome.npu_mome import ColumnParallelMOME
from omni_npu.layers.attention.npu_sparse_attentions import (
    MLASWAAttention,
    DSAAttention,
    MomeAttention,
)

from omni_npu.compilation.utils import (
    capture_graph_task,
    OP_FIA_SINK,
    OP_FIA_PIONEER,
)

from vllm.utils.torch_utils import direct_register_custom_op

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
        self.block_size_c8 = 2 * self.block_size
        self.on_ascend950 = on_ascend950()
        self._init_indexer_weights()
        self.quant_cache_dtype = ["hif8_ds_mla", "fp8_ds_mla", "int8_ds_mla"]

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
        if self.cache_config.cache_dtype in self.quant_cache_dtype:
            quant_output = self._apply_lightning_indexer_quant(*args, **kwargs)
            return quant_output
        else:
            unquant_output = self._apply_lightning_indexer_unquant(*args, **kwargs)
            return unquant_output

    def _apply_lightning_indexer_cp(self, *args, **kwargs):
        if self.cache_config.cache_dtype in self.quant_cache_dtype:
            quant_output = self._apply_lightning_indexer_cp_quant(*args, **kwargs)
            return quant_output
        else:
            unquant_output = self._apply_lightning_indexer_cp_unquant(*args, **kwargs)
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
            actual_seq_lengths_query=metadata.query_cumlens,
            actual_seq_lengths_key=metadata.seq_lens,
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
                key=kv_cache[1].unsqueeze(-2) if len(kv_cache[1].shape) == 3 else kv_cache[1],
                weights=weights,
                query_dequant_scale=q_scale,
                key_dequant_scale=kv_cache[2],
                actual_seq_lengths_query=metadata.query_cumlens,
                actual_seq_lengths_key=metadata.seq_lens,
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
        elif self.cache_config.cache_dtype in ["int8_ds_mla"]:
            q_int8, q_scale = torch_npu.npu_dynamic_quant(q)
            return torch_npu.torch.ops.custom.npu_ai_infra_quant_lightning_indexer(
                query=q_int8,
                key=kv_cache[1][..., :128].unsqueeze(2),
                weights=weights.to(torch.float16),
                query_dequant_scale=q_scale.to(torch.float16),
                key_dequant_scale=kv_cache[1][..., 128:].view(torch.float16),
                actual_seq_lengths_query=metadata.query_cumlens,
                actual_seq_lengths_key=metadata.seq_lens,
                block_table=metadata.block_table,
                query_quant_mode=0,
                key_quant_mode=0,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
            )

    def _apply_lightning_indexer_cp_unquant(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        sp_manager: Optional[MLACommonMetadata] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        actual_seq_lengths_query, actual_seq_lengths_kv, _, block_table = sp_manager.cp_attn_meta()

        return torch.ops.custom.npu_lightning_indexer_enhance(
            query=q,
            key=kv_cache[1].unsqueeze(2),
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_kv,
            block_table=block_table,
            layout_key="PA_BSND",
            layout_query="TND",
            sparse_count=self.index_topk,
            sparse_mode=3,
            sparse_block_size=1,
            sparse_block_mode=False,
        )[0]

    def _apply_lightning_indexer_cp_quant(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        sp_manager: Optional[MLACommonMetadata] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        actual_seq_lengths_query, actual_seq_lengths_kv, _, block_table = sp_manager.cp_attn_meta()

        if self.cache_config.cache_dtype in ["int8_ds_mla"]:
            q_int8, q_scale = torch_npu.npu_dynamic_quant(q)
            return torch_npu.torch.ops.custom.npu_ai_infra_quant_lightning_indexer(
                query=q_int8,
                key=kv_cache[1][..., :128].unsqueeze(2),
                weights=weights.to(torch.float16),
                query_dequant_scale=q_scale.to(torch.float16),
                key_dequant_scale=kv_cache[1][..., 128:].view(torch.float16),
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_kv,
                block_table=block_table,
                query_quant_mode=0,
                key_quant_mode=0,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
            )

    def _update_indexer_cache_unquant(
        self,
        k: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> bool:

        slot_indices = torch.stack([
            slot_mapping // self.block_size,
            slot_mapping % self.block_size,
            ], dim=1,
        )
        # TODO: need fix
        torch.ops.custom.npu_ai_infra_scatter_block_update_(
            kv_cache[1],
            slot_indices,
            k.view(-1, k.shape[-1]),
        )
        return True

    def _update_indexer_cache_quant(
        self,
        k: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> bool:

        slot_indices_c8 = torch.stack([
            slot_mapping // self.block_size_c8,
            slot_mapping % self.block_size_c8,
            ], dim=1,
        )

        if self.on_ascend950 and self.cache_config.cache_dtype in ["hif8_ds_mla"]:
            k_scale = torch.ones(
                (k.shape[0], 1), dtype=torch.float32, device=k.device,
            )
            k_hif8 = torch_npu.npu_dtype_cast(k, torch_npu.hifloat8)

            torch_npu.npu_scatter_nd_update_(
                kv_cache[1].view(torch.int8),
                slot_indices_c8,
                k_hif8.view(torch.int8),
            )
            torch_npu.npu_scatter_nd_update_(
                kv_cache[2],
                slot_indices_c8,
                k_scale,
            )
            return True

        elif self.cache_config.cache_dtype in ["int8_ds_mla"]:
            k_int8, k_scale = torch_npu.npu_dynamic_quant(k)
            k_scale_fp16 = k_scale.to(torch.float16).view(-1, 1)
            k_scale_bytes = k_scale_fp16.view(torch.int8)
            k_packed = torch.cat([k_int8, k_scale_bytes], dim=-1)

            torch.ops.custom.npu_ai_infra_scatter_block_update_(
                kv_cache[1],
                slot_indices_c8,
                k_packed.view(-1, k_packed.shape[-1]),
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
            attn_metadata.slot_mapping,
            kv_cache,
        )

        return self._apply_lightning_indexer(
            q,
            weights,
            attn_metadata,
            kv_cache,
        )

    def forward_cp(
        self,
        cp_x: torch.Tensor,
        sp_x: torch.Tensor,
        q_lora: torch.Tensor,
        sp_cos: torch.Tensor,
        sp_sin: torch.Tensor,
        cp_cos: torch.Tensor,
        cp_sin: torch.Tensor,
        sp_manager: Optional[SPManager] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        q = self.wq_b(q_lora)
        q = q.view(-1, self.index_n_heads, self.index_head_dim)
        q_pe, q_nope = torch.split(
            q,
            [self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim],
            dim=-1,
        )
        q_pe = torch_npu.npu_rotary_mul(
            q_pe.view(-1, 1, self.index_n_heads, self.qk_rope_head_dim),
            cp_cos.view(-1, 1, 1, self.qk_rope_head_dim),
            cp_sin.view(-1, 1, 1, self.qk_rope_head_dim),
        ).squeeze(1)

        k = self.wk(sp_x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k,
            [self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim],
            dim=-1,
        )
        k_pe = torch_npu.npu_rotary_mul(
            k_pe.view(-1, 1, 1, self.qk_rope_head_dim),
            sp_cos.view(-1, 1, 1, self.qk_rope_head_dim),
            sp_sin.view(-1, 1, 1, self.qk_rope_head_dim),
        ).squeeze(1).squeeze(1)

        k = torch.cat([k_pe, k_nope], dim=-1)
        k = sp_manager.ag_tokens(k)

        weights = self.weights_proj(cp_x)
        self._update_indexer_cache(
            k,
            sp_manager.cp_slot_mapping,
            kv_cache,
        )

        q = torch.cat([q_pe, q_nope], dim=-1)
        topk_indices = self._apply_lightning_indexer_cp(
            q,
            weights,
            sp_manager,
            kv_cache,
        )

        return topk_indices


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
            # SWA layer
            pos_in_swa = self.swa_layers.index(self.layer_idx)
            self.sliding_window = self.sliding_window_list[pos_in_swa]
            self.is_dsa_layer = False
        elif self.layer_idx >= config.num_hidden_layers:
            # MTP layer
            self.sliding_window = self.sliding_window_list[-1]
            self.is_dsa_layer = False
        elif hasattr(config, "index_topk") and config.index_topk > 0:
            # DSA layer
            self.sliding_window = None
            self.is_dsa_layer = True
            self.index_topk = config.index_topk
        else:
            # MLA layer
            # set a very large sliding window to disable the sliding window attention and fall back to global attention
            self.sliding_window = max(1024 * 1024, self.aligned_window_size + 1)
            self.is_dsa_layer = False
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
        self.num_local_heads = (
            num_heads if self.is_dsa_layer and model_extra_config.parall_config.ena_context_parallel else num_heads // self.tp_size
        )
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
        assert model_extra_config.operator_opt_config.use_noncontiguous_kv
        self.enable_flashcomm2 = model_extra_config.parall_config.enable_flashcomm2
        self.dummy_value_cache = torch.zeros(
            (1, cache_config.block_size, 1, self.kv_lora_rank),
            device='npu',
            dtype=torch.bfloat16,
        )
        self.dummy_value_cache_hif8 = torch.zeros(
            (1, cache_config.block_size, 1, 656),
            device='npu',
            dtype=torch.uint8,
        )

        self.quant_cache_dtype = ["hif8_ds_mla", "fp8_ds_mla", "int8_ds_mla"]
        self.block_size = self.cache_config.block_size
        self.block_size_c8 = 2 * self.block_size

        self._init_MLA_weights()
        self._init_rotary_emb()
        self._init_param_sinks()
        self._align_pagesize()
        self._init_attention_layers()
        self._init_mome_layer()

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

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
            disable_tp=True if self.is_dsa_layer and model_extra_config.parall_config.ena_context_parallel else False,
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
            disable_tp=True if self.is_dsa_layer and model_extra_config.parall_config.ena_context_parallel else False,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            reduce_results=False,
            prefix=f"{self.layer_name}.o_proj",
            return_bias=False,
            disable_tp=True if (self.is_dsa_layer and model_extra_config.parall_config.ena_context_parallel) or (self.enable_flashcomm2 and not self.is_dsa_layer) else False,
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
        # assert self.cache_config.block_size == self.param_sink_number
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
            assert self.num_heads % self.tp_size == 0, \
                "For MoME attention, num_heads should be divisible by tp_size."
            if self.is_dsa_layer and model_extra_config.parall_config.ena_context_parallel:
                o_mome_cache_shape = (self.num_heads * self.v_head_dim,)
            else:
                o_mome_cache_shape = (self.num_heads * self.v_head_dim // self.tp_size,)
            self.mome_state_shapes = (
                (self.q_lora_rank,),
                (self.kv_lora_rank,),
                o_mome_cache_shape,
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

        self.cache_dtype_str = self.cache_config.cache_dtype
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

        if self.is_dsa_layer:
            attn_kwargs.update({
                "indexer": self.indexer,
                "indexer_head_dim": self.indexer.index_head_dim,
                "cache_dtype_str": self.cache_dtype_str,
                "page_size_padded": self.page_size_padded,
            })
            if self.cache_dtype_str in ["fp8_ds_mla", "hif8_ds_mla", "int8_ds_mla"]:
                attn_kwargs.update({
                    "block_size": self.block_size_c8,
                })
            self.attn = DSAAttention(**attn_kwargs)
        else:
            attn_kwargs.update({
                "cache_dtype_str": self.cache_dtype_str,
                "page_size_padded": self.page_size_padded,
                "sliding_window": self.aligned_window_size 
                                  if self.sliding_window <= self.aligned_window_size
                                  else None,
            })
            self.attn = MLASWAAttention(**attn_kwargs)

    def _init_mome_layer(self):
        if not self.use_mome:
            return

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
            disable_tp=True if self.is_dsa_layer and model_extra_config.parall_config.ena_context_parallel else False,
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
                dsa_page_size = self.block_size_c8 * (656 + 128 + 4)
            elif cache_dtype_str == "int8_ds_mla":
                # Quant case: 512 int8 + 64 bf16 + 4 fp32 + 128 int8 + 1 bf16
                dsa_page_size = self.block_size_c8 * (656 + 128 + 2)
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

        required_page_sizes = [mla_page_size]
        if dsa_page_size is not None:
            required_page_sizes.append(dsa_page_size)
        if mome_page_size is not None:
            required_page_sizes.append(mome_page_size)

        target_page_size = max(required_page_sizes)

        return target_page_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.vllm.npu_pangu_forward(
            hidden_states=hidden_states,
            cos=cos,
            sin=sin,
            layer_name=self.prefix,
        )

    def _prepare_phase_inputs(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        phase: str,
    ):
        num_decode_tokens = attn_metadata.num_decode_tokens

        if phase == "prefill":
            # first phase: backup originals
            attn_metadata.origin_slot_mapping = attn_metadata.slot_mapping.clone()
            attn_metadata.orig_num_actual_tokens = attn_metadata.num_actual_tokens
            num_actual_tokens = attn_metadata.num_actual_tokens

            sliced_hidden = hidden_states[num_decode_tokens:num_actual_tokens, ...]
            sliced_cos = cos[num_decode_tokens:num_actual_tokens, ...]
            sliced_sin = sin[num_decode_tokens:num_actual_tokens, ...]
            attn_metadata.prefill.slot_mapping = attn_metadata.origin_slot_mapping[num_decode_tokens:num_actual_tokens]
            attn_metadata.slot_mapping = attn_metadata.prefill.slot_mapping
            attn_metadata.saved_decode = attn_metadata.decode
            attn_metadata.decode = None
            attn_metadata.num_actual_tokens = num_actual_tokens - num_decode_tokens
        else:
            saved_decode = getattr(attn_metadata, 'saved_decode', None)
            if saved_decode is not None:
                attn_metadata.decode = attn_metadata.saved_decode
            origin_slot_mapping = attn_metadata.origin_slot_mapping

            sliced_hidden = hidden_states[:num_decode_tokens, ...]
            sliced_cos = cos[:num_decode_tokens, ...]
            sliced_sin = sin[:num_decode_tokens, ...]
            attn_metadata.decode.slot_mapping = origin_slot_mapping[:num_decode_tokens]
            attn_metadata.slot_mapping = attn_metadata.decode.slot_mapping
            attn_metadata.saved_prefill = attn_metadata.prefill
            attn_metadata.prefill = None
            attn_metadata.num_actual_tokens = num_decode_tokens
        return sliced_hidden, sliced_cos, sliced_sin

    def _restore_phase_metadata(self, attn_metadata: MLACommonMetadata):
        saved_prefill = getattr(attn_metadata, 'saved_prefill', None)
        if saved_prefill is not None:
            attn_metadata.prefill = saved_prefill
        attn_metadata.slot_mapping = attn_metadata.origin_slot_mapping
        attn_metadata.num_actual_tokens = attn_metadata.orig_num_actual_tokens

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

        if self.enable_flashcomm2 and not self.is_dsa_layer:
            # FlashComm2.0 dummy: simulate all_to_all + full o_proj
            x = attn_output.view(self.tp_size, -1, attn_output.shape[-1])
            output = torch.empty_like(x)
            torch.distributed.all_to_all_single(output.flatten(), x.flatten(), group=get_tp_group().device_group)
            attn_output = output.transpose(0, 1).reshape(attn_output.shape[0] // self.tp_size, -1)

        hidden_states = self.o_proj(attn_output)

        if self.tp_size > 1:
            if self.enable_flashcomm2 and not self.is_dsa_layer:
                pass  # all_to_all + full o_proj already complete
            elif self.all2all_backend == "naive":
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
        mome_metadata: Optional[NPUMomeAttentionMetadata] = None,
    ) -> torch.Tensor:

        q_nope, q_pe, kv_cache, topk_indices = self._mla_prolog(
            hidden_states,
            cos,
            sin,
            attn_metadata,
            mome_metadata, 
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
            attn_output = self._apply_SWA_attention_decode(
                q_nope,
                q_pe,
                kv_cache,
                attn_metadata,
            )

        return self._mla_epilog(attn_output, attn_metadata, mome_metadata)

    def _apply_MOME(
        self,
        x: torch.Tensor, 
        layer: ColumnParallelMOME, 
        kv_index: int = 0, 
        attn_metadata: Optional[MLACommonMetadata] = None,
        mome_metadata: Optional[NPUMomeAttentionMetadata] = None,
    ):
        if attn_metadata is None or mome_metadata is None:
            # warm up run
            return x

        x, x_padded = torch.split(
            x, [attn_metadata.num_actual_tokens, x.shape[0] - attn_metadata.num_actual_tokens], dim=0
        )
        
        width = self.mome_kernel_width
        kv_cache = self.mome_attn.kv_cache[get_forward_context().virtual_engine]

        # todo: support continuous batching
        
        if mome_metadata.num_prefills > 0:
            x = layer.forward_prefill(
                x, 
                kv_cache[kv_index][:, :width-1], 
                mome_metadata.cache_indices, 
                mome_metadata.query_start_loc, 
            )
        else:
            x = layer.forward_decode(
                x, 
                kv_cache[kv_index], 
                mome_metadata.cache_indices, 
                mome_metadata.query_start_loc, 
                num_accepted_tokens=mome_metadata.num_accepted_tokens, 
                pad_slot_id=mome_metadata.pad_slot_id, 
            )

        x = torch.cat([x, x_padded], dim=0)
        return x

    def _apply_SWA_attention_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: Optional[MLACommonMetadata] = None,
    ) -> torch.Tensor:

        sink_k_nope = self.kv_a_layernorm(self.param_sink_compressed_kv).unsqueeze(1)
        sink_k_pe = self.param_sink_k_pe.unsqueeze(1)

        query_cumlens = attn_metadata.decode.query_cumlens
        num_actual_tokens = query_cumlens[-1]
        kwargs = {
            "query": q_nope[:num_actual_tokens],
            "key": kv_cache[0],
            "value": kv_cache[0],
            "query_rope": q_pe[:num_actual_tokens],
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

        num_tokens = q_nope.size(0)
        attn_output_shape = [self.num_local_heads, num_tokens, self.kv_lora_rank]
        attn_output = torch.empty(attn_output_shape, device=q_nope.device, dtype=q_nope.dtype)
        softmax_lse = torch.empty((num_tokens, self.num_local_heads, 1), device=q_nope.device, dtype=torch.float32)
        if self.on_ascend950:
            kwargs.update({
                "num_heads": self.num_local_heads,
                "actual_seq_lengths": query_cumlens,
                "actual_seq_lengths_kv": attn_metadata.decode.seq_lens,
                "scale": self.scaling,
            })
            forward_context = get_forward_context()
            if forward_context.capturing:
                capture_graph_task(
                    op_desc=OP_FIA_PIONEER,
                    op_kwargs=kwargs,
                    out_tensors=[attn_output, softmax_lse],
                    num_tokens=num_tokens,
                    layer_name=self.attn.layer_name,
                )
            else:
                attn_output[:, :num_actual_tokens] = torch_npu._npu_attention_pioneer(**kwargs)[0]
        else:
            kwargs.update({
                "num_query_heads": self.num_local_heads,
                "actual_seq_qlen": query_cumlens,
                "actual_seq_kvlen": attn_metadata.decode.seq_lens,
                "softmax_scale": self.scaling,
            })
            forward_context = get_forward_context()
            if forward_context.capturing:
                capture_graph_task(
                    op_desc=OP_FIA_SINK,
                    op_kwargs=kwargs,
                    out_tensors=[attn_output, softmax_lse],
                    num_tokens=num_tokens,
                    layer_name=self.attn.layer_name,
                )
            else:
                attn_output[:, :num_actual_tokens] = torch.ops.custom.npu_fused_infer_attention_sink(**kwargs)[0]

        attn_output = attn_output.view(self.num_local_heads, -1, self.kv_lora_rank) 
        attn_output = (
            torch.matmul(attn_output, self.attn.impl.W_UV)
                .transpose(1, 0)
                .reshape(-1, self.num_local_heads * self.v_head_dim)
        )

        return attn_output

    def _apply_DSA_attention_cp(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        topk_indices: torch.Tensor,
        sp_manager: Optional[SPManager] = None,
    ) -> torch.Tensor:
        actual_seq_lengths_query, actual_seq_lengths_kv, _, block_table = sp_manager.cp_attn_meta()

        q = torch.cat([q_nope, q_pe], dim=-1)

        sink_k_nope = self.kv_a_layernorm(self.param_sink_compressed_kv).unsqueeze(1)
        sink_k_pe = self.param_sink_k_pe.unsqueeze(1)
        sink_kv = torch.cat([sink_k_nope, sink_k_pe], dim=-1)

        if self.cache_config.cache_dtype in ["int8_ds_mla"]:
            attn_output = torch.ops.custom.npu_ai_infra_kv_quant_sparse_flash_attention(
                query=q,
                key=kv_cache[0].unsqueeze(2),
                value=kv_cache[0].unsqueeze(2),
                sparse_indices=topk_indices,
                scale_value=self.scaling,
                key_quant_mode=2,
                value_quant_mode=2,
                sparse_block_size=1,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                key_sink=sink_kv,
                value_sink=sink_k_nope,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                block_table=block_table,
                attention_mode=2,
                quant_scale_repo_mode=1,
                tile_size=128,
                rope_head_dim=64,
            )
        else:
            attn_output = torch.ops.custom.npu_ai_infra_sparse_flash_attention_pioneer(
                query=q,
                key=kv_cache[0].unsqueeze(2),
                value=self.dummy_value_cache,
                sparse_indices=topk_indices,
                scale_value=self.scaling,
                sparse_block_size=1,
                block_table=block_table,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
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

    def _apply_MOME_prefill_cp(
        self,
        x: torch.Tensor,
        layer: ColumnParallelMOME,
        kv_index: int,
        mome_metadata: Optional[NPUMomeAttentionMetadata] = None,
        sp_manager: Optional[SPManager] = None,
    ) -> torch.Tensor:
        merged_x = sp_manager.mome_suffix_exchange(x)
        merged_x = sp_manager.broadcast_mome_req_tails_from_rank0(merged_x)
        kv_cache = self.mome_attn.kv_cache[get_forward_context().virtual_engine]
        cache = kv_cache[kv_index][:, :self.mome_kernel_width - 1]
        cache_indices = mome_metadata.cache_indices
        merged_x = layer.forward_prefill(
            merged_x,
            cache,
            cache_indices,
            sp_manager.cp_mome_query_start_loc,
        )
        return sp_manager.mome_split_and_cat(merged_x)

    def _forward_prefill_cp(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
        mome_metadata: Optional[NPUMomeAttentionMetadata] = None,
    ) -> torch.Tensor:
        sp_manager: SPManager = (
            attn_metadata.prefill.sp_manager
            if attn_metadata is not None
            else DummySPManager(get_tp_group()))
        cos = cos[:attn_metadata.num_actual_tokens]
        sin = sin[:attn_metadata.num_actual_tokens]
        lazy_init_cos_sin(sp_manager, cos, sin, init_zigzag=True)
        sp_cos, sp_sin = sp_manager.sp_cos, sp_manager.sp_sin
        cp_cos, cp_sin = sp_manager.cp_cos, sp_manager.cp_sin
        sp_x, cp_x = hidden_states, sp_manager.sp_to_cp(hidden_states)

        ### Q stream begins ###
        q_lora = self.q_a_proj(cp_x)
        if self.use_mome:
            q_lora = self._apply_MOME_prefill_cp(
                q_lora,
                self.qa_conv,
                0,
                mome_metadata,
                sp_manager,
            )
        q_lora = self.q_a_layernorm(q_lora)
        q = self.q_b_proj(q_lora)
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        q_nope = q_nope.transpose(0, 1) \
                    .reshape(self.num_local_heads, -1, self.qk_nope_head_dim)
        q_nope = (
            torch.matmul(q_nope, self.attn.impl.W_UK_T)
                .transpose(1, 0)
                .reshape(-1, self.num_local_heads, self.kv_lora_rank)
        )

        q_pe = torch_npu.npu_rotary_mul(
            q_pe.view(-1, 1, self.num_local_heads, self.qk_rope_head_dim),
            cp_cos.view(-1, 1, 1, self.qk_rope_head_dim),
            cp_sin.view(-1, 1, 1, self.qk_rope_head_dim),
            rotary_mode="half" if not self.rope_interleaved else "interleave",
        ).squeeze(1)
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()
        ### Q stream ends ###

        # get KV cache for this layer
        kv_cache = self.attn.kv_cache[get_forward_context().virtual_engine]
        ### Indexer stream begins ###
        topk_indices = self.indexer.forward_cp(
            cp_x,
            sp_x,
            q_lora,
            sp_cos,
            sp_sin,
            cp_cos,
            cp_sin,
            sp_manager,
            kv_cache,
        )
        ### Indexer stream ends ###

        ### KV stream begins ###
        kv = self.kv_a_proj_with_mqa(sp_x)
        kv = sp_manager.ag_tokens(kv)
        k_nope, k_pe = torch.split(
            kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        if self.use_mome:
            k_nope = self._apply_MOME(
                k_nope,
                self.compresskv_conv,
                1,
                attn_metadata,
                mome_metadata,
            )

        kv = torch.cat([k_nope, k_pe], dim=-1)

        kwargs = {
            "kv": kv.view(-1, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim),
            "k_cache": None,
            "ckv_cache": kv_cache[0].unsqueeze(2),
            "gamma": self.kv_a_layernorm.weight,
            "cos": cos.view(-1, 1, 1, self.qk_rope_head_dim),
            "sin": sin.view(-1, 1, 1, self.qk_rope_head_dim),
            "index": sp_manager.cp_slot_mapping,
            "epsilon": self.kv_a_layernorm.variance_epsilon,
            "cache_mode": "PA",
            "rotary_mode": "half" if not self.rope_interleaved else "interleave-half",
            "quant_mode": "none",
            "is_output_kv": True,
        }

        if self.cache_config.cache_dtype in ["int8_ds_mla"]:
            kwargs.update({
                "quant_mode": "pertile128",
            })

        k_pe, k_nope = torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2(**kwargs)
        ### KV stream ends ###

        attn_output = self._apply_DSA_attention_cp(
            q_nope,
            q_pe,
            kv_cache,
            topk_indices,
            sp_manager,
        )

        if self.use_mome:
            attn_output = self._apply_MOME_prefill_cp(
                attn_output,
                self.o_conv,
                2,
                mome_metadata,
                sp_manager,
            )

        hidden_states = self.o_proj(attn_output)
        return sp_manager.cp_to_sp(hidden_states)

    def _forward_prefill(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
        mome_metadata: Optional[NPUMomeAttentionMetadata] = None,
    ) -> torch.Tensor:

        if self.is_dsa_layer:
            q_nope, q_pe, kv_cache, topk_indices = self._mla_prolog(
                hidden_states,
                cos,
                sin,
                attn_metadata,
                mome_metadata, 
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
                mome_metadata,
            )
            attn_output = self._apply_SWA_attention_prefill(
                q_nope,
                q_pe,
                k_up_nope,
                k_pe,
                v_up,
                attn_metadata,
            )

        return self._mla_epilog(attn_output, attn_metadata, mome_metadata)

    def _forward_prefill_FC2(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
        mome_metadata: Optional[NPUMomeAttentionMetadata] = None,
    ) -> torch.Tensor:
        """FlashComm2.0 prefill path: MLA prolog -> SWA attention -> FC2 epilog."""
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens

        prefill_hidden = hidden_states[num_decode_tokens:num_actual_tokens]
        prefill_cos = cos[num_decode_tokens:num_actual_tokens]
        prefill_sin = sin[num_decode_tokens:num_actual_tokens]

        # get KV cache for this layer
        kv_cache = self.attn.kv_cache[get_forward_context().virtual_engine]

        ### Q stream begins ###
        q_lora = self.q_a_proj(prefill_hidden)
        if self.use_mome:
            q_lora = self._apply_MOME(
                q_lora,
                self.qa_conv,
                0,
                attn_metadata,
                mome_metadata,
            )
        q_lora = self.q_a_layernorm(q_lora)
        q = self.q_b_proj(q_lora)
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )
        # FC2 is non-DSA prefill-only: no W_UK_T absorption needed

        q_pe = torch_npu.npu_rotary_mul(
            q_pe.view(-1, 1, self.num_local_heads, self.qk_rope_head_dim),
            prefill_cos.view(-1, 1, 1, self.qk_rope_head_dim),
            prefill_sin.view(-1, 1, 1, self.qk_rope_head_dim),
            rotary_mode="half" if not self.rope_interleaved else "interleave",
        ).squeeze(1)
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()
        ### Q stream ends ###

        ### KV stream begins ###
        kv = self.kv_a_proj_with_mqa(prefill_hidden)
        k_nope, k_pe = torch.split(
            kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        if self.use_mome:
            k_nope = self._apply_MOME(
                k_nope,
                self.compresskv_conv,
                1,
                attn_metadata,
                mome_metadata,
            )

        k_up_nope, k_pe, v_up = self._npu_kvrmsnorm_rope_cache(
            k_nope,
            k_pe,
            kv_cache,
            prefill_cos,
            prefill_sin,
            attn_metadata,
            None,
        )
        ### KV stream ends ###

        # --- SWA Attention ---
        attn_output = self._apply_SWA_attention_prefill(
            q_nope,
            q_pe,
            k_up_nope,
            k_pe,
            v_up,
            attn_metadata,
        )

        # --- FC2 MLA Epilog ---
        if self.use_mome:
            attn_output = self._apply_MOME(
                attn_output,
                self.o_conv,
                2,
                attn_metadata,
                mome_metadata,
            )

        # Write raw attn [num_prefill, local_heads*v_dim] into full N_total buffer
        # so all_to_all distributes tokens matching _maybe_padding_and_slice.
        attn_buf = torch.zeros(
            hidden_states.shape[0], attn_output.shape[-1],
            device=hidden_states.device, dtype=hidden_states.dtype)
        attn_buf[num_decode_tokens:num_actual_tokens] = attn_output
        # all_to_all: [tp_size, N_local, local_dim] -> [N_local, num_heads*v_dim]
        attn_buf = attn_buf.view(self.tp_size, -1, attn_output.shape[-1])
        output = torch.empty_like(attn_buf)
        torch.distributed.all_to_all_single(output.flatten(), attn_buf.flatten(), group=get_tp_group().device_group)
        attn_output = output.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)
        return self.o_proj(attn_output)

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
        
        # Note: 
        # Currently, attn_metadata.prefill.seq_lens is constructed as the "true" sequence lengths.
        # We pass actual_seq_lengths_kv=query_cumlens instead, as the ops need cumulative sum. 
        # Need to fix this when chunked prefill or prefix caching is enabled. 

        if self.on_ascend950:
            query = torch.cat([q_nope, q_pe], dim=-1)
            key = torch.cat([k_nope, k_pe], dim=-1)
            sink_key = torch.cat([sink_k_nope, sink_k_pe], dim=-1)
            kwargs = {
                "query": query,
                "key": key,
                "value": v,
                "actual_seq_lengths": attn_metadata.prefill.query_cumlens,
                "actual_seq_lengths_kv": attn_metadata.prefill.query_cumlens,
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
                "actual_seq_kvlen": attn_metadata.prefill.query_cumlens,
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
            attn_output = torch_npu._npu_kv_quant_sparse_flash_attention_pioneer(
                query=q,
                key=kv_cache[0].unsqueeze(2) if len(kv_cache[0].shape) == 3 else kv_cache[0],
                value=self.dummy_value_cache_hif8,
                sparse_indices=topk_indices,
                scale_value=self.scaling,
                key_quant_mode=2,
                value_quant_mode=2,
                block_table=metadata.block_table,
                actual_seq_lengths_query=metadata.query_cumlens,
                actual_seq_lengths_kv=metadata.seq_lens,
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
                key_sink=sink_kv,
                value_sink=sink_k_nope
            )
        elif self.cache_config.cache_dtype in ["int8_ds_mla"]:
            attn_output = torch.ops.custom.npu_ai_infra_kv_quant_sparse_flash_attention(
                query=q,
                key=kv_cache[0].unsqueeze(2),
                value=kv_cache[0].unsqueeze(2),
                sparse_indices=topk_indices,
                scale_value=self.scaling,
                key_quant_mode=2,
                value_quant_mode=2,
                sparse_block_size=1,
                actual_seq_lengths_query=metadata.query_cumlens,
                actual_seq_lengths_kv=metadata.seq_lens,
                key_sink=sink_kv,
                value_sink=sink_k_nope,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                block_table=metadata.block_table,
                attention_mode=2,
                quant_scale_repo_mode=1,
                tile_size=128,
                rope_head_dim=64,
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
                actual_seq_lengths_query=metadata.query_cumlens,
                actual_seq_lengths_kv=metadata.seq_lens,
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


    def _mla_prolog(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
        mome_metadata: Optional[NPUMomeAttentionMetadata] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor], # DSA/MLA/SWA absorb
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]: # MLA/SWA non-absorb

        # get KV cache for this layer
        kv_cache = self.attn.kv_cache[get_forward_context().virtual_engine]

        ### Q stream begins ###
        q_lora = self.q_a_proj(hidden_states)
        if self.use_mome:
            q_lora = self._apply_MOME(
                q_lora,
                self.qa_conv,
                0,
                attn_metadata,
                mome_metadata, 
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
                1, 
                attn_metadata,
                mome_metadata,
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
        if self.is_dsa_layer and self.cache_config.cache_dtype in self.quant_cache_dtype:
            return self._npu_kvrmsnorm_rope_cache_quant(*args, **kwargs)
        else:
            return self._npu_kvrmsnorm_rope_cache_unquant(*args, **kwargs)

    def _npu_kvrmsnorm_rope_cache_quant(
        self,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata],
        topk_indices: Optional[torch.Tensor] = None,
    ):
        # DSA layer c8 shape
        if self.on_ascend950 and self.cache_config.cache_dtype in ["hif8_ds_mla"]:
            actual_seq_kvlen = attn_metadata.slot_mapping.shape[0]
            k_nope = k_nope[:actual_seq_kvlen, ...]
            k_pe = k_pe[:actual_seq_kvlen, ...]
            cos = cos[:actual_seq_kvlen, ...]
            sin = sin[:actual_seq_kvlen, ...]

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
            slot_indices = torch.stack([
                attn_metadata.slot_mapping // self.block_size,
                attn_metadata.slot_mapping % self.block_size,
                ], dim=1,
            )
            torch_npu.npu_scatter_nd_update_(
                kv_cache[0].view(torch.int8),
                slot_indices,
                kv.view(torch.int8)
            )

            return kv_cache, topk_indices
        elif self.cache_config.cache_dtype in ["int8_ds_mla"]:
            actual_seq_kvlen = attn_metadata.slot_mapping.shape[0]
            k_nope = k_nope[:actual_seq_kvlen, ...]
            k_pe = k_pe[:actual_seq_kvlen, ...]
            cos = cos[:actual_seq_kvlen, ...]
            sin = sin[:actual_seq_kvlen, ...]

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
                "quant_mode": "pertile128",
                "is_output_kv": True,
                "k_cache": None,
                "ckv_cache": kv_cache[0].unsqueeze(2),
            }
            k_pe, k_nope = torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2(**kwargs)
            return kv_cache, topk_indices

    def _naive_kvrmsnorm_rope_cache(
        self,
        kv: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: MLACommonMetadata,
        cos: torch.Tensor,
        sin: torch.Tensor,
        update_k_cache: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Naive torch equivalent of npu_ai_infra_kv_rmsnorm_rope_cache_v2 for Ascend 950."""
        kv_4d = kv.view(-1, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim)
        k_nope = kv_4d[..., :self.kv_lora_rank]
        k_pe = kv_4d[..., self.kv_lora_rank:]

        # RMSNorm on k_nope
        k_nope = torch_npu.npu_rms_norm(k_nope, self.kv_a_layernorm.weight, self.kv_a_layernorm.variance_epsilon)[0]

        # Rotary embedding on k_pe
        rotary_mode = "half" if not self.rope_interleaved else "interleave"
        k_pe = torch_npu.npu_rotary_mul(k_pe, cos, sin, rotary_mode=rotary_mode)

        # Scatter update caches
        slot_indices = torch.stack([
            attn_metadata.slot_mapping // self.block_size,
            attn_metadata.slot_mapping % self.block_size,
        ], dim=1)

        torch_npu.npu_scatter_nd_update_(
            kv_cache[0],
            slot_indices,
            k_nope.squeeze(1).squeeze(1),
        )
        if update_k_cache:
            torch_npu.npu_scatter_nd_update_(
                kv_cache[1],
                slot_indices,
                k_pe.squeeze(1).squeeze(1),
            )

        return k_pe.squeeze(1).squeeze(1), k_nope.squeeze(1).squeeze(1)

    def _npu_kvrmsnorm_rope_cache_unquant(
        self,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata],
        topk_indices: Optional[torch.Tensor] = None,
    ):
        # the rest cases are unquantized
        actual_seq_kvlen = attn_metadata.slot_mapping.shape[0]
        k_nope = k_nope[:actual_seq_kvlen, ...]
        k_pe = k_pe[:actual_seq_kvlen, ...]
        cos = cos[:actual_seq_kvlen, ...]
        sin = sin[:actual_seq_kvlen, ...]

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
            if self.on_ascend950:
                k_pe, k_nope = self._naive_kvrmsnorm_rope_cache(
                    kv, kv_cache, attn_metadata,
                    cos.view(-1, 1, 1, self.qk_rope_head_dim),
                    sin.view(-1, 1, 1, self.qk_rope_head_dim),
                    update_k_cache=True,
                )
            else:
                kwargs.update({
                    "k_cache": kv_cache[1].unsqueeze(2),
                    "ckv_cache": kv_cache[0].unsqueeze(2),
                })
                k_pe, k_nope = torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2(**kwargs)

            return kv_cache, topk_indices

        else:
            # MLA/SWA non-absorb shape
            if self.on_ascend950:
                k_pe, k_nope = self._naive_kvrmsnorm_rope_cache(
                    kv, kv_cache, attn_metadata,
                    cos.view(-1, 1, 1, self.qk_rope_head_dim),
                    sin.view(-1, 1, 1, self.qk_rope_head_dim),
                    update_k_cache=True,
                )
            else:
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
        mome_metadata: Optional[NPUMomeAttentionMetadata] = None,
    ) -> torch.Tensor:

        if self.use_mome:
            attn_output = self._apply_MOME(
                attn_output,
                self.o_conv,
                2,
                attn_metadata,
                mome_metadata,
            )

        if self.enable_flashcomm2 and not self.is_dsa_layer:
            # Decode / mixed batch path: o_proj has disable_tp=True (full weight
            # [HS, num_heads*v_dim]), but attn_output has local heads only.
            # Zero-pad input to full heads dim so we can use self.o_proj directly
            # (supports INT8 quantized weights). Only local heads are non-zero,
            # so the result is a partial contribution — reduce_scatter sums all ranks. 
            tp_rank = get_tp_group().rank_in_group
            local_dim = self.num_local_heads * self.v_head_dim
            full_input = torch.zeros(
                attn_output.shape[0], self.num_heads * self.v_head_dim,
                device=attn_output.device, dtype=attn_output.dtype)
            full_input[:, tp_rank * local_dim : (tp_rank + 1) * local_dim] = attn_output
            return self.o_proj(full_input)

        hidden_states = self.o_proj(attn_output)

        return hidden_states


def npu_pangu_forward(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]    
    attn_metadata = get_forward_context().attn_metadata
    if isinstance(attn_metadata, dict):
        mome_metadata = attn_metadata.get(f"{self.prefix}.mome")
        attn_metadata = attn_metadata.get(f"{self.prefix}.attn")
    else:
        mome_metadata = None

    if attn_metadata is None:
        return self._forward_dummy(
            hidden_states,
        )
    else:
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0

        enable_cp = model_extra_config.parall_config.ena_context_parallel and self.is_dsa_layer \
                    and not has_decode and num_actual_tokens > attn_metadata.num_prefills * self.tp_size * 2
        enable_flashcomm2 = self.enable_flashcomm2 and not self.is_dsa_layer

        if self.tp_size > 1:
            if not self.all2all_backend == "naive" and not enable_cp:
                hidden_states = get_tp_group().all_gather(hidden_states, dim=0)

        if has_decode and has_prefill:
            prefill_hidden_states, prefill_cos, prefill_sin = self._prepare_phase_inputs(
                hidden_states, cos, sin, attn_metadata, 
                phase="prefill",
            )
            hidden_states[num_decode_tokens:num_actual_tokens] = self._forward_prefill(
                prefill_hidden_states,
                prefill_cos,
                prefill_sin,
                attn_metadata,
                mome_metadata.prefill,
            )

            decode_hidden_states, decode_cos, decode_sin = self._prepare_phase_inputs(
                hidden_states, cos, sin, attn_metadata, 
                phase="decode",
            )
            hidden_states[:num_decode_tokens] = self._forward_decode(
                decode_hidden_states,
                decode_cos,
                decode_sin,
                attn_metadata,
                mome_metadata.decode,
            )
            
            self._restore_phase_metadata(attn_metadata)

        elif attn_metadata.prefill is not None:
            if enable_cp:
                assert self.all2all_backend != "naive", "Context parallel is not supported with naive all2all backend"
                return self._forward_prefill_cp(
                    hidden_states,
                    cos,
                    sin,
                    attn_metadata,
                    mome_metadata,
                )
            elif enable_flashcomm2:
                return self._forward_prefill_FC2(
                    hidden_states,
                    cos,
                    sin,
                    attn_metadata,
                    mome_metadata,
                )
            else:
                hidden_states[num_decode_tokens:num_actual_tokens] = self._forward_prefill(
                    hidden_states[num_decode_tokens:num_actual_tokens],
                    cos[num_decode_tokens:num_actual_tokens],
                    sin[num_decode_tokens:num_actual_tokens],
                    attn_metadata,
                    mome_metadata,
                )
        else:
            hidden_states[:num_decode_tokens] = self._forward_decode(
                hidden_states[:num_decode_tokens],
                cos[:num_decode_tokens],
                sin[:num_decode_tokens],
                attn_metadata,
                mome_metadata,
            )
        if self.tp_size > 1:
            if self.all2all_backend == "naive":
                hidden_states = get_tp_group().all_reduce(hidden_states)
            elif not enable_cp:
                hidden_states = get_tp_group().reduce_scatter(hidden_states, dim=0)
        return hidden_states


def npu_pangu_forward_fake(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="npu_pangu_forward",
    op_func=npu_pangu_forward,
    mutates_args=[],
    fake_impl=npu_pangu_forward_fake,
    dispatch_key="PrivateUse1",
)