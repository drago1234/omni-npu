# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Optional, Tuple

import torch
from torch import nn
import torch_npu
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm.platforms import current_platform
from vllm.model_executor.models.utils import extract_layer_index
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    split_tensor_along_last_dim,
)
from vllm.config import VllmConfig, CacheConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.utils import set_weight_attrs
from vllm.attention.layer import MLAAttention
from vllm.logger import init_logger
logger = init_logger(__name__)

try:
    from vllm.model_executor.layers.attention.static_sink_attention import StaticSinkMLAAttention
except ImportError:
    logger.warning("StaticSinkMLAAttention has not being defined, skipping...")

try:
    from vllm.model_executor.layers.mome import AggregateConv
except ImportError:
    logger.warning("AggregateConv has not being defined, skipping...")

try: 
    from vllm.model_executor.layers.npumome import MomeAttention
except ImportError:
    logger.warning("MomeAttention has not being defined, skipping...")

from omni_npu.attention.backends.dsa import NPUDSAMetadata
from omni_npu.layers.utils import named_stream
from omni_npu.attention.backends.utils import (
    SPManager,
    DummySPManager,
    lazy_init_cos_sin,
)
from omni_npu.v1.layers.utils import yarn_get_mscale, calculate_page_size_padded
from omni_npu.v1.layers.linear import (
    RowParallelFlashCommLinear,
    ColumnParallelFlashCommLinear,
    ReplicatedFlashCommLinear
)
from omni_npu.model_config.config_loader.loader import  model_extra_config
from omni_npu.v1.utils import current_stream
from omni_npu.plugin_decorators import dsa_attn_decorator

class Indexer(torch.nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        sink_len: int = 0,
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
        self.wq_b = ReplicatedFlashCommLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wk = ReplicatedFlashCommLinear(
            hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wk",
        )
        if sink_len > 0:
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
            self.weights_proj = ReplicatedFlashCommLinear(
                hidden_size, self.n_head, bias=False, quant_config=None, prefix=f"{prefix}.weights_proj"
            )
        else:
            self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
            self.weights_proj = ReplicatedFlashCommLinear(
                hidden_size, self.n_head, quant_config=None, prefix=f"{prefix}.weights_proj"
            )
        self.sink_len = sink_len

    def _apply_rope(
        self,
        x: torch.Tensor,   # TND
        cos: torch.Tensor, # BNSD
        sin: torch.Tensor, # BNSD
    ) -> torch.Tensor:     # TND
        assert x.dim() == 3 # TND
        R, D, N = self.rope_dim, self.head_dim, x.size(1)
        pe, nope = torch.split(x, [R, D - R], dim=-1)
        pe = pe.view(-1, N, 1, R) # BNSD
        if getattr(self.config, "indexer_rope_interleave", False):
            pe = torch_npu.npu_interleave_rope(pe, cos, sin)
        else:
            pe = torch_npu.npu_rotary_mul(pe, cos, sin)
        pe = pe.view(-1, N, R) # TND
        return torch.cat([pe, nope], dim=-1) # TND

    def _li_prolog_ext(
        self,
        wx: torch.Tensor, # TD
        qr: torch.Tensor, # TD
        kx: torch.Tensor, # TD
        q_cos_sin: tuple, # BNSD, BNSD
        k_cos_sin: tuple, # BNSD, BNSD
    ) -> tuple[torch.Tensor]:
        assert qr.size(0) == q_cos_sin[0].size(0)
        assert kx.size(0) == k_cos_sin[0].size(0)
        N, D = self.n_head, self.head_dim
        wi = self.weights_proj(wx)[0]         # TN
        qi = self.wq_b(qr)[0].view(-1, N, D)  # TND
        qi = self._apply_rope(qi, *q_cos_sin) # TND
        ki = self.wk(kx)[0]                   # TD
        ki = self.k_norm(ki).view(-1, 1, D)   # T1D
        ki = self._apply_rope(ki, *k_cos_sin) # T1D
        return wi, qi, ki # TN, TND, T1D

    def _li_prolog(
        self,
        x: torch.Tensor,   # TD
        qr: torch.Tensor,  # TD
        cos: torch.Tensor, # BNSD
        sin: torch.Tensor, # BNSD
    ) -> tuple[torch.Tensor]:
        assert qr.size(0) == x.size(0)
        return self._li_prolog_ext(x, qr, x, (cos, sin), (cos, sin))

    def _update_cache(
        self,
        ki: torch.Tensor, # T1D
        slots: torch.Tensor,
        ki_cache: torch.Tensor, # [*, pg, 1, D]
    ):
        D = self.head_dim
        block_size = ki_cache.shape[1]
        slot_indices = torch.stack([
            slots // block_size,
            slots % block_size,
            ], dim=1,
        )
        torch_npu.npu_scatter_nd_update_(
            ki_cache,
            slot_indices,
            ki.view(-1, D),
        )

    def _apply_lightning_indexer(
        self,
        wi: torch.Tensor,       # [T, N]
        qi: torch.Tensor,       # [T, N, D]
        ki_cache: torch.Tensor, # [*, pg, 1, D]
        q_cumlens: torch.Tensor = None,   # int32 [B]
        kv_lens: torch.Tensor = None,     # int32 [B]
        block_table: torch.Tensor = None, # int32 [T, *]
    ) -> torch.Tensor: # int32 [T, 1, K]
        if any(it is None for it in [q_cumlens, kv_lens, block_table]):
            return None
        if model_extra_config.operator_opt_config.use_noncontiguous_kv:
            return torch.ops.custom.npu_lightning_indexer_enhance(
                query=qi,
                key=ki_cache.unsqueeze(2),
                weights=wi,
                actual_seq_lengths_query=q_cumlens,
                actual_seq_lengths_key=kv_lens,
                block_table=block_table,
                layout_key="PA_BSND",
                layout_query="TND",
                sparse_count=self.topk_tokens,
                sparse_mode=3,
                sparse_block_size=1,
                sparse_block_mode=False,
            )[0]

        num_sink_blocks = self.sink_len // self.vllm_config.cache_config.block_size
        block_table = block_table[:, num_sink_blocks:]

        return torch_npu.npu_lightning_indexer(
            weights=wi,
            query=qi,
            key=ki_cache,
            actual_seq_lengths_query=q_cumlens,
            actual_seq_lengths_key=kv_lens - self.sink_len,
            block_table=block_table,
            layout_key="PA_BSND",
            layout_query="TND",
            sparse_count=self.topk_tokens,
            sparse_mode=3
        )[0]

    def forward(
        self,
        x: torch.Tensor,   # TD
        qr: torch.Tensor,  # TD
        cos: torch.Tensor, # BNSD
        sin: torch.Tensor, # BNSD
        attn_metadata: NPUDSAMetadata,
        ki_cache: torch.Tensor, # [*, pg, 1, D]
    ) -> tuple[torch.Tensor]:
        wi, qi, ki = self._li_prolog(x, qr, cos, sin)
        self._update_cache(ki, attn_metadata.slot_mapping, ki_cache)
        tok_idx = self._apply_lightning_indexer(
            wi, qi, ki_cache,
            q_cumlens=attn_metadata.query_cumlens.to(torch.int32),
            kv_lens=attn_metadata.seq_lens.to(torch.int32),
            block_table=attn_metadata.block_table,
        )
        return tok_idx, ki


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
        self.num_local_heads = num_heads if model_extra_config.parall_config.ena_context_parallel else num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.prefix = prefix
        self.quant_symbol = quant_config is not None
        self._init_wuk_t_uv = False
        self.is_pd_disagg = vllm_config.kv_transfer_config is not None

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedFlashCommLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_a_proj",
            )
            self.kv_a_proj_with_mqa = ReplicatedFlashCommLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedFlashCommLinear(
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
                disable_tp=model_extra_config.parall_config.ena_context_parallel,
            )
        else:
            self.q_proj = ColumnParallelFlashCommLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
                disable_tp=model_extra_config.parall_config.ena_context_parallel,
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelFlashCommLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
            disable_tp=model_extra_config.parall_config.ena_context_parallel,
        )
        self.o_proj = RowParallelFlashCommLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            disable_tp=model_extra_config.parall_config.ena_context_parallel,
        )

        self.rope_interleaved = getattr(config,"rope_interleaved", True)
        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
            is_neox_style = False # Deepseek V3.2
        else:
            is_neox_style = True # GLM 5 (for generating neox style sin and cos caches. gptj style will be applied by the npu_interleave_rope operator)

        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=is_neox_style,
        )

        if (
            config.rope_parameters["rope_type"] != "default"
            and config.rope_parameters["rope_type"] == "deepseek_yarn"
        ):
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale
        
        # sink
        self.param_sink_number = getattr(config, "param_sink_number", 0)
        self.param_sink_with_value = getattr(config, "param_sink_with_value", False)

        self.indexer = Indexer(
            vllm_config,
            config,
            hidden_size,
            q_lora_rank,
            quant_config,
            cache_config,
            self.param_sink_number,
            f"{prefix}.indexer",
        )

        self.num_speculative_tokens = 0 if not vllm_config.speculative_config else vllm_config.speculative_config.num_speculative_tokens

        # MOME
        if getattr(config, "use_mome", False):
            if model_extra_config.operator_opt_config.use_noncontiguous_kv:
                num_extra_token = 1 if self.is_pd_disagg else 0
                fake_num_spec_tokens = max(self.num_speculative_tokens, num_extra_token)
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
                self.kernel_size = getattr(config, 'router_sliding_window', 0)
                self.cache_dtype_str = None

                page_size_padded, block_size_padded = calculate_page_size_padded(
                    cache_config=vllm_config.cache_config,
                    cache_dtype_str=None,
                    config=config,
                    mome_state_shapes=self.mome_state_shapes,
                    mome_state_dtypes=self.mome_state_dtypes,
                    kernel_size=self.kernel_size,
                    fake_spec_tokens=fake_num_spec_tokens,
                )

                mome_kwargs = {
                    "kernel_size": self.kernel_size,
                    "num_spec_tokens": fake_num_spec_tokens,
                    "state_dtypes": self.mome_state_dtypes,
                    "state_shapes": self.mome_state_shapes,
                    "quant_config": None,
                    "vllm_config": vllm_config,
                    "prefix": f"{prefix}.conv",
                    "page_size_padded": page_size_padded,
                }
                self.conv = MomeAttention(**mome_kwargs)
            else:
                self.qa_conv = AggregateConv(self.q_lora_rank, config, vllm_config, output_parallel=False, attn_prefix=f"{prefix}.attn")
                self.compresskv_conv = AggregateConv(self.kv_lora_rank, config, vllm_config, output_parallel=False, attn_prefix=f"{prefix}.attn")
                self.o_conv = AggregateConv(self.num_local_heads * self.v_head_dim, config, vllm_config, output_parallel=True, attn_prefix=f"{prefix}.attn")
                page_size_padded = None
                block_size_padded = vllm_config.cache_config.block_size
        else:
            self.qa_conv = None
            self.compresskv_conv = None
            self.o_conv = None
            self.conv = None
            page_size_padded = None
            block_size_padded = vllm_config.cache_config.block_size

        if self.param_sink_number == 0:
            assert self.q_b_proj.tp_size == self.kv_b_proj.tp_size
            self.attn = MLAAttention(
                num_heads=self.num_heads // self.q_b_proj.tp_size,
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
        else:
            self.attn = StaticSinkMLAAttention(
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
                sink_len=self.param_sink_number,
                page_size_padded=page_size_padded,
                block_size_padded=block_size_padded,
            )

        if self.param_sink_number > 0:
            self.param_sink_k_pe = torch.nn.Parameter(
                torch.empty(
                    (
                        self.param_sink_number,
                        self.qk_rope_head_dim,
                    ),
                    device=current_platform.device_type,
                    dtype=config.torch_dtype,
                )
            )
            set_weight_attrs(
                self.param_sink_k_pe,
                {
                    "output_dim": 1,
                    "weight_loader": self.sink_kv_weight_loader,
                },
            )
            if self.param_sink_with_value:
                self.param_sink_compressed_kv = torch.nn.Parameter(
                    torch.empty(
                        (
                            self.param_sink_number,
                            self.kv_lora_rank,
                        ),
                        device=current_platform.device_type,
                        dtype=config.torch_dtype,
                    )
                )
                set_weight_attrs(
                    self.param_sink_compressed_kv,
                    {
                        "output_dim": 1,
                        "weight_loader": self.sink_kv_weight_loader,
                    },
                )
            else:
                self.param_sink_compressed_kv = torch.zeros(
                    (
                        self.param_sink_number,
                        self.kv_lora_rank,
                    ),
                    device=current_platform.device_type,
                    dtype=config.torch_dtype,
                )
        # To enable dummy run with out weight
        self.post_weight_load()

        self.dummy_value_cache = torch.zeros(
            (1, block_size_padded, 1, self.kv_lora_rank),
            device='npu',
            dtype=torch.bfloat16,
        )
        
        self.use_mlaprolog = model_extra_config.operator_opt_config.enable_mlaprolog
        self.use_omni_cache = model_extra_config.operator_opt_config.use_omni_cache
        self.layer_idx = extract_layer_index(self.prefix)

        self.tp_size = get_tensor_model_parallel_world_size() if not model_extra_config.operator_opt_config.enable_dsa else 1
        self.actual_seq_lengths = {}

        gear_list = [1]
        if vllm_config.npu_compilation_config.decode_gear_list is not None:
            gear_list = vllm_config.npu_compilation_config.decode_gear_list
        elif vllm_config.compilation_config.cudagraph_capture_sizes is not None:
            gear_list = vllm_config.compilation_config.cudagraph_capture_sizes

        for batch_size in gear_list:
            self.actual_seq_lengths[batch_size] = (1 + self.num_speculative_tokens) * \
                                                  torch.arange(1, batch_size * self.tp_size // (
                                                              1 + self.num_speculative_tokens) + 1, dtype=torch.int64,
                                                               device=current_platform.device_type)
    def forward(
        self,
        x: torch.Tensor,   # TD
        cos: torch.Tensor, # BNSD
        sin: torch.Tensor, # BNSD
    ) -> torch.Tensor:
        full_hidden_states = x
        total_tokens = full_hidden_states.shape[0]

        def _pad_output_to_input_tokens(attn_output: torch.Tensor) -> torch.Tensor:
            output_tokens = attn_output.shape[0]
            if output_tokens == total_tokens:
                return attn_output
            if output_tokens > total_tokens:
                raise RuntimeError(
                    f"npu_mla_forward output tokens ({output_tokens}) exceed input tokens ({total_tokens})"
                )
            full_output = full_hidden_states.clone()
            full_output[:output_tokens, ...] = attn_output
            return full_output

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        kv_cache = self.attn.kv_cache[forward_context.virtual_engine]

        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[f"{self.prefix}.attn"]

        if self.param_sink_number > 0 and not model_extra_config.operator_opt_config.use_noncontiguous_kv:
            assert self.attn.sink_k_pe is not None and self.attn.sink_compressed_kv is not None, (
                "sink_k_pe and sink_compressed_kv have not been prepared"
            )
            if not self.attn.sink_populated:
                self_kv_cache = self.attn.kv_cache[forward_context.virtual_engine]
                if self_kv_cache is not None and len(self_kv_cache) > 0:
                    self.attn.populate_sink_kv(self_kv_cache[0], self_kv_cache[1])

        if attn_metadata is None:
            if model_extra_config.parall_config.ena_context_parallel:
                return self._forward_prefill_cp(x, cos, sin, attn_metadata, kv_cache)
            else:
                return self._forward_prefill(x, cos, sin, attn_metadata, kv_cache)
        
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0

        if has_decode and has_prefill:
            prefill_hs = x[num_decode_tokens:num_actual_tokens]
            prefill_cos = cos[num_decode_tokens:num_actual_tokens]
            prefill_sin = sin[num_decode_tokens:num_actual_tokens]
            attn_metadata.prefill.slot_mapping = attn_metadata.slot_mapping[num_decode_tokens:num_actual_tokens]
            prefill_output = self._forward_prefill(prefill_hs, prefill_cos, prefill_sin, attn_metadata.prefill, kv_cache, pd_mixed_flag=True)

            decode_hs = x[:num_decode_tokens]
            decode_cos = cos[:num_decode_tokens]
            decode_sin = sin[:num_decode_tokens]
            attn_metadata.decode.slot_mapping = attn_metadata.slot_mapping[:num_decode_tokens]
            pd_mixed_flag = 2 if num_decode_tokens > attn_metadata.num_decodes else 1 # short prefill in decode or pure decode
            decode_output = self._forward_decode(decode_hs, decode_cos, decode_sin, attn_metadata.decode, kv_cache, pd_mixed_flag=pd_mixed_flag)

            mixed_output = torch.cat([decode_output, prefill_output], dim=0)
            return _pad_output_to_input_tokens(mixed_output)

        if has_prefill:
            if model_extra_config.parall_config.ena_context_parallel:
                return self._forward_prefill_cp(x, cos, sin, attn_metadata, kv_cache)
            else:
                prefill_hs = x[num_decode_tokens:num_actual_tokens]
                prefill_cos = cos[num_decode_tokens:num_actual_tokens]
                prefill_sin = sin[num_decode_tokens:num_actual_tokens]
                attn_metadata.prefill.slot_mapping = attn_metadata.slot_mapping[num_decode_tokens:num_actual_tokens]
                prefill_output = self._forward_prefill(prefill_hs, prefill_cos, prefill_sin, attn_metadata.prefill, kv_cache)
                return _pad_output_to_input_tokens(prefill_output)
        else:
            decode_hs = x[:num_decode_tokens]
            decode_cos = cos[:num_decode_tokens]
            decode_sin = sin[:num_decode_tokens]
            attn_metadata.decode.slot_mapping = attn_metadata.slot_mapping[:num_decode_tokens]
            decode_output = self._forward_decode(decode_hs, decode_cos, decode_sin, attn_metadata.decode, kv_cache)
            return _pad_output_to_input_tokens(decode_output)

    def _q_absorb(
        self,
        q_lora: torch.Tensor, # TD
        cos: torch.Tensor,    # BNSD
        sin: torch.Tensor,    # BNSD
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Q = self.qk_nope_head_dim
        R = self.qk_rope_head_dim
        tok = q_lora.size(0)
        q = self.q_b_proj(q_lora)[0].view(tok, -1, Q + R) # TND
        q_nope, q_pe = torch.split(q, [Q, R], dim=-1)     # TND
        if self.attn.impl.W_UK_T.shape[-1] % 128 != 0 or self.attn.impl.W_UK_T.shape[-2] % 128 != 0:
            q_nope = (q_nope.transpose(0, 1) @ self.attn.impl.W_UK_T).transpose(1, 0)
        else:
            q_nope = torch_npu.npu_transpose_batchmatmul(
                q_nope.transpose(0, 1),       # TND -> NTD
                weight=self.attn.impl.W_UK_T, # [Q, L]
                perm_y=(1, 0, 2),             # NTD -> TND
            )
        if not self.rope_interleaved:
            q_pe = torch_npu.npu_rotary_mul(
                q_pe.view(tok, -1, 1, R), # BNSD
                cos, sin,                 # BNSD
            ).view(tok, -1, R)            # TND
        else:
            q_pe = torch_npu.npu_interleave_rope(
                q_pe.view(tok, -1, 1, R), # BNSD
                cos, sin,                 # BNSD
            ).view(tok, -1, R)            # TND
        return q_nope, q_pe

    def _kv_norm_rope_cache(
        self,
        latent_kv: torch.Tensor, # TD
        cos: torch.Tensor,       # BNSD
        sin: torch.Tensor,       # BNSD
        attn_metadata: NPUDSAMetadata = None,
        kv_cache: tuple[torch.Tensor] = None,
        fused_op: bool = True,
    ) -> tuple: # [*, pg, 1, L], [*, pg, 1, R], T1D, T1D
        R, L = self.qk_rope_head_dim, self.kv_lora_rank
        no_cache = kv_cache is None or attn_metadata is None
        if no_cache or not fused_op:
            latent_kv = latent_kv.view(-1, L + R)                 # TD
            k_nope, k_pe = torch.split(latent_kv, [L, R], dim=-1) # TD
            k_nope = self.kv_a_layernorm(k_nope).view(-1, 1, L)   # T1D
            k_pe = torch_npu.npu_interleave_rope(
                k_pe.view(-1, 1, 1, R), # BNSD
                cos, sin,               # BNSD
            ).view(-1, 1, R)            # T1D

            if no_cache:
                return None, None, k_nope, k_pe

            def cache_kv(x: torch.Tensor, cache: torch.Tensor):
                slots = attn_metadata.slot_mapping.view(-1, 1)
                cache = cache.view(-1, 1, cache.size(-1)) # T1D
                torch_npu.npu_scatter_nd_update_(cache, slots, x)
            cache_kv(k_nope, kv_cache[0])
            cache_kv(k_pe, kv_cache[1])
            return kv_cache[0], kv_cache[1], k_nope, k_pe
        else:
            if model_extra_config.operator_opt_config.use_noncontiguous_kv:
                rope_cache, nope_cache = torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2(
                    latent_kv.view(-1, 1, 1, L + R),
                    self.kv_a_layernorm.weight,
                    cos,
                    sin,
                    attn_metadata.slot_mapping,
                    k_cache=None,
                    ckv_cache=kv_cache[0].unsqueeze(2),
                    k_rope_scale=None,
                    k_rope_offset=None,
                    epsilon=self.kv_a_layernorm.variance_epsilon,
                    cache_mode="PA_NZ" if model_extra_config.operator_opt_config.kv_nz else "PA",
                    rotary_mode="half" if not self.rope_interleaved else "interleave",
                    quant_mode="none",
                    is_output_kv=True
                )
                # nope_cache.unsqueeze_(2)
                # rope_cache.unsqueeze_(2)
                k_nope, k_pe = nope_cache, rope_cache
            else:
                rope_cache, nope_cache, k_pe, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
                    latent_kv.view(-1, 1, 1, L + R), # BNSD
                    self.kv_a_layernorm.weight,
                    cos, sin, # BNSD
                    attn_metadata.slot_mapping,
                    kv_cache[1].view(-1, 128, 1, R),
                    kv_cache[0].view(-1, 128, 1, L),
                    epsilon=self.kv_a_layernorm.variance_epsilon,
                    cache_mode="PA",
                    is_output_kv=True,
                ) # -> [*, pg, 1, L], [*, pg, 1, R], BNSD, BNSD
            k_nope = k_nope.view(-1, 1, L)
            k_pe = k_pe.view(-1, 1, R)
            return nope_cache, rope_cache, k_nope, k_pe

    @dsa_attn_decorator
    def _apply_attention( # absorb
        self,
        q_nope: torch.Tensor, # [T, N, D]
        q_pe: torch.Tensor,   # [T, N, R]
        k_nope: torch.Tensor, # [*, pg, 1, L]
        k_pe: torch.Tensor,   # [*, pg, 1, R]
        q_cumlens: torch.Tensor = None,   # int32 [B]
        kv_lens: torch.Tensor = None,     # int32 [B]
        topk_idx: torch.Tensor = None,    # int32 [T, 1, K]
        block_table: torch.Tensor = None, # int32 [T, *]
        kv_cache: torch.Tensor = None,
        attn_metadata: NPUDSAMetadata = None,
    ) -> torch.Tensor: # [T, N, L]
        if None in [q_cumlens, kv_lens, block_table, topk_idx]:
            return torch.zeros_like(q_nope) # dummy
        if model_extra_config.operator_opt_config.use_noncontiguous_kv:
            sink_k_pe = self.param_sink_k_pe.unsqueeze(1)
            sink_k_nope = self.attn.sink_compressed_kv.unsqueeze(1)
            sink_kv = torch.cat([sink_k_nope, sink_k_pe], dim=-1)
            return torch.ops.custom.npu_ai_infra_sparse_flash_attention_pioneer(
                query=torch.cat([q_nope, q_pe], dim=-1),
                key=kv_cache[0].unsqueeze(2),
                value=self.dummy_value_cache,
                sparse_indices=topk_idx,
                scale_value=self.scaling,
                sparse_block_size=1,
                block_table=block_table,
                actual_seq_lengths_query=q_cumlens,
                actual_seq_lengths_kv=kv_lens,
                pre_tokens=(1<<63)-1,
                next_tokens=(1<<63)-1,
                attention_mode=2,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                key_sink=sink_kv.contiguous(),
                value_sink=sink_k_nope.contiguous(),
            )[0]

        return torch_npu.npu_sparse_flash_attention(
            query=q_nope,            # [T, N, L]
            key=k_nope,              # [*, pg, 1, L]
            value=k_nope,            # [*, pg, 1, L]
            query_rope=q_pe,         # [T, N, R]
            key_rope=k_pe,           # [*, pg, 1, R]
            sparse_indices=topk_idx, # [T, 1, 2048]
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            block_table=block_table,
            actual_seq_lengths_query=q_cumlens,
            actual_seq_lengths_kv=kv_lens,
            scale_value=self.scaling,
            attention_mode=2,
            sparse_mode=3,
        )[0] # -> [T, N, L]

    def _mla_epilog(self, out: torch.Tensor, reorg: bool=False):
        assert out.dim() == 3

        out = torch_npu.npu_transpose_batchmatmul(
            out.transpose(0, 1),        # TND -> NTD
            weight=self.attn.impl.W_UV, # [N, L, V]
            perm_y=(1, 0, 2),           # NTD -> TND
        ).reshape(out.size(0), -1)      # [T, NV]

        if reorg:
            tp_size = get_tp_group().world_size
            assert self.o_proj.tp_size == tp_size
            D = self.num_local_heads * self.v_head_dim
            sp_out = out.view(-1, tp_size, D)                 # [T, n, NV]
            sp_out = sp_out.transpose(0, 1).contiguous() # [n, T, NV]
            tp_out = torch.empty_like(sp_out)
            torch.distributed.all_to_all_single(
                tp_out.view(tp_size, -1),
                sp_out.view(tp_size, -1),
                [1] * tp_size, [1] * tp_size,
                group=get_tp_group().device_group,
            )
            out = tp_out.view(-1, D) # [nT, NV]
        return out

    def _forward_prefill(
        self,
        x: torch.Tensor,   # TD
        cos: torch.Tensor, # BNSD
        sin: torch.Tensor, # BNSD
        attn_metadata: NPUDSAMetadata = None,
        kv_cache: tuple[torch.Tensor] = None,
        pd_mixed_flag: bool = False,
    ) -> torch.Tensor:
        only_prefill = True if pd_mixed_flag else False
        q_cumlens, kv_lens, block_table, topk_idx = None, None, None, None
        if attn_metadata:
            kv_lens = attn_metadata.seq_lens.to(torch.int32)
            q_cumlens = attn_metadata.query_cumlens.to(torch.int32)
            block_table = attn_metadata.block_table

        # seq_parallel = model_extra_config.parall_config.ena_seq_parallel
        seq_parallel = False
        sp_manager: SPManager = (
            attn_metadata.sp_manager
            if attn_metadata is not None
            else DummySPManager(get_tp_group())
        ) if seq_parallel else None

        if sp_manager:
            lazy_init_cos_sin(sp_manager, cos, sin)

        if sp_manager:
            cur_stream = torch.npu.current_stream()
            com_stream = named_stream("comm_stream")
            com_stream.wait_stream(cur_stream)
            cur_stream.wait_stream(com_stream)

        latent_kv = self.kv_a_proj_with_mqa(x)[0] # TD
        if model_extra_config.operator_opt_config.use_noncontiguous_kv:
            if self.conv is not None:
                kv_c, k_pe = latent_kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                kv_c = self.conv(kv_c, state_indice=1, is_prefill=True)
                latent_kv = torch.cat([kv_c, k_pe], dim=-1)
        else:
            if self.compresskv_conv is not None:
                kv_c, k_pe = latent_kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                kv_c = self.compresskv_conv(kv_c, only_prefill=only_prefill) + kv_c
                if not self.rope_interleaved:
                    k_pe = k_pe.view(-1, 2, self.qk_rope_head_dim // 2) \
                            .transpose(-1, -2) \
                            .reshape(-1, self.qk_rope_head_dim)
                latent_kv = torch.cat([kv_c, k_pe], dim=-1)

        if sp_manager:
            com_stream.wait_stream(cur_stream)
            with torch.npu.stream(com_stream):
                latent_kv = sp_manager.ag_tokens(latent_kv)

        q_lora = self.q_a_proj(x)[0]        # TD
        if model_extra_config.operator_opt_config.use_noncontiguous_kv:
            if self.conv is not None:
                q_lora = self.conv(q_lora, state_indice=0, is_prefill=True)
        else:
            if self.qa_conv is not None:
                q_lora = self.qa_conv(q_lora, only_prefill=only_prefill) + q_lora
        q_lora = self.q_a_layernorm(q_lora) # TD

        if sp_manager:
            cur_stream.wait_stream(com_stream)
        k_nope, k_pe, tnd_k_nope, tnd_k_pe = self._kv_norm_rope_cache(
            latent_kv, cos, sin,
            attn_metadata, kv_cache,
            fused_op=True)

        wi, qi, ki = self.indexer._li_prolog(
            x, q_lora,
            cos=sp_manager.sp_cos if sp_manager else cos,
            sin=sp_manager.sp_sin if sp_manager else sin,
        )
        if sp_manager:
            com_stream.wait_stream(cur_stream)
            with torch.npu.stream(com_stream):
                ki = sp_manager.ag_tokens(ki)
            cur_stream.wait_stream(com_stream)
        if attn_metadata:
            if model_extra_config.operator_opt_config.use_noncontiguous_kv:
                self.indexer._update_cache(ki, attn_metadata.slot_mapping, kv_cache[1])
            else:
                self.indexer._update_cache(ki, attn_metadata.slot_mapping, kv_cache[2])

        if sp_manager:
            with torch.npu.stream(com_stream):
                q_lora = sp_manager.ag_tokens(q_lora)
            cur_stream.wait_stream(com_stream)
        q_nope, q_pe = self._q_absorb(q_lora, cos, sin)

        if sp_manager:
            with torch.npu.stream(com_stream):
                wi = sp_manager.ag_tokens(wi)
                qi = sp_manager.ag_tokens(qi)
            cur_stream.wait_stream(com_stream)
        if attn_metadata:
            if model_extra_config.operator_opt_config.use_noncontiguous_kv:
                topk_idx = self.indexer._apply_lightning_indexer(
                    wi, qi, kv_cache[1], q_cumlens, kv_lens, block_table)
            else:
                topk_idx = self.indexer._apply_lightning_indexer(
                    wi, qi, kv_cache[2], q_cumlens, kv_lens, block_table)
            if self.param_sink_number and (not model_extra_config.operator_opt_config.use_noncontiguous_kv):
                sink_indices = torch.arange(self.param_sink_number, device=topk_idx.device,
                                            dtype=topk_idx.dtype).expand(topk_idx.shape[0], 1, self.param_sink_number)
                mask = (topk_idx != -1).to(topk_idx.dtype)
                topk_idx = torch.concat((sink_indices, topk_idx + mask * self.param_sink_number), dim=2)

        if sp_manager:
            com_stream.wait_stream(cur_stream)

        attn_out = self._apply_attention(
            q_nope, q_pe, # [T, N, D]
            k_nope, k_pe, # [*, pg, 1, D]
            q_cumlens, kv_lens,
            topk_idx,    # int32 [T, 1, K]
            block_table, # int32 [T, *]
            kv_cache,
            attn_metadata=attn_metadata,
        ) # [T, N, L]

        if sp_manager:
            attn_out = sp_manager.align_tokens(attn_out)
        
        out = self._mla_epilog(attn_out)
        if model_extra_config.operator_opt_config.use_noncontiguous_kv:
            if self.conv is not None:
                out = get_tp_group().all_gather(out, dim=1)
                out = self.conv(out, state_indice=2, is_prefill=True)
                if self.o_proj.tp_size > 1:
                    out = split_tensor_along_last_dim(out, num_partitions=self.o_proj.tp_size)
                    out = out[self.o_proj.tp_rank].contiguous()
        else:
            if self.o_conv is not None:
                out = self.o_conv(out, only_prefill=only_prefill) + out
        return self.o_proj(out)[0]

    def _apply_mome_prefill_cp(
        self,
        x: torch.Tensor,
        state_indice: int,
        sp_manager: Optional[SPManager] = None,
    ) -> torch.Tensor:
        assert not model_extra_config.operator_opt_config.merge_q_kv_conv, "merge_q_kv_conv is not supported when prefill cp is enabled"
        assert model_extra_config.operator_opt_config.use_noncontiguous_kv, "use_noncontiguous_kv is required when prefill cp is enabled"

        merged_x = sp_manager.mome_suffix_exchange(x)
        merged_x = sp_manager.broadcast_mome_req_tails_from_rank0(merged_x)

        forward_context = get_forward_context()
        metadata = forward_context.attn_metadata
        if metadata is None:
            return x
        kv_cache = self.conv.kv_cache[forward_context.virtual_engine]

        mome_metadata = metadata[self.conv.prefix]
        if mome_metadata.prefill is None:
            return x

        merge_conv = self.conv.qa_conv
        if state_indice == 1:
            merge_conv = self.conv.compresskv_conv
        elif state_indice == 2:
            merge_conv = self.conv.o_conv

        merged_x = merge_conv.forward_prefill(
            x=merged_x,
            conv_states=kv_cache[state_indice][:, :self.conv.kernel_size - 1],
            cache_indices=mome_metadata.prefill.cache_indices,
            query_start_loc=sp_manager.cp_mome_query_start_loc,
        )

        return sp_manager.mome_split_and_cat(merged_x)

    def _forward_prefill_cp(
        self,
        x: torch.Tensor,   # TD
        cos: torch.Tensor, # BNSD
        sin: torch.Tensor, # BNSD
        attn_metadata: NPUDSAMetadata = None,
        kv_cache: tuple[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert self.q_b_proj.tp_size == 1  # full head required
        assert self.kv_b_proj.tp_size == 1 # full head required
        # assert model_extra_config.parall_config.ena_seq_parallel # dependency
        assert not self.use_omni_cache and kv_cache is not None
        """
        SP here refers to standard SP splitting, i.e., splitting tokens with ceil division
        based on sp_size.

        CP refers to zigzag-form SP splitting, aimed at adjusting query distribution to
        balance attention computation across ranks.
        """
        sp_manager: SPManager = (
            attn_metadata.prefill.sp_manager
            if attn_metadata is not None
            else DummySPManager(get_tp_group()))

        lazy_init_cos_sin(sp_manager, cos, sin, init_zigzag=True)

        sp_cos_sin = (sp_manager.sp_cos, sp_manager.sp_sin)
        cp_cos_sin = (sp_manager.cp_cos, sp_manager.cp_sin)
        sp_x, cp_x = x, sp_manager.sp_to_cp(x) # TD

        q_lora = self.q_a_proj(cp_x)[0]     # TD, cp
        q_lora = self._apply_mome_prefill_cp(q_lora, state_indice=0, sp_manager=sp_manager)
        q_lora = self.q_a_layernorm(q_lora) # TD, cp
        q_nope, q_pe = self._q_absorb(q_lora, *cp_cos_sin) # TND, full head, cp

        kv = self.kv_a_proj_with_mqa(sp_x)[0] # TD, sp
        kv = sp_manager.ag_tokens(kv)         # TD
        kv_c, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        if self.conv is not None:
            kv_c = self.conv(kv_c, state_indice=1, is_prefill=True)
        kv = torch.cat([kv_c, k_pe], dim=-1)
        _, _, k_nope, k_pe = self._kv_norm_rope_cache(
            kv, cos, sin, attn_metadata, kv_cache, fused_op=True)

        wi, qi, ki = self.indexer._li_prolog_ext(
            cp_x, q_lora, sp_x, cp_cos_sin, sp_cos_sin)
        ki = sp_manager.ag_tokens(ki)
        if attn_metadata:
            if hasattr(attn_metadata.prefill, "cache_fn"):
                if model_extra_config.operator_opt_config.use_noncontiguous_kv:
                    attn_metadata.prefill.cache_fn(ki.view(-1, ki.size(-1)), kv_cache[1])
                else:
                    attn_metadata.prefill.cache_fn(ki.view(-1, ki.size(-1)), kv_cache[2])
            else:
                if model_extra_config.operator_opt_config.use_noncontiguous_kv:
                    self.indexer._update_cache(ki, attn_metadata.slot_mapping, kv_cache[1])
                else:
                    self.indexer._update_cache(ki, attn_metadata.slot_mapping, kv_cache[2])

        q_cumlens, kv_lens, _, blk_table = sp_manager.cp_attn_meta()
        topk_idx = self.indexer._apply_lightning_indexer(
            wi, qi,             # TND, cp
            kv_cache[1] if attn_metadata is not None else None,
            q_cumlens, kv_lens, # int32 [2B]
            blk_table,          # int32 [T, *]
        ) # int32 [T, 1, K] or None for dummy_run

        attn_out = self._apply_attention(
            q_nope, q_pe,       # TND, full head, cp
            k_nope, k_pe,       # [*, pg, 1, D]
            q_cumlens, kv_lens, # int32 [2B]
            topk_idx,           # int32 [T, 1, K]
            blk_table,          # int32 [T, *]
            kv_cache,
        )

        cp_out = self._mla_epilog(attn_out, reorg=self.o_proj.tp_size > 1)
        cp_out = self._apply_mome_prefill_cp(cp_out, state_indice=2, sp_manager=sp_manager)
        cp_out= self.o_proj(cp_out)[0]
        return sp_manager.cp_to_sp(cp_out) if attn_metadata else cp_out

    def _forward_mlaprolog(
        self,
        hidden_states,
        cos,
        sin,
        kv_cache,
        attn_metadata
    ):
        bs, _ = hidden_states.view(-1, hidden_states.shape[-1]).shape
        q_nope, q_pe, dequant_scale_q_nope, q_norm, dequant_scale_q_norm = torch_npu.npu_mla_prolog_v3(
            token_x=hidden_states.view(bs, 1, -1),
            weight_dq=self.q_a_proj.weight,                 # BF16, NZ
            weight_uq_qr=self.q_b_proj.weight,              # BF16, NZ
            weight_uk=self.attn.impl.W_UK_T,                # BF16, ND
            weight_dkv_kr=self.kv_a_proj_with_mqa.weight,   # BF16, NZ
            rmsnorm_gamma_cq=self.q_a_layernorm.weight,
            rmsnorm_gamma_ckv=self.kv_a_layernorm.weight,
            rope_sin=sin.squeeze(1),
            rope_cos=cos.squeeze(1),
            kv_cache=kv_cache[0],
            kr_cache=kv_cache[1],
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
        x: torch.Tensor,   # TD
        cos: torch.Tensor, # BNSD
        sin: torch.Tensor, # BNSD
        attn_metadata: NPUDSAMetadata = None,
        kv_cache: tuple[torch.Tensor] = None,
        pd_mixed_flag: int = 0,
    ) -> torch.Tensor:
        force_decode = True if pd_mixed_flag == 1 else False
        short_prefill = True if pd_mixed_flag == 2 else False
        # TODO: support decode ena_seq_parallel in the future
        # assert not model_extra_config.parall_config.ena_seq_parallel

        if self.use_mlaprolog:
            q_nope, q_pe, q_lora, k_nope, k_pe, _, _ = self._forward_mlaprolog(
                x, cos, sin, kv_cache, attn_metadata)
        else:
            q_lora = self.q_a_proj(x)[0]        # TD
            if model_extra_config.operator_opt_config.use_noncontiguous_kv:
                if self.conv is not None:
                    q_lora = self.conv(q_lora, state_indice=0)
            else:
                if self.qa_conv is not None:
                    q_lora = self.qa_conv(q_lora, force_decode=force_decode, short_prefill=short_prefill) + q_lora
            q_lora = self.q_a_layernorm(q_lora) # TD
            q_nope, q_pe = self._q_absorb(q_lora, cos, sin) # TND

            kv = self.kv_a_proj_with_mqa(x)[0]  # TD
            if model_extra_config.operator_opt_config.use_noncontiguous_kv:
                if self.conv is not None:
                    kv_c, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                    kv_c = self.conv(kv_c, state_indice=1)
                    kv = torch.cat([kv_c, k_pe], dim=-1)
            else:
                if self.compresskv_conv is not None:
                    kv_c, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                    kv_c = self.compresskv_conv(kv_c, force_decode=force_decode, short_prefill=short_prefill) + kv_c
                    if not self.rope_interleaved:
                        k_pe = k_pe.view(-1, 2, self.qk_rope_head_dim // 2) \
                            .transpose(-1, -2) \
                            .reshape(-1, self.qk_rope_head_dim)
                    kv = torch.cat([kv_c, k_pe], dim=-1)
            k_nope, k_pe, _, _ = self._kv_norm_rope_cache(
                kv, cos, sin, attn_metadata, kv_cache,
            ) # -> [*, pg, 1, D]

        if model_extra_config.operator_opt_config.use_noncontiguous_kv:
            topk_idx = self.indexer(x, q_lora, cos, sin, attn_metadata, kv_cache[1])[0]
        else:
            topk_idx = self.indexer(x, q_lora, cos, sin, attn_metadata, kv_cache[2])[0]
        if self.param_sink_number and (not model_extra_config.operator_opt_config.use_noncontiguous_kv):
            sink_indices = torch.arange(self.param_sink_number, device=topk_idx.device,
                                        dtype=topk_idx.dtype).expand(topk_idx.shape[0], 1, self.param_sink_number)
            mask = (topk_idx != -1).to(topk_idx.dtype)
            topk_idx = torch.concat((sink_indices, topk_idx + mask * self.param_sink_number), dim=2)

        attn_out = self._apply_attention(
            q_nope, q_pe, # [T, N, D]
            k_nope, k_pe, # [*, pg, 1, D]
            q_cumlens=attn_metadata.query_cumlens.to(torch.int32),
            kv_lens=attn_metadata.seq_lens.to(torch.int32),
            block_table=attn_metadata.block_table,
            topk_idx=topk_idx, # int32 [T, 1, K]
            kv_cache=kv_cache,
        ) # [T, N, L]

        out = self._mla_epilog(attn_out)
        if model_extra_config.operator_opt_config.use_noncontiguous_kv:
            if self.conv is not None:
                out = get_tp_group().all_gather(out, dim=1)
                out = self.conv(out, state_indice=2)
                if self.o_proj.tp_size > 1:
                    out = split_tensor_along_last_dim(out, num_partitions=self.o_proj.tp_size)
                    out = out[self.o_proj.tp_rank].contiguous()
        else:
            if self.o_conv is not None:
                out = self.o_conv(out, force_decode=force_decode, short_prefill=short_prefill) + out
        return self.o_proj(out)[0]

    def post_weight_load(self) -> None:
        if self._init_wuk_t_uv and getattr(self.attn.impl, "W_UK_T", None) is not None:
            is_weight_nz = getattr(self.kv_b_proj.weight, "is_weight_nz", False)
            if is_weight_nz:
                self.kv_b_proj.weight.data = torch_npu.npu_format_cast(self.kv_b_proj.weight.data, torch_npu.Format.ND)
            self.attn.impl.process_weights_after_loading(self.kv_b_proj.weight.dtype)
            if is_weight_nz:
                self.kv_b_proj.weight.data = torch_npu.npu_format_cast(self.kv_b_proj.weight.data, torch_npu.Format.FRACTAL_NZ)
        else:
            self._init_wuk_t_uv = True
        if getattr(self, 'param_sink_number', 0) > 0:
            if getattr(self, "kv_a_layernorm", None) is not None:
                param_sink_compressed_kv = self.kv_a_layernorm(self.param_sink_compressed_kv)
            else:
                param_sink_compressed_kv = self.param_sink_compressed_kv
            self.attn.update_sink_kv(self.param_sink_k_pe, param_sink_compressed_kv)

    def sink_kv_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
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
                tp_size = getattr(self, "tp_size", 1)
                assert final_shape[output_dim] % tp_size == 0
                final_shape[output_dim] = final_shape[output_dim] // tp_size
            param.materialize(final_shape, dtype=loaded_weight.dtype)
        param_data = param.data
        if output_dim is not None and not is_sharded_weight:
            shard_size = param_data.shape[output_dim]
            tp_rank = getattr(self, "tp_rank", 0)
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
