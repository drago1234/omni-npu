# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Optional, Union

import torch
import torch_npu
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm.platforms import current_platform
from vllm.distributed import (
    get_tensor_model_parallel_world_size, 
    get_tp_group,
)
from vllm.config import VllmConfig, CacheConfig, get_current_vllm_config
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.attention.layer import MLAAttention
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import set_weight_attrs
from vllm.logger import init_logger
logger = init_logger(__name__)
try: # UT won't include pangu_sink_swa_mla patches
    from vllm.model_executor.layers.attention.static_sink_attention import StaticSinkMLAAttention, PanguSinkAttentionBase
except ImportError:
    logger.warning("PanguSinkAttentionBase has not being defined, skipping...")
    class PanguSinkAttentionBase:
        pass

try:
    from vllm.model_executor.layers.npumome import MomeAttention
except ImportError:
    logger.warning("MomeAttention has not being defined, skipping...")

try:
    from vllm.model_executor.layers.mome import AggregateConv
except ImportError:
    logger.warning("AggregateConv has not being defined, skipping...")

from omni_npu.attention.backends.mla import NPUMLAImpl, NPUMLAMetadata
from omni_npu.v1.layers.utils import (
    yarn_get_mscale,
    named_stream,
)
from omni_npu.v1.layers.linear import (
    ColumnParallelFlashCommLinear,
    RowParallelFlashCommLinear,
)
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.attention import ops
try:
    import omni_training_custom_ops
except:
    logger.warning_once("Failed to import omni_training_custom_ops")
try:
    import omni_custom_ops
except:
    logger.warning_once("Failed to import omni_custom_ops")

from omni_npu.compilation.utils import (
    capture_graph_task,
    OP_FIA_V1,
    OP_FIA_SINK,
)

from vllm.utils.torch_utils import direct_register_custom_op


KVCACHE_NZ_DIM = 16


class NPUDeepseekMLAAttention(PanguSinkAttentionBase, torch.nn.Module):
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
        block_size_padded: int = 128,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        page_size_padded: int | None = None,
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
        self.block_size_padded = block_size_padded
        self._init_wuk_t_uv = False

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

        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelFlashCommLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
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
            disable_tp=model_extra_config.operator_opt_config.use_noncontiguous_kv,
        )

        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
        self.rope_interleaved = getattr(config,"rope_interleaved", True)
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False if self.rope_interleaved else True,
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
        # SWA
        layer_idx = extract_layer_index(prefix)
        sliding_window = None
        if hasattr(config, "sliding_window") or hasattr(config, "sliding_window_list"): # same sliding window size or different
            if not hasattr(config, "swa_layers") or layer_idx in config.swa_layers:     # all swa layer or partly
                if hasattr(config, "sliding_window_list") and hasattr(config, "swa_layers") and layer_idx in config.swa_layers:
                    sliding_window = config.sliding_window_list[config.swa_layers.index(layer_idx)]
                elif hasattr(config, "sliding_window"):
                    sliding_window = config.sliding_window
        self.sliding_window = sliding_window
        self.num_spec_tokens = vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config is not None else 0
        # MOME
        if getattr(config, "use_mome", False):
            self.merge_q_kv_conv = model_extra_config.operator_opt_config.merge_q_kv_conv
            if model_extra_config.operator_opt_config.use_noncontiguous_kv:
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
                mome_kwargs = {
                    "kernel_size": self.kernel_size,
                    "num_spec_tokens": self.num_spec_tokens,
                    "state_dtypes": self.mome_state_dtypes,
                    "state_shapes": self.mome_state_shapes,
                    "quant_config": None,
                    "cache_config": vllm_config.cache_config,
                    "prefix": f"{prefix}.conv",
                    "page_size_padded": page_size_padded,
                }
                self.conv = MomeAttention(**mome_kwargs)
            else:
                self.qa_conv = AggregateConv(self.q_lora_rank, config, vllm_config, output_parallel=False, attn_prefix=f"{prefix}.attn")
                self.compresskv_conv = AggregateConv(self.kv_lora_rank, config, vllm_config, output_parallel=False, attn_prefix=f"{prefix}.attn")
                if self.merge_q_kv_conv:
                    self.merge_conv = AggregateConv(self.q_lora_rank + self.kv_lora_rank, config, vllm_config, output_parallel=False, attn_prefix=f"{prefix}.attn")
                else:
                    self.merge_conv = None
                self.o_conv = AggregateConv(self.num_local_heads * self.v_head_dim, config, vllm_config, output_parallel=True, attn_prefix=f"{prefix}.attn")
        else:
            self.conv = None
            self.qa_conv = None
            self.compresskv_conv = None
            self.merge_conv = None
            self.o_conv = None
            self.merge_q_kv_conv = False

        if self.param_sink_number == 0:
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
                use_sparse=False,
                indexer=None,
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
                use_sparse=False,
                indexer=None,
                sink_len=self.param_sink_number,
                sliding_window=self.sliding_window,
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
                    "weight_loader": self.weight_loader,
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
                        "weight_loader": self.weight_loader,
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

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.vllm.npu_mla_forward(
            hidden_states=hidden_states, 
            cos=cos,
            sin=sin,
            layer_name=self.prefix)

    def _forward_decode(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional['NPUMLADecodeMetadata'] = None,
        pd_mixed_flag: int = 0,
        layer_name: str = "",
    ) -> torch.Tensor:
        force_decode = True if pd_mixed_flag == 1 else False
        short_prefill = True if pd_mixed_flag == 2 else False
        kv_cache = self.attn.kv_cache[get_forward_context().virtual_engine]
        nz_block_size = 16

        q_lora = self.q_a_proj(hidden_states)[0]
        kv = self.kv_a_proj_with_mqa(hidden_states)[0]

        if model_extra_config.operator_opt_config.use_noncontiguous_kv:
            if self.conv is not None:
                kv_c, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                if self.merge_q_kv_conv:
                    merge_data = torch.cat([q_lora, kv_c], dim=-1)
                    merge_conv = self.conv(merge_data, state_indice=3)
                    q_lora, kv_c = merge_conv.split(
                        [self.q_lora_rank, self.kv_lora_rank],
                        dim=-1,
                    )
                else:
                    kv_c = self.conv(kv_c, state_indice=1)
            kv = torch.cat([kv_c, k_pe], dim=-1)

            if self.conv is not None and not self.merge_q_kv_conv:
                q_lora = self.conv(q_lora, state_indice=0)
        else:
            if self.compresskv_conv is not None or self.merge_conv is not None:
                kv_c, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                if self.merge_q_kv_conv:
                    merge_data = torch.cat([q_lora, kv_c], dim=-1)
                    merge_conv = self.merge_conv(merge_data, force_decode=force_decode, short_prefill=short_prefill) + merge_data
                    q_lora, kv_c = merge_conv.split(
                        [self.q_lora_rank, self.kv_lora_rank],
                        dim=-1,
                    )
                else:
                    kv_c = self.compresskv_conv(kv_c, force_decode=force_decode, short_prefill=short_prefill) + kv_c
                if not self.rope_interleaved:
                    k_pe = k_pe.view(-1, 2, self.qk_rope_head_dim // 2) \
                        .transpose(-1, -2) \
                        .reshape(-1, self.qk_rope_head_dim)
                kv = torch.cat([kv_c, k_pe], dim=-1)

            if self.qa_conv is not None and self.merge_conv is None:
                q_lora = self.qa_conv(q_lora, force_decode=force_decode, short_prefill=short_prefill) + q_lora

        q_norm = self.q_a_layernorm(q_lora)
        q = self.q_b_proj(q_norm)[0]

        bsz, _ = q.shape
        q = q.view(bsz, self.num_local_heads, 1, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1) # b,n,s,d
        q_nope = q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim).transpose(0, 1) # n, bs, d
        q_nope = torch_npu.npu_transpose_batchmatmul(q_nope, self.attn.impl.W_UK_T, perm_y=(1, 0, 2))
        q_nope = q_nope.view(bsz, 1, self.num_local_heads, -1)

        block_num, block_size, _ = kv_cache[0].shape
        if model_extra_config.operator_opt_config.use_noncontiguous_kv:
            k_rope, k_nope = torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2(
                kv.unsqueeze(1).unsqueeze(1),
                self.kv_a_layernorm.weight,
                cos,
                sin,
                attn_metadata.slot_mapping,
                kv_cache[1].unsqueeze(2),
                kv_cache[0].unsqueeze(2),
                k_rope_scale=None,
                k_rope_offset=None,
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode="PA_NZ" if model_extra_config.operator_opt_config.kv_nz else "PA",
                rotary_mode="half" if not self.rope_interleaved else "interleave",
                quant_mode="none",
                is_output_kv=True
            )
        else:
            k_rope, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
                kv.unsqueeze(1).unsqueeze(1),
                self.kv_a_layernorm.weight,
                cos,
                sin,
                attn_metadata.slot_mapping,
                kv_cache[1].unsqueeze(2),
                kv_cache[0].unsqueeze(2),
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode="PA_NZ" if model_extra_config.operator_opt_config.kv_nz else "PA",
            )

            if model_extra_config.operator_opt_config.kv_nz:
                k_nope = k_nope.view(block_num, 1, self.kv_lora_rank // nz_block_size, block_size, nz_block_size)
                k_rope = k_rope.view(block_num, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)
            else:
                k_nope = k_nope.view(block_num, block_size, self.kv_lora_rank)
                k_rope = k_rope.view(block_num, block_size, self.qk_rope_head_dim)
        if not self.rope_interleaved:
            q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin)
        else:
            q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin) # BNSD
        q_nope = q_nope.view(bsz, self.num_local_heads, self.kv_lora_rank)
        q_pe = q_pe.view(bsz, self.num_local_heads, -1)

        if self.param_sink_number > 0:
            query_heads = 1 << (self.num_local_heads - 1).bit_length()
            pad_len = query_heads - self.num_local_heads
            q_nope_pad = q_nope.new_empty((q_nope.shape[0], pad_len, q_nope.shape[-1]))
            q_nope = torch.cat([q_nope, q_nope_pad], dim=1)
            q_pe_pad = q_pe.new_empty((q_pe.shape[0], pad_len, q_pe.shape[-1]))
            q_pe = torch.cat([q_pe, q_pe_pad], dim=1)
        else:
            query_heads = self.num_local_heads 

        NPUMLAImpl.ensure_decode_attn_mask()
        num_tokens = q_nope.size(0)
        forward_context = get_forward_context()
        if self.param_sink_number > 0:
            if self.sliding_window is not None:
                window_size = self.sliding_window-1
            else:
                window_size = NPUMLAImpl.MAX_WINDOW_SIZE
            actual_query_cumlens = attn_metadata.query_cumlens
            kwargs = {
                "query": q_nope[:actual_query_cumlens[-1]], 
                "query_rope": q_pe[:actual_query_cumlens[-1]],
                "key": kv_cache[0], 
                "value": kv_cache[0],
                "key_rope": kv_cache[1],
                "num_query_heads": query_heads,
                "num_key_value_heads": 1,
                "input_layout": "TND",
                "softmax_scale": self.scaling,
                "block_table": attn_metadata.block_table,
                "block_size": self.block_size_padded,
                "actual_seq_qlen": actual_query_cumlens,
                "actual_seq_kvlen": attn_metadata.seq_lens,
                "atten_mask": NPUMLAImpl.SHARE_MASK_TRIL_SPARSE,
                "sparse_mode": 4,
                "pre_tokens": window_size,
                "next_tokens": 0,
            }
            if model_extra_config.operator_opt_config.use_noncontiguous_kv:
                kwargs.update(
                    {"key_sink" : self.attn.sink_compressed_kv.unsqueeze(1),
                    "value_sink" : self.attn.sink_compressed_kv.unsqueeze(1),
                    "key_rope_sink" : self.attn.sink_k_pe.unsqueeze(1)}
                )
            else:
                kwargs.update(
                    {"sink_number": self.param_sink_number}
                )
            attn_output_shape = (num_tokens, query_heads, self.kv_lora_rank)
            attn_output = torch.empty(attn_output_shape, dtype=q_nope.dtype, device=q_nope.device)
            softmax_lse = torch.empty(num_tokens, dtype=q_nope.dtype, device=q_nope.device)
            if forward_context.capturing:
                capture_graph_task(
                    op_desc=OP_FIA_SINK,
                    op_kwargs=kwargs,
                    out_tensors=[attn_output, softmax_lse],
                    num_tokens=num_tokens,
                    layer_name=layer_name,
                )
            else:
                attn_output[:actual_query_cumlens[-1]] = torch.ops.custom.npu_fused_infer_attention_sink(
                    **kwargs
                )[0]
            attn_output = attn_output.transpose(0, 1).contiguous() # TND -> NTD
        else:
            actual_query_cumlens = attn_metadata.query_cumlens
            sparse_mode = 3
            input_layout = "TND_NTD"
            attn_output_shape = (self.num_local_heads, num_tokens, self.kv_lora_rank)
            attn_mask = NPUMLAImpl.SHARE_MASK_TRIL_SPARSE
            num_key_value_heads = 1
            block_size = 128
            kwargs = {
                "query": q_nope[:actual_query_cumlens[-1]],
                "key": k_nope,
                "value": k_nope,
                "query_rope": q_pe[:actual_query_cumlens[-1]],
                "key_rope": k_rope,
                "num_heads": self.num_local_heads,
                "num_key_value_heads": num_key_value_heads,
                "input_layout": input_layout,
                "atten_mask": attn_mask,
                "sparse_mode": sparse_mode,
                "scale": self.scaling,
                "antiquant_mode": 0,
                "antiquant_scale": None,
                "block_table": attn_metadata.block_table,
                "block_size": block_size,
                "actual_seq_lengths": actual_query_cumlens,
                "actual_seq_lengths_kv": attn_metadata.seq_lens,
            }
            attn_output = torch.zeros(attn_output_shape, dtype=q_nope.dtype, device=q_nope.device)
            softmax_lse = torch.empty(num_tokens, dtype=q_nope.dtype, device=q_nope.device)
            if forward_context.capturing:
                capture_graph_task(
                    op_desc=OP_FIA_V1,
                    op_kwargs=kwargs,
                    out_tensors=[attn_output, softmax_lse],
                    num_tokens=num_tokens,
                    layer_name=layer_name,
                )
            else:
                attn_output[:,:actual_query_cumlens[-1],:] = torch.ops.npu.npu_fused_infer_attention_score(**kwargs)[0]

        if self.param_sink_number > 0:
            attn_output = attn_output[:self.num_local_heads]

        # Apply UV, (N, B, L) @ W_UV (N, L, V) -> (N, B, V)
        attn_output = attn_output.view(self.num_local_heads, bsz, self.kv_lora_rank) # adapter BSND_NBSD
        attn_output = torch_npu.npu_transpose_batchmatmul(attn_output, self.attn.impl.W_UV, perm_y=(1, 0, 2))
        attn_output = attn_output.reshape(bsz, 1, -1).view(-1, self.num_local_heads * self.v_head_dim)

        if model_extra_config.operator_opt_config.use_noncontiguous_kv:
            if self.conv is not None:
                attn_output = get_tp_group().all_gather(attn_output, dim=1)
                attn_output = self.conv(attn_output, state_indice=2)
        else:
            if self.o_conv is not None:
                attn_output = self.o_conv(attn_output, force_decode=force_decode, short_prefill=short_prefill) + attn_output

        return self.o_proj.forward(attn_output)[0]

    def _forward_prefill(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[Union['NPUMLAPrefillMetadata', 'NPUMLAMetadata']] = None,
        pd_mixed_flag: bool = False,
    ) -> torch.Tensor:
        only_prefill = True if pd_mixed_flag else False
        q = self.q_a_proj(hidden_states)[0]
        attn_output = q.new_empty(
            q.shape[0],
            self.num_local_heads,
            self.v_head_dim)

        if attn_metadata is None: # for memory usage recording in dummy_run
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            latent_cache = latent_cache.view(-1, 1, latent_cache.size(-1))
            kv_a, k_pe = torch.split(latent_cache, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            if model_extra_config.operator_opt_config.use_noncontiguous_kv:
                if self.conv is not None:
                    kv_a = self.conv(kv_a, state_indice=1, is_prefill=True)
            else:
                if self.compresskv_conv is not None or self.merge_conv is not None:
                    kv_a = self.compresskv_conv(kv_a) + kv_a

            kv_a = self.kv_a_layernorm(kv_a)
            k_pe = k_pe.unsqueeze(2)
            k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin)
            k_pe = k_pe.squeeze(2)
            attn_output.fill_(0)
            attn_output = attn_output.view(-1, self.num_local_heads * self.v_head_dim)
            if model_extra_config.operator_opt_config.use_noncontiguous_kv:
                if self.conv is not None:
                    attn_output = get_tp_group().all_gather(attn_output, dim=1)
                    attn_output = self.conv(attn_output, state_indice=2, is_prefill=True)
            else:
                if self.o_conv is not None:
                    attn_output = self.o_conv(attn_output) + attn_output

            output = self.o_proj.forward(attn_output)[0]
            return output

        actual_seq_kvlen = attn_metadata.seq_lens
        actual_seq_qlen = attn_metadata.query_cumlens
        if attn_metadata.max_query_len > 1:
            attn_mask = self.attn.impl.SHARE_MASK_TRIL_SPARSE
            sparse_mode = 3
        else:
            attn_mask = None
            sparse_mode = 0

        cur_stream = torch.npu.current_stream()
        sub_stream = named_stream("mla_sub_stream")
        sub_stream.wait_stream(cur_stream)

        with torch.npu.stream(sub_stream):
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            if model_extra_config.operator_opt_config.use_noncontiguous_kv:
                if self.conv is not None:
                    kv_c, k_pe = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                    if self.merge_q_kv_conv:
                        merge_data = torch.cat([q, kv_c], dim=-1)
                        merge_conv = self.conv(merge_data, state_indice=3, is_prefill=True)
                        q, kv_c = merge_conv.split(
                            [self.q_lora_rank, self.kv_lora_rank],
                            dim=-1,
                        )
                    else:
                        kv_c = self.conv(kv_c, state_indice=1, is_prefill=True)
                    latent_cache = torch.cat([kv_c, k_pe], dim=-1)
            else:
                if self.compresskv_conv is not None:
                    kv_c, k_pe = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                    if self.merge_q_kv_conv:
                        merge_data = torch.cat([q, kv_c], dim=-1)
                        merge_conv = self.merge_conv(merge_data, only_prefill=only_prefill) + merge_data
                        q, kv_c = merge_conv.split(
                            [self.q_lora_rank, self.kv_lora_rank],
                            dim=-1,
                        )
                    else:
                        kv_c = self.compresskv_conv(kv_c, only_prefill=only_prefill) + kv_c
                    if not self.rope_interleaved:
                        k_pe = k_pe.view(-1, 2, self.qk_rope_head_dim // 2) \
                            .transpose(-1, -2) \
                            .reshape(-1, self.qk_rope_head_dim)
                    latent_cache = torch.cat([kv_c, k_pe], dim=-1)

        if model_extra_config.operator_opt_config.use_noncontiguous_kv:
            if self.conv is not None and not self.merge_q_kv_conv:
                q = self.conv(q, state_indice=0, is_prefill=True)
        else:
            if self.qa_conv is not None and self.merge_conv is None:
                q = self.qa_conv(q, only_prefill=only_prefill) + q

        q = self.q_a_layernorm(q)
        if self.quant_symbol:
            q, pertoken_scale = torch_npu.npu_dynamic_quant(q)
            q = {'x_int8': q, 'pertoken_scale': pertoken_scale}

        cur_stream.wait_stream(sub_stream)
        sub_stream.wait_stream(cur_stream)

        q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        with torch.npu.stream(sub_stream):
            kv_cache = self.attn.kv_cache[get_forward_context().virtual_engine]
            if model_extra_config.operator_opt_config.use_noncontiguous_kv:
                k_pe, kv_a = torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2(
                    latent_cache.view(-1, 1, 1, 576),
                    self.kv_a_layernorm.weight,
                    cos.view(-1, 1, 1, self.qk_rope_head_dim),
                    sin.view(-1, 1, 1, self.qk_rope_head_dim),
                    attn_metadata.slot_mapping,
                    kv_cache[1].unsqueeze(2),
                    kv_cache[0].unsqueeze(2),
                    k_rope_scale=None,
                    k_rope_offset=None,
                    epsilon=self.kv_a_layernorm.variance_epsilon,
                    cache_mode="PA_NZ" if model_extra_config.operator_opt_config.kv_nz else "PA",
                    rotary_mode="half" if not self.rope_interleaved else "interleave",
                    quant_mode="none",
                    is_output_kv=True
                )
            else:
                _, _, k_pe, kv_a = torch_npu.npu_kv_rmsnorm_rope_cache(
                    latent_cache.view(-1, 1, 1, 576), # bnsd
                    self.kv_a_layernorm.weight,
                    cos.view(-1, 1, 1, self.qk_rope_head_dim),
                    sin.view(-1, 1, 1, self.qk_rope_head_dim),
                    attn_metadata.slot_mapping,
                    kv_cache[1].unsqueeze(2),
                    kv_cache[0].unsqueeze(2),
                    k_rope_scale=None,
                    k_rope_offset=None,
                    epsilon=self.kv_a_layernorm.variance_epsilon,
                    cache_mode="PA_NZ" if model_extra_config.operator_opt_config.kv_nz else "PA",
                    is_output_kv=True
                )

        cur_stream.wait_stream(sub_stream)
        sub_stream.wait_stream(cur_stream)

        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim],  dim=-1)
        q_pe = q_pe.unsqueeze(2)
        if not self.rope_interleaved:
            q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin)
        else:
            q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin) # BNSD
        q_pe = q_pe.squeeze(2) # BSH
        with torch.npu.stream(sub_stream):
            prefill_kv_a = kv_a[:actual_seq_qlen[-1]]
            prefill_k_pe = k_pe[:actual_seq_qlen[-1]]
            # When sink tokens are used, we need to insert cached sink tokens at the beginning of each sequence
            if not model_extra_config.operator_opt_config.use_noncontiguous_kv:
                if self.param_sink_number > 0:
                    prefill_k_pe = prefill_k_pe.squeeze(2).squeeze(1)
                    prefill_k_pe = self._insert_tensor_by_start_loc(
                        prefill_k_pe,
                        self.attn.sink_k_pe,
                        attn_metadata.query_start_loc,
                    )
                    prefill_kv_a = prefill_kv_a.squeeze(2).squeeze(1)
                    prefill_kv_a = self._insert_tensor_by_start_loc(
                        prefill_kv_a,
                        self.attn.sink_compressed_kv,
                        attn_metadata.query_start_loc,
                    )
            else:
                sink_kv = self.kv_b_proj.forward(self.attn.sink_compressed_kv)[0]
            kv = self.kv_b_proj.forward(prefill_kv_a)[0]

        cur_stream.wait_stream(sub_stream)

        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        prefill_k_rope = prefill_k_pe.view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_local_heads, 1)

        if model_extra_config.operator_opt_config.use_noncontiguous_kv:
            sink_k_pe = self.attn.sink_k_pe.view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_local_heads, 1)
            sink_k_nope, sink_v = torch.split(
                sink_kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim),
                [self.qk_nope_head_dim, self.v_head_dim],
                dim=-1,
            )
        else:
            if self.param_sink_number > 0:
                # When sink tokens are used, the actual sequence lengths for key and value are different.
                num_prefills = len(actual_seq_qlen)
                sink_len_offset = [self.param_sink_number * (i + 1) for i in range(num_prefills)]
                kv_cumlens = [x + y for x, y in zip(actual_seq_qlen, sink_len_offset)]
            else:
                kv_cumlens = actual_seq_qlen

        if self.param_sink_number > 0:
            if self.sliding_window is not None:
                window_size = self.sliding_window-1
            else:
                window_size = NPUMLAImpl.MAX_WINDOW_SIZE

            kwargs = {
                "query": q_nope[:actual_seq_qlen[-1]].contiguous(),
                "query_rope": q_pe[:actual_seq_qlen[-1]],
                "key": k_nope.contiguous(),
                "value": v.contiguous(),
                "key_rope": prefill_k_rope,
                "num_query_heads": self.num_local_heads,
                "num_key_value_heads": self.num_local_heads,
                "input_layout": "TND",
                "softmax_scale": self.scaling,
                "sparse_mode": 4,
                "atten_mask": NPUMLAImpl.SHARE_MASK_TRIL_SPARSE,
                "actual_seq_qlen": actual_seq_qlen,
                "pre_tokens": window_size,
                "next_tokens": 0,
            }
            if model_extra_config.operator_opt_config.use_noncontiguous_kv:
                kwargs.update(
                    {"actual_seq_kvlen": actual_seq_qlen,
                    "key_sink": sink_k_nope.contiguous(),
                    "value_sink": sink_v.contiguous(),
                    "key_rope_sink": sink_k_pe}
                )
            else:
                kwargs.update(
                    {"actual_seq_kvlen": kv_cumlens,
                    "sink_number": self.param_sink_number,}
                )
            attn_output[:actual_seq_qlen[-1]] = torch.ops.custom.npu_fused_infer_attention_sink(**kwargs)[0]
        else:
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
                actual_seq_lengths_kv=kv_cumlens,
                scale=self.scaling,
                next_tokens=0
            )[0]

        attn_output = attn_output.view(-1, self.num_local_heads * self.v_head_dim)

        if model_extra_config.operator_opt_config.use_noncontiguous_kv:
            if self.conv is not None:
                attn_output = get_tp_group().all_gather(attn_output, dim=1)
                attn_output = self.conv(attn_output, state_indice=2, is_prefill=True)
        else:
            if self.o_conv is not None:
                attn_output = self.o_conv(attn_output, only_prefill=only_prefill) + attn_output

        output = self.o_proj.forward(attn_output)[0]
        return output

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
        if model_extra_config.operator_opt_config.use_noncontiguous_kv:
            if self.merge_q_kv_conv and self.conv is not None:
                self.conv.merge_conv.weight.data = torch.cat([self.conv.qa_conv.weight.data, self.conv.compresskv_conv.weight.data], dim=1).contiguous()
        else:
            if self.merge_q_kv_conv and self.merge_conv is not None:
                self.merge_conv.merge_conv.weight.data = torch.cat([self.qa_conv.merge_conv.weight.data, self.compresskv_conv.merge_conv.weight.data], dim=0).contiguous()
                self.merge_conv.conv_weight = self.merge_conv.merge_conv.weight.data.squeeze(1).transpose(0, 1).contiguous()

    @staticmethod
    def _insert_tensor_by_start_loc(
        raw_tensor: torch.Tensor, insert_segment: torch.Tensor, start_loc: list[int]
    ) -> torch.Tensor:
        segment_len = insert_segment.shape[0]
        num_inserts = len(start_loc) - 1
        total_len = segment_len * num_inserts + raw_tensor.shape[0]
        offset = 0
        # allocate result tensor
        result = torch.empty(total_len, *raw_tensor.shape[1:], device=raw_tensor.device, dtype=raw_tensor.dtype)

        for i in range(num_inserts):
            # write insert segment to result
            result[offset:offset+segment_len] = insert_segment
            offset += segment_len
            # write raw tensor to result
            seg_len = start_loc[i + 1] - start_loc[i]
            result[offset:offset+seg_len] = raw_tensor[start_loc[i]:start_loc[i+1]]
            offset += seg_len

        return result

def npu_mla_forward(  
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    full_hidden_states = hidden_states
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
    self = forward_context.no_compile_layers[layer_name]
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[f"{self.prefix}.attn"]

    if not model_extra_config.operator_opt_config.use_noncontiguous_kv:
        if self.param_sink_number > 0:
            assert self.attn.sink_k_pe is not None and self.attn.sink_compressed_kv is not None, (
                "sink_k_pe and sink_compressed_kv have not been prepared"
            )
            if not self.attn.sink_populated:
                self_kv_cache = self.attn.kv_cache[forward_context.virtual_engine]
                if self_kv_cache is not None and len(self_kv_cache) > 0:
                    self.attn.populate_sink_kv(self_kv_cache[0], self_kv_cache[1])

    def _maybe_quant(hs):
        if not self.quant_symbol:
            return hs
        x_int8, scale = torch_npu.npu_dynamic_quant(hs)
        return {"x_int8": x_int8, "pertoken_scale": scale}

    if attn_metadata is None:
        return self._forward_prefill(_maybe_quant(hidden_states), cos, sin, attn_metadata)

    num_actual_tokens = attn_metadata.num_actual_tokens
    num_decode_tokens = attn_metadata.num_decode_tokens
    has_decode = attn_metadata.num_decodes > 0
    has_prefill = attn_metadata.num_prefills > 0

    if has_decode and has_prefill:
        prefill_hs = hidden_states[num_decode_tokens:num_actual_tokens]
        prefill_cos = cos[num_decode_tokens:num_actual_tokens]
        prefill_sin = sin[num_decode_tokens:num_actual_tokens]
        attn_metadata.prefill.slot_mapping = attn_metadata.slot_mapping[num_decode_tokens:num_actual_tokens]
        prefill_output = self._forward_prefill(_maybe_quant(prefill_hs), prefill_cos, prefill_sin, attn_metadata.prefill, pd_mixed_flag=True)

        decode_hs = hidden_states[:num_decode_tokens]
        decode_cos = cos[:num_decode_tokens]
        decode_sin = sin[:num_decode_tokens]
        attn_metadata.decode.slot_mapping = attn_metadata.slot_mapping[:num_decode_tokens]
        pd_mixed_flag = 2 if num_decode_tokens > attn_metadata.num_decodes else 1 # short prefill in decode or pure decode 
        decode_output = self._forward_decode(
            _maybe_quant(decode_hs),
            decode_cos,
            decode_sin,
            attn_metadata.decode,
            pd_mixed_flag=pd_mixed_flag,
            layer_name=f"{layer_name}.attn",
        )

        mixed_output = torch.cat([decode_output, prefill_output], dim=0)
        return _pad_output_to_input_tokens(mixed_output)

    if has_prefill:
        prefill_hs = hidden_states[num_decode_tokens:num_actual_tokens]
        prefill_cos = cos[num_decode_tokens:num_actual_tokens]
        prefill_sin = sin[num_decode_tokens:num_actual_tokens]
        attn_metadata.prefill.slot_mapping = attn_metadata.slot_mapping[num_decode_tokens:num_actual_tokens]
        prefill_output = self._forward_prefill(_maybe_quant(prefill_hs), prefill_cos, prefill_sin, attn_metadata.prefill)
        return _pad_output_to_input_tokens(prefill_output)
    else:
        decode_hs = hidden_states[:num_decode_tokens]
        decode_cos = cos[:num_decode_tokens]
        decode_sin = sin[:num_decode_tokens]
        attn_metadata.decode.slot_mapping = attn_metadata.slot_mapping[:num_decode_tokens]
        decode_output = self._forward_decode(
            _maybe_quant(decode_hs),
            decode_cos,
            decode_sin,
            attn_metadata.decode,
            layer_name=f"{layer_name}.attn",
        )
        return _pad_output_to_input_tokens(decode_output)
    
    
def npu_mla_forward_fake(   
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="npu_mla_forward",
    op_func=npu_mla_forward,
    mutates_args=[],
    fake_impl=npu_mla_forward_fake,
    dispatch_key="PrivateUse1",
)
