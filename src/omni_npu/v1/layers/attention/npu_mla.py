# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Optional, Union

import torch
import torch_npu
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm.platforms import current_platform
from vllm.distributed import get_tensor_model_parallel_world_size
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
    from vllm.model_executor.layers.mome import AggregateConv
except ImportError:
    logger.warning("PanguSinkAttentionBase has not being defined, skipping...")
    class PanguSinkAttentionBase:
        pass

from omni_npu.attention.backends.mla import NPUMLAImpl, NPUMLAMetadata
from omni_npu.v1.layers.utils import (
    yarn_get_mscale, 
    named_stream,
)
from omni_npu.v1.layers.linear import (
    ColumnParallelFlashCommLinear,
    RowParallelFlashCommLinear,
)
from omni_npu.v1.models.config_loader.loader import model_extra_config
from omni_npu.attention import ops
import omni_training_custom_ops
import omni_custom_ops

from omni_npu.compilation.utils import (
    capture_multi_fia_graph_size,
    capture_multi_fia_sink_graph_size,
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
        # MOME
        if getattr(config, "use_mome", False):
            self.qa_conv = AggregateConv(self.q_lora_rank, config, vllm_config, output_parallel=False, attn_prefix=f"{prefix}.attn")
            self.compresskv_conv = AggregateConv(self.kv_lora_rank, config, vllm_config, output_parallel=False, attn_prefix=f"{prefix}.attn")
            self.o_conv = AggregateConv(self.num_local_heads * self.v_head_dim, config, vllm_config, output_parallel=True, attn_prefix=f"{prefix}.attn")
        else:
            self.qa_conv = None
            self.compresskv_conv = None
            self.o_conv = None

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
        pd_mixed_flag: bool = False,
    ) -> torch.Tensor:
        force_decode = True if pd_mixed_flag else False
        kv_cache = self.attn.kv_cache[get_forward_context().virtual_engine]
        nz_block_size = 16

        q_lora = self.q_a_proj(hidden_states)[0]
        kv = self.kv_a_proj_with_mqa(hidden_states)[0]

        if self.compresskv_conv is not None:
            kv_c, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            kv_c = self.compresskv_conv(kv_c, force_decode=force_decode) + kv_c
            if not self.rope_interleaved:
                k_pe = self.even_odd_indexing(k_pe)
            kv = torch.cat([kv_c, k_pe], dim=-1)

        if self.qa_conv is not None:
            q_lora = self.qa_conv(q_lora, force_decode=force_decode) + q_lora
        q_norm = self.q_a_layernorm(q_lora)
        q = self.q_b_proj(q_norm)[0]

        bsz, _ = q.shape
        q = q.view(bsz, self.num_local_heads, 1, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1) # b,n,s,d
        q_nope = q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim).transpose(0, 1) # n, bs, d
        q_nope = torch_npu.npu_transpose_batchmatmul(q_nope, self.attn.impl.W_UK_T, perm_y=(1, 0, 2))
        q_nope = q_nope.view(bsz, 1, self.num_local_heads, -1)

        block_num, block_size, _ = kv_cache[0].shape
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
            q_pe = self.even_odd_indexing(q_pe)
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
            kwargs = {
                "query": q_nope, 
                "key": kv_cache[0], 
                "value": kv_cache[0],
                "query_rope": q_pe,
                "key_rope": kv_cache[1],
                "num_query_heads": query_heads,
                "num_key_value_heads": 1,
                "input_layout": "TND",
                "softmax_scale": self.scaling,
                "block_table": attn_metadata.block_table,
                "block_size": 128,
                "actual_seq_qlen": attn_metadata.query_cumlens,
                "actual_seq_kvlen": attn_metadata.seq_lens,
                "atten_mask": NPUMLAImpl.DECORE_ATTN_MASK,
                "sparse_mode": 4,
                "sink_number": self.param_sink_number,
                "pre_tokens": window_size,
                "next_tokens": 0,
            }
            attn_output_shape = (num_tokens, query_heads, self.kv_lora_rank)
            attn_output = torch.empty(attn_output_shape, dtype=q_nope.dtype, device=q_nope.device)
            softmax_lse = torch.empty(num_tokens, dtype=q_nope.dtype, device=q_nope.device)
            if forward_context.capturing:
                capture_multi_fia_sink_graph_size(
                    attn_output=attn_output,
                    softmax_lse=softmax_lse ,
                    num_tokens=num_tokens,
                    const_args=kwargs)
                attn_output = attn_output.transpose(0, 1).contiguous()
            else:
                attn_output = torch.ops.custom.npu_fused_infer_attention_sink(
                    **kwargs
                )[0].transpose(0, 1).contiguous() # TND -> NTD
        else:
            sparse_mode = 3
            input_layout = "TND_NTD"
            attn_output_shape = (self.num_local_heads, num_tokens, self.kv_lora_rank)
            attn_mask = NPUMLAImpl.DECORE_ATTN_MASK
            num_key_value_heads = 1
            block_size = 128
            kwargs = {
                "query": q_nope,
                "key": k_nope,
                "value": k_nope,
                "query_rope": q_pe,
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
                "actual_seq_lengths": attn_metadata.query_cumlens,
                "actual_seq_lengths_kv": attn_metadata.seq_lens,
            }
            attn_output = torch.empty(attn_output_shape, dtype=q_nope.dtype, device=q_nope.device)
            softmax_lse = torch.empty(num_tokens, dtype=q_nope.dtype, device=q_nope.device)
            if forward_context.capturing:
                capture_multi_fia_graph_size(
                    attn_output=attn_output,
                    softmax_lse=softmax_lse ,
                    num_tokens=num_tokens,
                    const_args=kwargs)
            else:
                attn_output = torch.ops.npu.npu_fused_infer_attention_score(**kwargs)[0]

        if self.param_sink_number > 0:
            attn_output = attn_output[:self.num_local_heads]

        # Apply UV, (N, B, L) @ W_UV (N, L, V) -> (N, B, V)
        attn_output = attn_output.view(self.num_local_heads, bsz, self.kv_lora_rank) # adapter BSND_NBSD
        attn_output = torch_npu.npu_transpose_batchmatmul(attn_output, self.attn.impl.W_UV, perm_y=(1, 0, 2))
        attn_output = attn_output.reshape(bsz, 1, -1).view(-1, self.num_local_heads * self.v_head_dim)
        if self.o_conv is not None:
            attn_output = self.o_conv(attn_output, force_decode=force_decode) + attn_output
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
            if self.compresskv_conv is not None:
                kv_a = self.compresskv_conv(kv_a) + kv_a
            kv_a = self.kv_a_layernorm(kv_a)
            k_pe = k_pe.unsqueeze(2)
            k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin)
            k_pe = k_pe.squeeze(2)
            attn_output.fill_(0)
            attn_output = attn_output.view(-1, self.num_local_heads * self.v_head_dim)
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

        if self.qa_conv is not None:
            q = self.qa_conv(q, only_prefill=only_prefill) + q
        q = self.q_a_layernorm(q)
        if self.quant_symbol:
            q, pertoken_scale = torch_npu.npu_dynamic_quant(q)
            q = {'x_int8': q, 'pertoken_scale': pertoken_scale}
        with torch.npu.stream(sub_stream):
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            if self.compresskv_conv is not None:
                kv_c, k_pe = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                kv_c = self.compresskv_conv(kv_c, only_prefill=only_prefill) + kv_c
                if not self.rope_interleaved:
                    k_pe = self.even_odd_indexing(k_pe)
                latent_cache = torch.cat([kv_c, k_pe], dim=-1)

        cur_stream.wait_stream(sub_stream)
        sub_stream.wait_stream(cur_stream)

        q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        with torch.npu.stream(sub_stream):
            kv_cache = self.attn.kv_cache[get_forward_context().virtual_engine]
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
            q_pe = self.even_odd_indexing(q_pe)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin) # BNSD
        q_pe = q_pe.squeeze(2) # BSH
        with torch.npu.stream(sub_stream):
            prefill_kv_a = kv_a[:actual_seq_kvlen[-1]]
            prefill_k_pe = k_pe[:actual_seq_kvlen[-1]]
            # When sink tokens are used, we need to insert cached sink tokens at the beginning of each sequence
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
            kv = self.kv_b_proj.forward(prefill_kv_a)[0]

        cur_stream.wait_stream(sub_stream)

        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        prefill_k_rope = prefill_k_pe.view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_local_heads, 1)

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
            output = torch.ops.custom.npu_fused_infer_attention_sink(
                q_nope[:actual_seq_qlen[-1]],
                k_nope,
                v,
                query_rope=q_pe[:actual_seq_qlen[-1]],
                key_rope=prefill_k_rope,
                num_query_heads=self.num_local_heads,
                num_key_value_heads=self.num_local_heads,
                input_layout="TND",
                sparse_mode=4,
                atten_mask=NPUMLAImpl.SHARE_MASK_TRIL_SPARSE,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=kv_cumlens,
                softmax_scale=self.scaling,
                sink_number=self.param_sink_number,
                pre_tokens=window_size,
                next_tokens=0,
            )[0]
            attn_output[:actual_seq_qlen[-1]] = output
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
                actual_seq_lengths_kv=actual_seq_kvlen,
                scale=self.scaling,
                next_tokens=0
            )[0]

        attn_output = attn_output.view(-1, self.num_local_heads * self.v_head_dim)
        if self.o_conv is not None:
            attn_output = self.o_conv(attn_output, only_prefill=only_prefill) + attn_output
        output = self.o_proj.forward(attn_output)[0]
        return output

    def post_weight_load(self) -> None:
        if getattr(self, 'param_sink_number', 0) > 0:
            if getattr(self, "kv_a_layernorm", None) is not None:
                param_sink_compressed_kv = self.kv_a_layernorm(self.param_sink_compressed_kv)
            else:
                param_sink_compressed_kv = self.param_sink_compressed_kv
            self.attn.update_sink_kv(self.param_sink_k_pe, param_sink_compressed_kv)

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

    @staticmethod
    def even_odd_indexing(x):
        """
        使用索引实现：偶数位置←前半部分，奇数位置←后半部分
        """
        *prefix, dim = x.shape
        assert dim % 2 == 0, "最后维度必须是偶数"
        
        half = dim // 2
        
        result = torch.zeros_like(x)
        
        result[..., 0::2] = x[..., :half]
        result[..., 1::2] = x[..., half:]
        
        return result

def npu_mla_forward(  
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    attn_metadata = forward_context.attn_metadata
    if self.quant_symbol:
        hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
        hidden_states = {'x_int8': hidden_states, 'pertoken_scale': pertoken_scale}
                
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[f"{self.prefix}.attn"]

    if self.param_sink_number > 0:
        assert self.attn.sink_k_pe is not None and self.attn.sink_compressed_kv is not None, (
            "sink_k_pe and sink_compressed_kv have not been prepared"
        )
        if not self.attn.sink_populated:
            self_kv_cache = self.attn.kv_cache[forward_context.virtual_engine]
            if self_kv_cache is not None and len(self_kv_cache) > 0:
                self.attn.populate_sink_kv(self_kv_cache[0], self_kv_cache[1])

    if attn_metadata is None:
        return self._forward_prefill(hidden_states, cos, sin, attn_metadata)

    num_actual_toks = attn_metadata.num_actual_tokens
    has_decode = attn_metadata.num_decodes > 0
    has_prefill = attn_metadata.num_prefills > 0
    num_decode_tokens = attn_metadata.num_decode_tokens

    if has_decode and has_prefill:
        prefill_hidden_states = hidden_states[num_decode_tokens:num_actual_toks, ...]
        prefill_cos = cos[num_decode_tokens:num_actual_toks, ...]
        prefill_sin = sin[num_decode_tokens:num_actual_toks, ...]
        attn_metadata.prefill.slot_mapping = attn_metadata.slot_mapping[num_decode_tokens:num_actual_toks]
        prefill_output = self._forward_prefill(prefill_hidden_states, prefill_cos, prefill_sin, attn_metadata.prefill, pd_mixed_flag=True)

        decode_hidden_states = hidden_states[:num_decode_tokens]
        decode_cos = cos[:num_decode_tokens]
        decode_sin = sin[:num_decode_tokens]
        attn_metadata.decode.slot_mapping = attn_metadata.slot_mapping[:num_decode_tokens]
        decode_output = self._forward_decode(decode_hidden_states, decode_cos, decode_sin, attn_metadata.decode, pd_mixed_flag=True)

        return torch.cat([decode_output, prefill_output], dim=0)

    if attn_metadata.prefill is not None:
        attn_metadata.prefill.slot_mapping = attn_metadata.slot_mapping
        return self._forward_prefill(hidden_states, cos, sin, attn_metadata.prefill)
    else:
        attn_metadata.decode.slot_mapping = attn_metadata.slot_mapping
        return self._forward_decode(hidden_states, cos, sin, attn_metadata.decode)
    
    
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
