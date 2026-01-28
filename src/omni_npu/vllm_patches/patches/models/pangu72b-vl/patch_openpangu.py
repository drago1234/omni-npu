from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention.layer import AttentionType
from vllm.attention.layers.static_sink_attention import StaticSinkAttention
from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import (
    extract_layer_index,
    is_pp_missing_parameter,
)
from vllm.transformers_utils.config import set_default_rope_theta
from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from vllm.model_executor.models import openpangu
from vllm.model_executor.models.openpangu import (
    OpenPanguMoE,
    OpenPanguMLAAttention,
    OpenPanguEmbeddedAttention,
    OpenPanguMoEModel,
    OpenPanguDecoderLayer,
    OpenPanguModel,
    OpenPanguMLP,
    check_ffn_act_fn,
) 
from vllm.logger import init_logger
logger = init_logger(__name__)
 

@register_patch("OpenPanguMoEPatch", OpenPanguMoE)
class OpenPanguMoEPatch(VLLMPatch):
    _attr_names_to_apply = ['__init__']

    
    def __init__(
        self,
        config: PretrainedConfig,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        #####patch start: for pangu72B-VL
        nn.Module.__init__(self)
        #####patch end

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

        #####patch start: for pangu72B-VL
        if (
            hasattr(config, "router_enable_expert_bias")
            and config.router_enable_expert_bias
        ):
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32)
            )
        else:
            self.gate.e_score_correction_bias = None
        #####patch end

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
            self.shared_experts = OpenPanguMLP(
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


@register_patch("OpenPanguMLAAttentionPatch", OpenPanguMLAAttention)
class OpenPanguMLAAttentionPatch(VLLMPatch):
    _attr_names_to_apply = ['__init__']

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
        super().__init__()
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

        # TODO: remove hard coding
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

            #####patch start: for pangu718B
            indexer_rotary_emb=None,
            #####patch end

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


@register_patch("openpanguPatch", openpangu)
class openpanguPatch(VLLMPatch):
    _attr_names_to_apply = ['OpenPanguSinkAttention', 'PanguProMoEV2ForCausalLM']


    #####patch start: for pangu72B-VL
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

            # 检查EnginerArgs && vllm_config.cache_config 里是否有该变量
            from vllm.config import get_current_vllm_config
            vllm_config = get_current_vllm_config()
            is_fused_op_enabled = getattr(vllm_config.cache_config, "enable_kv_rmsnorm_rope_cache", False)
            from vllm.engine.arg_utils import EngineArgs
            logger.info(f"[openpanguPatch] enable_kv_rmsnorm_rope_cache status: \
                                vllm_config_obj:{is_fused_op_enabled} \
                                engine_args_obj: {getattr(EngineArgs, "enable_kv_rmsnorm_rope_cache", False)}")

            # Set result as vllm_config value
            self.enable_kv_rmsnorm_rope_cache = is_fused_op_enabled

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
                # Number of KV heads is less than TP size, so we replicate
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
            self._init_rotary_emb(
                config, rope_parameters=rope_parameters, quant_config=quant_config)

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

            # Special case for loading scales off disk, which often do not
            # have a shape (such as in the case of AutoFP8).
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
            cos, sin = None, None
            if self.enable_kv_rmsnorm_rope_cache:
                q, _, cos, sin = self.rotary_emb(positions, q.contiguous(), output_cos_sin=True)
            else:
                k = self.k_layernorm(k.view(-1, self.num_kv_heads, self.head_dim))
                q, k, _, _ = self.rotary_emb(positions, q.contiguous(), key=k)

            q = q.view(-1, self.q_size)
            k = k.view(-1, self.k_size)

            if self.enable_kv_rmsnorm_rope_cache:
                self.attn.impl.set_kv_rmsnorm_rope_params(self.k_layernorm.weight, self.k_layernorm.variance_epsilon, cos, sin, self.enable_kv_rmsnorm_rope_cache, self.v_channels)

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
            rope_parameters: dict[str, Any] | None,
            quant_config: QuantizationConfig | None,
        ) -> None:
            is_neox_style = False
            rope_parameters = {"partial_rotary_factor": self.qk_rope_dim / self.head_dim}

            from vllm.model_executor.layers.rotary_embedding import get_rope_wrapper
            self.rotary_emb = get_rope_wrapper(
                self.head_dim,
                max_position=self.max_position_embeddings,
                rotary_dim=self.qk_rope_dim,
                base=config.rope_theta,
                rope_scaling=getattr(config, "rope_scaling", None),
                is_neox_style=is_neox_style,
                num_hidden_layers_cache=config.num_hidden_layers
            )

        def sink_key_interleave(self, weight):
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

            # K & Q 保持一致，都做interleave(奇偶交叉处理)
            param_sink_key = self.sink_key_interleave(param_sink_key) 
            self.attn.update_sink_kv(param_sink_key, self.param_sink_value)
    #####patch end

    openpangu.OpenPanguSinkAttention = OpenPanguSinkAttention


    #####patch start: for pangu72B-VL
    class PanguProMoEV2ForCausalLM(OpenPanguMoEModel):
        pass
    #####patch end


@register_patch("OpenPanguDecoderLayerPatch", OpenPanguDecoderLayer)
class OpenPanguDecoderLayerPatch(VLLMPatch):
    _attr_names_to_apply = ['__init__']

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        vllm_config: VllmConfig,
    ) -> None:
        #####patch start: for pangu72B-VL
        nn.Module.__init__(self)
        #####patch end

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

        #####patch start: for pangu72B-VL
        self.use_sink_attention = (
            hasattr(config, "param_sink_number") and config.param_sink_number > 0
        )
        #####patch end

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

        #####patch start: for pangu72B-VL
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
            self.self_attn = openpangu.OpenPanguSinkAttention(
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
        #####patch end

        else:
            attention_bias = getattr(config, "attention_bias", False) or getattr(
                config, "bias", False
            )
            bias_o_proj = attention_bias
            if hasattr(config, "qkv_bias"):
                attention_bias = config.qkv_bias
            # By default, PanguEmbedded uses causal attention
            # as it is a decoder-only model.
            # You can override the HF config with `is_causal=False` to enable
            # bidirectional attention, which is used in some embedding models
            if getattr(config, "is_causal", True):
                attn_type = AttentionType.DECODER
            else:
                attn_type = AttentionType.ENCODER_ONLY
            self.self_attn = OpenPanguEmbeddedAttention(
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
            self.mlp = OpenPanguMLP(
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
            self.pre_mlp_layernorm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_mlp_layernorm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

@register_patch("OpenPanguModelPatch", OpenPanguModel)
class OpenPanguModelPatch(VLLMPatch):
    _attr_names_to_apply = ['load_weights', 'post_weight_load']

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
                    continue  # skip spec decode layers for main model

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

                #####patch start: for pangu72B-VL
                if name.endswith("e_score_correction_bias"):
                    name = name.replace(
                        "e_score_correction_bias", "gate.e_score_correction_bias"
                    )
                #####patch end

                if name is None:
                    continue

                #####patch start: for pangu72B-VL
                if name not in params_dict:
                    continue
                #####patch end

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        #####patch start: for pangu72B-VL
        self.post_weight_load()
        #####patch end

        return loaded_params

    #####patch start: for pangu72B-VL
    def post_weight_load(self) -> None:
        for name, module in self.named_modules():
            if module is self:
                continue
            if hasattr(module, "post_weight_load"):
                module.post_weight_load()
    #####patch end


@register_patch("OpenPanguMoEModelPatch", OpenPanguMoEModel)
class OpenPanguMoEModelPatch(VLLMPatch):
    _attr_names_to_apply = ['set_eplb_state']


    #####patch start: for pangu72B-VL
    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        for layer_idx, layer in enumerate(self.moe_layers):
            # Register the expert weights.
            self.expert_weights.append(layer.get_expert_weights())
            layer.set_eplb_state(
                moe_layer_idx=layer_idx,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )
    #####patch end