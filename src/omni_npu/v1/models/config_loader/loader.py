# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass, field, fields, asdict
from typing import Any
import json
import os
import torch
import torch_npu

# import logging
from vllm.logger import init_logger

from .features import apply_eager_mode_config, apply_omni_cache

logger = init_logger(__name__)

default_config_path = os.path.normpath(os.path.join(os.path.abspath(__file__), '../../configs'))


def load_model_extra_config(model_config, vllm_config, scheduler_config):
    model_name, quant_type= parse_hf_config(model_config.hf_config)
    is_pd_disaggregation = False
    is_prefill_node = None
    if os.getenv('ROLE', None):
        is_pd_disaggregation = True
        is_prefill_node = True if os.getenv('ROLE', None)=='prefill' else False
    if vllm_config.additional_config is not None:
        enable_pd_elastic_scaling = vllm_config.additional_config.get("enable_pd_elastic_scaling", False)
        enable_low_latency=vllm_config.additional_config.get("enable_low_latency", False)
        enable_omni_cache = vllm_config.additional_config.get("enable_omni_cache", False)
    else:
        enable_pd_elastic_scaling = False
        enable_low_latency = False
        enable_omni_cache = False

    if model_config.enforce_eager:
        graph_mode = 'eager_mode'
    elif vllm_config.npu_compilation_config.use_gegraph:
        graph_mode = 'ge_graph'
    else:
        graph_mode = 'acl_graph'

    enable_chunked_prefill = scheduler_config.enable_chunked_prefill
    enable_eplb=vllm_config.parallel_config.enable_eplb
    
    device_name = torch_npu.npu.get_device_name(0)

    if device_name.startswith("Ascend910B"):
        hardware_platform = "A2"
    elif device_name.startswith("Ascend910"):
        hardware_platform = "A3"
    else:
        raise ValueError(f"Unsupported device: {device_name}. Only Ascend910/Ascend910B are supported.")
    
    update_task_config(
        model_name = model_name,
        hardware_platform = hardware_platform,
        is_pd_disaggregation = is_pd_disaggregation,
        is_prefill_node = is_prefill_node,
        quant_type = quant_type,
        prefill_node_num = int(os.getenv("PREFILL_POD_NUM", 1)),
        decode_node_num = int(os.getenv("DECODE_POD_NUM", 1)),
        enable_eplb = enable_eplb,
        enable_chunked_prefill = enable_chunked_prefill,
        enable_low_latency = enable_low_latency,
        graph_mode = graph_mode,
        enable_pd_elastic_scaling = enable_pd_elastic_scaling,
        enable_omni_cache = enable_omni_cache
    )
    _validate_config(vllm_config.additional_config)
    _print_model_config()

@dataclass
class TaskConfig:
    model_name: str = "deepseek_v3"
    hardware_platform: str = "A3"
    is_pd_disaggregation: bool = True
    is_prefill_node: bool = True # decode_node when it's False
    quant_type: str = "w8a8c16"
    prefill_node_num: int = 0
    decode_node_num: int = 0
    enable_eplb: bool = False
    enable_pd_elastic_scaling: bool = False # 是否支持动态扩缩容，开启时使用默认一套配置
    enable_chunked_prefill: bool = False
    graph_mode: str = "eager_mode"
    enable_low_latency: bool = False
    enable_omni_cache: bool = False


@dataclass
class ModelParallelConfig:
    dense_mlp_tp_size: int = 1
    o_proj_tp_size: int = 1
    attn_sp_size: int = 1
    redundancy_shared_expert_num: int = 0
    attn_dies: int = 0
    enable_share_expert_tp: bool = False
    eh_proj_tp_size: int = 1
    layer_parallel_config: dict[str, Any] = field(default_factory=dict)
    input_split: bool = False

 
@dataclass
class ModelOperatorOptConfig:
    enable_kv_rmsnorm_rope_cache: bool = True
    prefill_moe_all_to_all: bool = True
    moe_multi_stream_tune: bool = False
    best_ep: bool = False
    merge_qkv: bool = False
    two_stage_comm: bool = False
    gmm_nz: bool = False
    unquant_bmm_nz: bool = False
    decode_moe_dispatch_combine: bool = True
    decode_flash_comm_1: bool = True # decode节点开启FlashComm1优化
    enable_super_kernel: bool = False
    enable_prefill_micro_batch: bool = False
    enable_mlaprolog: bool = False
    cast_w2_scale_f32: bool = False
    control_accept_rate: float = -1 # <0 or >1 不控制, >=0 and <=1 控制MTP开启时接受率为该值，几乎必然导致输出结果异常，仅保证只投机1个token时满足这一数值
    mla_multistream_limit_core: str = '' # 空字符串代表不开启多流分核，形如'20|36'代表主流分配的AIC和AIV核数分别为20和36
    shared_experts_to_gmm: bool = False # 当redundancy_shared_expert_num > 0时，共享专家使用GMM代替BMM进行计算（限定收益场景：EP288 + 单die bs >= 48，仅针对Decode阶段）
    enable_gmm_swiglu_quant: bool = False # 当redundancy_shared_expert_num > 0时，使用npu_grouped_matmul_swiglu_quant_v2融合算子
    mtp_remove_redundant_kv: bool = False # MTP场景下，去除FIA算子对同一请求的冗余KV cache搬运，当前不支持与Omni Attention同时使用
    enable_prefetch: bool = True # 是否开启预取
    expert_gate_up_prefetch: int = 50 # 默认预取大小为 50Mb；如果是权重是BF16型，设置为 30Mb
    expert_down_prefetch: int = 28 # 当权重是w8a8且ep_size > 64 时，默认预取大小为 28Mb，否则为0
    dense_mlp_prefetch: int = 56 # 默认预取大小为 56Mb
    lm_head_prefetch: int = 135 # 默认预取大小为 135Mb
    attn_prefetch: int = 96 # 默认预取大小为 96Mb
    shared_expert_gate_up_prefetch: int = 28
    shared_expert_down_prefetch: int = 14

    enable_round_pipeline_comm: bool = False
    enable_pipeline_comm: bool = False
    prefill_enable_long_seq: bool = False
    prefill_enable_mla_alltoall: bool = False
    prefill_enable_mla_alltoall_local: bool = False
    fa_quant: bool = False
    use_omni_cache: bool = False
    enable_dsa: bool=True


    def __post_init__(self):

        # Check the dependencies of use_prefetch and prefetch_Mb
        if not self.enable_prefetch:
            self.expert_gate_up_prefetch = 0
            self.expert_down_prefetch = 0
            self.attn_prefetch = 0
            self.dense_mlp_prefetch = 0
            self.lm_head_prefetch = 0
            self.shared_expert_gate_up_prefetch = 0
            self.shared_expert_down_prefetch = 0
            logger.warning(f"[WARNING] When enable_prefetch is false, prefetch_Mb must be set to 0.")

        if os.getenv("ENABLE_OMNI_CACHE", "0") == "1":
            self.use_omni_cache = True

        # Check for mutually exclusive configuration options
        if self.enable_pipeline_comm and \
                self.enable_round_pipeline_comm:
            raise ValueError(
                "Conflicting communication configuration: "
                "'enable_pipeline_comm' and 'enable_round_pipeline_comm' cannot both be True. "
                "Please disable one of these communication modes."
            )
        
        if self.unquant_bmm_nz:
            # if use weight nz, this config must be True
            torch.npu.config.allow_internal_format = True

@dataclass 
class ModelExtraConfig:
    parall_config: ModelParallelConfig = field(default_factory = ModelParallelConfig)
    operator_opt_config: ModelOperatorOptConfig = field(default_factory = ModelOperatorOptConfig)
    task_config: TaskConfig = field(default_factory = TaskConfig)


model_extra_config = ModelExtraConfig()

def filter_dict_by_dataclass(dataclass_type, data_dict):
    valid_keys = {f.name for f in fields(dataclass_type)}
    return {k: v for k, v in data_dict.items() if k in valid_keys}

def update_task_config(**kwargs):
    global model_extra_config
    task_config = model_extra_config.task_config
    if task_config is None:
        task_config = TaskConfig()
    if kwargs:
        for key, value in kwargs.items():
            if hasattr(task_config, key):
                setattr(task_config, key, value)

    _init_model_extra_config(task_config)


def parse_hf_config(hf_config):
    
    vars_hf_config = vars(hf_config)

    matches = []
    match_hf_configs_path = os.path.join(default_config_path,'match_hf_configs.json')

    match_hf_configs_data = _loader_configs_data(match_hf_configs_path)

    for model_name, model_params in match_hf_configs_data.items():
        # Check if all extracted_params match model parameters
        is_match = True
        for key, value in model_params.items():
            # If model doesn't have this parameter or parameter values don't match
            if key not in vars_hf_config or vars_hf_config[key] != value:
                is_match = False
                break
        
        if is_match:
            matches.append(model_name)

    # Check matching results
    if len(matches) == 0:
        model_name = hf_config.model_type
    elif len(matches) > 1:
        if hf_config.model_type == "deepseek_v3":
            model_name = "deepseek_v3" 
        elif hf_config.model_type == "deepseek_v32": 
            model_name = "deepseek_v32"
        else:
            raise RuntimeError(f"[ERROR] Multiple matching model names found: {matches}. Unable to determine the correct model name.")
    else:
        model_name = matches[0]

    if hasattr(hf_config, "quantization_config") and hf_config.quantization_config['format'].strip() == 'int-quantized':
        weights_type = hf_config.quantization_config["config_groups"]["group_0"]["weights"]["num_bits"]
        if isinstance(weights_type, dict):
            num_bits_values = weights_type.values()
            weights_type = f"{min(num_bits_values)}"

        input_activations_type = hf_config.quantization_config["config_groups"]["group_0"]["input_activations"]["num_bits"]
        if isinstance(input_activations_type, dict):
            num_bits_values = input_activations_type.values()
            input_activations_type = f"{min(num_bits_values)}"
        
        kv_cache_scheme_type = hf_config.quantization_config["kv_cache_scheme"]
        quant_type = f"w{weights_type}a{input_activations_type}"
        if kv_cache_scheme_type == "Opti-C8":
            quant_type = quant_type+"_fa_c8"
        elif isinstance(kv_cache_scheme_type, dict):
            num_bits_values = kv_cache_scheme_type["num_bits"]
            quant_type = f"{quant_type}c{num_bits_values}"
        else:
            quant_type = f"{quant_type}c16"
    else:
        quant_type = "bf16"

    return model_name, quant_type

def _init_model_extra_config(task_config):

    config_data = _get_best_practice_config(task_config)

    setattr(model_extra_config, 'task_config', task_config)

    if config_data:

        parall_config = ModelParallelConfig(**filter_dict_by_dataclass(ModelParallelConfig,config_data['model_parallel_config']))
        operator_opt_config = ModelOperatorOptConfig(**filter_dict_by_dataclass(ModelOperatorOptConfig, config_data['operator_optimization_config']))

        setattr(model_extra_config, 'parall_config', parall_config)
        setattr(model_extra_config, 'operator_opt_config', operator_opt_config)

    else:
        # Set default configs if no config data found
        setattr(model_extra_config, 'parall_config', ModelParallelConfig())
        setattr(model_extra_config, 'operator_opt_config', ModelOperatorOptConfig())



def _loader_configs_data(file_path):
    try:
        with open(file_path, 'r') as f:
            configs_data = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"[ERROR] Invalid JSON format in config file: {e}")
    except KeyError as e:
        raise RuntimeError(f"[ERROR] Missing required key in config data: {e}")
    except TypeError as e:
        raise RuntimeError(f"[ERROR] Config structure mismatch or incorrect field types: {e}")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Unexpected error while loading model extra config: {e}")

    return configs_data


def _get_best_practice_config(task_config):
    
    performance_mode = "low_latency" if task_config.enable_low_latency else "high_throughout"

    configs_data = _loader_configs_data(os.path.join(default_config_path,f'{performance_mode}/best_practice_configs.json'))

    configs_list = None
    for data in configs_data:
        if data["model"] == task_config.model_name and \
            data["hardware"] == task_config.hardware_platform and \
                data["precision"] == task_config.quant_type:
                    configs_list = data["configs"]
                    break

    if task_config.is_pd_disaggregation and not task_config.enable_pd_elastic_scaling:
        pd_scheme = f'{task_config.prefill_node_num}P{task_config.decode_node_num}D'
    elif task_config.is_pd_disaggregation and task_config.enable_pd_elastic_scaling:
        pd_scheme = "pd_elastic_scaling"
    else:
        pd_scheme = 'hybrid'

    task_info = f'{task_config.model_name}_{task_config.quant_type}_{task_config.hardware_platform}_{pd_scheme}'
    
    if not configs_list:
        logger.warning(
            f"The configuration for {task_config.model_name}_{task_config.quant_type} "
            f"on {task_config.hardware_platform} with performance mode '{performance_mode}' "
            f"was not found in best_practice_configs.json. Loading default configuration."
        )
        return None
    else:
        files_data = configs_list.get(pd_scheme, None)
        if not files_data:
            logger.warning(
                f"The configuration for {task_info} with performance mode '{performance_mode}' "
                f"was not found in best_practice_configs.json. Loading default configuration."
            )
            return None

        if task_config.is_pd_disaggregation and task_config.is_prefill_node:
            model_config_file_path = files_data.get("prefill_config_file")
        elif task_config.is_pd_disaggregation and not task_config.is_prefill_node:
            model_config_file_path = files_data.get("decode_config_file")
        else:
            model_config_file_path = files_data.get("config_file")

        best_practice_model_config_path = os.path.join(default_config_path, f'{performance_mode}/{model_config_file_path}')

        if not os.path.exists(best_practice_model_config_path):
            raise RuntimeError(f"[ERROR] Task {task_info} requires configuration file {best_practice_model_config_path}, but not found.")
        else:
            logger.info(
                f"The task about {task_info} load configuration file from {best_practice_model_config_path}")
            config_data = _loader_configs_data(best_practice_model_config_path)
        
        return config_data


def _validate_config(additional_config):
    global model_extra_config
    apply_eager_mode_config(model_extra_config)
    apply_omni_cache(additional_config)

def _print_model_config():
    try:
        model_info = json.dumps(asdict(model_extra_config), indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        model_info = repr(model_extra_config)
        logger.warning(f"Failed to JSON-serialize model_extra_config: {e}")
    logger.info(f"ModelExtraConfig: {model_info}")
    


    





