# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import math

def get_p_start_rank(p_tp_size, p_dp_size, d_tp_size, d_dp_size, d_node_num, cur_d_node, cur_d_rank):
    # only support full tp in prefill.
    if p_dp_size != 1:
        raise ValueError('p_dp_size must be 1')

    # Parameter validation
    if p_tp_size <= 0 or d_tp_size <= 0 or d_dp_size <= 0 or d_node_num <= 0:
        raise ValueError('p_tp_size, d_tp_size, d_dp_size, d_node_num must be positive')

    if cur_d_node < 0 or cur_d_node >= d_node_num:
        raise ValueError('cur_d_node < 0 or cur_d_node >= d_node_num')

    if cur_d_rank < 0:
        raise ValueError('cur_d_rank < 0')

    # Calculate device information
    devices_per_node = d_dp_size * d_tp_size
    if cur_d_rank >= devices_per_node:
        raise ValueError('cur_d_rank >= devices_per_node')

    # Calculate KV group information
    kv_group_size = d_tp_size
    if p_tp_size % kv_group_size != 0:
        raise ValueError('p_tp_size % kv_group_size != 0')

    # Calculate current device's position in DP group
    cur_d_dp = cur_d_rank // d_tp_size
    cur_d_tp = cur_d_rank % d_tp_size

    # as same as get_p_start_rank_v1 in deepseek.
    global_dp_group = cur_d_node + cur_d_dp * d_node_num
    total_dp_groups = d_node_num * d_dp_size

    # Calculate prefill information
    p_replica_groups = p_tp_size // kv_group_size
    total_kv_groups = p_dp_size * p_replica_groups

    # Calculate connection step size (ensure KV group are evenly distrubuted)
    link_group_step = max(1, math.ceil(total_kv_groups / total_dp_groups))
    kv_group_index = (global_dp_group * link_group_step) % total_kv_groups

    # Calculate starting prefill rank for KV group
    p_dp_index = kv_group_index // p_replica_groups
    replica_index = kv_group_index % p_replica_groups

    # Calculate the prefill rank for current decode device
    stride = p_tp_size // kv_group_size
    offset = replica_index + cur_d_tp * stride
    return p_dp_index * p_tp_size + offset


def get_config_from_dict_or_env(config, config_var_name, env_var_name, default_value, value_type):
    env_value = os.environ.get(env_var_name, None)
    if isinstance(config, dict):
        args_value = config.get(config_var_name, None)
    else:
        args_value = getattr(config, config_var_name, None)
    if env_value is None and args_value is None:
        if default_value is None:
            raise ValueError(f"ENV {env_var_name} or args {config_var_name} should not be None.")
        else:
            value = default_value
    # ENV first
    elif env_value is not None:
        value = env_value
    else:
        value = args_value
    return value_type(value)