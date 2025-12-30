# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import pytest
import tempfile
import traceback
import importlib
import socket
from typing import Callable, Any, List, Tuple, Dict
import torch.multiprocessing as mp
import torch
import unittest
from unittest.mock import Mock, patch

TEST_SEED = 0

def parse_ascend_devices():
    # Get the environment variable, default to empty string if not found
    env_val = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')
    
    if not env_val.strip():
        # Handle case where env var is missing or empty
        return 0, [0, 1]
    try:
        # Split by comma and convert to integers
        visible_die_list = [int(x.strip()) for x in env_val.split(',') if x.strip()]
        first_die_no = visible_die_list[0] if visible_die_list else None
    except ValueError as e:
        print(f"Error parsing ASCEND_RT_VISIBLE_DEVICES: {e}, using default values.")
        return 0, [0, 1]

    return first_die_no, visible_die_list

def _persistent_worker_loop(device: int, rank: int, world_size: int, temp_file_path: str, 
                            task_queue: mp.Queue, result_queue: mp.Queue, master_port: int,
                            layer_parallel_config: Dict):
    try:
        # 1. Apply Patches Immediately
        # from omni_npu.adaptors.vllm.patches.model_patch import patch_all
        # patch_all()

        # 2. Set Configuration BEFORE loading/reloading layers
        # from omni_npu.models.config_loader.loader import model_extra_config
        # model_extra_config.parall_config.dense_mlp_tp_size = world_size
        # model_extra_config.parall_config.o_proj_tp_size = world_size

        # 3. CRITICAL: Reload the layer module
        # import omni_npu.layers.linear
        # importlib.reload(omni_npu.layers.linear)

        # 4. Setup Distributed Environment
        from vllm import distributed as vllm_dist
        from vllm.utils.system_utils import update_environment_variables

        from omni_npu.v1.distributed.parallel_state_ext import ensure_layer_parallel_initialized
        
        torch.npu.set_device(device)
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
        update_environment_variables({
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port), # <--- Use the dynamic port
        })

        vllm_dist.init_distributed_environment(
            distributed_init_method=f"file://{temp_file_path}",
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            backend="hccl",
        )
        
        vllm_dist.initialize_model_parallel(tensor_model_parallel_size=world_size)

        vllm_config = Mock()
        vllm_config.parallel_config = Mock(local_rank=rank)

        with patch(
            "omni_npu.v1.distributed.parallel_state_ext._load_layer_parallel_config_from_vllm",
            return_value=(vllm_config, layer_parallel_config),
        ):
            ensure_layer_parallel_initialized(backend="hccl")
        
        # Verify TP Size
        current_tp = vllm_dist.parallel_state.get_tensor_model_parallel_world_size()
        if current_tp != world_size:
            raise RuntimeError(f"Distributed Init Failed: Expected TP={world_size}, got {current_tp}")

        # 5. Signal Ready
        result_queue.put("READY")

        # 6. Task Loop
        while True:
            task = task_queue.get()
            if task is None: break
            
            func, args, kwargs = task
            
            try:
                torch.manual_seed(TEST_SEED)
                func(device, rank, world_size, *args, **kwargs)
                result_queue.put(None) 
            except Exception:
                tb = traceback.format_exc()
                result_queue.put(RuntimeError(f"Rank {rank} failed:\n{tb}"))
                
    except Exception:
        tb = traceback.format_exc()
        result_queue.put(RuntimeError(f"Worker Startup Failed Rank {rank}:\n{tb}"))
    finally:
        try:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except:
            pass

@pytest.fixture(scope="function")
def distributed_worker_pool():
    world_size = 2
    ctx = mp.get_context('spawn') # Use spawn context

    def run_task(func, *args, config=None, **kwargs):

        task_queues = [ctx.Queue() for _ in range(world_size)]
        result_queues = [ctx.Queue() for _ in range(world_size)]
        
        with tempfile.NamedTemporaryFile(delete=False) as tfile:
            temp_file_path = tfile.name

        # Find a free port on the host
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            master_port = s.getsockname()[1]

        layer_parallel_config = config
        processes = []
        first_die_no, visible_die_list = parse_ascend_devices()
        assert first_die_no is not None, "ASCEND_RT_VISIBLE_DEVICES is not set or empty."
        assert len(visible_die_list) >= world_size, "Not enough visible devices for the requested world size." 
        for rank in range(world_size):
            device = visible_die_list[rank]
            p = ctx.Process(
                target=_persistent_worker_loop,
                # Pass master_port to the worker
                args=(device, rank, world_size, temp_file_path, task_queues[rank], result_queues[rank], master_port,
                      layer_parallel_config)
            )
            p.start()
            processes.append(p)

        try:
            for i, q in enumerate(result_queues):
                res = q.get(timeout=30)
                if res != "READY":
                    raise RuntimeError(f"Worker {i} failed to start: {res}")
        except Exception as e:
            for p in processes: p.terminate()
            raise e

        for q in task_queues:
            q.put((func, args, kwargs))
        
        errors = []
        for q in result_queues:
            res = q.get()
            if res is not None:
                errors.append(res)
        
        for q in task_queues: q.put(None)
        for p in processes: 
            p.join(timeout=5)
            if p.is_alive(): p.terminate()
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        if errors:
            raise errors[0]

    yield run_task