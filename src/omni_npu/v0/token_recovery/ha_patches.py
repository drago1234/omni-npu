import functools
import threading
import time
from vllm.logger import init_logger
from omni_npu.v0.token_recovery.envs import ENV
from omni_npu.v0.token_recovery import ha_monitor
import torch
import torch_npu
from vllm import utils

logger = init_logger("vllm")

UPDATE_FAILURE_SERVER_INFO = "update_failure_server_info"
GET_WORKER_SERVER_ADDRESS = "get_worker_server_address"
STOP_DEVICE = "stop_device"
RESTART_DEVICE = "restart_device"
REINIT_PROCESS_GROUP = "reinit_process_group"
UPDATE_HA_CMD = "update_ha_cmd"

HTTP_MAX_PORT = 65535
ProcessGroups = []
lock = threading.Lock()

# register this method to every RayWorkerWrapper
def update_failure_server_info(self, ip, port):
    ha_monitor.HA_SERVER_IP = ip
    ha_monitor.HA_SERVER_PORT = port
    logger.info(f"local_rank={self.local_rank}, success receive server port {port}.")
    return self.local_rank


def get_worker_server_address(self):
    server_ip = utils.get_ip()
    server_port = int(ENV.ha_port) + 1 + self.local_rank
    if server_port > HTTP_MAX_PORT:
        raise ValueError(f"server_port must be in 0-{HTTP_MAX_PORT}.")
    return f"{server_ip}:{server_port}"

def register_process_group(process_group):
    ProcessGroups.append(process_group)

def stop_device(self):
    with lock:
        start_time = time.time()
        logger.info(f"local_rank={self.local_rank}, start stop_device.")
        torch_npu.npu.stop_device(self.local_rank)
        logger.info(f"local_rank={self.local_rank}, finish stop_device, time_used: {time.time() - start_time}.")


def restart_device(self, rebuild_all_resources=False):
    start_time = time.time()
    logger.info(f"local_rank={self.local_rank}, start restart_device.")
    torch_npu.npu.restart_device(self.local_rank, rebuild_all_resources=rebuild_all_resources)
    logger.info(f"local_rank={self.local_rank}, finish restart_device, time_used: {time.time() - start_time}.")


def reinit_process_group(self):
    start_time = time.time()
    logger.info(f"local_rank={self.local_rank}, start reinit_process_group.")
    torch.distributed.reinit_process_group(group=None, rebuild_link=False)
    for process_group in ProcessGroups:
        torch.distributed.reinit_process_group(process_group, False)
    logger.info(f"local_rank={self.local_rank}, finish reinit_process_group, time_used: {time.time() - start_time}.")

def is_token_recompute(outputs) -> bool:
    strategy = outputs.get("opt", "")
    if strategy == "recompute":
        return True
    else:
        raise Exception("token recover failed, exit process")

def update_ha_cmd(self, cmd):
    start_time = time.time()
    ha_monitor.HA_CMD = cmd
    logger.info(f"local_rank={self.local_rank}, finish update_ha_cmd to {cmd}, time_used: {time.time() - start_time}.")


def start_ha_server_for_dp(port):
    from omni.adaptors.vllm.token_recovery.ha_server import start_server
    start_server(port)

def start_ha_cmd_wrapper(func):
    # works on FxWorker.init_device
    @functools.wraps(func)
    def wrapper(cls, *args, **kwargs):
        res = func(cls, *args, **kwargs)
        from omni.adaptors.vllm.token_recovery.ha_worker_server import start_server
        start_server(cls)
        return res

    return wrapper