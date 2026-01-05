"""
vLLM Fault Monitoring Module

This module provides decorators methods to capture exceptions in vLLM distributed inference.
When an exception occurs, the decorator reports the exception information to ha_server, then enters an infinite waiting loop.
"""
import functools
import threading
import time
from datetime import datetime

import requests

import torch
import torch_npu
from torch_npu.npu._recovery import check_npu_tensor_is_safe, update_npu_tensor_to_safe

from vllm.logger import init_logger
from omni_npu.v0.token_recovery.common import FailureInfo, WorkerStatus, WORKER_STATUS_DEVICE_RESTARTED
from omni_npu.v0.token_recovery.common import FAULT_URL_FORMAT, WORKER_STATUS_URL_FORMAT, HA_CMD_KEEP_RUN, HA_CMD_RECOVERY
from omni_npu.v0.token_recovery.common import WORKER_STATUS_DEVICE_FORCE_STOPPED
from omni_npu.v0.token_recovery.ha_server import update_ha_msg, get_ha_msg, clear_ha_msg
from omni_npu.v0.token_recovery.envs import ENV

logger = init_logger("vllm")

# Global variables for fault monitoring server IP and port
HA_SERVER_IP = ENV.ha_server_ip
HA_SERVER_PORT = ENV.ha_port

DEVICE_FORCE_STOP_ERROR_CODE = 'FORCE STOP'

MEM_ERROR_UNKOWN = 1
MEM_ERROR_TEMP = 2
MEM_ERROR_DATA = 3

INFERENCE_STEP = 0

# Cmd received from
HA_CMD = None
KEEP_RUN_EVENT = threading.Event()
RUN_OPT = ""

RECOMPUTE = False

def _get_server_address():
    if HA_SERVER_IP is None or HA_SERVER_PORT is None:
        raise ValueError("HA_SERVER_IP and HA_SERVER_PORT can not be None")
    return f"{HA_SERVER_IP}:{HA_SERVER_PORT}"


def report_fault(fault_info):
    if fault_info is None:
        raise ValueError("fault_info can not be None")
    logger.info(f"start report fault info.")
    response = requests.post(
        FAULT_URL_FORMAT.format(_get_server_address()),
        json=fault_info.model_dump(),
        timeout=60
    )
    if response and response.ok:
        logger.info(f"success report fault info: {fault_info}")


def report_status(worker_status):
    if worker_status is None:
        raise ValueError("worker_status can not be None")
    logger.info(f"start report worker status.")
    response = requests.post(
        WORKER_STATUS_URL_FORMAT.format(_get_server_address()),
        json=worker_status.model_dump(),
        timeout=60
    )
    if response and response.ok:
        logger.info(f"success report worker status: {worker_status}")


def _get_target_error_codes():
    return [
        "FORCE STOP",
        "507057",
        "SUSPECT REMOTE ERROR"
    ]


def _make_fault_info(worker, error_code, error_msg):
    if worker is None:
        raise ValueError("worker can not be None")
    failure_info = FailureInfo()
    failure_info.error_code = error_code
    failure_info.error_msg = error_msg
    failure_info.rank = getattr(worker, "rank")
    failure_info.local_rank = getattr(worker, "local_rank")
    failure_info.step_count = INFERENCE_STEP
    return failure_info


def _make_status_info(worker, status, err_code=None, uce_flag=None, uce_blocks=None):
    from omni.adaptors.vllm.token_recovery.ha_worker_server import WORKER_ADDRESS
    worker_status = WorkerStatus()
    worker_status.worker_server_ip = WORKER_ADDRESS
    worker_status.rank = getattr(worker, "rank")
    worker_status.local_rank = getattr(worker, "local_rank")
    worker_status.status = status
    worker_status.err_code = err_code
    worker_status.uce_flag = uce_flag
    worker_status.uce_blocks = uce_blocks
    return worker_status


def _identity(self):
    from omni.adaptors.vllm.token_recovery import ha_worker_server
    return f"rank={ha_worker_server.RANK}"


def _do_local_recovery(self):
    start_time = time.time()
    logger.info(f"{_identity(self)} start _do_local_recovery")
    self.restart_device()
    time.sleep(0.5)
    self.reinit_process_group()
    logger.info(f"{_identity(self)} finish _do_local_recovery, time_used: {time.time() - start_time}")


def _clear_ha_cmd():
    global HA_CMD
    HA_CMD = None

def token_recover_wrapper(func):
    # works on vllm Worker
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not ENV.use_ha:
            return func(self, *args, **kwargs)

        try:
            result = func(self, *args, **kwargs)
            global RECOMPUTE
            RECOMPUTE = False
            return result
        except Exception as origin_exp:
            error_msg = str(origin_exp)
            logger.error(f"{_identity(self)} {error_msg}")
            curr_error_codes = [elm for elm in _get_target_error_codes() if elm in error_msg]
            if not curr_error_codes:
                logger.error(
                    f"{_identity(self)} skip report for error_code not match. {error_msg}")
                raise origin_exp
            curr_error_code = curr_error_codes[0]

            if RECOMPUTE:
                # continuous failures are not supported
                logger.error(f"{_identity(self)} Continuous failures are not supported.")
                raise origin_exp

            # update inference step count
            global INFERENCE_STEP
            INFERENCE_STEP += 1
            try:
                # 1. report fault info
                failure_info = _make_fault_info(self, curr_error_code, error_msg)
                report_fault(failure_info)

                # 2. stop_device
                start = time.time()
                logger.info(f"stop_device in wrapper")
                self.stop_device()
                logger.info(f"finish stop_device in wrapper used:{time.time() - start}")

                # 3. check uce and report worker status FORCE_STOPPED
                uce_flag, uce_blocks = get_uce_blocks(self, curr_error_code)
                worker_status = _make_status_info(self, WORKER_STATUS_DEVICE_FORCE_STOPPED, curr_error_code, uce_flag, uce_blocks)
                report_status(worker_status)

                logger.info(f"{_identity(self)} _waiting_do_recovery")
                # 4. waiting recovery
                global KEEP_RUN_EVENT
                if KEEP_RUN_EVENT.wait(100):
                    logger.info(f"{_identity(self)} _waiting_do_recovery end")
                    KEEP_RUN_EVENT.clear()
                _waiting_do_recovery(self, origin_exp)

                # 5. waiting keep_run
                if KEEP_RUN_EVENT.wait(10):
                    logger.info(f"{_identity(self)} keep_run_loop end")
                    KEEP_RUN_EVENT.clear()
                return _waiting_keep_run(self, origin_exp, uce_blocks)

            except Exception as new_exp:
                # something wrong when process fault, just log and re-raise origin exception
                if new_exp != origin_exp:
                    logger.error(f"{_identity(self)} error process exp in {func.__name__}, error: {new_exp}")
                raise origin_exp
            finally:
                _clear_ha_cmd()

    return wrapper

def _waiting_do_recovery(self, origin_exp):
    global HA_CMD
    if not HA_CMD == HA_CMD_RECOVERY:
        raise origin_exp
    logger.info(f"{_identity(self)} receive do_recovery cmd and start do_local_recovery.")
    # restart_device & reinit_process_group
    _do_local_recovery(self)
    # 3. report worker status DEVICE_RESTARTED
    report_status(_make_status_info(self, WORKER_STATUS_DEVICE_RESTARTED))

def _waiting_keep_run(self, origin_exp, uce_blocks):
    global HA_CMD
    if not HA_CMD == HA_CMD_KEEP_RUN:
        raise origin_exp
    global RECOMPUTE
    if RUN_OPT == "recompute":
        RECOMPUTE = True
    # just return sth after fault process
    if not self.is_driver_worker:
        return {}
    req_list = []
    update_ha_msg({"opt": RUN_OPT, "abort_reqs": req_list})
    return {"opt": RUN_OPT, "abort_reqs": req_list}

def get_uce_blocks(self, curr_error_code):
    if curr_error_code != "UCE ERROR":
        return False, []
    # occur UCE ERROR
    logger.info(f"{_identity(self)} find uce error, do more check")
    error_type = torch.npu.check_uce_in_memory(self.local_rank)
    logger.info(f"{_identity(self)} mem uce error_type {error_type}")
    if error_type == MEM_ERROR_UNKOWN:
        return True, [-1]
    if error_type == MEM_ERROR_TEMP:
        return False, []
    uce_block_set = set()
    logger.info(f"{_identity(self)}  error_type is MEM_ERROR_DATA, need do more check")
    # 检查是否是kv cache发生UCE ERROR
    kv = self.gpu_cache[0]
    for layer in range(len(kv)):
        target = kv[layer]['kv_cache']
        is_safe = check_npu_tensor_is_safe(target)
        logger.info(f"{_identity(self)} target shape layer:{layer} block: is_safe:{is_safe}")
        if is_safe:
            continue
        uce_addr = torch_npu.npu._get_uce_addr()
        logger.info(f"{_identity(self)} target shape layer:{layer} uce addr {uce_addr}")
        update_npu_tensor_to_safe(target)
        for uce_info in uce_addr:
            ptr = uce_info.get("ptr")
            size = uce_info.get("size")
            if ptr and size:
                logger.info(f"start ptr {ptr} end ptr  {ptr + size}")
                for i in range(len(target)):
                    block_ptr = target[i].data_ptr()
                    if block_ptr >= ptr and block_ptr <= ptr + size:
                        uce_block_set.add(i)
    if len(uce_block_set) > 0:
        return True, list(uce_block_set) # kv cache occur uce error
    return True, [-1] # weight occur uce error


def asnyc_llm_engine_execute_model_wrapper(func):
    # works on asnyc_llm_engine.execute_model()
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        output = await func(self, *args, **kwargs)
        msg = get_ha_msg()
        if msg:
            logger.info(f"generate token failed msg {msg}, jump to next step.")
            strategy = msg.get("opt")
            req_list = msg.get("abort_reqs")
            if strategy == "exit":
                raise Exception("exit process")
            if strategy == "abort":
                for req in req_list:
                    self.abort_request(req)
            clear_ha_msg()
        return output

    return wrapper
