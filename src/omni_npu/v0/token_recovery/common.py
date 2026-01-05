"""
vLLM故障监控系统的共享组件
"""
from typing import Optional

from pydantic import BaseModel

# server info
FAULT_URL_FORMAT = "http://{}/failures"
WORKER_STATUS_URL_FORMAT = "http://{}/worker_status"

# worker status info
WORKER_STATUS_DEVICE_AWAKE = "DEVICE_AWAKE"
WORKER_STATUS_DEVICE_FORCE_STOPPED = "DEVICE_FORCE_STOPPED"
WORKER_STATUS_DEVICE_RESTARTED = "DEVICE_RESTARTED"
WORKER_STATUS_DEVICE_READY = "DEVICE_READY"

# cmd for workers
HA_CMD_KEEP_RUN = "keep_run"
HA_CMD_EXIT_RETRY = "exit_retry"
HA_CMD_RECOVERY = "do_recovery"

# waiting time
WORKER_WAITING_STOP_TIME = 100
WORKER_WAITING_RECOVERY_TIME = 10


# 定义故障信息模型
class FailureInfo(BaseModel):
    error_code: Optional[str] = None
    error_msg: Optional[str] = None
    actor_name: Optional[str] = None
    rank: Optional[int] = None
    local_rank: Optional[int] = None
    node_ip: Optional[str] = None
    step_count: Optional[int] = None
    timestamp: Optional[float] = None


class WorkerStatus(BaseModel):
    worker_server_ip: Optional[str] = None
    rank: Optional[int] = None
    local_rank: Optional[int] = None
    status: Optional[str] = None
    err_code: Optional[str] = None
    uce_flag: Optional[bool] = None
    uce_blocks: Optional[list] = None
    timestamp: Optional[float] = None
    call_back_url: Optional[str] = None


class  FaultStatics:
    def __init__(self):
        self.largest_fault_step = -1
        self.largest_processed_step = -1

    def update_by_fault(self, fault_info: FailureInfo):
        if fault_info and fault_info.step_count and fault_info.step_count > self.largest_fault_step:
            self.largest_fault_step = fault_info.step_count

    def update_largest_processed_step(self, step_count):
        self.largest_processed_step = step_count

    def update_step(self, largest_fault_step, largest_processed_step):
        self.largest_fault_step = largest_fault_step
        self.largest_processed_step = largest_processed_step

    def is_first_process(self):
        return self.largest_processed_step == 0

    def has_new_step_fault(self):
        return self.largest_fault_step - self.largest_processed_step >= 1

    def __str__(self):
        return f"largest_fault_step: {self.largest_fault_step}, largest_processed_step: {self.largest_processed_step}"