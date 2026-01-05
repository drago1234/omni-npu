"""
vLLM 故障监控服务器
提供一个轻量级HTTP服务器用于收集故障信息
"""


import logging
import threading
import time
import queue
import uvicorn

from datetime import datetime
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from vllm.logger import init_logger

from omni_npu.v0.token_recovery.common import (
    FaultStatics,
    FailureInfo,
    WorkerStatus,
)
from omni_npu.v0.token_recovery.ha_processor import HAProcessor

logger = init_logger("vllm")

# sharded fault msg queue
FAULT_QUEUE = queue.Queue()
FAULT_STATICS = FaultStatics()
WORKER_STATUS = {}

# token recover processor
HA_PROCESSOR = None

# fault report server
SERVER_THREAD = None
PROCESSOR_THREAD = None
HA_MASTER_MSG = {}

def get_ha_msg():
    return HA_MASTER_MSG


def update_ha_msg(msg):
    HA_MASTER_MSG.update(msg)


def clear_ha_msg():
    HA_MASTER_MSG.clear()

app = FastAPI()

@app.post('/failures')
async def report_failure(request: Request) -> JSONResponse:
    failure_data = await request.json()
    logger.info(f"receive worker failure report: {failure_data}")
    if failure_data is None or (failure_data is not None and not isinstance(failure_data, dict)):
        raise ValueError("failure_data should be dict")
    failure_info = FailureInfo.model_validate(failure_data)
    failure_info.timestamp = time.time()
    global FAULT_STATICS
    FAULT_STATICS.update_by_fault(failure_info)
    return JSONResponse({"status": "success"})


@app.post('/worker_status')
async def worker_status(request: Request) -> JSONResponse:
    status_data = await request.json()
    logger.info(f"receive worker status report: {status_data}")
    if status_data is None or (status_data is not None and not isinstance(status_data, dict)):
        raise ValueError("status_data should be dict")
    worker_info = WorkerStatus.model_validate(status_data)
    worker_info.timestamp = time.time()
    if worker_info.rank is None:
        raise ValueError("worker_info's rank should not be None")
    WORKER_STATUS[worker_info.worker_server_ip] = worker_info
    return JSONResponse({"status": "success"})


@app.post('/report_worker_server_info')
async def report_worker_server_info(request: Request) -> JSONResponse:
    status_data = await request.json()
    logger.info(f"receive worker server info report: {status_data}")
    if status_data is None or (status_data is not None and not isinstance(status_data, dict)):
        raise ValueError("status_data should be dict")
    HA_PROCESSOR.worker_server_list.append(status_data["address"])
    return JSONResponse({"status": "success"})


def start_failure_server(port=4999):
    uvicorn.run(app, host="0.0.0.0", port=port)


def start_server(port=4999):
    global HA_PROCESSOR, PROCESSOR_THREAD, FAULT_STATICS

    HA_PROCESSOR = HAProcessor(FAULT_QUEUE, FAULT_STATICS, WORKER_STATUS)
    PROCESSOR_THREAD = HA_PROCESSOR.start()

    global SERVER_THREAD
    SERVER_THREAD = threading.Thread(target=start_failure_server, args=(port,))
    SERVER_THREAD.daemon = True
    SERVER_THREAD.start()
    logger.info(f"success start ha server at port: {port}")