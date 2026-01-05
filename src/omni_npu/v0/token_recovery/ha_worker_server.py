"""
vLLM worker server
"""
import signal
import threading
import time
import uvicorn
import requests

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from vllm.logger import init_logger
from vllm.distributed import get_world_group
from omni_npu.v0.token_recovery.envs import ENV

logger = init_logger("vllm")

# actor
CLS = None
# npu rank
RANK = None
# worker server thread
SERVER_THREAD = None
# worker server addr
WORKER_ADDRESS = None

app = FastAPI()

@app.post('/update_ha_cmd')
async def update_ha_cmd(request: Request) -> JSONResponse:
    data = await request.json()
    logger.info(f"rank:{RANK} receive update_ha_cmd report: {data}")
    if data is None or (data is not None and not isinstance(data, dict)):
        raise ValueError("data should be dict")
    start_time = time.time()
    from omni_npu.v0.token_recovery import ha_monitor
    cmd = data.get("cmd")
    ha_monitor.HA_CMD = cmd
    ha_monitor.RUN_OPT = data.get("opt")
    ha_monitor.KEEP_RUN_EVENT.set()
    logger.info(f"rank={RANK}, finish update_ha_cmd to {cmd}, time_used: {time.time() - start_time}")
    return JSONResponse({"status": f"success, rank:{RANK}"})


@app.post('/stop_device')
async def stop_device(request: Request) -> JSONResponse:
    start_time = time.time()
    CLS.stop_device()
    logger.info(f"rank={RANK}, finish stop_device, time_used: {time.time() - start_time}")
    return JSONResponse({"status": f"stop_device, rank:{RANK}"})

def stop_device_signal_handler(sigum, frame):
    if sigum == signal.SIGUSR1:
        logger.info("receive stop sign and start to stop device")
        CLS.stop_device()

def _start_worker_server(port):
    uvicorn.run(app, host="0.0.0.0", port=port)

def start_server(cls):
    global CLS
    CLS = cls
    rank = get_world_group().rank

    from vllm import utils
    server_ip = utils.get_ip()

    port = ENV.ha_port + 1 + rank
    global RANK
    RANK = rank
    global SERVER_THREAD
    SERVER_THREAD = threading.Thread(target=_start_worker_server, args=(port,))
    SERVER_THREAD.daemon = True
    SERVER_THREAD.start()
    logger.info(f"success start ha cmd server at port: {port}")

    global WORKER_ADDRESS
    WORKER_ADDRESS = f"{server_ip}:{port}"

    signal.signal(signal.SIGUSR1, stop_device_signal_handler)

    try:
        # update cmd address
        if ENV.ha_server_ip and ENV.ha_port:
            url = f"http://{ENV.ha_server_ip}:{ENV.ha_port}/report_worker_server_info"
            response = requests.post(url, json={"address": WORKER_ADDRESS}, timeout=2)
            if response and response.ok:
                logger.info(f"success report_worker_server_info, res: {response.text}")
    except Exception as e:
        logger.exception("update cmd address error", str(e))