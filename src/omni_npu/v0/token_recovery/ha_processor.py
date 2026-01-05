"""
vLLM高可用处理器

负责处理收集到的故障信息，可以执行日志记录、通知发送等操作
"""

import logging
import threading
import time
import asyncio
import aiohttp

from datetime import datetime

from vllm.logger import init_logger

from omni_npu.v0.token_recovery.common import (
    FaultStatics,
    HA_CMD_KEEP_RUN,
    HA_CMD_RECOVERY,
    HA_CMD_EXIT_RETRY,
    WORKER_STATUS_DEVICE_FORCE_STOPPED,
    HA_CMD_EXIT_RETRY,
    WORKER_WAITING_STOP_TIME,
    WORKER_WAITING_RECOVERY_TIME, WORKER_STATUS_DEVICE_RESTARTED,
)

logger = init_logger("vllm")

processing_active = True

ERR_CODES_CATEGORY = {"507057", "FORCE STOP", "SUSPECT REMOTE ERROR"}

def safe_get(obj, attr_name):
    if obj and hasattr(obj, attr_name):
        return getattr(obj, attr_name)
    logger.info(f"failed get attr {attr_name} from object {obj}")
    return None


class HAProcessor:
    """高可用处理器类，负责处理故障信息"""

    def __init__(self, failure_queue, fault_statics: FaultStatics, WORKER_STATUS: dict):
        """
        初始化高可用处理器

        Args:
            failure_queue (queue.Queue): 故障信息队列
        """
        self.failure_queue = failure_queue
        self.processor_thread = None
        self.is_running = False
        self.fault_statics = fault_statics
        self.worker_status = WORKER_STATUS
        self.worker_server_list = []

    def start(self):
        """启动故障处理线程"""
        if self.is_running:
            logger.info("ha processor already running")
            return self.processor_thread
        self.is_running = True
        self.processor_thread = threading.Thread(target=self._process_failures)
        self.processor_thread.daemon = True
        self.processor_thread.start()
        logger.info("success start ha processor")
        return self.processor_thread

    def stop(self):
        """停止故障处理线程"""
        self.is_running = False
        self.processing_active = False
        if self.processor_thread:
            self.processor_thread.join(timeout=-1)
            logger.info("stop ha processor")

    def _process_failures(self):
        """处理故障信息"""
        while self.is_running and processing_active:
            has_new_fault = self.fault_statics.has_new_step_fault()
            if not has_new_fault:
                time.sleep(1)
                continue
            curr_process_fault_step = self.fault_statics.largest_fault_step
            logger.info(f"start process fault step: {curr_process_fault_step}")
            try:
                # 1. just awake all workers (when return just means all stop_device hs finished)
                start_time = time.time()
                logger.info("start to awake all workers")
                self._workers_awake(HA_CMD_KEEP_RUN)

                # 2. wait until all workers awake
                all_workers = self.worker_server_list.copy()
                all_workers_count = len(all_workers)
                all_workers_stopped = False
                not_stopped_workers = []
                while time.time() - start_time < WORKER_WAITING_STOP_TIME:
                    stopped_workers = [v.worker_server_ip for k,v in self.worker_status.items()
                                       if v.status == WORKER_STATUS_DEVICE_FORCE_STOPPED and
                                       start_time - v.timestamp < WORKER_WAITING_STOP_TIME]
                    if len(stopped_workers) >= all_workers_count:
                        all_workers_stopped = True
                        logger.info(f"all {all_workers_count} workers has report force stopped.")
                        break
                    else:
                        not_stopped_workers = [elm for elm in all_workers if elm not in stopped_workers]
                        logger.warning(f"{len(not_stopped_workers)}/{all_workers_count} not report force stopped, "
                                     f"detail: {not_stopped_workers}")
                    time.sleep(1)

                # 3.send recover or exit
                logger.info(f"waiting for workers report stopped end, time_used: {time.time() - start_time}")
                opt, uce_blocks = self._get_keep_run_opt()
                self._workers_update_ha_cmd(HA_CMD_RECOVERY if all_workers_stopped else HA_CMD_EXIT_RETRY,
                                            opt, uce_blocks)
                if not all_workers_stopped:
                    logger.error(f"all workers stopped failed, unstopped workers: {not_stopped_workers}, "
                                 f"token recover exit.")
                    break
                logger.info(f"all workers has stopped, start to exec recovery")

                start_time = time.time()
                all_workers_recovered = False
                not_recovered_workers = []
                while time.time() - start_time < WORKER_WAITING_RECOVERY_TIME:
                    recovered_workers = [v.worker_server_ip for k, v in self.worker_status.items()
                                       if v.status == WORKER_STATUS_DEVICE_RESTARTED and
                                       start_time - v.timestamp < WORKER_WAITING_RECOVERY_TIME]
                    if len(recovered_workers) >= all_workers_count:
                        all_workers_recovered = True
                        logger.info(f"all {all_workers_count} workers has report recovered.")
                        break
                    else:
                        not_recovered_workers = [elm for elm in all_workers if elm not in recovered_workers]
                        logger.info(f"{len(not_recovered_workers)}/{all_workers_count} not report recovered, "
                                    f"detail: {not_recovered_workers}")
                    time.sleep(1)

                # 4.send token recompute or exit
                logger.info(f"waiting for workers report recovered end, time_used: {time.time() - start_time}")
                self._workers_update_ha_cmd(HA_CMD_KEEP_RUN if all_workers_recovered else HA_CMD_EXIT_RETRY,
                                            opt, uce_blocks)
                if not all_workers_recovered:
                    logger.error(f"all workers recovered failed, unrecovered workers: {not_recovered_workers}, "
                                 f"token recover exit.")
                    break
                self.fault_statics.update_largest_processed_step(curr_process_fault_step)
                logger.info(f"all workers has recovered, token recover finished.")
            except Exception as e:
                logger.error(f"error when process fault_step: {curr_process_fault_step}, error: {e}")

    def _workers_awake(self, cmd):
        logger.info("start_workers_awake")
        asyncio.run(self._batch_post_workers("/stop_device", {"cmd": cmd}))

    def _get_keep_run_opt(self):
        # ha.common.WorkerStatus
        status_info_list = [v for k, v in self.worker_status.items() if v.status == WORKER_STATUS_DEVICE_FORCE_STOPPED]
        err_codes = set()
        uce_flag = False
        uce_blocks = set()
        for info in status_info_list:
            if info.err_code:
                err_codes.add(info.err_code)
            if info.uce_flag:
                uce_flag = True
            if info.uce_blocks:
                uce_blocks.update(set(info.uce_blocks))

        if uce_flag:
            if -1 in uce_blocks or "-1" in uce_blocks:
                return "exit", uce_blocks
            return "abort", uce_blocks
        # if 507057，FORCE_STOP, OPT = "recompute"
        if not err_codes - ERR_CODES_CATEGORY:
            return "recompute", uce_blocks
        return "exit", uce_blocks

    def _workers_update_ha_cmd(self, cmd, opt, uce_blocks):
        logger.info("start_workers_update_ha_cmd")
        asyncio.run(self._batch_post_workers("/update_ha_cmd",
                                             {"cmd": cmd, "opt": opt, "uce_blocks": list(uce_blocks)}))

    async def _batch_post_workers(self, method, body):
        urls = [f"http://{server}{method}" for server in self.worker_server_list]
        async with aiohttp.ClientSession() as session:
            tasks = [_try_fetch_url(session, url, body) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(result)
                else:
                    logger.info(result)

async def _try_fetch_url(session, url, json):
    try:
        return await _fetch_url(session, url, json)
    except Exception as e:
        return f"Error fetching {url}:{str(e)}"

async def _fetch_url(session, url, json):
    start = time.time()
    msg = "success"
    try:
        async with session.post(url, json=json, timeout=10) as response:
            response.raise_for_status()
            return await response.text()
    except Exception as e:
        msg = str(e)
        logger.error(f"_fetch_url {url} cost: {time.time() - start} msg: {msg}")
        raise RuntimeError(f"_fetch_url {url} cost: {time.time() - start} msg: {msg}")