import asyncio
from fastapi import APIRouter, HTTPException
from .basic_models import CodeRunningBenchmarkTask, CodeRunningTaskStatus
from .task_manager import GLOBAL_TASK_MANAGER
from datetime import datetime
from loguru import logger
from pydantic import BaseModel
import os

import psutil
import time

# 增加logger的文件输出
logger.add("http_service.log", rotation="100 MB")


router = APIRouter()

class MonitorValue(BaseModel):
    total_requests: int
    total_responses: int

MONITOR_VALUE = MonitorValue(total_requests=0, total_responses=0)

async def get_system_usage():
    """
    获取系统CPU和内存使用率
    
    Returns:
        dict: 包含CPU和内存使用率的字典
    """
    # 获取内存使用率
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    # 获取CPU使用率（interval参数指定测量时间间隔）
    # interval=1 表示测量1秒内的平均使用率
    cpu_percent = await asyncio.to_thread(psutil.cpu_percent, interval=1)
    
    return {
        'memory_percent': memory_percent,
        'cpu_percent': cpu_percent,
        'memory_used_gb': memory.used / (1024**3),
        'memory_total_gb': memory.total / (1024**3)
    }


@router.post("/run_code_replace_benchmark", response_model=CodeRunningTaskStatus)
async def run_benchmark(task: CodeRunningBenchmarkTask):
    logger.info(f"Get New Code Running Task {task.task_name}")
    current_usage = await get_system_usage()
    logger.info(f"Current System Usage: CPU {current_usage['cpu_percent']}%, Memory {current_usage['memory_percent']}% ({current_usage['memory_used_gb']:.2f}GB/{current_usage['memory_total_gb']:.2f}GB)")

    current_running_tasks = MONITOR_VALUE.total_requests - MONITOR_VALUE.total_responses
    logger.info(f"ONPROCESSING: {current_running_tasks}, TOTAL_REQUESTS: {MONITOR_VALUE.total_requests}, TOTAL_RESPONSES: {MONITOR_VALUE.total_responses}")
    start_time = time.time()
    while time.time() - start_time < task.timeout:
        if current_running_tasks >= 400:
            logger.warning(f"Too many concurrent tasks ({current_running_tasks}), rejecting new task {task.task_name}")
            await asyncio.sleep(10)
            # raise HTTPException(status_code=429, detail="Too many concurrent tasks, please try again later.")
            continue
        current_compiling_tasks = GLOBAL_TASK_MANAGER.get_current_compiling_tasks_count()
        if current_compiling_tasks >= 400:
            logger.warning(f"Too many concurrent compiling tasks ({current_compiling_tasks}), rejecting new task {task.task_name}")
            await asyncio.sleep(10)
            # raise HTTPException(status_code=429, detail="Too many concurrent compiling tasks, please try again later.")
            continue
        break

    MONITOR_VALUE.total_requests += 1
    try:
        result = await GLOBAL_TASK_MANAGER.run_code_running_task(task)
    except Exception as e:
        logger.exception(f"Error when running code running task {task.task_name}: {e}")
        MONITOR_VALUE.total_responses += 1
        raise HTTPException(status_code=500, detail=str(e))
    logger.info(f"Code Running Task Finished")
    MONITOR_VALUE.total_responses += 1
    return result


@router.post("/check_folder_not_exist")
async def check_folder_not_exist(path: str):
    assert path.startswith("/data/")
    if not os.path.exists(path):
        print(f"Folder {path} not exist")
        return {"status": "success"}
    else:
        print(f"Folder {path} exist")
        return {"status": "failed"}


@router.get("/health")
async def health_check():
    total_requests = MONITOR_VALUE.total_requests
    total_responses = MONITOR_VALUE.total_responses
    current_compiling_tasks = GLOBAL_TASK_MANAGER.get_current_compiling_tasks_count()
    system_usage = await get_system_usage()
    return {
        "status": "ok",
        "timestamp": datetime.now(),
        "total_requests": total_requests,
        "total_responses": total_responses,
        "current_compiling_tasks": current_compiling_tasks,
        "system_usage": system_usage,
    }

print(get_system_usage())
