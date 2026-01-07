import asyncio
import os
import shutil
import signal
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import redis
from colorist import bg_green, bg_red
from loguru import logger

from .basic_models import CodeRunningBenchmarkTask, CodeRunningTaskStatus
from .cuda_utils import check_gpu_available

TMP_DIR = os.environ["TMP_DIR"]
TORCH_CUDA_ARCH_LIST = os.environ["TORCH_CUDA_ARCH_LIST"]
MACHINE = os.environ["MACHINE"]
bg_red(f'TORCH_CUDA_ARCH_LIST={TORCH_CUDA_ARCH_LIST}, MACHINE={MACHINE}')


def clean_redis():
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.delete(f"reval_configs")


def add_task_to_redis(config: str):
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.rpush("eval_configs", config)


clean_redis()


class TaskManager:
    def __init__(self):
        self.current_compiling_tasks_count = 0
        self.current_gpu_tasks_count = 0
        self.compile_finish_but_not_run_count = 0
        clean_redis()

    def get_current_compiling_tasks_count(self) -> int:
        return self.current_compiling_tasks_count
    def get_current_gpu_tasks_count(self) -> int:
        return self.current_gpu_tasks_count

    async def run_code_running_task(
        self, task: CodeRunningBenchmarkTask
    ) -> CodeRunningTaskStatus:
        start_time = datetime.now()
        # 1. 获取磁盘挂载的映射和result路径（result路径是在磁盘映射里的，但是为了方便我们也返回）
        # 其中result路径存放compile的结果以及最后输出结果
        assert task.compile_command
        self.current_compiling_tasks_count += 1
        temp_proj_dir, temp_exec_base_dir = await self.get_volume_mapping(task)

        # 2. 如果有compile命令，先运行compile命令
        if task.compile_command:
            logger.info(f"Compile code")
            compile_start_time = time.time()
            compile_successed, compile_result = await self.run_compile_command_adopt_error(
                task=task, temp_proj_dir=temp_proj_dir
            )
            self.current_compiling_tasks_count -= 1
            compile_end_time = time.time()
            logger.info(f"Compile finished in {compile_end_time - compile_start_time:.2f} seconds")
            with open("compile_times.txt", "a") as f:
                f.write(f"{task.task_name}\t{compile_end_time - compile_start_time:.2f}\n")

            if not compile_successed:
                return compile_result
            else:
                assert compile_result is None
        else:
            self.current_compiling_tasks_count -= 1
        self.compile_finish_but_not_run_count += 1

        # 3. 构建config
        config = {"mnk": "1024_1024_1024", "bm": -1, "bn": -1, "bk": -1, "base_dir": "/mgfs/shared/Group_GY/wenchao/e-hgemm-32/results"}

        # 4. 放入
        logger.info(f"Running code running task {task.task_name}")
        result = await self.run_to_get_a_new_result_with_error(task, temp_exec_base_dir=temp_exec_base_dir)

        # 5. 释放GPU
        bg_green(f"Task finished.")
        return result

    async def run_compile_command_adopt_error(self, task: CodeRunningBenchmarkTask, temp_proj_dir: Path) -> tuple[bool, Optional[CodeRunningTaskStatus]]:
        start_time = datetime.now()
        try:
            compile_successed, compile_logs = await self.run_compile_command(
                task, temp_proj_dir=temp_proj_dir
            )
            if compile_successed:
                compile_successed, compile_result = True, None
            else:
                logger.warning(f"Compile filed for task {task.task_name}")
                output_file_path = Path(f"errors/{uuid.uuid4().hex}_compile.txt")
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file_path, "w") as f:
                    f.write(f"Fail to run compile command for task {task.task_name}\n")
                    f.write(f"compile_logs: \n{compile_logs}\n")
                logger.error(f"Please check the error file: {output_file_path}")
                compile_successed, compile_result = False, CodeRunningTaskStatus(
                    task_id=task.task_id,
                    start_time=start_time,
                    end_time=datetime.now(),
                    status="failed",
                    exec_time_seconds=int((datetime.now() - start_time).total_seconds()),
                    exec_time_human_readable=str(timedelta(seconds=int((datetime.now() - start_time).total_seconds()))),
                    result={},
                    other_logs="",
                    docker_logs=compile_logs,
                    machine=MACHINE,
                )
        except Exception as e:
            stack = traceback.format_exc()
            bg_red(f"Failed to run compile command for task {task.task_name}")
            logger.error(f"Fail to run compile command for task {task.task_name}")
            output_file_path = Path(f"errors/{uuid.uuid4().hex}_compile.txt")
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, "w") as f:
                f.write(f"Fail to run compile command for task {task.task_name}\n")
                f.write(str(e) + "\n" + str(stack))
            logger.error(f"Please check the error file: {output_file_path}")
            compile_successed, compile_result = False, CodeRunningTaskStatus(
                task_id=task.task_id,
                start_time=start_time,
                end_time=datetime.now(),
                status="failed",
                exec_time_seconds=int((datetime.now() - start_time).total_seconds()),
                exec_time_human_readable=str(timedelta(seconds=int((datetime.now() - start_time).total_seconds()))),
                result={},
                other_logs=str(e) + "\n\n" + str(stack),
                docker_logs="",
                machine=MACHINE,
            )
        return compile_successed, compile_result

    async def run_compile_command(self, task: CodeRunningBenchmarkTask, temp_proj_dir: Path) -> tuple[bool, str]:
        commands = [
            f"cd {temp_proj_dir}",
            task.compile_command
        ]
        progress = await asyncio.create_subprocess_shell(
            " && ".join(commands),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # 将stderr重定向到stdout
            # 关键点 1：创建一个新的进程组
            preexec_fn=os.setsid
        )
        try:
            stdout, _ = await asyncio.wait_for(progress.communicate(), timeout=task.timeout)
        except asyncio.TimeoutError:
            # 关键点 2：杀死整个进程组
            try:
                # 获取该子进程的组 ID 并发送 SIGKILL
                os.killpg(os.getpgid(progress.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass # 进程可能已经自己结束了
            await asyncio.wait_for(progress.communicate(), timeout=60)
            return False, "Compile command timed out"
        except Exception as e:
            # 关键点 2：杀死整个进程组
            try:
                # 获取该子进程的组 ID 并发送 SIGKILL
                os.killpg(os.getpgid(progress.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass # 进程可能已经自己结束了
            await asyncio.wait_for(progress.communicate(), timeout=60)
            return False, f"Compile command failed when communicating: {str(e)}"

        output = stdout.decode("utf-8")
        if progress.returncode != 0:
            return False, f"Compile command failed with return code {progress.returncode}\n Logs:\n{output}"
        return True, output

    async def run_to_get_a_new_result_with_error(self, task: CodeRunningBenchmarkTask, temp_exec_base_dir: Path) -> CodeRunningTaskStatus:
        start_time = datetime.now()
        try:
            result = await self.run_to_get_a_new_result(task, temp_exec_base_dir)
        except Exception as e:
            stack = traceback.format_exc()
            logger.error(f"Fail to run code running task {task.task_name}")
            print(str(stack))
            result = CodeRunningTaskStatus(
                task_id=task.task_id,
                start_time=start_time,
                end_time=datetime.now(),
                status="failed",
                exec_time_seconds=int((datetime.now() - start_time).total_seconds()),
                exec_time_human_readable=str(timedelta(seconds=int((datetime.now() - start_time).total_seconds()))),
                result={},
                other_logs=str(e) + "\n" + str(stack),
                docker_logs="",
                machine=MACHINE,
            )
        if result.status != "success":
            bg_red(f"Failed to run code running task {task.task_name}")
            logger.error(f"Fail to run code running task {task.task_name}")
            output_file_path = Path(f"errors/{uuid.uuid4().hex}.txt")
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, "w") as f:
                f.write(f"Fail to run code running task {task.task_name}\n")
                f.write(f"docker_logs: {result.docker_logs}\n")
                f.write(f"other_logs: {result.other_logs}\n")
            logger.error(f"Please check the error file: {output_file_path}")
        else:
            bg_green(f"status: {result.status}")
            logger.info(f"status: {result.status}")
        return result

    async def run_to_get_a_new_result(self, task: CodeRunningBenchmarkTask, temp_exec_base_dir: Path) -> CodeRunningTaskStatus:
        # 获取设备ID
        start_time = datetime.now()
        task.run_command += f" --base_dir {str(temp_exec_base_dir)}"
        add_task_to_redis(task.run_command)
        finish_file = temp_exec_base_dir / "finished.txt"
        error_file = temp_exec_base_dir / "error.txt"
        start_file = temp_exec_base_dir / "start.txt"
        real_start_time = -1
        while 1:
            await asyncio.sleep(1)
            if start_file.exists() and real_start_time == -1:
                print(f"Real start on {temp_exec_base_dir}")
                real_start_time = time.time()
                self.current_gpu_tasks_count += 1
                self.compile_finish_but_not_run_count -= 1
            if finish_file.exists():
                self.current_gpu_tasks_count -= 1
                break
            if error_file.exists():
                self.current_gpu_tasks_count -= 1
                break
            if real_start_time > -1 and time.time() - real_start_time > 40:
                logger.error(f"Code running command timed out for task {task.task_name}, time={time.time() - real_start_time}")
                return CodeRunningTaskStatus(
                    task_id=task.task_id,
                    start_time=start_time,
                    end_time=datetime.now(),
                    status="failed",
                    exec_time_seconds=int((datetime.now() - start_time).total_seconds()),
                    exec_time_human_readable=str(timedelta(seconds=int((datetime.now() - start_time).total_seconds()))),
                    result={},
                    other_logs="Code running command timed out",
                    docker_logs="",
                    machine=MACHINE,
                )

        exec_result = {}
        if finish_file.exists():
            status = "success"
            for filename in task.result_filename_list:
                with open(temp_exec_base_dir / filename, "r") as f:
                    exec_result[filename] = f.read()
            error_logs = ""
        else:
            status = "failed"
            if error_file.exists():
                with open(error_file, "r") as f:
                    error_logs = f.read()
            else:
                error_logs = "Unknown error"
        result = CodeRunningTaskStatus(
            task_id=task.task_id,
            start_time=start_time,
            end_time=datetime.now(),
            status=status,
            exec_time_seconds=int((datetime.now() - start_time).total_seconds()),
            exec_time_human_readable=str(timedelta(seconds=int((datetime.now() - start_time).total_seconds()))),
            result=exec_result,
            docker_logs=str(error_logs),
            other_logs="",
            machine=MACHINE,
        )
        return result

    async def get_volume_mapping(self, task: CodeRunningBenchmarkTask) -> tuple[Path, Path]:
        # 需要强制exec_base_dir是相对路径
        assert task.exec_base_dir.is_relative_to(task.origin_project_dir), f"exec_base_dir {task.exec_base_dir} must be relative to origin_project_dir {task.origin_project_dir}"

        temp_proj_dir = Path(tempfile.mkdtemp(prefix=f"task_{uuid.uuid4().hex}_", dir=TMP_DIR))
        # copy task.origin_project_dir to temp_proj_dir
        logger.info(f"Copying project directory from {task.origin_project_dir} to {temp_proj_dir}")
        # 异步拷贝
        await asyncio.to_thread(shutil.copytree, task.origin_project_dir, temp_proj_dir, dirs_exist_ok=True)

        # 将新代码映射到容器中
        for file_path_and_code in task.file_path_and_code_list:
            origin_file_path = file_path_and_code.file_path
            offset = origin_file_path.relative_to(task.origin_project_dir)
            tmp_file_path = temp_proj_dir / offset
            with open(tmp_file_path, "w") as f:
                f.write(file_path_and_code.code)
        
        temp_exec_base_dir = temp_proj_dir / (task.exec_base_dir.relative_to(task.origin_project_dir))
        shutil.rmtree(temp_exec_base_dir, ignore_errors=True)
        temp_exec_base_dir.mkdir(parents=True, exist_ok=True)
        return temp_proj_dir, temp_exec_base_dir
    
            
GLOBAL_TASK_MANAGER = TaskManager()
