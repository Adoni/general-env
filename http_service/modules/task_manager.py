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

from colorist import bg_green, bg_red
from loguru import logger

from .basic_models import CodeRunningBenchmarkTask, CodeRunningTaskStatus
from .cuda_utils import check_gpu_available

TMP_DIR = os.environ["TMP_DIR"]
TORCH_CUDA_ARCH_LIST = os.environ["TORCH_CUDA_ARCH_LIST"]
MACHINE = os.environ["MACHINE"]
bg_red(f'TORCH_CUDA_ARCH_LIST={TORCH_CUDA_ARCH_LIST}, MACHINE={MACHINE}')


class TaskManager:
    def __init__(self):
        gpu_id_list = list(range(0, 8))
        self.device_queue = asyncio.Queue(maxsize=len(gpu_id_list))
        for i in gpu_id_list:
            self.device_queue.put_nowait(f"{i}")
            # assert i != 7, "GPU 7 需要保留给编译使用"
        self.current_compiling_tasks_count = 0

    def get_current_compiling_tasks_count(self) -> int:
        return self.current_compiling_tasks_count

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
        # 3. 找到一个空闲的GPU
        logger.info(f"Finding a device")
        try:
            device_id = await asyncio.wait_for(self.device_queue.get(), timeout=task.timeout)
        except asyncio.TimeoutError:
            bg_red(f"Failed to find a device for task {task.task_name}")
            return CodeRunningTaskStatus(
                task_id=task.task_id,
                start_time=start_time,
                end_time=datetime.now(),
                status="failed",
                exec_time_seconds=(int((datetime.now() - start_time).total_seconds())),
                exec_time_human_readable=str(datetime.now() - start_time).split(".")[0],
                result={},
                other_logs="Timeout when waiting for a free device",
                docker_logs="",
                machine=MACHINE,
            )
        logger.info(f"found a device: {device_id}")
        assert check_gpu_available(device_id), f"Device {device_id} is not empty"
        bg_green(f"Device {device_id} is empty. Occupy it now!")

        # 4. 运行代码
        logger.info(f"Running code running task {task.task_name}")
        result = await self.run_to_get_a_new_result_with_error(
            task, device_id=device_id, temp_proj_dir=temp_proj_dir, temp_exec_base_dir=temp_exec_base_dir
        )

        # 5. 释放GPU
        await asyncio.sleep(10)  # 等待1秒，确保GPU被释放
        try:
            await self.force_release_gpu(device_id)  # 强制释放GPU
        except Exception as e:
            logger.error(f"Failed to force release GPU {device_id}: {str(e)}")
        self.device_queue.put_nowait(device_id)
        bg_green(f"Task finished. Release device {device_id} now!")
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

    async def run_to_get_a_new_result_with_error(self, task: CodeRunningBenchmarkTask, device_id: str, temp_proj_dir: Path, temp_exec_base_dir: Path) -> CodeRunningTaskStatus:
        start_time = datetime.now()
        try:
            result = await self.run_to_get_a_new_result(task, device_id, temp_proj_dir, temp_exec_base_dir)
        except Exception as e:
            stack = traceback.format_exc()
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

    async def run_to_get_a_new_result(self, task: CodeRunningBenchmarkTask, device_id: int, temp_proj_dir: Path, temp_exec_base_dir: Path) -> CodeRunningTaskStatus:
        # 获取设备ID
        start_time = datetime.now()
        task.run_command += f" --gpu_device_id {device_id} --base_dir {str(temp_exec_base_dir)}"
        commands = [
            f"cd {temp_proj_dir}",
            task.run_command
        ]
        progress = await asyncio.create_subprocess_shell(
            " && ".join(commands),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT  # 将stderr重定向到stdout
        )
        try:
            stdout, _ = await asyncio.wait_for(progress.communicate(), timeout=task.timeout)
        except asyncio.TimeoutError:
            progress.kill()
            await asyncio.wait_for(progress.communicate(), timeout=60)
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
        except Exception as e:
            progress.kill()
            await asyncio.wait_for(progress.communicate(), timeout=60)
            return CodeRunningTaskStatus(
                task_id=task.task_id,
                start_time=start_time,
                end_time=datetime.now(),
                status="failed",
                exec_time_seconds=int((datetime.now() - start_time).total_seconds()),
                exec_time_human_readable=str(timedelta(seconds=int((datetime.now() - start_time).total_seconds()))),
                result={},
                other_logs=f"Code running command failed when communicating: {str(e)}",
                docker_logs="",
                machine=MACHINE,
            )
        logs = stdout.decode("utf-8")

        exec_result = {}
        if progress.returncode != 0:
            status = "failed"
        else:
            status = "success"
            for filename in task.result_filename_list:
                with open(temp_exec_base_dir / filename, "r") as f:
                    exec_result[filename] = f.read()
        
        result = CodeRunningTaskStatus(
            task_id=task.task_id,
            start_time=start_time,
            end_time=datetime.now(),
            status=status,
            exec_time_seconds=int((datetime.now() - start_time).total_seconds()),
            exec_time_human_readable=str(timedelta(seconds=int((datetime.now() - start_time).total_seconds()))),
            result=exec_result,
            docker_logs=str(logs),
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
    
    async def force_release_gpu(self, device_id: str):
        """使用nvidia-smi异步强制释放GPU"""
        # 异步获取指定GPU上的进程PID
        proc = await asyncio.create_subprocess_exec(
            'nvidia-smi', 
            f'--id={device_id}',
            '--query-compute-apps=pid',
            '--format=csv,noheader',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            raise Exception(f"nvidia-smi执行失败: {stderr.decode()}")
        
        # 解析PID列表
        pids = [pid.strip() for pid in stdout.decode().split('\n') if pid.strip()]
        
        # 并发终止所有进程
        tasks = []
        for pid in pids:
            task = asyncio.create_subprocess_exec(
                'kill', '-9', pid,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            tasks.append(task)
        
        # 等待所有终止操作完成
        processes = await asyncio.gather(*tasks, return_exceptions=True)
        
        killed_pids = []
        for i, proc in enumerate(processes):
            if isinstance(proc, Exception):
                print(f"终止进程 {pids[i]} 失败: {proc}")
            else:
                await proc.communicate()
                if proc.returncode == 0:
                    killed_pids.append(pids[i])
                    print(f"已终止进程 PID: {pids[i]}")
                else:
                    print(f"终止进程 {pids[i]} 失败，返回码: {proc.returncode}")
        return {
            "success": True,
            "device_id": device_id,
            "killed_processes": killed_pids
        }
            
GLOBAL_TASK_MANAGER = TaskManager()
