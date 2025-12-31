from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path
from datetime import datetime
import uuid

class FilePathAndCode(BaseModel):
    file_path: Path
    code: str

class CodeRunningBenchmarkTask(BaseModel):
    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    task_name: str
    file_path_and_code_list: list[FilePathAndCode]  # 要评测的代码
    origin_project_dir: Path  # 代码运行的基础目录
    exec_base_dir: Path  # 存放结果、中间结果、编译等文件的基础目录, 必须是origin_project_dir的子目录
    result_filename_list: list[str]  # 结果文件名列表
    timeout: int  # 超时时间，单位秒
    compile_command: str  # 编译命令
    run_command: str  # 运行命令


class CodeRunningTaskStatus(BaseModel):
    task_id: str
    start_time: datetime
    end_time: datetime
    status: Literal["success", "failed"]
    exec_time_seconds: int
    exec_time_human_readable: str
    result: dict  # 结果
    docker_logs: str  # 日志
    other_logs: str
    machine: str