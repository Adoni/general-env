# Docker-based Code Running Task Manager Usage

This document explains how to use the updated Docker-based `run_to_get_a_new_result` method.

## Changes Made

1. **Added `docker_image_tag` field** to `CodeRunningBenchmarkTask`
2. **Replaced file overwriting** with Docker container execution
3. **Added temporary file creation** with random UUID names
4. **Implemented volume mapping** for file sharing between host and container

## Example Usage

### 1. Basic Task Definition

```python
from http_service.modules.basic_models import CodeRunningBenchmarkTask, FilePathAndCode
from pathlib import Path

# Define the files to be created/modified
file_path_and_code_list = [
    FilePathAndCode(
        file_path=Path("/path/to/model.py"),
        code="""
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)
"""
    ),
    FilePathAndCode(
        file_path=Path("/path/to/train.py"),
        code="""
import torch
from model import MyModel

model = MyModel()
# Training code here...
"""
    )
]

# Create the task
task = CodeRunningBenchmarkTask(
    task_name="my_benchmark_task",
    device_id="cuda:1",
    is_baseline_task=False,
    run_baseline_again=False,
    file_path_and_code_list=file_path_and_code_list,
    script_file_path=Path("/path/to/run_benchmark.sh"),
    result_saved_path=Path("/path/to/result.json"),
    timeout=300,
    docker_image_tag="my-pytorch-image:latest"  # New field
)
```

### 2. Docker Image Requirements

Your Docker image should:
- Have the necessary Python packages installed
- Include CUDA support if using GPU
- Have the script execution environment set up

Example Dockerfile:
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install torch torchvision torchaudio

# Set working directory
WORKDIR /workspace

# Copy your benchmark script
COPY run_benchmark.sh /workspace/
RUN chmod +x /workspace/run_benchmark.sh
```

### 3. Script File Example

Your `run_benchmark.sh` script should:
- Work with the randomly named files in `/workspace`
- Save results to `/workspace/result.json`

```bash
#!/bin/bash
cd /workspace

# The files will have random names, so you might need to find them
# or the calling code should know the expected names

# Run your benchmark
python3 train.py

# Save results
echo '{"accuracy": 0.95, "loss": 0.1}' > result.json
```

### 4. Running the Task

```python
from http_service.modules.task_manager import GLOBAL_TASK_MANAGER

# Run the task
status = await GLOBAL_TASK_MANAGER.run_code_running_task(task)
print(f"Task status: {status.status}")
print(f"Execution time: {status.exec_time_seconds} seconds")
print(f"Result: {status.result}")
```

## How It Works

1. **Temporary Directory Creation**: A temporary directory is created with a unique UUID prefix
2. **File Creation**: Each file in `file_path_and_code_list` is created with a random UUID name (preserving extension)
3. **Script Copying**: The script file is copied to the temporary directory
4. **Docker Execution**: Docker runs with the temporary directory mounted as `/workspace`
5. **Result Reading**: Results are read from the mapped result file
6. **Cleanup**: The temporary directory is automatically cleaned up

## Benefits

- **Isolation**: Each task runs in its own Docker container
- **No File Conflicts**: Random file names prevent conflicts
- **Clean Environment**: Fresh container for each execution
- **GPU Support**: Proper GPU isolation and management
- **Automatic Cleanup**: Temporary files are automatically removed

## Notes

- The script inside the Docker container should expect files to be in `/workspace`
- Result files should be saved to `/workspace/result.json`
- The Docker image must have all necessary dependencies installed
- GPU access is controlled via `--gpus device=X` flag 