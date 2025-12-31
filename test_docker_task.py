#!/usr/bin/env python3
"""
Test script for the Docker-based task manager
"""

import asyncio
import json
from pathlib import Path
from http_service.modules.basic_models import CodeRunningBenchmarkTask, FilePathAndCode
from http_service.modules.task_manager import GLOBAL_TASK_MANAGER

async def test_docker_task():
    """Test the Docker-based task execution"""
    
    # Create a simple test script
    test_script_content = """#!/bin/bash
cd /workspace
echo "Running test script in Docker container"
echo "Current directory: $(pwd)"
echo "Files in workspace:"
ls -la

# Create a simple result
echo '{"test_result": "success", "message": "Docker task completed successfully"}' > result.json
echo "Result file created"
"""
    
    # Create the test script file
    test_script_path = Path("/tmp/test_benchmark.sh")
    with open(test_script_path, "w") as f:
        f.write(test_script_content)
    test_script_path.chmod(0o755)  # Make executable
    
    # Create test files
    test_file_content = """import numpy as np
import json

def test_function():
    data = np.random.rand(10, 10)
    result = {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "shape": data.shape
    }
    return result

if __name__ == "__main__":
    result = test_function()
    print("Test completed successfully")
"""
    
    # Define the task
    task = CodeRunningBenchmarkTask(
        task_name="docker_test_task",
        device_id="cuda:1",  # You may need to change this based on your setup
        is_baseline_task=False,
        run_baseline_again=False,
        file_path_and_code_list=[
            FilePathAndCode(
                file_path=Path("/tmp/test_model.py"),
                code=test_file_content
            )
        ],
        script_file_path=test_script_path,
        result_saved_path=Path("/tmp/result.json"),
        timeout=60,
        docker_image_tag="python:3.9-slim"  # Use a simple Python image for testing
    )
    
    try:
        print("Starting Docker-based task...")
        status = await GLOBAL_TASK_MANAGER.run_code_running_task(task)
        
        print(f"Task completed!")
        print(f"Status: {status.status}")
        print(f"Execution time: {status.exec_time_seconds} seconds")
        print(f"STDOUT: {status.stdout}")
        print(f"STDERR: {status.stderr}")
        print(f"Result: {json.dumps(status.result, indent=2)}")
        
    except Exception as e:
        print(f"Error running task: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test files
        if test_script_path.exists():
            test_script_path.unlink()

if __name__ == "__main__":
    asyncio.run(test_docker_task()) 