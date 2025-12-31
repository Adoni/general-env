import subprocess


def check_gpu_available(device_id: int):
    try:
        # 使用nvidia-smi检查GPU使用情况
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        stdout = result.stdout.strip()
        
        for line in stdout.split('\n'):
            if line:
                gpu_idx, util, mem_used = line.split(', ')
                if int(gpu_idx) == device_id:
                    # 如果GPU利用率大于5%或内存使用大于100MB，认为被占用
                    if int(util) > 5 or int(mem_used) > 100:
                        print(f"GPU {device_id} is not available")
                        print(f"line: {line}")
                        print(f"stdout: {stdout}")
                        return False
        return True
    except Exception as e:
        print(f"Error checking GPU {device_id}: {e}")
        return False