import subprocess
import time

import psutil


def kill_env_service():
    kill_command = "ps aux | grep env-service.py | grep -v grep | awk '{print $2}' | xargs kill"
    try:
        print(f"Executing kill command: {kill_command}")
        subprocess.run(kill_command, shell=True, check=False)
        print("Success!")
    except Exception as e:
        print(f"Failed to execute kill command: {e}")

def monitor():
    print("Strict watchdog started. Monitoring memory usage...")
    while True:
        try:
            # 1. Check system memory usage
            vm = psutil.virtual_memory()
            if vm.percent > 90:
                print(f"System memory usage critical: {vm.percent}% > 90%. triggering restart.")
                kill_env_service()
                # Wait a bit after killing to allow restart/cleanup
                time.sleep(10)
                continue

            # 2. Check all processes
            triggered = False
            for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
                try:
                    mem_percent = proc.info.get('memory_percent')
                    if mem_percent is not None and mem_percent > 5:
                        print(f"Process {proc.info.get('name')} (PID: {proc.info.get('pid')}) uses {mem_percent:.2f}% memory (> 5%). triggering restart.")
                        kill_env_service()
                        triggered = True
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            if triggered:
                time.sleep(10)

        except Exception as e:
            print(f"Error in watchdog loop: {e}")
        
        time.sleep(5)

if __name__ == "__main__":
    monitor()
