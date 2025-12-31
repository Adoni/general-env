#!/usr/bin/env python3
"""
GPU Process Watchdog
Monitor GPU processes and kill those running longer than timeout minutes
Get process runtime directly from system
Only monitor GPUs 1-7, skip GPU 0
Identify GPU processes by finding commands containing 'gpu_device_id'
Also monitor processes with path '/mgfs/shared/Group_GY/wenchao/tmp/task_'
"""

import re
import signal
import subprocess
import sys
import time
from typing import List, Optional, Tuple

from loguru import logger
from tqdm import tqdm


class GPUProcessWatchdog:
    def __init__(self, timeout_minutes=10, check_interval=60):
        """
        Initialize GPU process watchdog
        
        Args:
            timeout_minutes (int): Process timeout in minutes
            check_interval (int): Check interval in seconds
        """
        self.timeout_minutes = timeout_minutes
        self.check_interval = check_interval
        self.running = True
        
        # 原有的GPU进程匹配模式
        self.gpu_pattern = re.compile(r'--gpu_device_id\s+(\d+)')
        
        # 新增:匹配 /mgfs/shared/Group_GY/wenchao/tmp/task_ 路径的进程
        self.task_pattern = re.compile(r'/data/tmp/task_')
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down watchdog...")
        self.running = False
    
    def get_monitored_processes(self) -> List[Tuple[int, str]]:
        """
        Get list of processes to monitor
        Monitors: GPU processes (GPUs 1-7) and task_ path processes
        
        Returns:
            List[Tuple[int, str]]: List of (PID, description) tuples
        """
        try:
            # Get command line info for all processes
            result = subprocess.run([
                'ps', '-e', '-o', 'pid=,command='
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"ps command failed: {result.stderr}")
                return []
            
            processes = []
            for line in tqdm(result.stdout.strip().split('\n'), desc="Scanning processes"):
                if not line.strip():
                    continue
                
                # Extract PID and command line
                parts = line.strip().split(maxsplit=1)
                if len(parts) < 2:
                    continue
                
                try:
                    pid = int(parts[0])
                    cmd = parts[1]
                    
                    # 检查是否是GPU进程 (GPUs 1-7)
                    gpu_match = self.gpu_pattern.search(cmd)
                    if gpu_match:
                        gpu_id = int(gpu_match.group(1))
                        if 0 <= gpu_id <= 7:
                            processes.append((pid, f"GPU {gpu_id}"))
                            logger.debug(f"Monitoring GPU process {pid} on GPU {gpu_id}")
                        else:
                            logger.debug(f"Skipping process {pid} on GPU {gpu_id}")
                        continue
                    
                    # 检查是否是task_路径的进程
                    if self.task_pattern.search(cmd):
                        processes.append((pid, "task_path"))
                        logger.debug(f"Monitoring task_ process {pid}")
                        continue

                    if "zero_one_correctness_check.py" in cmd or "randn_correctness_check.py" in cmd or "compare_with_baseline_docker.py" in cmd:
                        processes.append((pid, "task_path"))
                        logger.debug(f"Monitoring task_ process {pid}")
                        continue

                except ValueError:
                    continue
            
            return processes
        
        except subprocess.TimeoutExpired:
            logger.error("ps command timed out")
            return []
        except Exception as e:
            logger.error(f"Error getting process info: {e}")
            return []
    
    def get_process_runtime_info(self, pid: int) -> Optional[Tuple[str, str, float]]:
        """
        Get process runtime information
        
        Args:
            pid (int): Process ID
            
        Returns:
            Optional[Tuple[str, str, float]]: (process name, username, runtime in minutes) or None
        """
        try:
            # Use ps command to get process info: name, user, runtime (etime format)
            result = subprocess.run([
                'ps', '-p', str(pid), '-o', 'comm=,user=,etime=', '--no-headers'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return None
            
            line = result.stdout.strip()
            if not line:
                return None
            
            # Parse output: process_name username runtime
            parts = line.split()
            if len(parts) < 3:
                return None
            
            process_name = parts[0]
            username = parts[1]
            etime_str = parts[2]
            
            # Parse etime format to minutes
            runtime_minutes = self._parse_etime_to_minutes(etime_str)
            
            return process_name, username, runtime_minutes
        
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout getting info for process {pid}")
            return None
        except Exception as e:
            logger.warning(f"Error getting info for process {pid}: {e}")
            return None
    
    def _parse_etime_to_minutes(self, etime_str: str) -> float:
        """
        Parse ps etime format to minutes
        
        etime format could be:
        - MM:SS (minutes:seconds)
        - HH:MM:SS (hours:minutes:seconds) 
        - DD-HH:MM:SS (days-hours:minutes:seconds)
        
        Args:
            etime_str (str): ps etime output
            
        Returns:
            float: Runtime in minutes
        """
        try:
            total_seconds = 0
            
            # Handle DD-HH:MM:SS format
            if '-' in etime_str:
                days_part, time_part = etime_str.split('-', 1)
                total_seconds += int(days_part) * 24 * 3600
                etime_str = time_part
            
            # Handle HH:MM:SS or MM:SS format
            time_parts = etime_str.split(':')
            
            if len(time_parts) == 2:  # MM:SS
                minutes, seconds = map(int, time_parts)
                total_seconds += minutes * 60 + seconds
            elif len(time_parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = map(int, time_parts)
                total_seconds += hours * 3600 + minutes * 60 + seconds
            else:
                # Unknown format, return 0
                logger.warning(f"Unknown etime format: {etime_str}")
                return 0.0
            
            return total_seconds / 60.0  # Convert to minutes
        
        except Exception as e:
            logger.warning(f"Error parsing runtime '{etime_str}': {e}")
            return 0.0

    def kill_process(self, pid: int, process_name: str, username: str, description: str) -> bool:
        """
        Kill specified process
        
        Args:
            pid (int): Process ID
            process_name (str): Process name
            username (str): Username
            description (str): Process description
            
        Returns:
            bool: Whether process was successfully killed
        """
        try:
            # First try SIGTERM (graceful termination)
            subprocess.run(['kill', '-TERM', str(pid)], check=True)
            logger.info(f"Sending SIGTERM to process {pid} ({process_name}, user: {username}, {description})")
            
            # Wait 5 seconds, check if process terminated
            time.sleep(5)
            if not self._is_process_running(pid):
                logger.info(f"Process {pid} terminated normally")
                return True
            
            # If process still running, use SIGKILL (forceful termination)
            subprocess.run(['kill', '-KILL', str(pid)], check=True)
            logger.info(f"Sending SIGKILL to process {pid} ({process_name}, user: {username})")
            
            time.sleep(2)
            if not self._is_process_running(pid):
                logger.info(f"Process {pid} forcefully terminated")
                return True
            else:
                logger.warning(f"Failed to terminate process {pid}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error killing process {pid}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unknown error killing process {pid}: {e}")
            return False
    
    def _is_process_running(self, pid: int) -> bool:
        """Check if process is still running"""
        try:
            # Send signal 0 to check if process exists
            subprocess.run(['kill', '-0', str(pid)], 
                         capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def check_and_kill_processes(self):
        """Check and kill timed-out processes, starting with longest running"""
        monitored_processes = self.get_monitored_processes()
        
        if not monitored_processes:
            logger.info("No monitored processes found")
            return
        
        logger.info(f"Found {len(monitored_processes)} monitored processes")
        
        # Collect process info with runtime
        process_list = []
        for pid, description in monitored_processes:
            process_info = self.get_process_runtime_info(pid)
            
            if not process_info:
                continue
            
            process_name, username, runtime_minutes = process_info
            
            logger.info(f"Process {pid} ({description}, {process_name}, user: {username}) running for {runtime_minutes:.1f} minutes")
            
            # Only add processes that exceed timeout
            if runtime_minutes > self.timeout_minutes:
                process_list.append({
                    'pid': pid,
                    'description': description,
                    'process_name': process_name,
                    'username': username,
                    'runtime_minutes': runtime_minutes
                })
        
        if not process_list:
            logger.info("No processes exceeding timeout limit")
            return
        
        # Sort by runtime in descending order (longest first)
        process_list.sort(key=lambda x: x['runtime_minutes'], reverse=True)
        
        logger.info(f"Found {len(process_list)} processes exceeding {self.timeout_minutes} minutes timeout")
        logger.info("Killing processes in order of runtime (longest first):")
        
        # Kill processes in order
        for proc in process_list:
            logger.warning(
                f"Process {proc['pid']} ({proc['description']}, {proc['process_name']}, "
                f"user: {proc['username']}) running for {proc['runtime_minutes']:.1f} minutes, terminating..."
            )
            
            if self.kill_process(proc['pid'], proc['process_name'], proc['username'], proc['description']):
                logger.info(f"Successfully terminated process {proc['pid']}")
            else:
                logger.error(f"Failed to terminate process {proc['pid']}, will retry next check")
    
    def run(self):
        """Run monitoring loop"""
        logger.info(f"Starting process monitor, timeout: {self.timeout_minutes} minutes, check interval: {self.check_interval} seconds")
        logger.info(f"Monitoring: GPU processes (GPUs 1-7) and task_ path processes")
        logger.info("Processes will be killed in order of runtime (longest first)")
        
        while self.running:
            try:
                self.check_and_kill_processes()
                
                # Wait for next check
                for _ in range(self.check_interval):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Wait 10 seconds after error before continuing
        
        logger.info("Process monitor stopped")

def main():
    """Main function"""
    # Can be configured via command line args or environment variables
    import os
    
    timeout_minutes = int(os.environ.get('GPU_TIMEOUT_MINUTES', '10'))
    check_interval = int(os.environ.get('GPU_CHECK_INTERVAL', '60'))
    
    watchdog = GPUProcessWatchdog(timeout_minutes=timeout_minutes, check_interval=check_interval)
    
    try:
        watchdog.run()
    except Exception as e:
        logger.error(f"Error running program: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
