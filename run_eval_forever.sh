#!/bin/bash

while true; do
    # 启动服务
    source setup.sh
    echo "Starting my_service.py..."
    python http_service/env-service.py
    
    # 如果服务退出，打印错误信息并等待5秒后重启
    echo "Service crashed with exit code $?. Restarting in 5 seconds..."
    sleep 5
done