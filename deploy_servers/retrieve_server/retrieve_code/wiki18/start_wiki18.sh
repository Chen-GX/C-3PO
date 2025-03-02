#!/bin/bash

gunicorn_run=/opt/conda/envs/faiss/bin/gunicorn

APP_MODULE="wiki18_serve:app"   # 指向您的 Flask 应用对象的导入路径
WORKERS=1              # 设置工作进程数量
BIND="0.0.0.0:35004"   # 设置 Gunicorn 监听的服务器地址和端口号
LOG_LEVEL="info"       # 日志级别（debug, info, warning, error, critical）

# 运行 Gunicorn 服务器
$gunicorn_run -w $WORKERS -b $BIND --timeout 360 --log-level=$LOG_LEVEL $APP_MODULE





