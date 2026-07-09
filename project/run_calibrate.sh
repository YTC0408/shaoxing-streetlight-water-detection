#!/bin/bash
# 标定模式：为摄像头生成基准 mask
# 用法：bash run_calibrate.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

python src/daily_pipeline.py \
    --date $(date +%Y-%m-%d) \
    --config config/cameras.yaml \
    --data-dir data \
    --workers 4 \
    --calibrate

echo "标定完成"
