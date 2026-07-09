#!/bin/bash
# 每日路灯损伤检测流水线（Ubuntu）
# 建议加入 crontab：0 3 * * * /bin/bash /path/to/project/run_daily.sh >> /path/to/project/logs/cron.log 2>&1

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# 激活 conda 环境（若未使用 conda，可注释下面两行）
# source /opt/anaconda3/etc/profile.d/conda.sh
# conda activate shaoxing-yolo

# 处理昨日视频
YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)

python src/daily_pipeline.py \
    --date "$YESTERDAY" \
    --config config/cameras.yaml \
    --data-dir data \
    --workers 4

echo "[$YESTERDAY] 路灯检测完成"
