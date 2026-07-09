"""
时间感知工具：根据拍摄时间判断昼夜，选择合适的分析模式。
"""

from datetime import datetime, time
from typing import Tuple

import cv2
import numpy as np


def estimate_day_night(frame: np.ndarray, threshold: float = 50.0) -> str:
    """
    根据单帧平均亮度判断白天/夜晚。

    Returns:
        "day" 或 "night"
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    mean_brightness = float(np.mean(gray))
    return "night" if mean_brightness < threshold else "day"


def get_time_period(dt: datetime) -> str:
    """
    根据时间戳粗略划分时段，用于历史对比。

    Returns:
        "dawn", "day", "dusk", "night"
    """
    t = dt.time()
    if time(5, 0) <= t < time(7, 30):
        return "dawn"
    if time(7, 30) <= t < time(17, 0):
        return "day"
    if time(17, 0) <= t < time(20, 0):
        return "dusk"
    return "night"


def is_night_time(dt: datetime, night_start: time = time(19, 0),
                  night_end: time = time(6, 0)) -> bool:
    """根据时间判断是否处于夜间分析窗口。"""
    t = dt.time()
    if night_start <= night_end:
        return night_start <= t <= night_end
    return t >= night_start or t <= night_end


def parse_timestamp_from_path(path_str: str) -> datetime:
    """
    尝试从视频路径中解析时间戳。
    支持如 /data/videos/cam_001/2026-07-09_20-00-00.mp4
    """
    from pathlib import Path
    import re

    stem = Path(path_str).stem
    patterns = [
        r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})",
        r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2})",
        r"(\d{4}\d{2}\d{2})_(\d{2}\d{2}\d{2})",
        r"(\d{4}-\d{2}-\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            try:
                return datetime.strptime(match.group(0), "%Y-%m-%d_%H-%M-%S")
            except ValueError:
                pass
            try:
                return datetime.strptime(match.group(0), "%Y-%m-%d_%H-%M")
            except ValueError:
                pass
            try:
                return datetime.strptime(match.group(0), "%Y%m%d_%H%M%S")
            except ValueError:
                pass
            try:
                return datetime.strptime(match.group(0), "%Y-%m-%d")
            except ValueError:
                pass
    return datetime.fromtimestamp(Path(path_str).stat().st_mtime)
