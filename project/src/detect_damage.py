"""
路灯损伤检测：基于蒙板亮度与完整性分析。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.time_utils import get_time_period, is_night_time


def compute_mask_brightness(frame: np.ndarray, mask: np.ndarray) -> float:
    """计算蒙板区域内的平均亮度。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    total = cv2.sumElems(masked)[0]
    area = cv2.countNonZero(mask)
    return float(total / area) if area > 0 else 0.0


def compute_mask_shape(mask: np.ndarray) -> dict:
    """计算蒙板形状特征。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"area": 0, "perimeter": 0, "circularity": 0, "center": (0, 0)}

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

    moments = cv2.moments(cnt)
    cx = int(moments["m10"] / moments["m00"]) if moments["m00"] > 0 else 0
    cy = int(moments["m01"] / moments["m00"]) if moments["m00"] > 0 else 0

    return {
        "area": int(area),
        "perimeter": float(perimeter),
        "circularity": float(circularity),
        "center": (cx, cy),
    }


def load_reference(camera_id: str, lamp_id: str, calibration_dir: Path) -> dict:
    """加载基准 mask 和统计信息。"""
    ref_path = calibration_dir / camera_id / f"{lamp_id}_ref.json"
    if not ref_path.exists():
        return None
    with open(ref_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_brightness(current_brightness: float, reference: dict, cfg: dict) -> str:
    """根据当前亮度与历史基准判断状态。"""
    if reference is None:
        return "unknown"

    hist_brightness = reference.get("mean_brightness", current_brightness)
    if hist_brightness <= 0:
        return "unknown"

    ratio = current_brightness / hist_brightness
    if ratio < cfg.get("brightness_dim_ratio", 0.5):
        return "off_or_severe_dim"
    if ratio < cfg.get("brightness_warning_ratio", 0.7):
        return "dim"
    return "normal"


def analyze_integrity(current_shape: dict, reference: dict, cfg: dict) -> str:
    """根据 mask 面积和位置判断完整性。"""
    if reference is None:
        return "unknown"

    ref_area = reference.get("area", current_shape["area"])
    ref_center = reference.get("center", current_shape["center"])

    if ref_area <= 0:
        return "unknown"

    area_ratio = current_shape["area"] / ref_area
    if area_ratio < cfg.get("integrity_area_ratio", 0.5):
        return "damaged_or_missing"

    dx = current_shape["center"][0] - ref_center[0]
    dy = current_shape["center"][1] - ref_center[1]
    offset = np.sqrt(dx ** 2 + dy ** 2)
    if offset > cfg.get("integrity_offset_threshold", 15):
        return "displaced"

    return "normal"


def detect_damage(video_path: str, masks_meta: dict, output_dir: Path,
                  calibration_dir: Path, cfg: dict) -> List[dict]:
    """
    对视频和蒙板进行损伤检测。

    Returns:
        每个路灯的检测结果列表
    """
    camera_id = cfg.get("camera_id", "unknown")
    timestamp = datetime.fromisoformat(masks_meta["timestamp"])
    mode = masks_meta.get("mode", "night")
    is_night = is_night_time(timestamp)

    # 读取一帧代表当前画面
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return []

    results = []
    for lamp in masks_meta.get("lamps", []):
        lamp_id = lamp["lamp_id"]
        mask_path = Path(video_path).parent.parent.parent / lamp["mask_path"]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        reference = load_reference(camera_id, lamp_id, calibration_dir)

        current_brightness = compute_mask_brightness(frame, mask)
        current_shape = compute_mask_shape(mask)

        if is_night:
            brightness_status = analyze_brightness(current_brightness, reference, cfg)
        else:
            brightness_status = "daytime_skip"

        integrity_status = analyze_integrity(current_shape, reference, cfg)

        # 综合判定
        if brightness_status == "off_or_severe_dim" or integrity_status == "damaged_or_missing":
            overall = "fault"
        elif brightness_status == "dim" or integrity_status == "displaced":
            overall = "warning"
        else:
            overall = "normal"

        results.append({
            "camera_id": camera_id,
            "lamp_id": lamp_id,
            "timestamp": timestamp.isoformat(),
            "time_period": get_time_period(timestamp),
            "mode": mode,
            "brightness": round(current_brightness, 2),
            "brightness_status": brightness_status,
            "area": current_shape["area"],
            "circularity": round(current_shape["circularity"], 3),
            "center": current_shape["center"],
            "integrity_status": integrity_status,
            "overall_status": overall,
        })

    return results
