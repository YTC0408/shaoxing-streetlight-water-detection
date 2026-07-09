"""
基于 OpenCV 的自动蒙板生成模块。

核心思路：
1. 标定阶段：夜间利用路灯发光特性自动发现灯头位置，生成参考蒙板。
2. 日常阶段：加载参考蒙板作为 ROI，在白天/夜间分别提取当前帧的灯头区域。
3. 多帧共识，过滤车灯、闪电等瞬时亮斑。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.time_utils import is_night_time


def sample_frames(video_path: str, num_samples: int = 10) -> List[np.ndarray]:
    """从视频中均匀采样若干帧。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, min(num_samples, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def detect_bright_blobs(gray: np.ndarray, threshold: int,
                        min_area: int, max_area: int,
                        min_circularity: float) -> List[dict]:
    """检测灰度图中的高亮连通域。"""
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue

        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        if circularity < min_circularity:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = int(x + w / 2), int(y + h / 2)

        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        blobs.append({
            "center": (cx, cy),
            "bbox": (x, y, w, h),
            "area": int(area),
            "circularity": float(circularity),
            "mask": mask,
        })
    return blobs


def multi_frame_consensus(frames: List[np.ndarray], cfg: dict) -> List[dict]:
    """多帧共识：只在多数帧中稳定出现的目标才保留。"""
    if not frames:
        return []

    h, w = frames[0].shape[:2]
    accumulator = np.zeros((h, w), dtype=np.float32)

    threshold = cfg.get("night_brightness_threshold", 120)
    min_area = cfg.get("min_lamp_area", 30)
    max_area = cfg.get("max_lamp_area", 2000)
    min_circularity = cfg.get("min_circularity", 0.3)

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blobs = detect_bright_blobs(gray, threshold, min_area, max_area, min_circularity)
        for b in blobs:
            accumulator += (b["mask"] > 0).astype(np.float32)

    n_frames = len(frames)
    ratio = cfg.get("consensus_ratio", 0.6)
    consensus_mask = (accumulator / n_frames >= ratio).astype(np.uint8) * 255

    contours, _ = cv2.findContours(consensus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lamps = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = int(x + w / 2), int(y + h / 2)

        mask = np.zeros(consensus_mask.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        lamps.append({
            "center": (cx, cy),
            "bbox": (x, y, w, h),
            "area": int(area),
            "mask": mask,
        })
    return lamps


def load_reference_masks(camera_id: str, calibration_dir: Path) -> Optional[List[dict]]:
    """加载标定阶段生成的参考蒙板。"""
    ref_dir = calibration_dir / camera_id
    if not ref_dir.exists():
        return None

    refs = []
    for ref_file in sorted(ref_dir.glob("*_ref.json")):
        with open(ref_file, "r", encoding="utf-8") as f:
            ref = json.load(f)
        mask_path = ref.get("mask_path")
        if mask_path:
            full_path = calibration_dir.parent / mask_path
            mask = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                ref["mask"] = mask
                refs.append(ref)
    return refs if refs else None


def extract_daily_mask_from_reference(frame: np.ndarray, ref_mask: np.ndarray,
                                      is_night: bool, cfg: dict) -> np.ndarray:
    """
    基于参考蒙板提取当前帧的灯头区域。
    夜间用亮度阈值，白天用边缘/结构响应。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if is_night:
        threshold = cfg.get("night_brightness_threshold", 120)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    else:
        # 白天：在参考 ROI 内用自适应阈值提取灯体结构
        roi = cv2.bitwise_and(gray, gray, mask=ref_mask)
        binary = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        # 与参考蒙板取交集，避免背景干扰
        binary = cv2.bitwise_and(binary, ref_mask)

    return cv2.bitwise_and(binary, ref_mask)


def generate_masks(video_path: str, output_dir: Path, cfg: dict,
                   timestamp: datetime = None,
                   calibration_dir: Path = None) -> dict:
    """
    从视频中生成路灯蒙板。

    如果有参考蒙板（calibration_dir），则基于参考 ROI 提取当前蒙板；
    否则进入自动发现模式（仅夜间有效）。
    """
    camera_id = cfg.get("camera_id", "unknown")
    output_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now()

    is_night = is_night_time(timestamp)
    mode = "night" if is_night else "day"

    frames = sample_frames(video_path, num_samples=cfg.get("samples_per_hour", 6))
    if not frames:
        return {"camera_id": camera_id, "timestamp": timestamp.isoformat(),
                "mode": "failed", "lamps": []}

    reference_masks = None
    if calibration_dir is not None:
        reference_masks = load_reference_masks(camera_id, calibration_dir)

    if reference_masks is None:
        # 自动发现模式：适合首次标定
        if not is_night:
            return {"camera_id": camera_id, "timestamp": timestamp.isoformat(),
                    "mode": "day_no_reference", "lamps": []}
        lamps = multi_frame_consensus(frames, cfg)
    else:
        # 基于参考蒙板提取日常蒙板
        lamps = []
        for idx, ref in enumerate(reference_masks):
            ref_mask = ref["mask"]
            daily_masks = []
            for frame in frames:
                dm = extract_daily_mask_from_reference(frame, ref_mask, is_night, cfg)
                daily_masks.append(dm)

            # 多帧投票
            consensus = (np.mean(daily_masks, axis=0) >= 127).astype(np.uint8) * 255

            x, y, w, h = cv2.boundingRect(ref_mask)
            cx, cy = ref.get("center", (int(x + w / 2), int(y + h / 2)))
            area = int(cv2.countNonZero(consensus))

            lamp_id = ref.get("lamp_id", f"{camera_id}_lamp_{idx:03d}")
            mask_path = output_dir / f"{lamp_id}.png"
            cv2.imwrite(str(mask_path), consensus)

            lamps.append({
                "lamp_id": lamp_id,
                "center": (cx, cy),
                "bbox": (x, y, w, h),
                "area": area,
                "mask": consensus,
            })

    lamp_records = []
    for idx, lamp in enumerate(lamps):
        lamp_id = lamp.get("lamp_id", f"{camera_id}_lamp_{idx:03d}")
        mask_path = output_dir / f"{lamp_id}.png"
        if not mask_path.exists():
            cv2.imwrite(str(mask_path), lamp["mask"])

        record = {
            "lamp_id": lamp_id,
            "center": lamp["center"],
            "bbox": lamp["bbox"],
            "area": lamp["area"],
            "mask_path": str(mask_path.relative_to(output_dir.parent.parent)),
        }
        lamp_records.append(record)

    meta = {
        "camera_id": camera_id,
        "timestamp": timestamp.isoformat(),
        "mode": mode,
        "num_lamps": len(lamp_records),
        "lamps": lamp_records,
    }
    meta_path = output_dir / "masks.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta
