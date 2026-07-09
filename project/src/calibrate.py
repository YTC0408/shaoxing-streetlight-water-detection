"""
标定模块：用历史正常视频为每个摄像头生成基准蒙板和统计信息。

建议用夜间正常亮灯的视频运行一次，后续 daily_pipeline 会与此基准对比。
"""

import json
from pathlib import Path

import cv2
import numpy as np

from src.preprocess import generate_masks, sample_frames
from src.detect_damage import compute_mask_brightness, compute_mask_shape
from src.time_utils import parse_timestamp_from_path
from src.utils import load_config, get_camera_config


def calibrate_camera(camera_id: str, video_path: str, calibration_dir: Path,
                     cfg: dict) -> dict:
    """
    对单个摄像头进行标定，生成基准 mask 和亮度/形状统计。
    """
    calibration_dir.mkdir(parents=True, exist_ok=True)

    timestamp = parse_timestamp_from_path(video_path)
    output_dir = calibration_dir / camera_id / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    masks_meta = generate_masks(video_path, output_dir, cfg, timestamp)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    refs = {}
    for lamp in masks_meta.get("lamps", []):
        lamp_id = lamp["lamp_id"]
        mask_path = Path(video_path).parent.parent.parent / lamp["mask_path"]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        brightness = compute_mask_brightness(frame, mask) if ret else 0
        shape = compute_mask_shape(mask)

        ref = {
            "lamp_id": lamp_id,
            "camera_id": camera_id,
            "center": shape["center"],
            "area": shape["area"],
            "circularity": shape["circularity"],
            "mean_brightness": brightness,
            "mask_path": lamp["mask_path"],
        }

        ref_path = calibration_dir / camera_id / f"{lamp_id}_ref.json"
        with open(ref_path, "w", encoding="utf-8") as f:
            json.dump(ref, f, ensure_ascii=False, indent=2)
        refs[lamp_id] = ref

    return {"camera_id": camera_id, "num_references": len(refs), "references": refs}


def calibrate_all(config_path: Path, calibration_dir: Path,
                  video_mapping: dict) -> None:
    """
    批量标定多个摄像头。

    Args:
        video_mapping: {camera_id: video_path}
    """
    config = load_config(config_path)
    for camera_id, video_path in video_mapping.items():
        cfg = get_camera_config(config, camera_id)
        print(f"[标定] {camera_id}: {video_path}")
        result = calibrate_camera(camera_id, video_path, calibration_dir, cfg)
        print(f"  生成 {result['num_references']} 个基准 mask")
