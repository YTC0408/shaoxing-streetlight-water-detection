"""
每日流水线：为所有摄像头生成蒙板并检测损伤。

支持按日期批量处理，可并行加速。
"""

import argparse
import csv
import json
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

from src.calibrate import calibrate_camera
from src.detect_damage import detect_damage
from src.preprocess import generate_masks
from src.time_utils import parse_timestamp_from_path
from src.utils import load_config, get_camera_config, get_camera_defaults, setup_logging


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "cameras.yaml"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"


def discover_videos(raw_dir: Path, date_str: str) -> dict:
    """
    自动发现某日期下的所有摄像头视频。

    约定目录结构：data/raw/<camera_id>/<date>/<video>.mp4
    或 data/raw/<camera_id>/<video_with_date>.mp4
    """
    mapping = {}
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    for cam_dir in raw_dir.iterdir():
        if not cam_dir.is_dir():
            continue
        camera_id = cam_dir.name

        # 查找日期子目录或直接匹配文件名
        candidates = []
        date_dir = cam_dir / date_str
        if date_dir.exists():
            candidates.extend(date_dir.glob("*.mp4"))
            candidates.extend(date_dir.glob("*.avi"))

        for video in cam_dir.glob("*.mp4"):
            try:
                vt = parse_timestamp_from_path(str(video))
                if vt.date() == date_obj.date():
                    candidates.append(video)
            except Exception:
                pass

        if candidates:
            mapping[camera_id] = str(sorted(candidates)[0])

    return mapping


def process_camera(args) -> list:
    """
    单个摄像头的处理函数，供多进程调用。

    Args:
        args: (camera_id, video_path, config_path, data_dir, date_str)
    """
    camera_id, video_path, config_path, data_dir, date_str = args
    config = load_config(config_path)
    cfg = get_camera_config(config, camera_id)

    calibration_dir = data_dir / "calibration"
    masks_dir = data_dir / "masks" / date_str / camera_id
    reports_dir = data_dir / "reports" / date_str
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = parse_timestamp_from_path(video_path)

    try:
        # 生成当日蒙板（优先使用标定参考蒙板）
        masks_meta = generate_masks(video_path, masks_dir, cfg, timestamp,
                                    calibration_dir=calibration_dir)

        # 检测损伤
        results = detect_damage(video_path, masks_meta, masks_dir,
                                calibration_dir, cfg)

        # 保存单摄像头报告
        report_path = reports_dir / f"{camera_id}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return results
    except Exception as e:
        return [{"camera_id": camera_id, "error": str(e)}]


def merge_reports(reports_dir: Path, date_str: str) -> Path:
    """合并所有摄像头报告为总表。"""
    all_results = []
    for report_file in (reports_dir / date_str).glob("*.json"):
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                all_results.extend(data)

    # JSON 总表
    summary_json = reports_dir / date_str / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # CSV 总表
    if all_results:
        keys = all_results[0].keys()
        summary_csv = reports_dir / date_str / "summary.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)

    return summary_json


def main():
    parser = argparse.ArgumentParser(description="路灯损伤每日检测流水线")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"),
                        help="处理日期，格式 YYYY-MM-DD")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="摄像头配置文件路径")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help="数据根目录")
    parser.add_argument("--workers", type=int, default=4,
                        help="并行进程数，建议根据 CPU 核心数设置")
    parser.add_argument("--calibrate", action="store_true",
                        help="标定模式：用当日视频生成基准 mask")
    args = parser.parse_args()

    logger = setup_logging(DEFAULT_LOG_DIR, f"daily_pipeline_{args.date}")
    config_path = Path(args.config)
    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"

    if not config_path.exists():
        # 自动生成默认配置
        config_path.parent.mkdir(parents=True, exist_ok=True)
        from src.utils import save_config
        save_config({"defaults": get_camera_defaults({}), "cameras": {}}, config_path)
        logger.info(f"已生成默认配置文件: {config_path}")

    video_mapping = discover_videos(raw_dir, args.date)
    logger.info(f"发现 {len(video_mapping)} 个摄像头视频待处理")

    if not video_mapping:
        logger.warning("未找到任何视频，请检查 data/raw/<camera_id>/<date>/ 目录结构")
        return

    if args.calibrate:
        logger.info("进入标定模式")
        config = load_config(config_path)
        for camera_id, video_path in video_mapping.items():
            cfg = get_camera_config(config, camera_id)
            result = calibrate_camera(camera_id, video_path,
                                      data_dir / "calibration", cfg)
            logger.info(f"{camera_id}: 标定完成，{result['num_references']} 个灯")
        return

    # 准备多进程参数
    tasks = [
        (camera_id, video_path, str(config_path), data_dir, args.date)
        for camera_id, video_path in video_mapping.items()
    ]

    logger.info(f"开始处理，并行数: {args.workers}")
    with Pool(processes=args.workers) as pool:
        pool.map(process_camera, tasks)

    summary_path = merge_reports(data_dir / "reports", args.date)
    logger.info(f"日报已生成: {summary_path}")


if __name__ == "__main__":
    main()
