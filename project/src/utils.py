"""
通用工具函数。
"""

import logging
import sys
from pathlib import Path

import yaml


def setup_logging(log_dir: Path, name: str = "pipeline") -> logging.Logger:
    """配置日志输出到文件和控制台。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger


def load_config(config_path: Path) -> dict:
    """加载 YAML 配置文件。"""
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(config: dict, config_path: Path) -> None:
    """保存 YAML 配置文件。"""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


def get_camera_defaults(config: dict) -> dict:
    """获取默认配置。"""
    return config.get("defaults", {})


def get_camera_config(config: dict, camera_id: str) -> dict:
    """合并默认配置与单个摄像头配置。"""
    defaults = get_camera_defaults(config).copy()
    camera_cfg = config.get("cameras", {}).get(camera_id, {})
    defaults.update(camera_cfg)
    defaults["camera_id"] = camera_id
    return defaults
