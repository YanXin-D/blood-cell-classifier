"""config.yaml 加载与常用路径访问。"""

from __future__ import annotations

from pathlib import Path

import yaml

from utils.paths import PROJECT_ROOT, resolve_path


def load_config(config_path: Path | None = None) -> dict:
    path = config_path or PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_detection_cfg(cfg: dict) -> dict:
    return cfg.get("detection", {})


def get_train_cfg(cfg: dict) -> dict:
    return cfg.get("train", {})


def get_crops_dir(cfg: dict) -> Path:
    """推理/排序使用的 crop 目录（detection.crops_dir）。"""
    det = get_detection_cfg(cfg)
    return resolve_path(det.get("crops_dir", "data/all_cells_large"))


def get_raw_dir(cfg: dict) -> Path:
    """原视野图目录（detection.raw_dir）。"""
    det = get_detection_cfg(cfg)
    return resolve_path(det.get("raw_dir", "data/raw/all_cells"))


def get_output_dir(cfg: dict) -> Path:
    return resolve_path(cfg.get("output_dir", "outputs"))


def get_checkpoint_path(cfg: dict, name: str = "best_model.pt") -> Path:
    return resolve_path(cfg.get("save_dir", "checkpoints")) / name
