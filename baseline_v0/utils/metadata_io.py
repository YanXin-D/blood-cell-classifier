"""metadata.csv 读写与索引。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.constants import TRAINING_SOURCES
from utils.paths import PROJECT_ROOT, resolve_path, to_relative_path


def build_metadata_row(
    *,
    cell_id: str,
    image_id: str,
    crop_path: Path,
    phase_path: Path,
    center_x: float,
    center_y: float,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    area: int,
    mean_intensity: float,
    project_root: Path | None = None,
) -> dict:
    root = project_root or PROJECT_ROOT
    return {
        "cell_id": cell_id,
        "image_id": image_id,
        "crop_path": to_relative_path(crop_path, root),
        "bf_path": to_relative_path(phase_path, root),
        "phase_path": to_relative_path(phase_path, root),
        "fluor_path": to_relative_path(phase_path, root),
        "mask_id": f"mask_{cell_id}",
        "center_x": round(center_x, 2),
        "center_y": round(center_y, 2),
        "bbox_x1": x0,
        "bbox_y1": y0,
        "bbox_x2": x1 - 1,
        "bbox_y2": y1 - 1,
        "area": area,
        "perimeter": 0.0,
        "eccentricity": 0.0,
        "solidity": 1.0,
        "mean_intensity": round(mean_intensity, 4),
        "phase_mean": round(mean_intensity, 4),
        "label": -1,
    }


def load_training_metadata_index(project_root: Path | None = None) -> dict[str, pd.Series]:
    """加载所有训练源 metadata，按 cell_id 索引。"""
    root = project_root or PROJECT_ROOT
    index: dict[str, pd.Series] = {}
    for spec in TRAINING_SOURCES.values():
        path = resolve_path(spec["metadata"], root)
        if not path.exists():
            continue
        for _, row in pd.read_csv(path).iterrows():
            index[str(row["cell_id"])] = row
    return index
