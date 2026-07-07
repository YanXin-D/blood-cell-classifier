"""ROI 提取与原图定位 overlay。"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def extract_roi(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    roi_size: int,
) -> tuple[np.ndarray, int, int]:
    """以 (center_x, center_y) 为中心提取 roi_size×roi_size 区域。"""
    h, w = image.shape[:2]
    half = roi_size // 2
    cx, cy = int(round(center_x)), int(round(center_y))

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    roi = image[y1:y2, x1:x2].copy()

    # 边缘 padding 到目标尺寸
    pad_top = max(0, half - (cy - y1))
    pad_left = max(0, half - (cx - x1))
    pad_bottom = max(0, half - (y2 - cy))
    pad_right = max(0, half - (x2 - cx))

    if image.ndim == 2:
        roi = np.pad(
            roi,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
    else:
        roi = np.pad(
            roi,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    target = roi_size
    if roi.shape[0] > target:
        roi = roi[:target, :target]
    if roi.shape[1] > target:
        roi = roi[:, :target]
    if roi.shape[0] < target or roi.shape[1] < target:
        if roi.ndim == 2:
            padded = np.zeros((target, target), dtype=roi.dtype)
        else:
            padded = np.zeros((target, target, roi.shape[2]), dtype=roi.dtype)
        padded[: roi.shape[0], : roi.shape[1]] = roi
        roi = padded

    return roi, x1, y1


def circle_points(
    cx: float,
    cy: float,
    radius: float,
    n: int = 64,
) -> tuple[list[float], list[float]]:
    """生成圆周线点（用于 plotly scatter）。"""
    xs, ys = [], []
    for i in range(n + 1):
        t = 2 * math.pi * i / n
        xs.append(cx + radius * math.cos(t))
        ys.append(cy + radius * math.sin(t))
    return xs, ys


def roi_local_coords(
    center_x: float,
    center_y: float,
    roi_x0: int,
    roi_y0: int,
    bbox_x1: Optional[float] = None,
    bbox_y1: Optional[float] = None,
    bbox_x2: Optional[float] = None,
    bbox_y2: Optional[float] = None,
) -> dict:
    """将全图坐标转换为 ROI 局部坐标。"""
    lx = center_x - roi_x0
    ly = center_y - roi_y0
    out = {"center_x": lx, "center_y": ly}
    if bbox_x1 is not None:
        out["bbox"] = (
            bbox_x1 - roi_x0,
            bbox_y1 - roi_y0,
            bbox_x2 - roi_x0,
            bbox_y2 - roi_y0,
        )
    return out


def upscale_crop(
    crop: np.ndarray,
    scale: int = 4,
    method: str = "nearest",
) -> np.ndarray:
    """放大 crop 显示。"""
    from PIL import Image

    pil = Image.fromarray(crop)
    resample = Image.NEAREST if method == "nearest" else Image.BICUBIC
    new_size = (crop.shape[1] * scale, crop.shape[0] * scale)
    return np.array(pil.resize(new_size, resample=resample))
