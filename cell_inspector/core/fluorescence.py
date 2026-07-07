"""荧光图增强与伪彩色。"""

from __future__ import annotations

import numpy as np


def adjust_intensity(
    image: np.ndarray,
    gamma: float = 1.0,
    contrast: float = 1.0,
    brightness: float = 0.0,
) -> np.ndarray:
    """gamma / contrast / brightness 增强。"""
    img = image.astype(np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img / 255.0
    img = np.power(np.clip(img, 0, 1), 1.0 / max(gamma, 0.01))
    img = (img - 0.5) * contrast + 0.5 + brightness
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def apply_pseudocolor(image: np.ndarray, cmap: str = "green") -> np.ndarray:
    """单通道灰度转伪彩色 RGB。"""
    gray = image.astype(np.float32)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    gray = gray / max(gray.max(), 1.0)

    if cmap == "green":
        r = (gray * 80).astype(np.uint8)
        g = (gray * 255).astype(np.uint8)
        b = (gray * 80).astype(np.uint8)
    elif cmap == "red":
        r = (gray * 255).astype(np.uint8)
        g = (gray * 60).astype(np.uint8)
        b = (gray * 60).astype(np.uint8)
    elif cmap == "hot":
        r = np.clip(gray * 3, 0, 1)
        g = np.clip(gray * 3 - 1, 0, 1)
        b = np.clip(gray * 3 - 2, 0, 1)
        r, g, b = (r * 255).astype(np.uint8), (g * 255).astype(np.uint8), (b * 255).astype(np.uint8)
    else:
        v = (gray * 255).astype(np.uint8)
        return np.stack([v, v, v], axis=-1)

    return np.stack([r, g, b], axis=-1)


def compute_fluorescence_stats(
    fluor: np.ndarray,
    center_x: float,
    center_y: float,
    signal_size: int = 128,
    bg_margin: int = 16,
) -> dict[str, float]:
    """计算 128×128 ROI 内的荧光统计量与 SBR。"""
    from .roi import extract_roi

    if fluor.ndim == 3:
        fluor_gray = fluor.mean(axis=2).astype(np.float32)
    else:
        fluor_gray = fluor.astype(np.float32)

    signal_roi, _, _ = extract_roi(fluor_gray, center_x, center_y, signal_size)
    fluo_mean = float(signal_roi.mean())
    fluo_max = float(signal_roi.max())

    # 背景：signal ROI 外一圈 ring
    half = signal_size // 2
    cx, cy = int(round(center_x)), int(round(center_y))
    h, w = fluor_gray.shape
    x1 = max(0, cx - half - bg_margin)
    y1 = max(0, cy - half - bg_margin)
    x2 = min(w, cx + half + bg_margin)
    y2 = min(h, cy + half + bg_margin)
    outer = fluor_gray[y1:y2, x1:x2]
    inner_x1 = max(0, cx - half - x1)
    inner_y1 = max(0, cy - half - y1)
    inner_x2 = inner_x1 + signal_size
    inner_y2 = inner_y1 + signal_size
    mask = np.ones(outer.shape, dtype=bool)
    mask[inner_y1:inner_y2, inner_x1:inner_x2] = False
    bg_vals = outer[mask]
    bg_mean = float(bg_vals.mean()) if bg_vals.size else 1.0
    sbr = fluo_mean / max(bg_mean, 1e-6)

    return {
        "fluo_mean": fluo_mean,
        "fluo_max": fluo_max,
        "sbr": sbr,
        "signal_roi": signal_roi,
    }
