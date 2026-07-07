"""生成 Cell Inspector 示例 metadata 与合成图像。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sample_data"
N_CELLS = 30
IMG_W, IMG_H = 612, 512


def make_field(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(30, 90, (IMG_H, IMG_W), dtype=np.uint8)
    return base


def add_cell(img: np.ndarray, cx: int, cy: int, radius: int, intensity: int) -> dict:
    h, w = img.shape
    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    img[mask] = np.clip(img[mask].astype(int) + intensity, 0, 255).astype(np.uint8)
    ys, xs = np.where(mask)
    area = int(mask.sum())
    perimeter = 2 * np.pi * radius
    return {
        "center_x": cx,
        "center_y": cy,
        "bbox_x1": int(xs.min()),
        "bbox_y1": int(ys.min()),
        "bbox_x2": int(xs.max()),
        "bbox_y2": int(ys.max()),
        "area": area,
        "perimeter": perimeter,
        "eccentricity": 0.15,
        "solidity": 0.92,
        "mean_intensity": float(img[mask].mean()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    crops = OUT / "crops"
    bf_dir = OUT / "brightfield"
    phase_dir = OUT / "phase"
    fluor_dir = OUT / "fluorescence"
    for d in (crops, bf_dir, phase_dir, fluor_dir):
        d.mkdir(exist_ok=True)

    rows = []
    for i in range(N_CELLS):
        image_id = f"img_{i // 5:03d}"
        cell_id = f"cell_{i:05d}"
        seed = 1000 + i
        cx = 80 + (i * 37) % (IMG_W - 160)
        cy = 60 + (i * 53) % (IMG_H - 120)
        r = 12 + (i % 5) * 2

        bf = make_field(seed)
        phase = make_field(seed + 1)
        fluor = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        fluor_bg = make_field(seed + 2) // 4

        stats_bf = add_cell(bf, cx, cy, r, 40)
        stats_ph = add_cell(phase, cx, cy, r, 35)
        stats_fl = add_cell(fluor, cx, cy, r, 180 if i % 3 == 0 else 60)
        fluor = np.clip(fluor.astype(int) + fluor_bg, 0, 255).astype(np.uint8)

        bf_path = bf_dir / f"{image_id}.png"
        phase_path = phase_dir / f"{image_id}.png"
        fluor_path = fluor_dir / f"{image_id}.png"
        if not bf_path.exists():
            Image.fromarray(bf).save(bf_path)
            Image.fromarray(phase).save(phase_path)
            Image.fromarray(fluor).save(fluor_path)

        x1, y1, x2, y2 = stats_bf["bbox_x1"], stats_bf["bbox_y1"], stats_bf["bbox_x2"], stats_bf["bbox_y2"]
        pad = 4
        crop_arr = bf[max(0, y1 - pad) : y2 + pad + 1, max(0, x1 - pad) : x2 + pad + 1]
        crop_path = crops / f"{cell_id}.png"
        Image.fromarray(crop_arr).save(crop_path)

        rows.append(
            {
                "cell_id": cell_id,
                "image_id": image_id,
                "crop_path": str(crop_path.resolve()),
                "bf_path": str(bf_path.resolve()),
                "phase_path": str(phase_path.resolve()),
                "fluor_path": str(fluor_path.resolve()),
                "mask_id": f"mask_{i}",
                "center_x": cx,
                "center_y": cy,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "area": stats_bf["area"],
                "perimeter": stats_bf["perimeter"],
                "eccentricity": stats_bf["eccentricity"],
                "solidity": stats_bf["solidity"],
                "mean_intensity": stats_bf["mean_intensity"],
                "phase_mean": stats_ph["mean_intensity"],
                "label": -1,
            }
        )

    meta_path = OUT / "metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"已生成 {N_CELLS} 个细胞 -> {meta_path}")


if __name__ == "__main__":
    main()
