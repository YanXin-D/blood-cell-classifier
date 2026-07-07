"""在原视野图上绘制胎儿候选 bbox，每张原图一张可视化。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from utils.config import get_crops_dir, get_output_dir, get_raw_dir, load_config
from utils.constants import IMAGE_EXTENSIONS
from utils.inference import merge_predictions_with_metadata
from utils.paths import setup_import_path

setup_import_path()


def load_hits(cfg: dict, source: str) -> pd.DataFrame:
    output_dir = get_output_dir(cfg)

    if source == "candidates":
        path = output_dir / "candidates.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"未找到 {path}，请先运行 inference/detect_candidates.py"
            )
        return pd.read_csv(path)

    crops_dir = get_crops_dir(cfg)
    meta_path = crops_dir / "metadata.csv"
    pred_path = output_dir / "predictions.csv"
    if not meta_path.exists() or not pred_path.exists():
        raise FileNotFoundError("未找到 metadata 或 predictions.csv")
    meta = pd.read_csv(meta_path)
    pred = pd.read_csv(pred_path)
    return merge_predictions_with_metadata(meta, pred)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="原图 bbox 可视化")
    parser.add_argument(
        "--source",
        choices=("candidates", "predictions"),
        default="candidates",
        help="数据源",
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=None,
        help="仅 predictions 模式有效",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = get_raw_dir(cfg)
    out_dir = args.output_dir or (get_output_dir(cfg) / "fov_bbox")
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = load_hits(cfg, args.source)
    if args.source == "predictions":
        th = args.prob_threshold if args.prob_threshold is not None else 0.5
        merged = merged[merged["probability"] >= th]

    all_image_ids = sorted(
        p.stem for p in raw_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not all_image_ids:
        raise SystemExit(f"未在 {raw_dir} 找到原图")

    total_boxes = 0
    for image_id in all_image_ids:
        phase_path = raw_dir / f"{image_id}.png"
        if not phase_path.exists():
            continue

        img = Image.open(phase_path).convert("L")
        hits = merged[merged["image_id"] == image_id].sort_values(
            ["probability", "area"], ascending=[False, False]
        )
        total_boxes += len(hits)

        fig, ax = plt.subplots(figsize=(8, 6.8))
        ax.imshow(img, cmap="gray")
        for _, row in hits.iterrows():
            x1, y1, x2, y2 = (
                int(row["bbox_x1"]),
                int(row["bbox_y1"]),
                int(row["bbox_x2"]),
                int(row["bbox_y2"]),
            )
            w, h = x2 - x1 + 1, y2 - y1 + 1
            prob = float(row.get("probability", 1.0))
            color = plt.cm.RdYlGn(prob)
            ax.add_patch(
                patches.Rectangle(
                    (x1, y1), w, h, linewidth=1.8, edgecolor=color, facecolor="none"
                )
            )
            ax.text(
                x1, max(y1 - 3, 0), f"{prob:.2f} a{int(row['area'])}",
                color=color, fontsize=7, va="bottom",
            )

        ax.set_title(f"{image_id} | candidates={len(hits)}")
        ax.axis("off")
        out_path = out_dir / f"{image_id}_bbox.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out_path.name}: {len(hits)} boxes")

    print(f"\n完成: {len(all_image_ids)} 张 -> {out_dir}")
    print(f"共绘制 {total_boxes} 个 bbox")


if __name__ == "__main__":
    main()
