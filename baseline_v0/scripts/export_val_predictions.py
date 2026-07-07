"""在验证集上逐样本预测，导出 CSV 与 bbox 可视化。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image, ImageDraw

from utils.config import get_output_dir, load_config
from utils.constants import LABEL_TO_DIR
from utils.console import configure_utf8_stdio
from utils.metadata_io import load_training_metadata_index
from utils.model_loader import build_val_loader, load_model_for_inference
from utils.paths import PROJECT_ROOT, resolve_crop_path, setup_import_path

setup_import_path()
configure_utf8_stdio()


def collect_predictions(model, loader, device):
    rows = []
    model.eval()
    with torch.no_grad():
        for images, labels, names in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.numpy()

            for i, name in enumerate(names):
                cell_id = Path(name).stem
                true_label = int(labels_np[i])
                pred_label = int(preds[i])
                rows.append(
                    {
                        "cell_id": cell_id,
                        "cell_name": name,
                        "true_label": true_label,
                        "true_class": LABEL_TO_DIR[true_label],
                        "pred_label": pred_label,
                        "pred_class": LABEL_TO_DIR[pred_label],
                        "prob_fetal": round(float(probs[i, 1]), 6),
                        "prob_blood": round(float(probs[i, 0]), 6),
                        "correct": int(pred_label == true_label),
                    }
                )
    return rows


def enrich_with_metadata(rows: list[dict], meta_index: dict[str, pd.Series]) -> list[dict]:
    for row in rows:
        meta = meta_index.get(row["cell_id"])
        if meta is None:
            row.update(
                {
                    "image_id": "",
                    "source": "",
                    "bbox_x1": "",
                    "bbox_y1": "",
                    "bbox_x2": "",
                    "bbox_y2": "",
                    "area": "",
                    "crop_path": "",
                    "phase_path": "",
                }
            )
            continue

        crop_path = resolve_crop_path(meta["crop_path"])
        phase_path = resolve_crop_path(meta["phase_path"])
        source = "fetal_culture" if "fetal_culture" in crop_path.as_posix() else "blood_cell"
        row.update(
            {
                "image_id": str(meta["image_id"]),
                "source": source,
                "bbox_x1": int(meta["bbox_x1"]),
                "bbox_y1": int(meta["bbox_y1"]),
                "bbox_x2": int(meta["bbox_x2"]),
                "bbox_y2": int(meta["bbox_y2"]),
                "area": int(meta["area"]),
                "crop_path": str(crop_path),
                "phase_path": str(phase_path),
            }
        )
    return rows


def bbox_color(row: dict) -> tuple[str, str]:
    if row["correct"]:
        return "#2ecc71", "TP/TN"
    if row["true_label"] == 1 and row["pred_label"] == 0:
        return "#e67e22", "FN"
    if row["true_label"] == 0 and row["pred_label"] == 1:
        return "#e74c3c", "FP"
    return "#95a5a6", "?"


def draw_crop_images(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        crop_path = Path(str(row["crop_path"]))
        if not crop_path.exists():
            crop_path = PROJECT_ROOT / "data" / "val" / row["true_class"] / row["cell_name"]
        if not crop_path.exists():
            continue

        color, _ = bbox_color(row.to_dict())
        with Image.open(crop_path) as img:
            img = img.convert("RGB")
            draw = ImageDraw.Draw(img)
            w, h = img.size
            draw.rectangle([0, 0, w - 1, h - 1], outline=color, width=3)
            text = (
                f"T:{row['true_class']} P:{row['pred_class']} "
                f"p={row['prob_fetal']:.2f}"
            )
            draw.rectangle([0, 0, w, 14], fill="black")
            draw.text((2, 1), text, fill=color)
            out_name = f"{row['cell_id']}_{row['pred_class']}_{int(row['correct'])}.png"
            img.save(out_dir / out_name)


def draw_fov_images(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    usable = df[df["phase_path"].astype(str).str.len() > 0].copy()
    if usable.empty:
        print("无 metadata，跳过 FOV bbox 可视化")
        return

    for image_id, group in usable.groupby("image_id"):
        phase_path = Path(str(group.iloc[0]["phase_path"]))
        if not phase_path.exists():
            continue

        img = Image.open(phase_path).convert("L")
        fig, ax = plt.subplots(figsize=(8, 6.8))
        ax.imshow(img, cmap="gray")

        n_correct = int(group["correct"].sum())
        for _, row in group.iterrows():
            x1, y1, x2, y2 = (
                int(row["bbox_x1"]),
                int(row["bbox_y1"]),
                int(row["bbox_x2"]),
                int(row["bbox_y2"]),
            )
            w, h = x2 - x1 + 1, y2 - y1 + 1
            color, err_tag = bbox_color(row.to_dict())
            ax.add_patch(
                patches.Rectangle(
                    (x1, y1), w, h, linewidth=2.0, edgecolor=color, facecolor="none"
                )
            )
            if not row["correct"]:
                label = (
                    f"{err_tag} T:{row['true_class'][0]} "
                    f"P:{row['pred_class'][0]} {row['prob_fetal']:.2f}"
                )
                ax.text(x1, max(y1 - 3, 0), label, color=color, fontsize=7, va="bottom")

        ax.set_title(
            f"{image_id} | val_cells={len(group)} correct={n_correct}/{len(group)}"
        )
        ax.axis("off")
        fig.savefig(out_dir / f"{image_id}_val_bbox.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="导出验证集逐样本预测 CSV 与 bbox 图")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-crops", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = args.output_dir or (get_output_dir(cfg) / "val_bbox")
    csv_path = get_output_dir(cfg) / "val_predictions.csv"
    output_root.mkdir(parents=True, exist_ok=True)

    loader, _ = build_val_loader(cfg, device)
    model, _ = load_model_for_inference(cfg, device)

    rows = collect_predictions(model, loader, device)
    rows = enrich_with_metadata(rows, load_training_metadata_index())
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    n_total = len(df)
    n_correct = int(df["correct"].sum())
    tp = int(((df["true_label"] == 1) & (df["pred_label"] == 1)).sum())
    fp = int(((df["true_label"] == 0) & (df["pred_label"] == 1)).sum())
    fn = int(((df["true_label"] == 1) & (df["pred_label"] == 0)).sum())
    tn = int(((df["true_label"] == 0) & (df["pred_label"] == 0)).sum())

    draw_fov_images(df, output_root)
    if not args.skip_crops:
        draw_crop_images(df, output_root / "crops")

    print(f"验证集预测 CSV -> {csv_path}")
    print(f"FOV bbox 图 -> {output_root}")
    if not args.skip_crops:
        print(f"逐 crop 图 -> {output_root / 'crops'}")
    print(
        f"样本 {n_total} | 正确 {n_correct} ({n_correct / n_total:.4f}) | "
        f"TP={tp} FP={fp} FN={fn} TN={tn}"
    )


if __name__ == "__main__":
    main()
