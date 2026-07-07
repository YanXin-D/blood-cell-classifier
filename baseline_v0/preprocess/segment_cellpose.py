"""Cellpose 分割：全视野 phase 图 -> 单细胞 crop + metadata.csv。"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from preprocess.cellpose_model import create_segmentation_model, segment_image
from utils.config import load_config
from utils.constants import IMAGE_EXTENSIONS, SEGMENT_BATCH_PATHS
from utils.console import configure_utf8_stdio
from utils.metadata_io import build_metadata_row
from utils.paths import PROJECT_ROOT, resolve_path, setup_import_path

setup_import_path()
configure_utf8_stdio()


def get_segment_params(cfg: dict, batch: str | None) -> dict:
    seg = cfg["segment"]
    if batch and batch in seg and isinstance(seg[batch], dict):
        params = dict(seg[batch])
    elif batch and batch in SEGMENT_BATCH_PATHS:
        raise ValueError(f"config.yaml 缺少 segment.{batch} 参数")
    else:
        params = {
            k: seg[k]
            for k in (
                "diameter",
                "min_size",
                "crop_pad",
                "cellprob_threshold",
                "flow_threshold",
            )
            if k in seg
        }
    params["gpu"] = seg.get("gpu", True)
    params.setdefault("pretrained_model", "cyto3")
    return params


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def mask_bbox(mask: np.ndarray, cell_id: int, pad: int, h: int, w: int):
    ys, xs = np.where(mask == cell_id)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, h)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w)
    return x0, y0, x1, y1, float(xs.mean()), float(ys.mean()), int((mask == cell_id).sum())


def extract_crops(img: np.ndarray, mask: np.ndarray, pad: int):
    h, w = img.shape[:2]
    crops = []
    for cell_id in np.unique(mask):
        if cell_id == 0:
            continue
        x0, y0, x1, y1, cx, cy, area = mask_bbox(mask, int(cell_id), pad, h, w)
        crops.append(
            {
                "crop": img[y0:y1, x0:x1],
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "center_x": cx, "center_y": cy, "area": area,
            }
        )
    return crops


def save_qc_overlay(img: np.ndarray, mask: np.ndarray, out_path: Path):
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img, cmap="gray")
    ax.imshow(np.ma.masked_where(mask == 0, mask), cmap="nipy_spectral", alpha=0.35)
    ax.set_title(f"{out_path.stem} | cells={len(np.unique(mask)) - 1}")
    ax.axis("off")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def clear_output_dir(crops_dir: Path, qc_dir: Path) -> None:
    for folder in (crops_dir, qc_dir):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cellpose 分割 -> crop + metadata.csv")
    parser.add_argument("--batch", choices=list(SEGMENT_BATCH_PATHS), default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--crops-dir", type=Path, default=None)
    parser.add_argument("--qc-dir", type=Path, default=None)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if not args.batch:
        raise SystemExit("请指定 --batch fetal_culture 或 --batch blood_cell")

    cfg = load_config()
    seg_params = get_segment_params(cfg, args.batch)
    paths = SEGMENT_BATCH_PATHS[args.batch]

    input_dir = (args.input_dir or resolve_path(paths["input_dir"])).resolve()
    crops_dir = (args.crops_dir or resolve_path(paths["crops_dir"])).resolve()
    qc_dir = (args.qc_dir or resolve_path(paths["qc_dir"])).resolve()

    if args.clean:
        clear_output_dir(crops_dir, qc_dir)
    else:
        crops_dir.mkdir(parents=True, exist_ok=True)
        qc_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    if not images:
        raise SystemExit(f"未在 {input_dir} 找到图像，请先把原始视野图放入该目录")

    print(f"批次: {args.batch}")
    print(f"输入: {input_dir} ({len(images)} 张)")
    print(f"输出 crop: {crops_dir}")

    model = create_segmentation_model(
        gpu=seg_params["gpu"],
        pretrained_model=seg_params.get("pretrained_model", "cyto3"),
    )
    print(f"已加载模型: {model.name} ({model.backend})")
    all_rows = []
    total_cells = 0

    for img_path in images:
        img = np.array(Image.open(img_path).convert("L"))
        masks = segment_image(model, img, seg_params)
        n_cells = len(np.unique(masks)) - 1
        print(f"  {img_path.name}: {n_cells} 个细胞")

        save_qc_overlay(img, masks, qc_dir / f"{img_path.stem}_mask.png")

        image_id = img_path.stem
        for i, item in enumerate(extract_crops(img, masks, seg_params["crop_pad"]), start=1):
            crop_name = f"{img_path.stem}_cell_{i:03d}.png"
            crop_path = crops_dir / crop_name
            Image.fromarray(item["crop"]).save(crop_path)

            mean_int = float(item["crop"].mean()) if item["crop"].size else 0.0
            cell_id = f"{img_path.stem}_cell_{i:03d}"
            all_rows.append(
                build_metadata_row(
                    cell_id=cell_id,
                    image_id=image_id,
                    crop_path=crop_path,
                    phase_path=img_path,
                    center_x=item["center_x"],
                    center_y=item["center_y"],
                    x0=item["x0"],
                    y0=item["y0"],
                    x1=item["x1"],
                    y1=item["y1"],
                    area=item["area"],
                    mean_intensity=mean_int,
                    project_root=PROJECT_ROOT,
                )
            )
            total_cells += 1

    meta_path = crops_dir / "metadata.csv"
    if all_rows:
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"\n完成: 共提取 {total_cells} 个 crop")
    print(f"metadata -> {meta_path}")
    print(f"QC 叠加图 -> {qc_dir}")
    print("\n下一步: 用 Cell Inspector 标注")
    print(f"  metadata 路径: {meta_path}")


if __name__ == "__main__":
    main()
