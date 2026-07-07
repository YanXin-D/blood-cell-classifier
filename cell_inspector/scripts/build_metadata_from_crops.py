"""从 crop 文件夹生成 Cell Inspector 所需的 metadata.csv。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image

BASELINE_ROOT = Path(__file__).resolve().parents[2] / "baseline_v0"
sys.path.insert(0, str(BASELINE_ROOT))

from utils.constants import IMAGE_EXTENSIONS
from utils.metadata_io import build_metadata_row
from utils.paths import PROJECT_ROOT


def parse_image_id(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "image":
        return f"{parts[0]}_{parts[1]}"
    return stem


def build_row(cell_id: str, crop_path: Path) -> dict:
    with Image.open(crop_path) as img:
        w, h = img.size
        arr = img.convert("L")
        mean_int = float(sum(arr.getdata())) / (w * h) if w * h else 0.0

    return build_metadata_row(
        cell_id=cell_id,
        image_id=parse_image_id(crop_path.stem),
        crop_path=crop_path,
        phase_path=crop_path,
        center_x=w // 2,
        center_y=h // 2,
        x0=0,
        y0=0,
        x1=w,
        y1=h,
        area=w * h,
        mean_intensity=mean_int,
        project_root=PROJECT_ROOT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="从 crop 目录生成 metadata.csv")
    parser.add_argument(
        "--crops-dir",
        type=Path,
        default=BASELINE_ROOT / "data" / "all_cells_large",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--batch", type=str, default="fetal_culture")
    args = parser.parse_args()

    crops_dir = args.crops_dir.resolve()
    out_path = (args.out or crops_dir / "metadata.csv").resolve()

    files = sorted(p for p in crops_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise SystemExit(f"未在 {crops_dir} 找到图像")

    rows = [build_row(f"cell_{p.stem}", p) for p in files]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"批次: {args.batch}")
    print(f"生成 {len(rows)} 条记录 -> {out_path}")
    print("启动标注: streamlit run cell_inspector/app.py")
    print(f"metadata 路径填: {out_path}")


if __name__ == "__main__":
    main()
