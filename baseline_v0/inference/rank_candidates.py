"""按胎儿细胞概率降序排序，导出 Top 候选并复制图片。"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

from utils.config import get_crops_dir, get_output_dir, load_config
from utils.constants import IMAGE_EXTENSIONS
from utils.paths import setup_import_path

setup_import_path()


def find_image(crops_dir: Path, cell_name: str) -> Path | None:
    direct = crops_dir / cell_name
    if direct.exists():
        return direct
    stem = Path(cell_name).stem
    for ext in IMAGE_EXTENSIONS:
        candidate = crops_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def export_top(df: pd.DataFrame, n: int, output_dir: Path, crops_dir: Path):
    top = df.head(n)
    csv_path = output_dir / f"top{n}.csv"
    top.to_csv(csv_path, index=False)

    img_dir = output_dir / f"top{n}_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    copied, missing = 0, 0
    for cell_name in top["cell_name"]:
        src = find_image(crops_dir, cell_name)
        if src is None:
            missing += 1
            continue
        dst = img_dir / src.name
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            shutil.copy2(src, dst)
        copied += 1

    print(f"top{n}: {csv_path} | 复制 {copied} 张 | 缺失 {missing} 张")


def main():
    cfg = load_config()
    output_dir = get_output_dir(cfg)
    pred_path = output_dir / "candidates.csv"
    if not pred_path.exists():
        pred_path = output_dir / "predictions.csv"
    crops_dir = get_crops_dir(cfg)

    if not pred_path.exists():
        raise FileNotFoundError(
            "未找到 candidates.csv 或 predictions.csv，请先运行 detect_candidates.py"
        )

    df = pd.read_csv(pred_path)
    if "probability" in df.columns:
        df = df.sort_values(["probability", "area"], ascending=[False, False])
    else:
        df = df.sort_values("area", ascending=False)
    df = df.reset_index(drop=True)

    if "cell_name" not in df.columns and "cell_id" in df.columns:
        df["cell_name"] = df["cell_id"] + ".png"

    for n in (100, 500, 1000):
        export_top(df, n, output_dir, crops_dir)


if __name__ == "__main__":
    main()
