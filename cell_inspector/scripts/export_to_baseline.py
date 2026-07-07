"""将 Cell Inspector 标注结果导出到 baseline_v0 的 train/val 目录。"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import pandas as pd

BASELINE_ROOT = Path(__file__).resolve().parents[2] / "baseline_v0"
sys.path.insert(0, str(BASELINE_ROOT))

from utils.constants import CLASS_MAP
from utils.console import configure_utf8_stdio
from utils.paths import resolve_crop_path
from utils.split import assign_split_by_image_id, assign_split_random

configure_utf8_stdio()


def main() -> None:
    parser = argparse.ArgumentParser(description="导出标注到 baseline_v0 train/val")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-class", choices=["fetal", "blood"], default=None)
    parser.add_argument(
        "--split-by",
        choices=["random", "image_id"],
        default="image_id",
        help="image_id=按视野图划分（推荐）",
    )
    args = parser.parse_args()

    meta_path = args.metadata.resolve()
    labels_path = (args.labels or meta_path.parent / "labels.csv").resolve()
    baseline = args.baseline_root.resolve()
    rng = random.Random(args.seed)

    meta = pd.read_csv(meta_path)
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        label_map = dict(zip(labels["cell_id"], labels["label"]))
        meta["label"] = meta["cell_id"].map(label_map).fillna(meta["label"])

    labeled = meta[meta["label"].isin(CLASS_MAP.values())].copy()
    if labeled.empty:
        raise SystemExit("没有已标注细胞（label 需为 0 或 1）")

    stats = {"fetal": {"train": 0, "val": 0}, "blood": {"train": 0, "val": 0}}

    for label_val, class_name in ((1, "fetal"), (0, "blood")):
        if args.only_class and class_name != args.only_class:
            continue
        subset = labeled[labeled["label"] == label_val]
        if subset.empty:
            continue

        if args.split_by == "image_id":
            splits = assign_split_by_image_id(subset["image_id"], args.val_ratio, rng)
        else:
            splits = assign_split_random(subset.index, args.val_ratio, rng)

        for split_dir in (
            baseline / "data" / "train" / class_name,
            baseline / "data" / "val" / class_name,
        ):
            split_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in subset.iterrows():
            src = resolve_crop_path(row["crop_path"], baseline)
            split = splits.loc[idx]
            dst = baseline / "data" / split / class_name / src.name
            shutil.copy2(src, dst)
            stats[class_name][split] += 1

    print("导出完成:")
    for cls, counts in stats.items():
        print(f"  {cls}: train={counts['train']}, val={counts['val']}")
    print(f"目标目录: {baseline / 'data'}")


if __name__ == "__main__":
    main()
