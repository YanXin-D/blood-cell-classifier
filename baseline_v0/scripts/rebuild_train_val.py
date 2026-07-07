"""从 crops metadata + labels 按 image_id 重建 train/val（避免同图泄漏）。"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

from utils.config import get_output_dir, get_train_cfg, load_config
from utils.constants import LABEL_TO_DIR, TRAINING_SOURCES
from utils.console import configure_utf8_stdio
from utils.hard_negatives import import_hard_negatives
from utils.paths import PROJECT_ROOT, setup_import_path
from utils.split import (
    clear_split_dirs,
    copy_labeled_rows,
    load_labeled_rows,
    split_image_ids,
)

setup_import_path()
configure_utf8_stdio()


def rebuild(
    baseline: Path,
    val_ratio: float,
    seed: int,
    sources: dict,
    include_hard_negatives: bool,
    hard_negatives_max: int,
) -> dict:
    rng = random.Random(seed)
    data_root = baseline / "data"
    clear_split_dirs(data_root)
    stats = {"train": {0: 0, 1: 0}, "val": {0: 0, 1: 0}}
    split_report = []

    for source_name, spec in sources.items():
        meta_path = baseline / spec["metadata"]
        labels_path = baseline / spec["labels"]
        target_class = spec["target_class"]
        class_name = LABEL_TO_DIR[target_class]

        rows = load_labeled_rows(meta_path, labels_path, target_class)
        val_ids, train_ids = split_image_ids(rows["image_id"].tolist(), val_ratio, rng)
        split_report.append(
            {
                "source": source_name,
                "class": class_name,
                "train_images": len(train_ids),
                "val_images": len(val_ids),
                "cells": len(rows),
            }
        )

        for split_name, id_set in (("train", train_ids), ("val", val_ids)):
            subset = rows[rows["image_id"].isin(id_set)]
            n = copy_labeled_rows(subset, split_name, target_class, data_root, project_root=baseline)
            stats[split_name][target_class] += n

    if include_hard_negatives:
        added = import_hard_negatives(
            data_root / "train" / "blood",
            max_samples=hard_negatives_max,
            project_root=baseline,
        )
        stats["train"][0] += added
        if added:
            split_report.append(
                {"source": "all_cells_large", "class": "blood", "hard_negatives": added}
            )

    report_path = get_output_dir(load_config()) / "train_val_split_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(split_report).to_csv(report_path, index=False)

    return {"stats": stats, "report_path": report_path}


def main():
    parser = argparse.ArgumentParser(description="按 image_id 重建 baseline train/val")
    parser.add_argument("--baseline-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-hard-negatives", action="store_true")
    parser.add_argument("--hard-negatives-max", type=int, default=2000)
    args = parser.parse_args()

    cfg = load_config()
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    train_cfg = get_train_cfg(cfg)
    hard_max = int(train_cfg.get("hard_negatives_max", args.hard_negatives_max))

    result = rebuild(
        baseline=args.baseline_root.resolve(),
        val_ratio=args.val_ratio,
        seed=seed,
        sources=TRAINING_SOURCES,
        include_hard_negatives=not args.no_hard_negatives,
        hard_negatives_max=hard_max,
    )
    stats = result["stats"]
    print("重建完成（按 image_id 划分）:")
    print(
        f"  train: blood={stats['train'][0]}, fetal={stats['train'][1]} "
        f"(total={stats['train'][0] + stats['train'][1]})"
    )
    print(
        f"  val:   blood={stats['val'][0]}, fetal={stats['val'][1]} "
        f"(total={stats['val'][0] + stats['val'][1]})"
    )
    print(f"  报告: {result['report_path']}")


if __name__ == "__main__":
    main()
