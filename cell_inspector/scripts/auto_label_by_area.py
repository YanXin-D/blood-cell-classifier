"""按 area 阈值批量标注 metadata（血细胞 / 跳过）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "baseline_v0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASELINE))

from cell_inspector.core.metadata import LABEL_BLOOD, LABEL_SKIP, resolve_metadata_path
from utils.console import configure_utf8_stdio

configure_utf8_stdio()


def main() -> None:
    parser = argparse.ArgumentParser(description="按 area 阈值批量标注")
    parser.add_argument("--metadata", type=Path, required=True, help="metadata.csv 或所在目录")
    parser.add_argument("--threshold", type=float, default=120, help="area < threshold 为血细胞，否则跳过")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写 labels.csv")
    args = parser.parse_args()

    meta_path = resolve_metadata_path(args.metadata)
    labels_path = meta_path.parent / "labels.csv"
    df = pd.read_csv(meta_path)

    if "area" not in df.columns:
        raise SystemExit("metadata 缺少 area 列")

    blood_mask = df["area"] < args.threshold
    labels = pd.Series(LABEL_SKIP, index=df.index, dtype=int)
    labels.loc[blood_mask] = LABEL_BLOOD

    n_blood = int(blood_mask.sum())
    n_skip = int((~blood_mask).sum())
    print(f"metadata: {meta_path}")
    print(f"规则: area < {args.threshold} -> blood(0), area >= {args.threshold} -> skip(-1)")
    print(f"总计 {len(df)}: blood={n_blood}, skip={n_skip}")

    if args.dry_run:
        return

    out = pd.DataFrame({"cell_id": df["cell_id"], "label": labels})
    out.to_csv(labels_path, index=False)
    print(f"已写入: {labels_path}")


if __name__ == "__main__":
    main()
