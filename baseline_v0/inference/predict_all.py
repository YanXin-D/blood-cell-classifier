"""对 crop 目录全量 CNN 打分，输出 predictions.csv（兼容旧流水线）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

from utils.config import get_crops_dir, get_detection_cfg, get_output_dir, load_config
from utils.inference import merge_predictions_with_metadata
from utils.model_loader import score_all_cells
from utils.paths import setup_import_path

setup_import_path()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="对 crop 目录全量 CNN 打分")
    parser.add_argument("--crops-dir", type=Path, default=None, help="crop 目录")
    parser.add_argument("--output", type=Path, default=None, help="输出 CSV")
    parser.add_argument("--merge-metadata", action="store_true", help="合并 metadata.csv")
    args = parser.parse_args()

    cfg = load_config()
    det = get_detection_cfg(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    crops_dir = args.crops_dir or get_crops_dir(cfg)
    df = score_all_cells(crops_dir, cfg, device)

    if args.merge_metadata:
        meta_path = crops_dir / "metadata.csv"
        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            df = merge_predictions_with_metadata(meta, df)

    output_dir = get_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or output_dir / "predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"已预测 {len(df)} 张图像 -> {out_path}")


if __name__ == "__main__":
    main()
