"""候选检测：CNN 打分为主，形态规则（面积）为可选后处理。"""

from __future__ import annotations

import sys

import pandas as pd
import torch

from utils.config import get_crops_dir, get_detection_cfg, get_output_dir, load_config
from utils.console import configure_utf8_stdio
from utils.inference import describe_filters, filter_candidates, merge_predictions_with_metadata
from utils.model_loader import score_all_cells
from utils.paths import setup_import_path

setup_import_path()
configure_utf8_stdio()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="胎儿候选检测（CNN + 可选形态后处理）")
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="不跑 CNN，仅做形态过滤（需 morph_filter=true）",
    )
    parser.add_argument("--morph-filter", action="store_true", help="开启面积后处理")
    parser.add_argument("--no-morph-filter", action="store_true", help="关闭面积后处理")
    parser.add_argument("--min-probability", type=float, default=None, help="CNN 概率阈值")
    parser.add_argument("--min-area", type=int, default=None, help="面积阈值")
    args = parser.parse_args()

    cfg = load_config()
    det = dict(get_detection_cfg(cfg))
    if args.morph_filter:
        det["morph_filter"] = True
    if args.no_morph_filter:
        det["morph_filter"] = False
    if args.min_probability is not None:
        det["min_probability"] = args.min_probability
    if args.min_area is not None:
        det["min_area"] = args.min_area

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crops_dir = get_crops_dir(cfg)
    meta_path = crops_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"未找到 {meta_path}，请先分割：\n"
            f"  python preprocess/segment_cellpose.py --batch {det['segment_batch']} "
            f"--input-dir {det['raw_dir']} --crops-dir {det['crops_dir']} --clean"
        )

    meta = pd.read_csv(meta_path)
    use_model = det.get("use_model", True) and not args.skip_model
    if args.skip_model and not det.get("morph_filter", False):
        raise SystemExit("--skip-model 需要开启 morph_filter")

    pred = None
    if use_model:
        print(f"device: {device}")
        pred = score_all_cells(crops_dir, cfg, device)
        output_dir = get_output_dir(cfg)
        output_dir.mkdir(parents=True, exist_ok=True)
        scored_path = output_dir / "scored.csv"
        merge_predictions_with_metadata(meta, pred).to_csv(scored_path, index=False)
        print(f"全量打分 -> {scored_path}")

    hits = filter_candidates(meta, pred, det, use_model=use_model)

    output_dir = get_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "candidates.csv"
    hits.to_csv(out_path, index=False)

    n_fov = hits["image_id"].nunique() if len(hits) else 0
    print(f"策略: {describe_filters(det, use_model)}")
    print(
        f"候选: {len(hits)} 个（来自 {meta['image_id'].nunique()} 张视野图，"
        f"{n_fov} 张有候选）"
    )
    print(f"输出 -> {out_path}")

    if len(hits):
        print("\nTop 10:")
        cols = ["cell_id", "image_id", "area", "probability"]
        print(hits[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
