"""将混合血样 crop 作为 hard negative 加入训练集。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.config import get_train_cfg, load_config
from utils.console import configure_utf8_stdio
from utils.hard_negatives import import_hard_negatives
from utils.paths import PROJECT_ROOT, resolve_path, setup_import_path

setup_import_path()
configure_utf8_stdio()


def main():
    parser = argparse.ArgumentParser(description="导入混合血样 hard negatives")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="混合血样 metadata.csv（默认 config 中路径）",
    )
    parser.add_argument(
        "--train-blood-dir",
        type=Path,
        default=PROJECT_ROOT / "data/train/blood",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--prefix", type=str, default="hardneg_")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    train_cfg = get_train_cfg(cfg)
    max_samples = args.max_samples
    if max_samples is None:
        max_samples = int(train_cfg.get("hard_negatives_max", 2000))

    metadata_path = args.metadata
    if metadata_path is None:
        from utils.constants import HARD_NEGATIVES_METADATA
        metadata_path = resolve_path(HARD_NEGATIVES_METADATA)

    if not metadata_path.exists():
        raise SystemExit(f"未找到 {metadata_path}")

    copied = import_hard_negatives(
        args.train_blood_dir,
        metadata_path=metadata_path,
        max_samples=max_samples,
        prefix=args.prefix,
        dry_run=args.dry_run,
    )

    print(f"{'将复制' if args.dry_run else '已复制'}: {copied} -> {args.train_blood_dir}")
    if not args.dry_run and copied:
        print("下一步: python train/train.py 重新训练")


if __name__ == "__main__":
    main()
