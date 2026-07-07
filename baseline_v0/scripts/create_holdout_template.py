"""生成混合血样 holdout 标注包。"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from utils.constants import HARD_NEGATIVES_METADATA
from utils.console import configure_utf8_stdio
from utils.paths import PROJECT_ROOT, resolve_crop_path, resolve_path, setup_import_path

setup_import_path()
configure_utf8_stdio()


def main():
    parser = argparse.ArgumentParser(description="生成混合血样 holdout 标注模板")
    parser.add_argument(
        "--source-metadata",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/holdout/mixed_blood",
    )
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = args.source_metadata or resolve_path(HARD_NEGATIVES_METADATA)
    if not source.exists():
        raise SystemExit(f"未找到 {source}")

    meta = pd.read_csv(source)
    if args.sample_size > 0 and len(meta) > args.sample_size:
        meta = meta.sample(n=args.sample_size, random_state=args.seed)

    out_dir = args.output_dir.resolve()
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in meta.iterrows():
        src = resolve_crop_path(row["crop_path"])
        if not src.exists():
            continue
        dst = crops_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        new_row = dict(row)
        new_row["crop_path"] = str(dst.relative_to(PROJECT_ROOT))
        new_row["label"] = -1
        rows.append(new_row)

    out_meta = out_dir / "metadata.csv"
    pd.DataFrame(rows).to_csv(out_meta, index=False)
    pd.DataFrame({"cell_id": [r["cell_id"] for r in rows], "label": [-1] * len(rows)}).to_csv(
        out_dir / "labels.csv", index=False
    )

    print(f"holdout 标注包 -> {out_dir}")
    print(f"  crops: {len(rows)}")
    print(f"  metadata: {out_meta}")
    print("\nCell Inspector 启动:")
    print("  streamlit run cell_inspector/app.py")
    print(f"  metadata 路径填: {out_meta}")


if __name__ == "__main__":
    main()
