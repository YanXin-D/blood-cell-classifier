"""Hard negative 样本导入。"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from utils.constants import HARD_NEGATIVES_METADATA, IMAGE_EXTENSIONS
from utils.paths import PROJECT_ROOT, resolve_crop_path, resolve_path


def import_hard_negatives(
    train_blood_dir: Path,
    *,
    metadata_path: Path | None = None,
    max_samples: int = 2000,
    prefix: str = "hardneg_",
    dry_run: bool = False,
    project_root: Path | None = None,
) -> int:
    """从混合血样 metadata 复制 hard negatives 到 train/blood/。"""
    root = project_root or PROJECT_ROOT
    meta_path = metadata_path or resolve_path(HARD_NEGATIVES_METADATA, root)
    if not meta_path.exists():
        return 0

    meta = pd.read_csv(meta_path)
    rows = meta.head(max_samples if max_samples > 0 else len(meta))

    train_blood_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for _, row in rows.iterrows():
        src = resolve_crop_path(row["crop_path"], root)
        if src.suffix.lower() not in IMAGE_EXTENSIONS or not src.exists():
            continue
        dst = train_blood_dir / f"{prefix}{src.name}"
        if dst.exists():
            continue
        if not dry_run:
            shutil.copy2(src, dst)
        copied += 1
    return copied
