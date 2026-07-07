"""train/val 划分与 crop 复制。"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import pandas as pd

from utils.constants import LABEL_TO_DIR
from utils.paths import resolve_crop_path


def split_image_ids(
    image_ids: list[str],
    val_ratio: float,
    rng: random.Random,
) -> tuple[set[str], set[str]]:
    """按 image_id 划分，保证同视野图不跨 train/val。"""
    ids = list(dict.fromkeys(image_ids))
    rng.shuffle(ids)
    if len(ids) <= 1:
        return set(), set(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    n_val = min(n_val, len(ids) - 1)
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return val_ids, train_ids


def assign_split_by_image_id(
    image_ids: pd.Series,
    val_ratio: float,
    rng: random.Random,
) -> pd.Series:
    """为 DataFrame 行分配 train/val split 标签。"""
    unique_ids = image_ids.drop_duplicates().tolist()
    val_ids, train_ids = split_image_ids(unique_ids, val_ratio, rng)
    return image_ids.apply(lambda x: "val" if x in val_ids else "train")


def assign_split_random(
    index: pd.Index,
    val_ratio: float,
    rng: random.Random,
) -> pd.Series:
    """按 crop 随机划分（不推荐用于同图多 crop 场景）。"""
    indices = index.tolist()
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_ratio)) if len(indices) > 1 else 0
    split_map = {idx: ("val" if i < n_val else "train") for i, idx in enumerate(indices)}
    return index.to_series().map(split_map)


def clear_split_dirs(
    data_root: Path,
    splits: tuple[str, ...] = ("train", "val"),
    classes: tuple[str, ...] = ("blood", "fetal"),
) -> None:
    for split in splits:
        for cls in classes:
            folder = data_root / split / cls
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)


def copy_labeled_rows(
    rows: pd.DataFrame,
    split: str,
    class_label: int,
    data_root: Path,
    *,
    project_root: Path | None = None,
) -> int:
    """将标注 crop 复制到 data/{split}/{class}/。"""
    class_name = LABEL_TO_DIR[class_label]
    out_dir = data_root / split / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for _, row in rows.iterrows():
        src = resolve_crop_path(row["crop_path"], project_root)
        if not src.exists():
            continue
        shutil.copy2(src, out_dir / src.name)
        copied += 1
    return copied


def load_labeled_rows(
    metadata_path: Path,
    labels_path: Path,
    target_class: int,
) -> pd.DataFrame:
    meta = pd.read_csv(metadata_path)
    labels = pd.read_csv(labels_path)
    df = meta.merge(labels, on="cell_id", suffixes=("", "_lbl"))
    label_col = "label_lbl" if "label_lbl" in df.columns else "label"
    return df[df[label_col] == target_class].copy()
