"""细胞 metadata CSV 加载与索引。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

REQUIRED_COLUMNS = [
    "cell_id",
    "image_id",
    "crop_path",
    "bf_path",
    "phase_path",
    "fluor_path",
    "mask_id",
    "center_x",
    "center_y",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "mean_intensity",
    "phase_mean",
    "label",
]

FEATURE_COLUMNS = [
    "cell_id",
    "image_id",
    "center_x",
    "center_y",
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "mean_intensity",
    "phase_mean",
]

RESERVED_FEATURE_COLUMNS = [
    "cnn_probability",
    "cluster_id",
    "anomaly_score",
]

LABEL_FETAL = 1
LABEL_BLOOD = 0
LABEL_SKIP = -1


def resolve_metadata_path(path: str | Path) -> Path:
    """目录则自动补全为 metadata.csv。"""
    p = Path(path)
    if p.is_dir():
        return p / "metadata.csv"
    return p


class MetadataStore:
    """metadata 与 labels 的统一访问层。"""

    def __init__(self, metadata_path: Path, labels_path: Optional[Path] = None):
        self.metadata_path = resolve_metadata_path(metadata_path)
        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"未找到 metadata 文件: {self.metadata_path}\n"
                f"请填写 metadata.csv 完整路径，或包含该文件的目录。"
            )
        self.labels_path = Path(labels_path) if labels_path else self.metadata_path.parent / "labels.csv"
        self.df = self._load_metadata()
        self._merge_labels()
        self._build_indices()

    def _load_metadata(self) -> pd.DataFrame:
        df = pd.read_csv(self.metadata_path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"metadata 缺少列: {missing}")
        if "label" not in df.columns:
            df["label"] = LABEL_SKIP
        df["label"] = df["label"].fillna(LABEL_SKIP).astype(int)
        return df.reset_index(drop=True)

    def _merge_labels(self) -> None:
        if not self.labels_path.exists():
            return
        labels = pd.read_csv(self.labels_path)
        if "cell_id" not in labels.columns or "label" not in labels.columns:
            return
        label_map = dict(zip(labels["cell_id"], labels["label"]))
        self.df["label"] = self.df["cell_id"].map(label_map).fillna(self.df["label"]).astype(int)

    def _build_indices(self) -> None:
        self._cell_id_to_idx = {
            cid: i for i, cid in enumerate(self.df["cell_id"].tolist())
        }
        self._unlabeled_indices = self.df.index[self.df["label"] == LABEL_SKIP].tolist()
        self._image_ids = sorted(self.df["image_id"].unique().tolist())

    @property
    def n_cells(self) -> int:
        return len(self.df)

    @property
    def n_labeled(self) -> int:
        return int((self.df["label"] != LABEL_SKIP).sum())

    @property
    def n_unlabeled(self) -> int:
        return len(self._unlabeled_indices)

    def get_row(self, idx: int) -> pd.Series:
        return self.df.iloc[idx]

    def get_row_by_cell_id(self, cell_id) -> Optional[pd.Series]:
        idx = self._cell_id_to_idx.get(cell_id)
        if idx is None:
            return None
        return self.df.iloc[idx]

    def idx_from_cell_id(self, cell_id) -> Optional[int]:
        return self._cell_id_to_idx.get(cell_id)

    def filtered_indices(
        self,
        unlabeled_only: bool = False,
        image_id: Optional[str] = None,
    ) -> list[int]:
        indices = list(range(len(self.df)))
        if unlabeled_only:
            indices = [i for i in indices if self.df.iloc[i]["label"] == LABEL_SKIP]
        if image_id:
            indices = [i for i in indices if str(self.df.iloc[i]["image_id"]) == str(image_id)]
        return indices

    def refresh_labels(self) -> None:
        self._merge_labels()
        self._build_indices()

    def save_label(self, cell_id, label: int) -> None:
        idx = self._cell_id_to_idx.get(cell_id)
        if idx is not None:
            self.df.at[idx, "label"] = label
        records = []
        if self.labels_path.exists():
            existing = pd.read_csv(self.labels_path)
            records = existing.to_dict("records")
        updated = False
        for rec in records:
            if rec["cell_id"] == cell_id:
                rec["label"] = label
                updated = True
                break
        if not updated:
            records.append({"cell_id": cell_id, "label": label})
        out = pd.DataFrame(records)
        out.to_csv(self.labels_path, index=False)
        self._build_indices()

    def feature_table(self, row: pd.Series) -> pd.DataFrame:
        rows = []
        for col in FEATURE_COLUMNS:
            val = row.get(col, "")
            rows.append({"特征": col, "值": val})
        for col in RESERVED_FEATURE_COLUMNS:
            rows.append({"特征": col, "值": "（预留）"})
        return pd.DataFrame(rows)
