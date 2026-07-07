"""Atlas 自动归档接口（预留 ArcFace / Metric Learning）。"""

from __future__ import annotations

import shutil
from pathlib import Path

from .metadata import LABEL_BLOOD, LABEL_FETAL


class AtlasManager:
    """标注后自动复制 crop 到 atlas 目录。"""

    def __init__(self, atlas_root: Path, enabled: bool = True):
        self.atlas_root = Path(atlas_root)
        self.enabled = enabled
        self.fetal_dir = self.atlas_root / "fetal"
        self.blood_dir = self.atlas_root / "blood"

    def ensure_dirs(self) -> None:
        self.fetal_dir.mkdir(parents=True, exist_ok=True)
        self.blood_dir.mkdir(parents=True, exist_ok=True)

    def archive(self, crop_path: str | Path, cell_id: str, label: int) -> Path | None:
        """label=1 -> fetal/, label=0 -> blood/；skip 不复制。"""
        if not self.enabled:
            return None
        if label not in (LABEL_FETAL, LABEL_BLOOD):
            return None
        self.ensure_dirs()
        src = Path(crop_path)
        if not src.exists():
            return None
        dest_dir = self.fetal_dir if label == LABEL_FETAL else self.blood_dir
        suffix = src.suffix or ".png"
        dest = dest_dir / f"{cell_id}{suffix}"
        shutil.copy2(src, dest)
        return dest

    def remove(self, cell_id: str) -> None:
        """撤销时尝试删除 atlas 中对应文件。"""
        for d in (self.fetal_dir, self.blood_dir):
            if not d.exists():
                continue
            for p in d.glob(f"{cell_id}.*"):
                p.unlink(missing_ok=True)
