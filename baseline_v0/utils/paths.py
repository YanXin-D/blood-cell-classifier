"""路径解析：项目根目录、相对路径转换。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_import_path() -> Path:
    """确保 baseline_v0 在 sys.path 中，供脚本入口调用。"""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROJECT_ROOT


def resolve_path(rel: str | Path, base: Path | None = None) -> Path:
    """将 config 中的相对路径解析为绝对路径。"""
    p = Path(rel)
    if p.is_absolute():
        return p
    return (base or PROJECT_ROOT) / p


def to_relative_path(path: str | Path, base: Path | None = None) -> str:
    """存储为相对项目根的路径，便于跨机器协作。"""
    p = Path(path).resolve()
    root = (base or PROJECT_ROOT).resolve()
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def resolve_crop_path(crop_path: str | Path, base: Path | None = None) -> Path:
    """读取 metadata 中的 crop_path，支持相对/绝对路径。"""
    p = Path(crop_path)
    if p.is_absolute():
        return p
    return (base or PROJECT_ROOT) / p
