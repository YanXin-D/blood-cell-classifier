"""图像 lazy loading 与 LRU 缓存。"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


class LRUImageCache:
    """按路径缓存已加载图像，支持 prev/current/next 预取。"""

    def __init__(self, max_size: int = 24):
        self.max_size = max_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def clear(self) -> None:
        self._cache.clear()

    def get(self, path: str | Path) -> Optional[np.ndarray]:
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        arr = self._load(key)
        if arr is None:
            return None
        self._cache[key] = arr
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return arr

    def prefetch(self, paths: list[str | Path]) -> None:
        for p in paths:
            if p and str(p) not in self._cache:
                arr = self._load(str(p))
                if arr is not None:
                    self._cache[str(p)] = arr
                    if len(self._cache) > self.max_size:
                        self._cache.popitem(last=False)

    @staticmethod
    def _load(path: str) -> Optional[np.ndarray]:
        p = Path(path)
        if not p.exists():
            return None
        with Image.open(p) as img:
            if img.mode == "L":
                return np.array(img)
            return np.array(img.convert("RGB"))

    @staticmethod
    def placeholder(shape: tuple[int, int] = (256, 256), channels: int = 1) -> np.ndarray:
        if channels == 1:
            return np.full(shape, 40, dtype=np.uint8)
        return np.full((*shape, 3), 40, dtype=np.uint8)


# 全局缓存实例（Streamlit 单进程内复用）
GLOBAL_CACHE = LRUImageCache(max_size=36)
