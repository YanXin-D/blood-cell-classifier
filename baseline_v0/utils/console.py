"""Windows 终端 UTF-8 输出。"""

from __future__ import annotations

import sys


def configure_utf8_stdio() -> None:
    """避免 PowerShell/CMD 中文乱码。"""
    if sys.platform != "win32":
        return
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass
