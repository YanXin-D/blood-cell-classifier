"""一键准备训练数据：image_id 划分 + hard negatives。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    rc = run([sys.executable, "-u", "scripts/rebuild_train_val.py"])
    if rc != 0:
        return rc
    print("\n训练数据已就绪。下一步: python train/train.py", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
