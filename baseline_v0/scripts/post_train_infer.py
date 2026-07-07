"""训练后推理流水线：打分 → 候选 → 排序 → bbox 可视化。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STEPS = [
    [sys.executable, "-u", "inference/detect_candidates.py"],
    [sys.executable, "-u", "inference/rank_candidates.py"],
    [sys.executable, "-u", "inference/visualize_fov_bbox.py"],
]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="训练后推理流水线")
    parser.add_argument(
        "--morph-filter",
        action="store_true",
        help="开启形态后处理",
    )
    args = parser.parse_args()

    cmd0 = list(STEPS[0])
    if args.morph_filter:
        cmd0.append("--morph-filter")

    for cmd in [cmd0, *STEPS[1:]]:
        print("$", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
