"""下载 cyto3 权重（若缺失/损坏）后自动跑 blood_cell 分割。日志写入 outputs/nightly_blood_segment.log"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "outputs" / "nightly_blood_segment.log"
CYTO3_PATH = Path.home() / ".cellpose" / "models" / "cyto3"
CYTO3_URL = "https://www.cellpose.org/models/cyto3"
MIN_BYTES = 24_000_000


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def cyto3_ok() -> bool:
    if not CYTO3_PATH.exists():
        return False
    if CYTO3_PATH.stat().st_size < MIN_BYTES:
        return False
    try:
        import torch
        torch.load(CYTO3_PATH, map_location="cpu", weights_only=True)
        log(f"cyto3 验证通过: {CYTO3_PATH.stat().st_size / 1e6:.1f} MB")
        return True
    except Exception as e:
        log(f"cyto3 加载失败: {e}")
        return False


def download_cyto3() -> bool:
    models_dir = CYTO3_PATH.parent
    models_dir.mkdir(parents=True, exist_ok=True)

    # 清理损坏文件和临时下载
    if CYTO3_PATH.exists() and CYTO3_PATH.stat().st_size < MIN_BYTES:
        log(f"删除不完整 cyto3 ({CYTO3_PATH.stat().st_size / 1e6:.1f} MB)")
        CYTO3_PATH.unlink()

    for tmp in models_dir.glob("tmp*"):
        try:
            tmp.unlink()
        except OSError:
            pass

    tmp_path = models_dir / "cyto3.download"
    if tmp_path.exists():
        tmp_path.unlink()

    log(f"开始下载 cyto3 -> {CYTO3_PATH}")
    try:
        from cellpose.utils import download_url_to_file
        download_url_to_file(CYTO3_URL, str(tmp_path), progress=True)
        if not tmp_path.exists() or tmp_path.stat().st_size < MIN_BYTES:
            log(f"下载不完整: {tmp_path.stat().st_size / 1e6:.1f} MB")
            return False
        tmp_path.replace(CYTO3_PATH)
        log(f"下载完成: {CYTO3_PATH.stat().st_size / 1e6:.1f} MB")
        return True
    except Exception as e:
        log(f"下载异常: {e}")
        return False


def wait_and_download(max_wait_hours: float = 3.0) -> bool:
    deadline = time.time() + max_wait_hours * 3600
    attempt = 0
    while time.time() < deadline:
        if cyto3_ok():
            return True
        attempt += 1
        log(f"第 {attempt} 次下载尝试...")
        if download_cyto3() and cyto3_ok():
            return True
        log("等待 60 秒后重试...")
        time.sleep(60)
    return False


def run_segmentation() -> int:
    cmd = [
        sys.executable, "-u",
        str(ROOT / "preprocess" / "segment_cellpose.py"),
        "--batch", "blood_cell",
        "--clean",
    ]
    log(f"启动分割: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    log(f"分割结束, exit_code={proc.returncode}")
    return proc.returncode


def main() -> int:
    log("=== nightly blood_cell 分割任务开始 ===")
    if not wait_and_download():
        log("cyto3 下载/验证失败，分割未启动")
        return 1
    rc = run_segmentation()
    crops = ROOT / "data" / "crops" / "blood_cell"
    n_crops = len(list(crops.glob("*.png"))) if crops.exists() else 0
    meta = crops / "metadata.csv"
    log(f"结果: crops={n_crops}, metadata={'存在' if meta.exists() else '缺失'}")
    log("=== 任务结束 ===")
    return rc


if __name__ == "__main__":
    sys.exit(main())
