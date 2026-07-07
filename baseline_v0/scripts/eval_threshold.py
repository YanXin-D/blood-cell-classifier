"""在验证集上扫阈值，输出 PR 曲线与推荐 operating point。"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve

from utils.config import get_output_dir, load_config
from utils.console import configure_utf8_stdio
from utils.model_loader import build_val_loader, load_model_for_inference
from utils.paths import setup_import_path

setup_import_path()
configure_utf8_stdio()


def collect_scores(model, loader, device):
    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, y, _ in loader:
            out = model(images.to(device))
            p = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(y.numpy().tolist())
    return np.array(labels), np.array(probs)


def recommend_threshold(labels, probs, min_precision: float):
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    best = None
    for i, th in enumerate(thresholds):
        p, r = precisions[i + 1], recalls[i + 1]
        if p >= min_precision and (best is None or r > best["recall"]):
            best = {"threshold": float(th), "precision": float(p), "recall": float(r)}
    return best


def main():
    import argparse

    parser = argparse.ArgumentParser(description="验证集 PR 曲线与阈值推荐")
    parser.add_argument("--min-precision", type=float, default=0.9)
    args = parser.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = get_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader, _ = build_val_loader(cfg, device)
    model, _ = load_model_for_inference(cfg, device)

    labels, probs = collect_scores(model, loader, device)
    precisions, recalls, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recalls, precisions)

    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, label=f"PR AUC={pr_auc:.3f}")
    plt.xlabel("recall (fetal)")
    plt.ylabel("precision (fetal)")
    plt.title("Validation PR Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    pr_path = output_dir / "pr_curve_val.png"
    plt.savefig(pr_path, dpi=150)
    plt.close()

    rows = []
    for th in np.linspace(0.05, 0.99, 20):
        pred = (probs >= th).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "threshold": round(float(th), 3),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    sweep_path = output_dir / "threshold_sweep_val.csv"
    pd.DataFrame(rows).to_csv(sweep_path, index=False)

    rec = recommend_threshold(labels, probs, args.min_precision)
    print(f"验证集 PR 曲线 -> {pr_path}")
    print(f"阈值扫描表 -> {sweep_path}")
    if rec:
        print(
            f"推荐阈值 (precision>={args.min_precision}): "
            f"prob>={rec['threshold']:.3f} "
            f"(prec={rec['precision']:.3f}, rec={rec['recall']:.3f})"
        )
        det_path = output_dir / "recommended_threshold.txt"
        det_path.write_text(
            f"min_probability: {rec['threshold']:.4f}\n"
            f"precision: {rec['precision']:.4f}\n"
            f"recall: {rec['recall']:.4f}\n",
            encoding="utf-8",
        )
        print(f"已写入 -> {det_path}")
        print("提示: 将 detection.min_probability 更新为推荐值后重新运行推理")
    else:
        print(f"未找到 precision>={args.min_precision} 的阈值，请查看 {sweep_path}")


if __name__ == "__main__":
    main()
