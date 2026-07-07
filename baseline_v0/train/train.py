"""ResNet18 监督分类训练。"""

from __future__ import annotations

import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.cell_dataset import (
    CellDataset,
    detect_grayscale,
    get_train_transform,
    get_val_transform,
)
from models.resnet18_classifier import build_resnet18_classifier
from utils.config import get_train_cfg, load_config
from utils.console import configure_utf8_stdio
from utils.losses import FocalLoss
from utils.paths import PROJECT_ROOT, setup_import_path

setup_import_path()
configure_utf8_stdio()


def build_criterion(cfg: dict, class_weights: torch.Tensor, device: torch.device):
    train_cfg = get_train_cfg(cfg)
    loss_name = train_cfg.get("loss", "cross_entropy")
    if loss_name == "focal":
        return FocalLoss(
            alpha=class_weights.to(device),
            gamma=float(train_cfg.get("focal_gamma", 2.0)),
        )
    return nn.CrossEntropyLoss(weight=class_weights.to(device))


def build_train_loader(train_ds, cfg: dict, device: torch.device):
    train_cfg = get_train_cfg(cfg)
    kwargs = {
        "batch_size": cfg["batch_size"],
        "num_workers": cfg["num_workers"],
        "pin_memory": device.type == "cuda",
    }
    if train_cfg.get("use_weighted_sampler", False):
        counts = train_ds.get_class_counts()
        sample_weights = [
            1.0 / counts[label] if counts[label] > 0 else 0.0
            for _, label in train_ds.samples
        ]
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        return DataLoader(train_ds, sampler=sampler, shuffle=False, **kwargs)
    return DataLoader(train_ds, shuffle=True, **kwargs)


def metric_score(name: str, val_loss: float, metrics: dict) -> float:
    if name == "val_loss":
        return -val_loss
    if name in metrics:
        return metrics[name]
    raise ValueError(f"未知 best_metric: {name}")


def build_model(cfg: dict, in_channels: int, device: torch.device):
    train_cfg = get_train_cfg(cfg)
    return build_resnet18_classifier(
        num_classes=cfg["num_classes"],
        in_channels=in_channels,
        freeze_layer1=cfg["freeze_layer1"],
        freeze_layer2=cfg["freeze_layer2"],
        freeze_layer3=train_cfg.get("freeze_layer3", False),
    ).to(device)


def compute_class_weights(counts: dict, num_classes: int):
    total = sum(counts.values())
    weights = []
    for c in range(num_classes):
        n = counts.get(c, 0)
        weights.append(total / (num_classes * n) if n > 0 else 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(all_labels, all_preds):
    n = len(all_labels)
    if n == 0:
        return {
            "accuracy": 0.0, "macro_f1": 0.0,
            "blood_precision": 0.0, "blood_recall": 0.0,
            "fetal_precision": 0.0, "fetal_recall": 0.0, "fetal_f1": 0.0,
        }

    per_prec = precision_score(
        all_labels, all_preds, labels=[0, 1], average=None, zero_division=0
    )
    per_rec = recall_score(
        all_labels, all_preds, labels=[0, 1], average=None, zero_division=0
    )
    per_f1 = f1_score(all_labels, all_preds, labels=[0, 1], average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "blood_precision": float(per_prec[0]),
        "blood_recall": float(per_rec[0]),
        "fetal_precision": float(per_prec[1]),
        "fetal_recall": float(per_rec[1]),
        "fetal_f1": float(per_f1[1]),
    }


def run_epoch(model, loader, criterion, device, optimizer=None, desc="train"):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        for images, labels, _ in bar:
            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            bar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(all_labels)
    avg_loss = total_loss / n if n else 0.0
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics, all_labels, all_preds


def plot_loss_curve(history, output_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(labels, preds, output_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = ["blood(0)", "fetal(1)"]
    plt.xticks([0, 1], ticks)
    plt.yticks([0, 1], ticks)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.ylabel("true")
    plt.xlabel("pred")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    cfg = load_config()
    train_cfg = get_train_cfg(cfg)
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_dir = PROJECT_ROOT / cfg["train_dir"]
    val_dir = PROJECT_ROOT / cfg["val_dir"]
    save_dir = PROJECT_ROOT / cfg["save_dir"]
    output_dir = PROJECT_ROOT / cfg["output_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    grayscale = detect_grayscale(train_dir)
    print(f"grayscale: {grayscale}")

    train_ds = CellDataset(
        train_dir,
        transform=get_train_transform(cfg["img_size"], grayscale),
        grayscale=grayscale,
    )
    val_ds = CellDataset(
        val_dir,
        transform=get_val_transform(cfg["img_size"], grayscale),
        grayscale=grayscale,
    )

    train_counts = train_ds.get_class_counts()
    val_counts = val_ds.get_class_counts()
    print(
        f"训练集: {len(train_ds)} 张 (blood={train_counts[0]}, fetal={train_counts[1]})",
        flush=True,
    )
    print(
        f"验证集: {len(val_ds)} 张 (blood={val_counts[0]}, fetal={val_counts[1]})",
        flush=True,
    )
    print(
        f"epochs={cfg['epochs']} batch_size={cfg['batch_size']} lr={float(cfg['lr'])}",
        flush=True,
    )
    print(
        f"loss={train_cfg.get('loss', 'cross_entropy')} "
        f"best_metric={train_cfg.get('best_metric', 'val_loss')} "
        f"sampler={train_cfg.get('use_weighted_sampler', False)}",
        flush=True,
    )

    train_loader = build_train_loader(train_ds, cfg, device)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
    )

    in_channels = 1 if grayscale else 3
    model = build_model(cfg, in_channels, device)
    class_weights = compute_class_weights(train_ds.get_class_counts(), cfg["num_classes"])
    criterion = build_criterion(cfg, class_weights, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    writer = SummaryWriter(log_dir=str(PROJECT_ROOT / cfg["tensorboard_dir"]))

    history = {"train_loss": [], "val_loss": []}
    best_metric_name = train_cfg.get("best_metric", "val_loss")
    best_score = float("-inf")
    best_val_loss = float("inf")
    best_labels, best_preds = [], []
    patience = int(train_cfg.get("early_stop_patience", 0))
    stale_epochs = 0

    meta = {
        "grayscale": grayscale,
        "in_channels": in_channels,
        "img_size": cfg["img_size"],
        "num_classes": cfg["num_classes"],
        "best_metric": best_metric_name,
    }

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_metrics, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer,
            desc=f"epoch {epoch:03d} train",
        )
        val_loss, val_metrics, val_labels, val_preds = run_epoch(
            model, val_loader, criterion, device, desc=f"epoch {epoch:03d} val",
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"metrics/{k}", v, epoch)

        score = metric_score(best_metric_name, val_loss, val_metrics)
        improved = score > best_score
        print(
            f"epoch {epoch:03d}/{cfg['epochs']} | lr {current_lr:.2e} | "
            f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
            f"fetal_rec {val_metrics['fetal_recall']:.4f} | "
            f"fetal_prec {val_metrics['fetal_precision']:.4f} | "
            f"blood_rec {val_metrics['blood_recall']:.4f} | "
            f"macro_f1 {val_metrics['macro_f1']:.4f} | "
            f"{best_metric_name} {score:.4f}"
            f"{'  *best*' if improved else ''}",
            flush=True,
        )

        last_ckpt = {
            "model": model.state_dict(),
            "meta": meta,
            "epoch": epoch,
            "val_metrics": val_metrics,
        }
        torch.save(last_ckpt, save_dir / "last_model.pt")

        if improved:
            best_score = score
            best_val_loss = val_loss
            best_labels, best_preds = val_labels, val_preds
            stale_epochs = 0
            torch.save(last_ckpt, save_dir / "best_model.pt")
        elif patience > 0:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"early stop @ epoch {epoch} ({best_metric_name} 连续 {patience} 轮未提升)")
                break

    writer.close()
    plot_loss_curve(history, output_dir / "loss_curve.png")
    if best_labels:
        plot_confusion_matrix(best_labels, best_preds, output_dir / "confusion_matrix.png")

    print(f"训练完成，best {best_metric_name}: {best_score:.4f}, best val_loss: {best_val_loss:.4f}")
    print(f"已保存: {save_dir / 'best_model.pt'}, {save_dir / 'last_model.pt'}")

    best_ckpt = torch.load(save_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    val_loss, val_metrics, _, _ = run_epoch(model, val_loader, criterion, device)
    print(
        f"best_model 验证 | loss {val_loss:.4f} | "
        f"fetal_rec {val_metrics['fetal_recall']:.4f} | "
        f"fetal_prec {val_metrics['fetal_precision']:.4f} | "
        f"macro_f1 {val_metrics['macro_f1']:.4f}"
    )

    for script in ("scripts/eval_threshold.py", "scripts/export_val_predictions.py"):
        path = PROJECT_ROOT / script
        if path.exists():
            print(f"\n运行 {path.name}...", flush=True)
            subprocess.run([sys.executable, "-u", str(path)], cwd=PROJECT_ROOT, check=False)


if __name__ == "__main__":
    main()
