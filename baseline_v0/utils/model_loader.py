"""模型加载与批量推理。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.cell_dataset import (
    AllCellsDataset,
    CellDataset,
    detect_grayscale,
    get_inference_transform,
    get_val_transform,
)
from models.resnet18_classifier import build_resnet18_classifier
from utils.config import get_checkpoint_path, get_train_cfg
from utils.paths import PROJECT_ROOT


def load_checkpoint(cfg: dict, device: torch.device, ckpt_name: str = "best_model.pt") -> tuple[dict, dict]:
    ckpt_path = get_checkpoint_path(cfg, ckpt_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到模型: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    return ckpt, ckpt.get("meta", {})


def build_model_from_checkpoint(cfg: dict, meta: dict, device: torch.device):
    train_cfg = get_train_cfg(cfg)
    grayscale = meta.get("grayscale", False)
    in_channels = meta.get("in_channels", 1 if grayscale else 3)
    model = build_resnet18_classifier(
        num_classes=meta.get("num_classes", cfg["num_classes"]),
        in_channels=in_channels,
        freeze_layer1=cfg["freeze_layer1"],
        freeze_layer2=cfg["freeze_layer2"],
        freeze_layer3=train_cfg.get("freeze_layer3", False),
    ).to(device)
    return model, in_channels, grayscale


def load_model_for_inference(cfg: dict, device: torch.device, ckpt_name: str = "best_model.pt"):
    ckpt, meta = load_checkpoint(cfg, device, ckpt_name)
    model, _, _ = build_model_from_checkpoint(cfg, meta, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, meta


def score_all_cells(
    crops_dir: Path,
    cfg: dict,
    device: torch.device,
) -> pd.DataFrame:
    """对 crop 目录全量 CNN 打分，返回 cell_name / probability / prediction。"""
    model, meta = load_model_for_inference(cfg, device)
    grayscale = meta.get("grayscale", False)
    img_size = meta.get("img_size", cfg["img_size"])

    dataset = AllCellsDataset(
        crops_dir,
        transform=get_inference_transform(img_size, grayscale),
        grayscale=grayscale,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
    )

    rows = []
    with torch.no_grad():
        for images, names in loader:
            probs = torch.softmax(model(images.to(device)), dim=1)
            fetal_prob = probs[:, 1].cpu().numpy()
            preds = probs.argmax(dim=1).cpu().numpy()
            for name, prob, pred in zip(names, fetal_prob, preds):
                rows.append(
                    {
                        "cell_name": name,
                        "probability": float(prob),
                        "prediction": int(pred),
                    }
                )
    return pd.DataFrame(rows)


def build_val_loader(cfg: dict, device: torch.device):
    """构建验证集 DataLoader 及灰度信息。"""
    val_dir = PROJECT_ROOT / cfg["val_dir"]
    grayscale = detect_grayscale(val_dir)
    val_ds = CellDataset(
        val_dir,
        transform=get_val_transform(cfg["img_size"], grayscale),
        grayscale=grayscale,
    )
    loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
    )
    return loader, grayscale
