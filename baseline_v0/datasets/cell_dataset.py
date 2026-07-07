"""单细胞图像数据集：fetal=1, blood=0。"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.constants import CLASS_MAP, IMAGE_EXTENSIONS


def _is_grayscale(img: Image.Image) -> bool:
    if img.mode == "L":
        return True
    if img.mode != "RGB":
        img = img.convert("RGB")
    r, g, b = img.split()
    return r == g == b


def detect_grayscale(data_dir: Path) -> bool:
    for sub in CLASS_MAP:
        folder = data_dir / sub
        if not folder.exists():
            continue
        for p in folder.iterdir():
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                with Image.open(p) as img:
                    return _is_grayscale(img)
    return False


def _normalize(grayscale: bool):
    if grayscale:
        return transforms.Normalize(mean=[0.485], std=[0.229])
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_train_transform(img_size: int, grayscale: bool = False):
    jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3)
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
            ),
            transforms.RandomApply([jitter], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
            transforms.ToTensor(),
            _normalize(grayscale),
        ]
    )


def get_val_transform(img_size: int, grayscale: bool = False):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            _normalize(grayscale),
        ]
    )


def get_inference_transform(img_size: int, grayscale: bool = False):
    return get_val_transform(img_size, grayscale)


class CellDataset(Dataset):
    """监督训练/验证：data/{split}/{blood|fetal}/*.png。"""

    def __init__(self, data_dir, transform=None, grayscale=None):
        self.data_dir = Path(data_dir)
        self.samples = []
        self.transform = transform

        for class_name, label in CLASS_MAP.items():
            folder = self.data_dir / class_name
            if not folder.exists():
                continue
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((p, label))

        if not self.samples:
            raise ValueError(f"未在 {data_dir} 找到图像（需 fetal/ 与 blood/ 子目录）")

        if grayscale is None:
            with Image.open(self.samples[0][0]) as img:
                self.grayscale = _is_grayscale(img)
        else:
            self.grayscale = grayscale

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("L" if self.grayscale else "RGB")
            if self.transform:
                img = self.transform(img)
        return img, label, path.name

    def get_class_counts(self):
        counts = {0: 0, 1: 0}
        for _, label in self.samples:
            counts[label] += 1
        return counts


class AllCellsDataset(Dataset):
    """推理用：读取 crop 目录下全部图像。"""

    def __init__(self, data_dir, transform=None, grayscale=False):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.grayscale = grayscale
        self.samples = sorted(
            p for p in self.data_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.samples:
            raise ValueError(f"未在 {data_dir} 找到图像")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("L" if self.grayscale else "RGB")
            if self.transform:
                img = self.transform(img)
        return img, path.name
