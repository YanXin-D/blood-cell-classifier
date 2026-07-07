"""ResNet18 二分类：灰度图自动改 conv1，冻结 layer1/layer2。"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def build_resnet18_classifier(
    num_classes: int = 2,
    in_channels: int = 3,
    freeze_layer1: bool = True,
    freeze_layer2: bool = True,
    freeze_layer3: bool = False,
    dropout: float = 0.5,
):
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    if in_channels == 1:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            model.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.fc.in_features, num_classes),
    )

    if freeze_layer1:
        for param in model.layer1.parameters():
            param.requires_grad = False
    if freeze_layer2:
        for param in model.layer2.parameters():
            param.requires_grad = False
    if freeze_layer3:
        for param in model.layer3.parameters():
            param.requires_grad = False

    return model
