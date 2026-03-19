"""
VGG networks — Simonyan & Zisserman, ICLR 2015.
Used in He et al. as an ImageNet baseline reference (Table 1).

Variants:
    VGG-16    : conv blocks [2,2,3,3,3], no BN
    VGG-16-BN : same with Batch Normalization
    VGG-19    : conv blocks [2,2,4,4,4], no BN
    VGG-19-BN : same with Batch Normalization
"""

from typing import Sequence

import torch
import torch.nn as nn


# cfg: number of conv layers per block (5 blocks total)
_CONFIGS: dict[str, list[int]] = {
    "vgg16": [2, 2, 3, 3, 3],
    "vgg19": [2, 2, 4, 4, 4],
}

_CHANNELS: list[int] = [64, 128, 256, 512, 512]


def _make_features(cfg: Sequence[int], batch_norm: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = 3
    for block_idx, num_convs in enumerate(cfg):
        out_channels = _CHANNELS[block_idx]
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class VGG(nn.Module):
    """VGG with three fully-connected layers (original paper design)."""

    def __init__(self, cfg: Sequence[int], batch_norm: bool,
                 num_classes: int = 1000) -> None:
        super().__init__()
        self.features = _make_features(cfg, batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(self.features(x))
        x = torch.flatten(x, 1)
        return self.classifier(x)


def vgg16(num_classes: int = 1000) -> VGG:
    return VGG(_CONFIGS["vgg16"], batch_norm=False, num_classes=num_classes)


def vgg16_bn(num_classes: int = 1000) -> VGG:
    return VGG(_CONFIGS["vgg16"], batch_norm=True, num_classes=num_classes)


def vgg19(num_classes: int = 1000) -> VGG:
    return VGG(_CONFIGS["vgg19"], batch_norm=False, num_classes=num_classes)


def vgg19_bn(num_classes: int = 1000) -> VGG:
    return VGG(_CONFIGS["vgg19"], batch_norm=True, num_classes=num_classes)


# Supported ImageNet VGG models
IMAGENET_VGG_MODELS: dict[str, object] = {
    "vgg16":    vgg16,
    "vgg16_bn": vgg16_bn,
    "vgg19":    vgg19,
    "vgg19_bn": vgg19_bn,
}
