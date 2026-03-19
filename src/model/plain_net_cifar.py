"""
PlainNet for CIFAR-10 — He et al., CVPR 2016 (Fig. 6).

Degradation-problem baselines with same depth/width as CIFAR ResNets
but residual shortcuts removed.

Architecture:
    - Input 32x32 images (3x3 first conv, no maxpool)
    - 3 stages, each with n blocks; channels: 16 → 32 → 64
    - Total layers = 6n + 2

Variants (n → model name):
    n=3  → PlainNet-20
    n=5  → PlainNet-32
    n=7  → PlainNet-44
    n=9  → PlainNet-56
    n=18 → PlainNet-110
"""

import torch
import torch.nn as nn


class PlainBlockCIFAR(nn.Module):
    """Two 3x3 convs — no shortcut."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class PlainNetCIFAR(nn.Module):
    """CIFAR-10 plain network with 6n+2 layers (no skip connections)."""

    def __init__(self, n: int, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(16, 16, n, stride=1)
        self.layer2 = self._make_layer(16, 32, n, stride=2)
        self.layer3 = self._make_layer(32, 64, n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        self._init_weights()

    @staticmethod
    def _make_layer(in_planes: int, planes: int, n: int,
                    stride: int = 1) -> nn.Sequential:
        layers: list[nn.Module] = [PlainBlockCIFAR(in_planes, planes, stride)]
        for _ in range(1, n):
            layers.append(PlainBlockCIFAR(planes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def plain20(num_classes: int = 10) -> PlainNetCIFAR:
    return PlainNetCIFAR(n=3, num_classes=num_classes)


def plain32(num_classes: int = 10) -> PlainNetCIFAR:
    return PlainNetCIFAR(n=5, num_classes=num_classes)


def plain44(num_classes: int = 10) -> PlainNetCIFAR:
    return PlainNetCIFAR(n=7, num_classes=num_classes)


def plain56(num_classes: int = 10) -> PlainNetCIFAR:
    return PlainNetCIFAR(n=9, num_classes=num_classes)


def plain110(num_classes: int = 10) -> PlainNetCIFAR:
    return PlainNetCIFAR(n=18, num_classes=num_classes)


# Supported CIFAR-10 plain models
CIFAR_PLAIN_MODELS: dict[str, object] = {
    "plain20":  plain20,
    "plain32":  plain32,
    "plain44":  plain44,
    "plain56":  plain56,
    "plain110": plain110,
}
