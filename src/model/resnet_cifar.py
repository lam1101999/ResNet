"""
ResNet for CIFAR-10 — He et al., CVPR 2016 (Section 4.2).

Architecture:
    - Input 32x32 images (no maxpool, 3x3 first conv instead of 7x7)
    - 3 stages, each with n BasicBlocks; channels: 16 → 32 → 64
    - Total layers = 6n + 2

Variants (n → model name → reported error):
    n=3  → ResNet-20  → 8.75 %
    n=5  → ResNet-32  → 7.51 %
    n=7  → ResNet-44  → 7.17 %
    n=9  → ResNet-56  → 6.97 %
    n=18 → ResNet-110 → 6.43 %
"""

import torch
import torch.nn as nn


class BasicBlockCIFAR(nn.Module):
    """Two 3x3 conv layers with optional projection shortcut."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut: nn.Module
        if stride != 1 or in_planes != planes:
            # Option B: 1x1 projection to match dimensions
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNetCIFAR(nn.Module):
    """CIFAR-10 ResNet with 6n+2 layers (Section 4.2)."""

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
        layers = [BasicBlockCIFAR(in_planes, planes, stride)]
        for _ in range(1, n):
            layers.append(BasicBlockCIFAR(planes, planes))
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


def resnet20(num_classes: int = 10) -> ResNetCIFAR:
    return ResNetCIFAR(n=3, num_classes=num_classes)


def resnet32(num_classes: int = 10) -> ResNetCIFAR:
    return ResNetCIFAR(n=5, num_classes=num_classes)


def resnet44(num_classes: int = 10) -> ResNetCIFAR:
    return ResNetCIFAR(n=7, num_classes=num_classes)


def resnet56(num_classes: int = 10) -> ResNetCIFAR:
    return ResNetCIFAR(n=9, num_classes=num_classes)


def resnet110(num_classes: int = 10) -> ResNetCIFAR:
    return ResNetCIFAR(n=18, num_classes=num_classes)


# Supported CIFAR-10 ResNet models
CIFAR_RESNET_MODELS: dict[str, object] = {
    "resnet20":  resnet20,
    "resnet32":  resnet32,
    "resnet44":  resnet44,
    "resnet56":  resnet56,
    "resnet110": resnet110,
}
