"""
PlainNet for ImageNet — He et al., CVPR 2016 (Table 1).

Degradation-problem baselines with same depth/width as ResNet-18/34
but residual shortcuts removed.

Variants:
    PlainNet-18 : [2, 2, 2, 2]
    PlainNet-34 : [3, 4, 6, 3]
"""

import torch
import torch.nn as nn


class PlainBlock(nn.Module):
    """Two 3x3 convs, no shortcut."""

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


class PlainNet(nn.Module):
    """ImageNet plain network (7x7 stem, 4 stages)."""

    def __init__(self, layers: list[int], num_classes: int = 1000) -> None:
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64,  layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_layer(self, planes: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        layers: list[nn.Module] = []
        layers.append(PlainBlock(self.in_planes, planes, stride))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(PlainBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def plain18(num_classes: int = 1000) -> PlainNet:
    return PlainNet([2, 2, 2, 2], num_classes)


def plain34(num_classes: int = 1000) -> PlainNet:
    return PlainNet([3, 4, 6, 3], num_classes)


# Supported ImageNet plain models
IMAGENET_PLAIN_MODELS: dict[str, object] = {
    "plain18": plain18,
    "plain34": plain34,
}
