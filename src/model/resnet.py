"""
ResNet for ImageNet — He et al., CVPR 2016 (Table 1).

Variants:
    ResNet-18  : BasicBlock, [2, 2, 2, 2]
    ResNet-34  : BasicBlock, [3, 4, 6, 3]
    ResNet-50  : Bottleneck, [3, 4, 6, 3]
    ResNet-101 : Bottleneck, [3, 4, 23, 3]
    ResNet-152 : Bottleneck, [3, 8, 36, 3]
"""

import torch
import torch.nn as nn


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    """Two 3x3 conv layers with a residual shortcut (ResNet-18/34)."""
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample  # None when dims match

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class Bottleneck(nn.Module):
    """1x1 → 3x3 → 1x1 conv bottleneck with a residual shortcut (ResNet-50+)."""
    expansion: int = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = _conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = _conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    """ImageNet ResNet (Section 3 of the paper)."""

    def __init__(self, block: type[BasicBlock] | type[Bottleneck],
                 layers: list[int], num_classes: int = 1000) -> None:
        super().__init__()
        self.in_planes = 64

        # Stem: 7x7 conv, stride 2 → BN → ReLU → 3x3 maxpool stride 2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block: type[BasicBlock] | type[Bottleneck],
                    planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            # Option B: 1x1 projection shortcut when dims change
            downsample = nn.Sequential(
                _conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
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


def resnet18(num_classes: int = 1000) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes: int = 1000) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes: int = 1000) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def resnet101(num_classes: int = 1000) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def resnet152(num_classes: int = 1000) -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


# Supported ImageNet ResNet models
IMAGENET_RESNET_MODELS: dict[str, object] = {
    "resnet18":  resnet18,
    "resnet34":  resnet34,
    "resnet50":  resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}
