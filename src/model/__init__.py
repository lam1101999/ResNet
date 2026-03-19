from .resnet import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    IMAGENET_RESNET_MODELS,
)
from .resnet_cifar import (
    resnet20, resnet32, resnet44, resnet56, resnet110,
    CIFAR_RESNET_MODELS,
)
from .plain_net import plain18, plain34, IMAGENET_PLAIN_MODELS
from .plain_net_cifar import (
    plain20, plain32, plain44, plain56, plain110,
    CIFAR_PLAIN_MODELS,
)
from .vgg import vgg16, vgg16_bn, vgg19, vgg19_bn, IMAGENET_VGG_MODELS

# Combined registries — used by main.py and verify_all
CIFAR_MODELS: dict[str, object] = {
    **CIFAR_RESNET_MODELS,
    **CIFAR_PLAIN_MODELS,
}

IMAGENET_MODELS: dict[str, object] = {
    **IMAGENET_RESNET_MODELS,
    **IMAGENET_PLAIN_MODELS,
    **IMAGENET_VGG_MODELS,
}

__all__ = [
    # Individual factories
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "resnet20", "resnet32", "resnet44", "resnet56", "resnet110",
    "plain18", "plain34", "plain20", "plain32", "plain44", "plain56", "plain110",
    "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
    # Registries
    "CIFAR_MODELS", "IMAGENET_MODELS",
]
