"""
ImageNet data loaders via Hugging Face datasets (ILSVRC/imagenet-1k).

Train transforms (He et al., Section 4.1):
    - Random resized crop to 224×224
    - Random horizontal flip
    - Normalize with ImageNet channel statistics

Val transforms:
    - Resize shortest side to 256, center crop to 224×224
    - Normalize with ImageNet channel statistics
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class HFImageNetDataset(Dataset):
    """Wraps a Hugging Face ImageNet split as a PyTorch Dataset."""

    def __init__(self, hf_split, transform=None):
        self.data = hf_split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        image = sample["image"].convert("RGB")
        label = sample["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_imagenet_loaders(
    batch_size: int = 256,
    num_workers: int = 4,
    cache_dir: str | Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for ImageNet-1k via Hugging Face.

    Requires prior authentication:  uv run hf auth login
    and accepting terms at:         https://huggingface.co/datasets/ILSVRC/imagenet-1k
    """
    from datasets import load_dataset  # imported lazily to keep startup fast

    cache_dir = str(cache_dir) if cache_dir else None

    print("Loading ImageNet-1k from Hugging Face (this may take a while)...")
    train_hf = load_dataset(
        "ILSVRC/imagenet-1k",
        split="train",
        cache_dir=cache_dir,
    )
    val_hf = load_dataset(
        "ILSVRC/imagenet-1k",
        split="validation",
        cache_dir=cache_dir,
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_set = HFImageNetDataset(train_hf, transform=train_transform)
    val_set   = HFImageNetDataset(val_hf,   transform=val_transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
