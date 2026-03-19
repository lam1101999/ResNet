"""
Evaluation utilities — top-1 / top-5 accuracy on a DataLoader.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device = "cpu",
    topk: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Evaluate ``model`` on ``loader`` and return top-k accuracies.

    Returns
    -------
    dict mapping e.g. ``"top1"`` → accuracy percentage.
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    totals = {k: 0.0 for k in topk}
    n_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            batch = labels.size(0)
            n_samples += batch

            maxk = max(topk)
            _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))

            for k in topk:
                totals[k] += correct[:k].reshape(-1).float().sum().item()

    return {f"top{k}": totals[k] / n_samples * 100.0 for k in topk}
