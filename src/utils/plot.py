"""Training curve visualization."""

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # headless-safe backend — must be set before importing pyplot
import matplotlib.pyplot as plt


def plot_training_curves(
    train_losses: Sequence[float],
    train_accs: Sequence[float],
    val_accs: Sequence[float],
    model_name: str = "model",
    save_dir: str | Path = ".",
) -> None:
    """Plot loss and accuracy curves and save to ``save_dir``."""
    epochs = range(1, len(train_losses) + 1)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} — Training Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_accs, label="Train Acc")
    ax2.plot(epochs, val_accs,   label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{model_name} — Accuracy")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    out_path = save_dir / f"{model_name}_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves → {out_path}")
