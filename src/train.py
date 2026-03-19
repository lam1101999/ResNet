"""
Training loop — He et al., CVPR 2016, Section 3.4.

Optimizer : SGD, momentum=0.9, weight_decay=1e-4
Schedule  : lr starts at 0.1, divided by 10 at iterations 32k and 48k
            (≈ epoch 82 and 123 for CIFAR-10 with batch_size=128)
Total     : 64k iterations (~164 epochs)

Output layout (per model, under weights/<model_name>/)
-------------------------------------------------------
    last.pth     — saved every epoch (full state for resume)
    best.pth     — saved only when val accuracy improves
    metrics.csv  — epoch, train_loss, train_acc, val_acc, lr
    events.*     — TensorBoard logs  (tensorboard --logdir weights)
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_acc: float,
    history: dict[str, list[float]],
) -> None:
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc":         best_val_acc,
        "history":              history,
    }, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[int, float, dict[str, list[float]]]:
    """Load checkpoint into model/optimizer/scheduler; return (epoch, best_val_acc, history)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    history = ckpt.get("history", {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []})
    return ckpt["epoch"], ckpt["best_val_acc"], history


# ── Internal helpers ──────────────────────────────────────────────────────────

def _accuracy(output: torch.Tensor, target: torch.Tensor,
               topk: tuple[int, ...] = (1,)) -> list[float]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [
            correct[:k].reshape(-1).float().sum(0).item() * 100.0 / batch_size
            for k in topk
        ]


def _init_csv(path: Path, resume: bool) -> None:
    """Write CSV header only when starting fresh."""
    if not resume or not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])


def _append_csv(path: Path, epoch: int, train_loss: float,
                train_acc: float, val_loss: float, val_acc: float, lr: float) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.4f}",
                         f"{val_loss:.6f}", f"{val_acc:.4f}", f"{lr:.6f}"])


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 164,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    milestones: list[int] | None = None,
    lr_gamma: float = 0.1,
    device: str | torch.device = "cpu",
    weights_dir: str | Path = "weights",
    model_name: str = "model",
    start_epoch: int = 1,
    best_val_acc: float = 0.0,
    history: dict[str, list[float]] | None = None,
) -> dict[str, list[float]]:
    """Train ``model`` and return history dict.

    All outputs (checkpoints, TensorBoard events, metrics CSV) are saved
    under ``weights_dir/<model_name>/``.
    """
    if milestones is None:
        milestones = [82, 123]
    if history is None:
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    device = torch.device(device)
    model = model.to(device)

    # Each model gets its own subfolder
    model_dir = Path(weights_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    last_path = model_dir / "last.pth"
    best_path = model_dir / "best.pth"
    csv_path  = model_dir / "metrics.csv"

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum,
                    weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=lr_gamma)

    resuming = start_epoch > 1
    if resuming and last_path.exists():
        _, _, _ = load_checkpoint(last_path, model, optimizer, scheduler, device)
        print(f"Resumed optimizer/scheduler state from {last_path}")

    _init_csv(csv_path, resume=resuming)

    writer = SummaryWriter(log_dir=str(model_dir))

    print(f"Model dir   : {model_dir}")
    print(f"Last ckpt   : {last_path}")
    print(f"Best ckpt   : {best_path}")
    print(f"Metrics CSV : {csv_path}")
    print(f"TensorBoard : tensorboard --logdir {Path(weights_dir)}")

    for epoch in range(start_epoch, epochs + 1):
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:>3}/{epochs}  lr={current_lr:.2e}")

        # ---- training ----
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for images, labels in tqdm(train_loader, desc="  train", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc  += _accuracy(outputs, labels)[0]

        epoch_loss = running_loss / len(train_loader)
        epoch_acc  = running_acc  / len(train_loader)

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="  val  ", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                val_acc += _accuracy(outputs, labels)[0]
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        scheduler.step()

        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  train  loss={epoch_loss:.4f}  acc={epoch_acc:.2f}%")
        print(f"  val    loss={val_loss:.4f}  acc={val_acc:.2f}%")

        # ---- TensorBoard ----
        writer.add_scalars("Loss",     {"train": epoch_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": epoch_acc, "val": val_acc}, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        # ---- CSV ----
        _append_csv(csv_path, epoch, epoch_loss, epoch_acc, val_loss, val_acc, float(current_lr))

        # ---- save last (always) ----
        save_checkpoint(last_path, model, optimizer, scheduler,
                        epoch, best_val_acc, history)

        # ---- save best ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(best_path, model, optimizer, scheduler,
                            epoch, best_val_acc, history)
            print(f"  ✓ New best val acc {best_val_acc:.2f}% — saved to {best_path}")

    writer.close()
    print(f"\nBest val acc: {best_val_acc:.2f}%")
    return history
