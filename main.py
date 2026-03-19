"""
ResNet reimplementation — He et al., CVPR 2016.

Usage examples:
    # Single model
    python main.py --model resnet56 --dataset cifar10 --epochs 164

    # All CIFAR-10 models
    python main.py --model resnet20 resnet32 resnet44 resnet56 resnet110 plain20 plain32 plain44 plain56 plain110 --dataset cifar10 --epochs 100

    # All ImageNet models
    python main.py --model resnet18 resnet34 resnet50 resnet101 resnet152 plain18 plain34 vgg16 vgg16_bn vgg19 vgg19_bn --dataset imagenet --epochs 100 --batch-size 256

    # Resume interrupted run(s) — models already at --epochs are skipped
    python main.py --model resnet56 plain56 --dataset cifar10 --epochs 100 --resume

    # Evaluate a saved checkpoint
    python main.py --model resnet56 --dataset cifar10 --eval-only --checkpoint weights/resnet56/best.pth

    # Forward-pass shape check for all models
    python main.py --verify

TensorBoard:
    tensorboard --logdir weights
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.model import CIFAR_MODELS, IMAGENET_MODELS


# ── Verification ─────────────────────────────────────────────────────────────

def verify_all() -> None:
    cifar_dummy  = torch.zeros(2, 3, 32, 32)
    imgnet_dummy = torch.zeros(2, 3, 224, 224)

    print("── CIFAR-10 models (input 32×32, 10 classes) ──")
    for name, factory in CIFAR_MODELS.items():
        model = factory()
        model.eval()
        with torch.no_grad():
            out = model(cifar_dummy)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        status = "OK" if out.shape == (2, 10) else f"FAIL {out.shape}"
        print(f"  {name:<12}  output={out.shape}  params={params:.2f}M  [{status}]")

    print("\n── ImageNet models (input 224×224, 1000 classes) ──")
    for name, factory in IMAGENET_MODELS.items():
        model = factory()
        model.eval()
        with torch.no_grad():
            out = model(imgnet_dummy)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        status = "OK" if out.shape == (2, 1000) else f"FAIL {out.shape}"
        print(f"  {name:<12}  output={out.shape}  params={params:.2f}M  [{status}]")


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def build_model(model_name: str, num_classes: int) -> torch.nn.Module:
    registry = {**CIFAR_MODELS, **IMAGENET_MODELS}
    if model_name not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    return registry[model_name](num_classes=num_classes)


# ── Per-model train / eval ────────────────────────────────────────────────────

def run_one(model_name: str, num_classes: int, train_loader, test_loader,
            args, device: torch.device) -> None:
    # Folder name combines model + dataset so runs never collide
    save_name = f"{model_name}_{args.dataset}"

    model = build_model(model_name, num_classes)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {model_name}  dataset: {args.dataset}  ({n_params:.2f}M params)")

    if args.eval_only:
        ckpt_path = args.checkpoint or str(Path(args.weights_dir) / save_name / "best.pth")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded: {ckpt_path}  "
              f"(epoch {ckpt.get('epoch', '?')}, "
              f"val_acc {ckpt.get('best_val_acc', ckpt.get('val_acc', '?')):.2f}%)")
    else:
        start_epoch = 1
        best_val_acc = 0.0
        history = None

        if args.resume:
            last_path = Path(args.weights_dir) / save_name / "last.pth"
            if last_path.exists():
                from src.train import load_checkpoint
                from torch.optim import SGD
                from torch.optim.lr_scheduler import MultiStepLR
                _opt = SGD(model.parameters(), lr=args.lr,
                           momentum=args.momentum, weight_decay=args.weight_decay)
                _sch = MultiStepLR(_opt, milestones=args.milestones, gamma=0.1)
                start_epoch, best_val_acc, history = load_checkpoint(
                    last_path, model, _opt, _sch, device)
                start_epoch += 1
                print(f"Resuming from epoch {start_epoch}  "
                      f"(best val acc so far: {best_val_acc:.2f}%)")
                if start_epoch > args.epochs:
                    print(f"Already completed {args.epochs} epochs — skipping.")
                    return
            else:
                print(f"No checkpoint at {last_path}, starting from scratch.")

        from src.train import train
        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=args.epochs,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            milestones=args.milestones,
            device=device,
            weights_dir=args.weights_dir,
            model_name=save_name,
            start_epoch=start_epoch,
            best_val_acc=best_val_acc,
            history=history,
        )

        from src.utils import plot_training_curves
        plot_training_curves(
            train_losses=history["train_loss"],
            train_accs=history["train_acc"],
            val_accs=history["val_acc"],
            model_name=save_name,
            save_dir=Path(args.weights_dir) / save_name,
        )

    from src.evaluate import evaluate
    results = evaluate(model, test_loader, device=device, topk=(1, 5))
    print(f"── Test Results ({save_name}) ──")
    for k, v in results.items():
        print(f"  {k}: {v:.2f}%  (error: {100 - v:.2f}%)")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ResNet reimplementation — He et al., CVPR 2016",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--verify", action="store_true",
                        help="Run forward-pass shape check for all models and exit")
    parser.add_argument("--model", nargs="+", default=["resnet56"],
                        metavar="MODEL",
                        help="One or more model names to train sequentially")
    parser.add_argument("--dataset", default="cifar10",
                        choices=["cifar10", "imagenet"],
                        help="Dataset to use")
    parser.add_argument("--data-dir", default="./data",
                        help="Path to dataset root (cifar10 only)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--milestones", type=int, nargs="+", default=[82, 123],
                        help="Epochs at which to drop lr by 10×")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader num_workers")
    parser.add_argument("--device", default="auto",
                        help="'auto', 'cpu', 'cuda', or 'mps'")
    parser.add_argument("--weights-dir", default="weights",
                        help="Root directory for per-model weight folders")
    parser.add_argument("--plot-dir", default=None,
                        help="Directory for training curve plots "
                             "(default: weights/<model>/)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume each model from weights/<model>/last.pth; "
                             "models already at --epochs are skipped")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate")
    parser.add_argument("--checkpoint",
                        help="Checkpoint path for --eval-only (single model only); "
                             "defaults to weights/<model>/best.pth")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.verify:
        verify_all()
        return

    device = resolve_device(args.device)
    print(f"Device: {device}")

    # ── Data (loaded once, shared across all models) ───────────────────────
    if args.dataset == "cifar10":
        from src.data import get_cifar10_loaders
        train_loader, test_loader = get_cifar10_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        num_classes = 10
    elif args.dataset == "imagenet":
        from src.data import get_imagenet_loaders
        train_loader, test_loader = get_imagenet_loaders(
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        num_classes = 1000
    else:
        print(f"Dataset '{args.dataset}' not supported yet.", file=sys.stderr)
        sys.exit(1)

    # ── Train / evaluate each model sequentially ───────────────────────────
    for i, model_name in enumerate(args.model):
        print(f"\n{'='*60}")
        print(f"[{i + 1}/{len(args.model)}] {model_name}")
        print(f"{'='*60}")
        run_one(model_name, num_classes, train_loader, test_loader, args, device)

    if len(args.model) > 1:
        print(f"\nAll done. Trained {len(args.model)} models.")
        print(f"View all curves: tensorboard --logdir {args.weights_dir}")


if __name__ == "__main__":
    main()
