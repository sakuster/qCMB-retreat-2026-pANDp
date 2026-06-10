"""
train.py — Model training loop.

Handles class-imbalance weighting, learning-rate scheduling, and checkpoint
saving. The best model (by validation accuracy) is saved to results/best_model.pt
and loaded automatically by the evaluation step.
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from .model import PrionClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _class_weights(loader, num_classes: int, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency weights so rare classes are not dominated by
    the control group. Essential when dataset sizes are unequal across classes.
    """
    counts = torch.zeros(num_classes)
    for _, labels, _ in loader:
        for lbl in labels:
            counts[lbl] += 1
    # Samples with zero count are clamped to avoid division by zero
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    return weights.to(device)


def _run_epoch(model: PrionClassifier, loader, optimizer, criterion, device: torch.device, desc: str):
    """Single pass through loader. If optimizer is None, runs in eval mode."""
    training = optimizer is not None
    model.train(training)

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels, _ in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += images.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def train(config: dict, model: PrionClassifier, train_loader, val_loader) -> str:
    """
    Full training loop. Saves the best checkpoint and returns its file path.

    Training stops after config.training.epochs epochs. The model with the
    highest validation accuracy is kept — earlier epochs are discarded.
    """
    tr_cfg  = config["training"]
    out_dir = config["output"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "best_model.pt")

    use_gpu = tr_cfg.get("use_gpu", True)
    device  = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")
    if use_gpu and not torch.cuda.is_available():
        print("  Warning: use_gpu=true but no CUDA GPU found — falling back to CPU.")

    model = model.to(device)

    num_classes = model.classifier.out_features
    criterion   = nn.CrossEntropyLoss(weight=_class_weights(train_loader, num_classes, device))
    optimizer   = torch.optim.Adam(model.parameters(), lr=tr_cfg.get("learning_rate", 1e-4))
    # Halve the learning rate if validation loss plateaus for 5 epochs
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    epochs       = tr_cfg.get("epochs", 50)
    best_val_acc = 0.0

    print(f"  Training for {epochs} epoch(s)...\n")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(
            model, train_loader, optimizer, criterion, device,
            desc=f"Epoch {epoch:3d}/{epochs} [train]"
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, None, criterion, device,
            desc=f"Epoch {epoch:3d}/{epochs} [val  ]"
        )
        scheduler.step(val_loss)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            marker = "  ← best"

        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"train loss {train_loss:.4f}  acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f}  acc {val_acc:.3f}{marker}"
        )

    print(f"\n  Training complete. Best validation accuracy: {best_val_acc:.3f}")
    print(f"  Checkpoint saved to: {checkpoint_path}")
    return checkpoint_path
