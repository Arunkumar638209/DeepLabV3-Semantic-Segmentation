"""
Training script for Cityscapes semantic segmentation.

Features:
- Configurable via command-line arguments (overriding config.py defaults)
- Validation loop with mIoU + pixel accuracy after each epoch
- Polynomial LR decay (standard for segmentation fine-tuning)
- Optional class-weighted CrossEntropyLoss for imbalance handling
- Mixed precision training (AMP) for faster GPU utilization
- TensorBoard logging of train/val loss, mIoU, pixel accuracy
- Best model saved by validation mIoU (not train loss)
- Full checkpoint saving for training resumption
- Reproducible via seed setting

Usage:
    python src/train.py
    python src/train.py --epochs 10 --lr 5e-5 --batch-size 8
    python src/train.py --resume
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
from tqdm import tqdm

from cityscapes_dataset import CityscapesDataset
from config import (
    BATCH_SIZE, CHECKPOINT_PATH, FREEZE_BN, IGNORE_LABEL, IMAGE_DIR_TRAIN,
    IMAGE_DIR_VAL, IMAGE_SIZE, LEARNING_RATE, LOG_INTERVAL, MASK_DIR_TRAIN,
    MASK_DIR_VAL, MODEL_PATH, MODEL_SAVE_DIR, NUM_CLASSES, NUM_EPOCHS,
    NUM_WORKERS, PIN_MEMORY, POLY_POWER, SEED, STEP_GAMMA, STEP_SIZE,
    TENSORBOARD_DIR, USE_AMP, USE_CLASS_WEIGHTS, WEIGHT_DECAY,
)
from model import get_model


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_batchnorm(model: nn.Module) -> None:
    """Set all BatchNorm layers to eval mode (freeze running stats).

    When fine-tuning a pretrained model with small batch sizes, updating
    BatchNorm statistics with mini-batch statistics introduces noise.
    Freezing them keeps the ImageNet-learned statistics stable.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    ignore_label: int = 255,
) -> tuple:
    """Compute pixel accuracy and per-class IoU.

    Args:
        preds:        N×H×W predicted class labels.
        targets:      N×H×W ground-truth class labels.
        num_classes:  Number of classes.
        ignore_label: Label to ignore in metric computation.

    Returns:
        (pixel_accuracy, per_class_iou) where per_class_iou is a numpy array
        of shape (num_classes,).
    """
    valid = targets != ignore_label

    correct = ((preds == targets) & valid).sum()
    total = valid.sum()
    pixel_acc = correct / (total + 1e-10)

    intersection = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)

    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = targets == cls
        valid_mask = valid

        intersection[cls] = (pred_cls & target_cls & valid_mask).sum()
        union[cls] = (
            (pred_cls & valid_mask).sum()
            + (target_cls & valid_mask).sum()
            - intersection[cls]
        )

    iou = intersection / (union + 1e-10)
    return float(pixel_acc), iou


def polynomial_lr_lambda(epoch: int, max_epochs: int, power: float = 0.9):
    """Polynomial LR decay schedule: lr = base_lr × (1 - epoch/max_epochs)^power."""
    return (1.0 - epoch / max_epochs) ** power


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Run validation and compute loss, pixel accuracy, and mIoU."""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = outputs.argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(masks.cpu().numpy())

    avg_loss = total_loss / len(val_loader)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    pixel_acc, per_class_iou = compute_metrics(
        all_preds, all_targets, NUM_CLASSES, IGNORE_LABEL
    )
    miou = float(np.mean(per_class_iou))

    return {
        "loss": avg_loss,
        "pixel_acc": pixel_acc,
        "miou": miou,
        "per_class_iou": per_class_iou,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train DeepLabV3-ResNet50 on Cityscapes"
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Disable class-weighted loss")
    args = parser.parse_args()

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # --- Datasets --------------------------------------------------------
    print("\nLoading datasets...")
    train_dataset = CityscapesDataset(
        image_dir=IMAGE_DIR_TRAIN,
        mask_dir=MASK_DIR_TRAIN,
        image_size=IMAGE_SIZE,
        augment=not args.no_augment,
    )
    val_dataset = CityscapesDataset(
        image_dir=IMAGE_DIR_VAL,
        mask_dir=MASK_DIR_VAL,
        image_size=IMAGE_SIZE,
        augment=False,
    )
    print(f"Train: {len(train_dataset)} images")
    print(f"Val:   {len(val_dataset)} images")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # --- Model -----------------------------------------------------------
    print("\nBuilding model...")
    model = get_model().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # --- Loss ------------------------------------------------------------
    class_weights = None
    if USE_CLASS_WEIGHTS and not args.no_class_weights:
        print("\nComputing class weights from training masks...")
        class_weights = CityscapesDataset.compute_class_weights(
            MASK_DIR_TRAIN, IMAGE_SIZE, NUM_CLASSES
        ).to(device)
        print("Class weights:", class_weights.cpu().numpy().round(3))

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=IGNORE_LABEL,
    )

    # --- Optimizer & Scheduler -------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: polynomial_lr_lambda(
            epoch, args.epochs, POLY_POWER
        ),
    )

    # --- AMP scaler ------------------------------------------------------
    use_amp = USE_AMP and not args.no_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"\nMixed precision: {'enabled' if use_amp else 'disabled'}")

    # --- TensorBoard -----------------------------------------------------
    writer = None
    if HAS_TENSORBOARD:
        os.makedirs(TENSORBOARD_DIR, exist_ok=True)
        writer = SummaryWriter(log_dir=TENSORBOARD_DIR)
    else:
        print("Note: TensorBoard not installed. Install with: pip install tensorboard")

    # --- Resume ----------------------------------------------------------
    start_epoch = 0
    best_miou = 0.0

    if args.resume and os.path.exists(CHECKPOINT_PATH):
        print(f"\nResuming from {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("best_miou", 0.0)
        print(f"Resumed at epoch {start_epoch}, best mIoU: {best_miou:.4f}")

    # --- Create model save dir -------------------------------------------
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # --- Training loop ---------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Training for {args.epochs} epochs")
    print(f"Batch size: {args.batch_size} | LR: {args.lr}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if FREEZE_BN:
            freeze_batchnorm(model)

        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            leave=True,
        )

        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)["out"]
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / num_batches
        current_lr = optimizer.param_groups[0]["lr"]

        # --- Validation --------------------------------------------------
        val_metrics = validate(model, val_loader, criterion, device)

        # --- Logging -----------------------------------------------------
        print(
            f"\nEpoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val mIoU: {val_metrics['miou']:.4f} | "
            f"Val PixAcc: {val_metrics['pixel_acc']:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if writer is not None:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("mIoU/val", val_metrics["miou"], epoch)
            writer.add_scalar("PixelAccuracy/val", val_metrics["pixel_acc"], epoch)
            writer.add_scalar("LearningRate", current_lr, epoch)

        # --- Save best model ---------------------------------------------
        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ New best model saved (mIoU: {best_miou:.4f})")

        # --- Save checkpoint ---------------------------------------------
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_miou": best_miou,
            },
            CHECKPOINT_PATH,
        )

        scheduler.step()

    # --- Cleanup ---------------------------------------------------------
    if writer is not None:
        writer.close()

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation mIoU: {best_miou:.4f}")
    print(f"Best model saved to: {MODEL_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()