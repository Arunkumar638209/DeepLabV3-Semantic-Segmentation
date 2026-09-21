"""
Consolidated evaluation script for Cityscapes semantic segmentation.

Reports:
- Overall pixel accuracy
- Mean IoU (mIoU) across all 19 classes
- Per-class IoU with human-readable class names
- Saves results to outputs/evaluation_results.json

Usage:
    python src/evaluate.py
    python src/evaluate.py --model-path models/best_model.pth
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cityscapes_dataset import CityscapesDataset, CLASS_NAMES
from config import (
    BATCH_SIZE, IGNORE_LABEL, IMAGE_DIR_VAL, IMAGE_SIZE, MASK_DIR_VAL,
    MODEL_PATH, NUM_CLASSES, NUM_WORKERS, OUTPUT_DIR, PIN_MEMORY,
)
from model import get_model


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
    ignore_label: int = IGNORE_LABEL,
) -> dict:
    """Run full evaluation: pixel accuracy + per-class IoU + mIoU.

    Returns:
        Dictionary with keys: pixel_accuracy, miou, per_class_iou (dict).
    """
    model.eval()

    correct_pixels = 0
    total_pixels = 0
    intersection = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images = images.to(device)

            outputs = model(images)["out"]
            preds = outputs.argmax(1).cpu().numpy()
            masks_np = masks.numpy()

            valid = masks_np != ignore_label

            # Pixel accuracy
            correct_pixels += ((preds == masks_np) & valid).sum()
            total_pixels += valid.sum()

            # Per-class IoU
            for cls in range(num_classes):
                pred_cls = preds == cls
                mask_cls = masks_np == cls

                intersection[cls] += (pred_cls & mask_cls & valid).sum()
                union[cls] += (
                    (pred_cls & valid).sum()
                    + (mask_cls & valid).sum()
                    - (pred_cls & mask_cls & valid).sum()
                )

    pixel_accuracy = float(correct_pixels / (total_pixels + 1e-10))
    per_class_iou = intersection / (union + 1e-10)
    miou = float(np.mean(per_class_iou))

    # Build per-class results dict
    per_class_results = {}
    for cls in range(num_classes):
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class_{cls}"
        per_class_results[name] = round(float(per_class_iou[cls]), 4)

    return {
        "pixel_accuracy": round(pixel_accuracy, 4),
        "miou": round(miou, 4),
        "per_class_iou": per_class_results,
    }


def print_results(results: dict) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'='*50}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  Pixel Accuracy:  {results['pixel_accuracy']:.4f}")
    print(f"  Mean IoU (mIoU): {results['miou']:.4f}")
    print(f"{'='*50}")
    print(f"\n  {'Class':<20} {'IoU':>8}")
    print(f"  {'-'*28}")

    for class_name, iou in results["per_class_iou"].items():
        bar = "#" * int(iou * 20)
        print(f"  {class_name:<20} {iou:>8.4f}  {bar}")

    print(f"  {'-'*28}")
    print(f"  {'Mean IoU':<20} {results['miou']:>8.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DeepLabV3-ResNet50 on Cityscapes validation set"
    )
    parser.add_argument(
        "--model-path", type=str, default=MODEL_PATH,
        help="Path to saved model weights",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset ---------------------------------------------------------
    print("Loading validation dataset...")
    val_dataset = CityscapesDataset(
        image_dir=IMAGE_DIR_VAL,
        mask_dir=MASK_DIR_VAL,
        image_size=IMAGE_SIZE,
        augment=False,
    )
    print(f"Validation images: {len(val_dataset)}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # --- Model -----------------------------------------------------------
    print(f"Loading model from {args.model_path}")
    model = get_model(pretrained=False).to(device)

    from model import load_model_weights
    load_model_weights(model, args.model_path, device)

    # --- Evaluate --------------------------------------------------------
    results = evaluate(model, val_loader, device)
    print_results(results)

    # --- Save results ----------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()