"""
Inference script — run segmentation on a single image.

Produces:
1. A color-coded segmentation mask (pure mask)
2. An alpha-blended overlay on the original image
3. A version with a class legend

Usage:
    python src/inference.py --image test.jpg
    python src/inference.py --image test.jpg --output outputs/ --alpha 0.5
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from cityscapes_dataset import (
    CLASS_NAMES,
    CITYSCAPES_COLORMAP,
    colorize_mask,
    overlay_mask,
)
from config import IMAGE_SIZE, MODEL_PATH, NUM_CLASSES, OUTPUT_DIR
from model import get_model, load_model_weights


def load_model(model_path: str, device: torch.device):
    """Load trained model weights."""
    model = get_model(pretrained=False).to(device)
    load_model_weights(model, model_path, device)
    model.eval()
    return model


def preprocess_image(image: Image.Image, image_size: tuple) -> torch.Tensor:
    """Preprocess a PIL image for model input.

    Args:
        image:      PIL Image (RGB).
        image_size: (width, height) tuple.

    Returns:
        Tensor of shape (1, 3, H, W).
    """
    resized = image.resize(image_size, Image.BILINEAR)
    img_np = np.array(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    return tensor


def predict(model, image_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """Run model inference.

    Returns:
        H×W numpy array of predicted class labels.
    """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)["out"]
        prediction = output.argmax(1).squeeze(0).cpu().numpy()
    return prediction


def draw_legend(width: int = 200, height: int = 256) -> Image.Image:
    """Draw a vertical class legend with color swatches and names.

    Args:
        width:  Legend image width.
        height: Legend image height (will expand if needed).

    Returns:
        PIL Image with the legend.
    """
    num_classes = len(CLASS_NAMES)
    row_height = max(14, height // num_classes)
    legend_height = row_height * num_classes + 10

    legend = Image.new("RGB", (width, legend_height), (255, 255, 255))
    draw = ImageDraw.Draw(legend)

    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, (name, color) in enumerate(zip(CLASS_NAMES, CITYSCAPES_COLORMAP)):
        y = i * row_height + 5
        # Color swatch
        draw.rectangle(
            [5, y, 20, y + row_height - 2],
            fill=tuple(color),
            outline=(0, 0, 0),
        )
        # Class name
        draw.text((25, y), name, fill=(0, 0, 0), font=font)

    return legend


def main():
    parser = argparse.ArgumentParser(
        description="Run semantic segmentation on a single image"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--model-path", type=str, default=MODEL_PATH,
        help="Path to saved model weights",
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Overlay blending factor (0=image, 1=mask)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model ------------------------------------------------------
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, device)

    # --- Load and preprocess image ---------------------------------------
    print(f"Processing: {args.image}")
    original = Image.open(args.image).convert("RGB")
    image_tensor = preprocess_image(original, IMAGE_SIZE)

    # --- Predict ---------------------------------------------------------
    prediction = predict(model, image_tensor, device)
    print(f"Prediction shape: {prediction.shape}")

    # --- Colorize --------------------------------------------------------
    mask_rgb = colorize_mask(prediction)

    # Resize mask to original image size for overlay
    original_np = np.array(original.resize(IMAGE_SIZE, Image.BILINEAR))
    blended = overlay_mask(original_np, mask_rgb, alpha=args.alpha)

    # --- Save outputs ----------------------------------------------------
    os.makedirs(args.output, exist_ok=True)

    mask_path = os.path.join(args.output, "prediction_mask.png")
    overlay_path = os.path.join(args.output, "prediction_overlay.png")
    legend_path = os.path.join(args.output, "prediction_with_legend.png")

    Image.fromarray(mask_rgb).save(mask_path)
    print(f"  Mask saved:    {mask_path}")

    Image.fromarray(blended).save(overlay_path)
    print(f"  Overlay saved: {overlay_path}")

    # --- Create composite with legend ------------------------------------
    overlay_img = Image.fromarray(blended)
    legend_img = draw_legend(width=160, height=overlay_img.height)

    composite_width = overlay_img.width + legend_img.width + 10
    composite_height = max(overlay_img.height, legend_img.height)
    composite = Image.new("RGB", (composite_width, composite_height), (255, 255, 255))
    composite.paste(overlay_img, (0, 0))
    composite.paste(legend_img, (overlay_img.width + 10, 0))
    composite.save(legend_path)
    print(f"  Legend saved:  {legend_path}")

    print("\nInference complete!")


if __name__ == "__main__":
    main()