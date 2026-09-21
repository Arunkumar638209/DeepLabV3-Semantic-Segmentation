"""
Cityscapes dataset loader for semantic segmentation.

Handles:
- Pairing leftImg8bit images with gtFine_labelIds masks
- Resizing to the configured resolution
- Label remapping (raw IDs → 19 trainIds + ignore)
- Optional data augmentation (horizontal flip, color jitter)
- Centralized class names and color map for visualization
"""

import os
import glob

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from label_mapping import encode_segmap

# ---------------------------------------------------------------------------
# Cityscapes 19 evaluation classes — names and official colors
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    "road",           # 0
    "sidewalk",       # 1
    "building",       # 2
    "wall",           # 3
    "fence",          # 4
    "pole",           # 5
    "traffic light",  # 6
    "traffic sign",   # 7
    "vegetation",     # 8
    "terrain",        # 9
    "sky",            # 10
    "person",         # 11
    "rider",          # 12
    "car",            # 13
    "truck",          # 14
    "bus",            # 15
    "train",          # 16
    "motorcycle",     # 17
    "bicycle",        # 18
]

# Official Cityscapes color palette (RGB) indexed by trainId.
CITYSCAPES_COLORMAP = [
    [128, 64, 128],    # road
    [244, 35, 232],    # sidewalk
    [70, 70, 70],      # building
    [102, 102, 156],   # wall
    [190, 153, 153],   # fence
    [153, 153, 153],   # pole
    [250, 170, 30],    # traffic light
    [220, 220, 0],     # traffic sign
    [107, 142, 35],    # vegetation
    [152, 251, 152],   # terrain
    [70, 130, 180],    # sky
    [220, 20, 60],     # person
    [255, 0, 0],       # rider
    [0, 0, 142],       # car
    [0, 0, 70],        # truck
    [0, 60, 100],      # bus
    [0, 80, 100],      # train
    [0, 0, 230],       # motorcycle
    [119, 11, 32],     # bicycle
]


def colorize_mask(prediction: np.ndarray) -> np.ndarray:
    """Convert a trainId prediction map to an RGB color image.

    Args:
        prediction: H×W numpy array with values in {0..18}.

    Returns:
        H×W×3 numpy array (uint8) with Cityscapes colors.
    """
    h, w = prediction.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id, color in enumerate(CITYSCAPES_COLORMAP):
        color_image[prediction == cls_id] = color

    return color_image


def overlay_mask(image: np.ndarray, mask_rgb: np.ndarray,
                 alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend a color mask onto the original image.

    Args:
        image:    H×W×3 original image (uint8).
        mask_rgb: H×W×3 colorized mask (uint8).
        alpha:    Blending factor for the mask (0 = image only, 1 = mask only).

    Returns:
        H×W×3 blended image (uint8).
    """
    blended = (
        (1 - alpha) * image.astype(np.float32)
        + alpha * mask_rgb.astype(np.float32)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


class CityscapesDataset(Dataset):
    """PyTorch dataset for Cityscapes semantic segmentation.

    Expects the standard Cityscapes directory layout::

        leftImg8bit/
            train/ (or val/)
                <city>/
                    <city>_<seq>_<frame>_leftImg8bit.png
        gtFine/
            train/ (or val/)
                <city>/
                    <city>_<seq>_<frame>_gtFine_labelIds.png

    Args:
        image_dir:  Path to ``leftImg8bit/<split>`` directory.
        mask_dir:   Path to ``gtFine/<split>`` directory.
        image_size: (width, height) to resize images and masks.
        augment:    If True, apply random horizontal flip and color jitter.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: tuple = (512, 256),
        augment: bool = False,
    ):
        self.image_size = image_size
        self.augment = augment

        self.images = []
        self.masks = []

        cities = sorted(os.listdir(image_dir))

        for city in cities:
            image_paths = sorted(
                glob.glob(
                    os.path.join(image_dir, city, "*_leftImg8bit.png")
                )
            )

            for img_path in image_paths:
                file_name = os.path.basename(img_path)
                mask_name = file_name.replace(
                    "_leftImg8bit.png",
                    "_gtFine_labelIds.png"
                )
                mask_path = os.path.join(mask_dir, city, mask_name)

                if os.path.exists(mask_path):
                    self.images.append(img_path)
                    self.masks.append(mask_path)

        if len(self.images) == 0:
            raise RuntimeError(
                f"No image-mask pairs found.\n"
                f"  image_dir: {image_dir}\n"
                f"  mask_dir:  {mask_dir}"
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        # --- Load --------------------------------------------------------
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])

        # --- Resize ------------------------------------------------------
        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)

        # --- Augmentation (joint transforms) -----------------------------
        if self.augment:
            # Random horizontal flip (applied to both image and mask)
            if torch.rand(1).item() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Color jitter (image only — mask is categorical)
            color_jitter = T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            )
            image = color_jitter(image)

        # --- To numpy & remap labels ------------------------------------
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask)
        mask = encode_segmap(mask)

        # --- To tensor ---------------------------------------------------
        image = torch.from_numpy(image).permute(2, 0, 1)  # C×H×W
        mask = torch.from_numpy(mask).long()               # H×W

        return image, mask

    @staticmethod
    def compute_class_weights(
        mask_dir: str,
        image_size: tuple = (512, 256),
        num_classes: int = 19,
    ) -> torch.Tensor:
        """Compute inverse-frequency class weights from training masks.

        Scans all masks in ``mask_dir``, counts per-class pixel frequencies,
        and returns weights = median_freq / class_freq.  Classes with zero
        pixels receive weight 0 (they will also never appear in the loss
        because ``ignore_index=255`` is used).

        This addresses the severe class imbalance in Cityscapes where
        road and building dominate, while pole and traffic light have
        very few pixels.

        Args:
            mask_dir:    Path to ``gtFine/<split>`` directory.
            image_size:  (width, height) — masks are resized to match.
            num_classes: Number of evaluation classes (19 for Cityscapes).

        Returns:
            Float tensor of shape (num_classes,) with class weights.
        """
        counts = np.zeros(num_classes, dtype=np.float64)

        cities = sorted(os.listdir(mask_dir))

        for city in cities:
            mask_paths = sorted(
                glob.glob(
                    os.path.join(mask_dir, city, "*_gtFine_labelIds.png")
                )
            )

            for mask_path in mask_paths:
                mask = Image.open(mask_path)
                mask = mask.resize(image_size, Image.NEAREST)
                mask = np.array(mask)
                mask = encode_segmap(mask)

                for cls in range(num_classes):
                    counts[cls] += np.sum(mask == cls)

        # Median frequency balancing
        nonzero = counts[counts > 0]
        if len(nonzero) == 0:
            return torch.ones(num_classes, dtype=torch.float32)

        median_freq = np.median(nonzero)
        weights = np.zeros(num_classes, dtype=np.float32)

        for cls in range(num_classes):
            if counts[cls] > 0:
                weights[cls] = median_freq / counts[cls]

        return torch.from_numpy(weights)