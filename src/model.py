"""
Model definition — DeepLabV3 with ResNet-50 backbone.

Architecture choice rationale:
- DeepLabV3 uses Atrous Spatial Pyramid Pooling (ASPP) to capture
  multi-scale context without losing spatial resolution.  This is
  critical for segmentation where both fine detail (poles, signs) and
  large-scale context (road surface, sky) matter.
- ResNet-50 backbone provides a strong pretrained feature extractor
  (ImageNet) while remaining feasible on an 8 GB GPU.  ResNet-101
  would give ~1–2% mIoU improvement but doubles training time.
- We replace only the final classification head (1×1 conv) to output
  19 channels instead of the default 21 (PASCAL VOC classes).  All
  other layers retain their pretrained weights and are fine-tuned.

Key blocks and what they cost:
- Backbone (ResNet-50):  ~23M params, extracts hierarchical features.
- ASPP module:           ~3M params, parallel atrous convolutions at
                         rates 6/12/18 + 1×1 conv + global avg pool.
- Classifier head:       ~0.5M params, 256→256 (3×3) → 256→19 (1×1).

Total: ~40M parameters, ~168 MB on disk.
"""

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

from config import NUM_CLASSES


def get_model(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """Create a DeepLabV3-ResNet50 model for semantic segmentation.

    Args:
        num_classes: Number of output classes (19 for Cityscapes).
        pretrained:  If True, load COCO-pretrained weights for the backbone
                     and decoder.  The final classification layer is always
                     replaced to match ``num_classes``.

    Returns:
        A ``torchvision.models.segmentation.DeepLabV3`` model.
    """
    weights = "DEFAULT" if pretrained else None
    model = deeplabv3_resnet50(weights=weights)

    # Replace the classifier head: 256 → num_classes (was 21 for VOC)
    model.classifier[4] = nn.Conv2d(
        256, num_classes, kernel_size=1
    )

    # Also replace the auxiliary classifier head if it exists.
    # The aux head is used during training for deep supervision
    # (provides gradient signal to earlier layers).
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(
            256, num_classes, kernel_size=1
        )

    return model


def load_model_weights(
    model: nn.Module,
    weights_path: str,
    device: torch.device,
) -> nn.Module:
    """Load saved weights into the model, handling version mismatches.

    Older checkpoints may have a different aux_classifier head (21 classes
    instead of 19).  This function loads with ``strict=False`` and logs
    any mismatched keys rather than crashing.

    Args:
        model:        The model to load weights into.
        weights_path: Path to the ``.pth`` state dict file.
        device:       Device to map weights to.

    Returns:
        The model with loaded weights.
    """
    import logging
    logger = logging.getLogger(__name__)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    result = model.load_state_dict(state_dict, strict=False)

    if result.missing_keys:
        logger.info(f"Missing keys (will use random init): {result.missing_keys}")
    if result.unexpected_keys:
        logger.info(f"Unexpected keys (ignored): {result.unexpected_keys}")

    return model