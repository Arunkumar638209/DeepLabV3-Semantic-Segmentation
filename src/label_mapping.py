"""
Cityscapes label ID → trainId mapping.

Cityscapes defines 30+ raw label IDs but only 19 are used for evaluation.
The remaining labels are mapped to IGNORE_LABEL (255) and excluded from
loss computation and metric calculation.

Why 19 classes?
- The full 30+ labels include overlapping/ambiguous categories (e.g.,
  "ground" vs "road", "guard rail" vs "fence") and extremely rare classes
  (e.g., "caravan", "trailer") that don't provide reliable training signal.
- The 19-class mapping groups semantically meaningful classes that appear
  frequently enough for robust evaluation.
- This is the official Cityscapes benchmark protocol.
"""

import numpy as np

# Raw Cityscapes labelId → trainId (0–18).
# Any raw ID not in this dict is mapped to IGNORE_LABEL.
LABEL_MAPPING = {
    7: 0,    # road
    8: 1,    # sidewalk
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    17: 5,   # pole
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}

IGNORE_LABEL = 255


def encode_segmap(mask: np.ndarray) -> np.ndarray:
    """Convert a raw Cityscapes labelId mask to trainId mask.

    Pixels whose raw labelId is not in the 19 evaluation classes are
    set to IGNORE_LABEL (255), which is excluded from loss and metrics
    via ``ignore_index=255`` in CrossEntropyLoss.

    Args:
        mask: H×W numpy array of raw Cityscapes label IDs.

    Returns:
        H×W numpy array with values in {0..18, 255}.
    """
    encoded = np.full(mask.shape, IGNORE_LABEL, dtype=np.uint8)

    for raw_id, train_id in LABEL_MAPPING.items():
        encoded[mask == raw_id] = train_id

    return encoded