"""
Centralized configuration for the Cityscapes Semantic Segmentation project.

All hyperparameters, paths, and constants are defined here so that
no magic numbers are scattered across the codebase.
"""

import os

# ---------------------------------------------------------------------------
# Paths  (relative to project root — scripts should be run from project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATA_ROOT = PROJECT_ROOT  # Cityscapes dirs live at project root

IMAGE_DIR_TRAIN = os.path.join(DATA_ROOT, "leftImg8bit", "train")
MASK_DIR_TRAIN = os.path.join(DATA_ROOT, "gtFine", "train")

IMAGE_DIR_VAL = os.path.join(DATA_ROOT, "leftImg8bit", "val")
MASK_DIR_VAL = os.path.join(DATA_ROOT, "gtFine", "val")

MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_DIR, "checkpoint.pth")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
IMAGE_SIZE = (512, 256)  # (width, height) — 4:1 aspect preserved from 2048×1024
NUM_CLASSES = 19
IGNORE_LABEL = 255

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 4
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LR_SCHEDULER = "poly"        # "poly" | "step" | "cosine"
POLY_POWER = 0.9              # for polynomial LR decay
STEP_SIZE = 10                # for step LR
STEP_GAMMA = 0.1              # for step LR

NUM_WORKERS = 2
PIN_MEMORY = True

USE_CLASS_WEIGHTS = True      # inverse-frequency weighting for CrossEntropyLoss
USE_AMP = True                # automatic mixed precision (faster on GPU)

SEED = 42

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
BACKBONE = "resnet50"
PRETRAINED = True
FREEZE_BN = True              # freeze BatchNorm layers when fine-tuning

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_INTERVAL = 50             # print every N batches
TENSORBOARD_DIR = os.path.join(PROJECT_ROOT, "runs")
