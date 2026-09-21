"""
Cityscapes Semantic Segmentation — source package.

Modules:
    config               Centralized configuration (paths, hyperparameters)
    label_mapping        Raw Cityscapes labelId → trainId conversion
    cityscapes_dataset   Dataset class, color map, overlay utilities
    model                DeepLabV3-ResNet50 model definition
    train                Training pipeline
    evaluate             Evaluation pipeline (pixel accuracy + mIoU)
    inference            Single-image inference with overlay output
"""
