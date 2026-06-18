# Walkthrough: Semantic Scene Parsing on Cityscapes

> **A technical walkthrough of the architecture, design decisions, training process, and results.**

---

## 1. Problem Formulation

### What is semantic segmentation?

Unlike image classification (one label per image) or object detection (bounding boxes), **semantic segmentation** requires assigning a class label to **every pixel** in the image. This produces a dense output map the same size as the input.

### Why is it harder than classification?

| Challenge | Classification | Segmentation |
|---|---|---|
| Output dimensionality | 1 label | H×W labels |
| Spatial precision | Not needed | Per-pixel |
| Multi-scale objects | Less critical | Must handle poles AND buildings |
| Class imbalance | Moderate | Severe (road >> traffic light) |
| Compute cost | Low | High (dense prediction) |

### Why does it matter?

Semantic segmentation is a core perception task in:
- **Autonomous driving** — understanding the road scene (where is the road? where are pedestrians?)
- **Robotics** — navigating environments
- **Medical imaging** — segmenting tumors, organs

---

## 2. Dataset: Cityscapes

### Overview

| Property | Value |
|---|---|
| Domain | Urban street scenes (European cities) |
| Training images | 2,975 |
| Validation images | 500 |
| Test images | 1,525 (labels withheld) |
| Original resolution | 2048 × 1024 |
| Working resolution | 512 × 256 |
| Raw label classes | 30+ |
| Evaluation classes | 19 |

### Label Remapping: 30+ → 19 Classes

Cityscapes defines 30+ raw label IDs, but only **19** are used for the official benchmark. Why?

1. **Ambiguous categories**: "ground" vs "road", "guard rail" vs "fence" — merging reduces annotation noise
2. **Extremely rare classes**: "caravan", "trailer" appear in <0.01% of pixels — insufficient training signal
3. **Semantic grouping**: Related categories are merged for meaningful evaluation

Labels not in the 19 evaluation set are mapped to `ignore_index = 255` and **excluded from both loss computation and metric calculation**.

### Class Imbalance — The Central Data Challenge

The 19 classes have severely unequal pixel frequencies:

| Frequency Tier | Classes | ~Pixel Share |
|---|---|---|
| **Dominant** | road, building, vegetation | >60% combined |
| **Common** | sky, car, sidewalk | ~20% |
| **Moderate** | person, fence, wall, terrain | ~10% |
| **Rare** | pole, traffic sign, traffic light, rider, truck, bus, train, motorcycle, bicycle | <10% combined |

**Impact**: Without addressing this, the model learns to predict dominant classes everywhere (high pixel accuracy, terrible mIoU). Road alone can be >35% of all pixels.

**Solution**: Inverse-frequency class weighting in CrossEntropyLoss, which gives rare classes proportionally higher loss values.

---

## 3. Architecture Deep-Dive: DeepLabV3-ResNet50

### Why DeepLabV3?

The core challenge in segmentation is maintaining **spatial resolution** while having a **large receptive field** for context. Standard classification CNNs downsample by 32× (2048→64), losing too much spatial detail.

DeepLabV3 solves this with **Atrous (Dilated) Convolutions** — convolutions with gaps between kernel elements that increase the receptive field **without reducing resolution**.

### Architecture Overview

```
Input Image (3×256×512)
       │
       ▼
┌─────────────────────────┐
│  ResNet-50 Backbone     │  Pretrained on ImageNet
│  (Layers 1-4)           │  Output stride = 8 (not 32)
│  ~23M params            │  Last two blocks use atrous convolutions
└──────────┬──────────────┘
           │
           ▼  Feature maps: 2048×32×64
┌─────────────────────────┐
│  ASPP Module            │  Atrous Spatial Pyramid Pooling
│  ~3M params             │
│  ┌─────────────────┐    │
│  │ 1×1 Conv        │    │  Rate=1 (local features)
│  │ 3×3 Conv rate=6 │    │  Medium-range context
│  │ 3×3 Conv rate=12│    │  Wider context
│  │ 3×3 Conv rate=18│    │  Scene-level context
│  │ Global AvgPool  │    │  Image-level context
│  └───────┬─────────┘    │
│    Concatenate + 1×1     │
└──────────┬──────────────┘
           │
           ▼  256×32×64
┌─────────────────────────┐
│  Classifier Head        │
│  3×3 Conv → BN → ReLU  │
│  1×1 Conv → 19 channels │
│  Bilinear upsample 8×   │  Back to input resolution
└──────────┬──────────────┘
           │
           ▼
Output: 19×256×512 (logits per class per pixel)
```

### Why each block matters

| Block | Purpose | What it costs |
|---|---|---|
| **ResNet-50 backbone** | Hierarchical feature extraction (edges → textures → parts → objects) | 23M params, most of the compute |
| **Atrous convolutions** | Maintain resolution while expanding receptive field | Minimal extra params vs. standard convolutions |
| **ASPP** | Multi-scale context capture (a pole needs local features; a road needs scene-level context) | ~3M params |
| **1×1 classifier** | Map 256 feature channels → 19 class scores | 256×19 = ~5K params |
| **Bilinear upsample** | Recover spatial resolution (32×64 → 256×512) | Zero learnable params |

### Why ResNet-50 and not ResNet-101?

- ResNet-101 gives ~1–2% mIoU improvement on Cityscapes
- But it roughly **doubles** backbone computation and memory
- On an 8 GB GPU at 512×256, ResNet-50 fits comfortably; ResNet-101 requires reducing batch size
- **Decision**: ResNet-50 for the available compute budget. This is an explicit, defensible tradeoff.

### What we modified

Only the **final classification head** was changed:
```python
# Original (PASCAL VOC): 256 → 21
# Ours (Cityscapes):     256 → 19
model.classifier[4] = nn.Conv2d(256, 19, kernel_size=1)
```

All pretrained weights in the backbone and ASPP are retained and fine-tuned.

---

## 4. Training Pipeline

### Loss Function: CrossEntropyLoss

```
Loss = -Σ_c  w_c · y_c · log(p_c)
```

Where:
- `y_c` is the one-hot ground truth for class `c`
- `p_c` is the predicted probability (after softmax)
- `w_c` is the class weight (inverse of frequency)

**Key properties:**
- `ignore_index=255`: Void/unlabeled pixels are excluded from the loss — they contribute zero gradient
- Class weights: Computed from training set pixel frequencies using median frequency balancing

### Why not Dice loss?

Dice loss directly optimizes IoU and handles imbalance inherently. However:
- It can be **unstable** in early training (division by small numbers)
- Cross-entropy with class weights is more stable and well-understood
- In practice, CE + class weights achieves comparable results on Cityscapes
- **Recommendation for future work**: Use CE + Dice as a combined loss

### Optimizer: Adam

- Learning rate: 1e-4
- Weight decay: 1e-4 (L2 regularization)
- Adam's adaptive learning rates handle the different magnitudes of gradients in the backbone vs. decoder

### LR Schedule: Polynomial Decay

```
lr(epoch) = base_lr × (1 - epoch / max_epochs)^0.9
```

This is the standard schedule for segmentation fine-tuning (used in the original DeepLab papers). It provides aggressive initial learning that gradually slows.

### BatchNorm Freezing

When fine-tuning with small batch sizes (4), BatchNorm running statistics become noisy estimates that hurt performance. Freezing BN keeps the ImageNet-learned statistics.

### Data Augmentation

- **Random horizontal flip** (p=0.5): Doubles effective training set, encourages left-right invariance
- **Color jitter** (brightness/contrast/saturation/hue): Robustness to lighting variation

Both are applied **jointly** to image and mask (spatial transforms must be synchronized).

### Mixed Precision (AMP)

`torch.amp.autocast` runs forward pass in FP16 where safe, ~1.5× faster on CUDA with negligible accuracy impact.

---

## 5. Evaluation Metrics

### Pixel Accuracy

```
PixelAcc = (correctly classified pixels) / (total valid pixels)
```

**Limitation**: Dominated by large classes. A model that predicts "road" everywhere gets ~35% pixel accuracy for free.

### Mean Intersection-over-Union (mIoU)

```
IoU_c = TP_c / (TP_c + FP_c + FN_c)
mIoU  = (1/19) Σ IoU_c
```

**Why mIoU matters more than pixel accuracy:**
- Each class contributes **equally** regardless of pixel count
- A model that ignores rare classes will have low mIoU even with high pixel accuracy
- This is the primary metric on the Cityscapes benchmark

### Per-Class IoU

The most revealing metric. Expected pattern:
- **High IoU**: road (~90%), building (~85%), sky (~80%) — large, consistent regions
- **Medium IoU**: car (~70%), vegetation (~75%) — common but with edges
- **Low IoU**: pole (~30%), traffic sign (~40%), bicycle (~40%) — small, rare, hard to segment at low resolution

---

## 6. Resolution Tradeoff: 512×256 vs. 2048×1024

### Why not full resolution?

| Resolution | GPU Memory | Batch Size | Time/Epoch | Expected mIoU |
|---|---|---|---|---|
| 2048×1024 | ~20 GB | 1 | ~60 min | ~75% |
| 1024×512 | ~8 GB | 2 | ~15 min | ~70% |
| **512×256** | **~3 GB** | **4** | **~5 min** | **~55-60%** |

At 512×256:
- Small objects (poles, signs, traffic lights) shrink to just a few pixels
- Boundaries become blurry (aliasing from downsampling)
- But: training is 12× faster and fits on any GPU

**This is a deliberate, defensible scope decision** given compute constraints.

---

## 7. What I'd Do With More Time

### Immediate improvements (hours)
1. **ImageNet normalization** — proper mean/std preprocessing aligned with pretrained backbone
2. **Higher resolution** (1024×512) with gradient accumulation
3. **LR warmup** for the first 500 iterations

### Medium-term improvements (days)
4. **Combined CE + Dice loss** for better rare-class performance
5. **Test-time augmentation** (multi-scale + horizontal flip inference)
6. **DeepLabV3+ decoder** with low-level feature skip connections for sharper boundaries
7. **Confusion matrix visualization** to identify systematic misclassifications

### Production readiness (weeks)
8. **ONNX export** for deployment without PyTorch
9. **Cityscapes gtCoarse** semi-supervised pretraining
10. **EMA model weights** for smoother predictions
11. **Boundary-aware post-processing** (CRF or guided filtering)

---

## 8. Summary

| Decision | Choice | Rationale |
|---|---|---|
| Framework | PyTorch 2.12.0 | Requirement |
| Architecture | DeepLabV3-ResNet50 | Best tradeoff of accuracy/compute/understanding |
| Resolution | 512×256 | GPU budget constraint (documented) |
| Loss | CE + class weights | Stable, handles imbalance |
| LR schedule | Polynomial decay | Standard for segmentation |
| Augmentation | HFlip + ColorJitter | Minimal but effective |
| Metric | mIoU (primary) + pixel accuracy | Industry standard |
| Web UI | Streamlit | Fast to build, sufficient for demo |

The project prioritizes **understanding and defensibility** over raw performance. Every architectural and training decision can be traced to a concrete engineering rationale.
