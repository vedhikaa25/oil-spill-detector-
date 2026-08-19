# SpillSense AI — Intelligent Marine Oil Spill Detection

**Binary Semantic Segmentation of Oil Spills in Drone Imagery**

---

## 🎯 Project Overview

SpillSense AI is an AI-powered system for detecting oil spills in marine environments
using drone-captured RGB images. The system uses deep learning (U-Net + ResNet34) to
produce pixel-level segmentation masks that identify where oil is present in an image.

```
New RGB Drone Image
       ↓
  Preprocessing (resize, normalize)
       ↓
  U-Net + ResNet34 (segmentation model)
       ↓
  Binary Oil Spill Mask (oil = white, non-oil = black)
       ↓
  Oil Coverage Percentage + Visualization
```

---

## 📊 Dataset

**Oil Spill Drone Dataset** — 1,268 drone-captured RGB images with pixel-level annotations.

| Split      | Images | Purpose                                     |
|------------|--------|---------------------------------------------|
| Training   | 811    | Model learns from these images              |
| Validation | 203    | Monitor learning progress (not used for learning) |
| Test       | 254    | Final performance evaluation only           |

**Research Paper**: [https://arxiv.org/abs/2402.18202](https://arxiv.org/abs/2402.18202)
**Dataset Description**: [https://www.nature.com/articles/s41597-024-03993-8](https://www.nature.com/articles/s41597-024-03993-8)

### Annotation Classes

The dataset uses RGB colors for 4-class pixel-level annotation:

| Class       | RGB Color       | Binary Label |
|-------------|-----------------|--------------|
| Background  | (0, 0, 0)       | 0            |
| **Oil**     | **(255, 0, 124)** | **1**      |
| Others      | (255, 204, 51)  | 0            |
| Water       | (51, 221, 255)  | 0            |

For this project, we convert to **binary segmentation**: Oil = 1, Everything else = 0.

---

## 🏗️ Architecture: U-Net + ResNet34

### Why U-Net?

U-Net is designed for **pixel-level semantic segmentation** — exactly what we need.
Unlike image classification (which says "this image contains oil"), segmentation tells
us **where** the oil is at every pixel.

U-Net has two parts:
- **Encoder** (downsampling path): Compresses the image to understand *what* features
  are present (edges, textures, patterns).
- **Decoder** (upsampling path): Expands back to full resolution to classify *each pixel*.
- **Skip connections**: Bridge encoder and decoder to preserve fine spatial details.

### Why ResNet34 as the Encoder?

- Well-established CNN architecture with proven feature extraction capabilities
- ImageNet pretrained weights provide useful initial visual features (transfer learning)
- Lighter than deeper variants (ResNet50/101) — good balance of accuracy vs. speed
- Strong hierarchical feature extraction through residual connections

### Architecture Diagram

```
RGB Image (3 × 512 × 512)
       ↓
┌─────────────────────┐
│   ResNet34 Encoder   │  ← Pretrained on ImageNet
│   (Feature Extraction)│
│                       │
│   64 → 128 → 256 → 512 feature maps
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│   U-Net Decoder      │  ← Upsampling + Skip Connections
│   (Pixel Classification)│
│                       │
│   512 → 256 → 128 → 64 → 1 output channel
└─────────┬───────────┘
          ↓
Binary Oil Mask (1 × 512 × 512)
```

> **Note**: The original Oil Spill Drone Dataset research used a modified U-Net with
> an EfficientNet-B4 encoder. Our project intentionally uses U-Net + ResNet34 as our
> selected baseline architecture.

---

## 📉 Loss Function: BCE + Dice

We use a **combined loss** to handle class imbalance (oil often covers only a small
portion of an image):

```
total_loss = 0.5 × BCE_loss + 0.5 × Dice_loss
```

| Loss Function | Purpose |
|---------------|---------|
| **BCE** (Binary Cross-Entropy) | Per-pixel classification accuracy |
| **Dice Loss** | Overlap between predicted and true oil regions |

**Why both?** If oil covers only 5% of an image, a model predicting "no oil everywhere"
would be 95% accurate by BCE alone. Dice Loss specifically penalizes missed oil regions.

---

## 📏 Evaluation Metrics

| Metric     | What It Measures |
|------------|------------------|
| **Dice Score** | Overlap between prediction and ground truth (2 × intersection / union-like) |
| **IoU** (Intersection over Union) | Strictest overlap measure |
| **Precision** | Of pixels predicted as oil, how many truly are oil? |
| **Recall** | Of actual oil pixels, how many did we find? |
| **F1 Score** | Harmonic mean of precision and recall |

---

## 📁 Project Structure

```
SpillSense-AI/
├── dataset/                    # Oil Spill Drone Dataset
│   ├── train/
│   │   ├── images/            # 811 training RGB images
│   │   └── masks/             # 811 training annotation masks
│   ├── val/
│   │   ├── images/            # 203 validation images
│   │   └── masks/             # 203 validation masks
│   └── test/
│       ├── images/            # 254 test images
│       └── masks/             # 254 test masks
│
├── models/                    # Saved model checkpoints
├── outputs/                   # All generated outputs
│   ├── dataset_visualization/ # Dataset sample visualizations
│   ├── test_predictions/      # Test set prediction visualizations
│   └── predictions/           # Inference predictions on new images
├── logs/                      # Training logs
│
├── config.py                  # Central configuration (all settings)
├── verify_environment.py      # Stage 1: Environment verification
├── inspect_dataset.py         # Stage 2: Dataset inspection
├── visualize_dataset.py       # Stage 4: Dataset visualization
├── dataset.py                 # Stage 5: PyTorch Dataset + DataLoader
├── model.py                   # Stage 6: U-Net + ResNet34 model
├── utils.py                   # Utility functions (metrics, helpers)
├── train.py                   # Stage 8-10: Training pipeline
├── evaluate.py                # Stage 11: Test set evaluation
├── predict.py                 # Stage 12: Inference on new images
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Environment
```bash
python verify_environment.py
```

### 3. Inspect Dataset
```bash
python inspect_dataset.py
```

### 4. Train Model (Debug Mode)
```bash
python train.py --debug
```

### 5. Train Model (Full Training)
```bash
python train.py
```

### 6. Evaluate on Test Set
```bash
python evaluate.py
```

### 7. Run Inference on New Image
```bash
python predict.py --image path/to/image.jpg
```

---

## 🔧 Development Stages

| Stage | Task                     | Script                 | Status |
|-------|--------------------------|------------------------|--------|
| 1     | Environment setup        | `verify_environment.py`| ⬜     |
| 2     | Dataset inspection       | `inspect_dataset.py`   | ⬜     |
| 3     | Mask conversion          | (in dataset.py)        | ⬜     |
| 4     | Dataset visualization    | `visualize_dataset.py` | ⬜     |
| 5     | PyTorch Dataset/Loader   | `dataset.py`           | ⬜     |
| 6     | U-Net + ResNet34 model   | `model.py`             | ⬜     |
| 7     | Forward-pass test        | (in model.py)          | ⬜     |
| 8     | 5-epoch debug training   | `train.py --debug`     | ⬜     |
| 9     | Prediction visualization | (in train.py)          | ⬜     |
| 10    | Full training            | `train.py`             | ⬜     |
| 11    | Test evaluation          | `evaluate.py`          | ⬜     |
| 12    | Inference on new image   | `predict.py`           | ⬜     |
| 13    | Results analysis         | (manual)               | ⬜     |
| 14    | Documentation            | README.md              | ⬜     |

---

## 🔮 Future Modules (Not Yet Implemented)

- Oil spill severity estimation
- Environmental sensor integration
- Autonomous drone monitoring
- Bioremediation decision support

---

## 📚 References

1. Oil Spill Drone Dataset: [arXiv:2402.18202](https://arxiv.org/abs/2402.18202)
2. Dataset Description: [Nature Scientific Data](https://www.nature.com/articles/s41597-024-03993-8)
3. U-Net: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
4. ResNet: [He et al., 2015](https://arxiv.org/abs/1512.03385)
5. Segmentation Models PyTorch: [GitHub](https://github.com/qubvel/segmentation_models.pytorch)
