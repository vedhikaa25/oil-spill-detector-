# 🛢️ SpillSense AI — Complete Project Explanation

> **A guide-ready, comprehensive explanation of the entire project — what it does, how it works, and why every design decision was made.**

---

## 1. Problem Statement

### What Problem Are We Solving?

Oil spills in marine environments cause massive ecological damage — killing marine life, destroying habitats, and contaminating water sources. Early detection is critical for rapid response, but manually scanning vast ocean areas from drone footage is **slow and error-prone**.

### Our Solution

**SpillSense AI** is an AI-powered system that automatically detects oil spills in drone-captured images using **deep learning-based semantic segmentation**. Given a drone/satellite image of the ocean, the model outputs a **pixel-level mask** showing exactly **where** oil is present.

```
Input: RGB Drone Image (e.g., 1920×1080)
   ↓ Preprocessing (resize to 512×512, normalize)
   ↓ U-Net + ResNet34 (deep learning model)
   ↓ Sigmoid activation (probability map)
   ↓ Thresholding (> 0.5 = oil)
Output: Binary Mask (white = oil, black = not oil) + Oil Coverage %
```

### Why Segmentation, Not Classification?

- **Image Classification** would just say: *"This image contains oil"* — not useful for response teams
- **Semantic Segmentation** says: *"These exact pixels contain oil"* — tells responders **where** and **how much**

---

## 2. Dataset

### Oil Spill Drone Dataset

| Property | Detail |
|----------|--------|
| **Source** | [arXiv:2402.18202](https://arxiv.org/abs/2402.18202), [Nature Scientific Data](https://www.nature.com/articles/s41597-024-03993-8) |
| **Total images** | **1,268** drone-captured RGB images |
| **Image type** | Real drone photographs of marine environments |
| **Annotations** | Pixel-level RGB color masks (4 classes) |

### Dataset Splits

| Split | Images | Purpose |
|-------|--------|---------|
| **Training** | 811 | Model learns patterns from these images |
| **Validation** | 203 | Monitor learning progress during training (never used for learning) |
| **Test** | 254 | Final, unseen evaluation — measures true generalization |

> **Why 3 splits?** The test set acts as a "final exam" the model has never seen. If we only had train/val, we might overfit to the validation set by tuning hyperparameters against it. The test set gives an honest, unbiased performance measure.

### Annotation Color Scheme → Binary Conversion

The raw dataset masks use **4 RGB colors**:

| Class | RGB Color | Visual | Binary Label |
|-------|-----------|--------|-------------|
| Background | `(0, 0, 0)` | ⬛ Black | **0** |
| **Oil** | `(255, 0, 124)` | 🟪 Pink/Magenta | **1** ← what we detect |
| Others | `(255, 204, 51)` | 🟨 Yellow | **0** |
| Water | `(51, 221, 255)` | 🟦 Cyan | **0** |

**We convert this to binary segmentation**: Oil = 1 (white), Everything else = 0 (black).

The conversion logic (from [`utils.py`](file:///c:/Users/vedhi/.gemini/antigravity-ide/scratch/SpillSense-AI/utils.py#L33-L55)):
```python
def rgb_mask_to_binary(mask_rgb):
    oil_color = np.array((255, 0, 124), dtype=np.uint8)
    binary_mask = np.all(mask_rgb == oil_color, axis=-1).astype(np.uint8)
    return binary_mask
```
This checks each pixel: if all 3 channels (R, G, B) match the oil color → 1, otherwise → 0.

---

## 3. Complete Project Architecture

### File-by-File Breakdown

```
SpillSense-AI/
│
├── config.py                    ← Central configuration (single source of truth)
├── verify_environment.py        ← Stage 1: Pre-flight checks
├── inspect_dataset.py           ← Stage 2: Dataset validation
├── visualize_dataset.py         ← Stage 3-4: Mask conversion + visual verification
├── dataset.py                   ← Stage 5: PyTorch Dataset + DataLoader
├── model.py                     ← Stage 6-7: U-Net + ResNet34 model definition
├── utils.py                     ← Shared utility functions (metrics, helpers)
├── train.py                     ← Stage 8-10: Full training pipeline
├── evaluate.py                  ← Stage 11: Test set evaluation
├── predict.py                   ← Stage 12: Inference on new images
├── dashboard.html               ← Interactive visualization dashboard
├── SpillSense_AI_Colab_Training.ipynb  ← Google Colab notebook for GPU training
├── requirements.txt             ← Python dependencies
├── README.md                    ← Project documentation
│
├── dataset/                     ← Oil Spill Drone Dataset
│   ├── train/images/ (811)      
│   ├── train/masks/  (811)
│   ├── val/images/   (203)
│   ├── val/masks/    (203)
│   ├── test/images/  (254)
│   └── test/masks/   (254)
│
├── models/
│   └── best_model.pth           ← Trained model weights (~280 MB)
│
├── outputs/
│   ├── dataset_visualization/   ← Sample visualizations from the dataset
│   ├── test_predictions/        ← Model predictions on test set
│   ├── predictions/             ← Inference results on new images
│   └── training_curves.png      ← Loss/Dice/IoU plots over epochs
│
└── logs/
    └── training_log.csv         ← Epoch-by-epoch metrics log
```

---

## 4. Stage-by-Stage Walkthrough

### Stage 1: Environment Verification — [`verify_environment.py`](file:///c:/Users/vedhi/.gemini/antigravity-ide/scratch/SpillSense-AI/verify_environment.py)

**What it does:** A pre-flight checklist that verifies everything is set up correctly before starting any work.

**Checks performed:**
1. Python version ≥ 3.8
2. All 10 required packages are installed (PyTorch, torchvision, segmentation-models-pytorch, albumentations, OpenCV, NumPy, Matplotlib, scikit-learn, Pillow, tqdm)
3. GPU/CUDA availability
4. Dataset directory structure exists (train/val/test with images/ and masks/ each)
5. Dataset file counts match (each image has a corresponding mask)
6. Output directories can be created
7. PyTorch tensor operations work correctly
8. `config.py` imports without errors

**Why this matters:** Catches missing packages or broken paths early, rather than discovering them mid-training.

---

### Stage 2: Dataset Inspection — [`inspect_dataset.py`](file:///c:/Users/vedhi/.gemini/antigravity-ide/scratch/SpillSense-AI/inspect_dataset.py)

**What it does:** Deep validation of the dataset quality and integrity.

**What it checks:**
- File counts per split (images vs masks)
- Filename matching (images paired with masks by stem name)
- Image dimensions and formats (JPEG for images, PNG for masks)
- Corrupted file detection (forces full file read)
- Mask color analysis (confirms oil color `(255, 0, 124)` is present)
- Identifies unexpected colors from PNG compression artifacts

---

### Stage 3-4: Visualization — [`visualize_dataset.py`](file:///c:/Users/vedhi/.gemini/antigravity-ide/scratch/SpillSense-AI/visualize_dataset.py)

**What it does:** Visual sanity check of the mask conversion process.

For random samples, it generates side-by-side images:
```
[Original Drone Photo] → [RGB Annotation Mask] → [Binary Mask (oil=white)]
```

**Why this is critical:** If the mask conversion is wrong (e.g., oil shows as black instead of white), the model would learn the exact opposite of what we want. Visual inspection catches errors that code alone cannot.

Also generates a **color legend** chart showing the mapping from RGB colors to binary labels.

---

### Stage 5: Data Pipeline — [`dataset.py`](file:///c:/Users/vedhi/.gemini/antigravity-ide/scratch/SpillSense-AI/dataset.py)

**What it does:** Creates the PyTorch data loading infrastructure.

#### OilSpillDataset Class
A custom `torch.utils.data.Dataset` that:
1. Matches image-mask pairs by filename stem
2. Loads images (BGR→RGB via OpenCV)
3. Converts RGB masks to binary (oil=1, else=0)
4. Applies augmentation/preprocessing transforms
5. Returns tensors: `image [3, 512, 512]` + `mask [1, 512, 512]`

#### Data Augmentation (Training Only)

```python
# Training transforms — make the model robust
A.Resize(512, 512)                    # Standardize size
A.HorizontalFlip(p=0.5)              # Randomly flip left-right
A.VerticalFlip(p=0.5)                # Randomly flip up-down
A.RandomRotate90(p=0.5)              # Randomly rotate 90°
A.Affine(translate, scale, rotate)    # Slight shifts, zoom, rotation
A.RandomBrightnessContrast(p=0.5)    # Lighting variations
A.Normalize(ImageNet_mean, ImageNet_std)  # Match ResNet34's training stats
ToTensorV2()                          # Convert to PyTorch tensor
```

**Why augmentation?** With only 811 training images, the model could memorize specific images (overfitting). Augmentation creates many "virtual" variations — flipped, rotated, brightness-adjusted — so the model learns general oil patterns instead of memorizing specific images.

**Key detail:** The same geometric transforms are applied to BOTH image and mask simultaneously (if we flip the image, we must flip the mask too so they stay aligned).

**Mask resizing uses NEAREST interpolation** (not bilinear), because bilinear averaging would create values like 0.5 which corrupt our binary labels (should only be 0 or 1).

#### Validation/Test Transforms
Only resize + normalize. **No random augmentation** — we need deterministic, consistent evaluation.

#### DataLoader
Wraps the Dataset to provide:
- **Batching**: Groups 8 images together (batch_size=8)
- **Shuffling**: Randomizes order each epoch (training only)
- **Parallel loading**: Multiple CPU threads load data in background
- **Pin memory**: Faster CPU→GPU data transfer

---

### Stage 6-7: Model — [`model.py`](file:///c:/Users/vedhi/.gemini/antigravity-ide/scratch/SpillSense-AI/model.py)

**What it does:** Defines the U-Net + ResNet34 segmentation model.

#### Architecture: U-Net

```
RGB Image (3 × 512 × 512)
       ↓
┌──────────────────────────┐
│   ENCODER (ResNet34)      │  ← Pretrained on ImageNet (1.2M images)
│   Compresses + extracts   │
│   features at each level  │
│                           │
│   512→256→128→64→32→16    │  (spatial dimensions shrink)
│   64→128→256→512 channels │  (feature channels grow)
└────────────┬─────────────┘
             │
     Skip Connections ──────────── Bridge fine details
             │
┌────────────┴─────────────┐
│   DECODER (U-Net)         │  ← Upsamples back to original size
│   Classifies each pixel   │
│                           │
│   16→32→64→128→256→512    │  (spatial dimensions grow back)
│   512→256→128→64→1 channel│  (converge to 1 output channel)
└────────────┬─────────────┘
             ↓
Binary Oil Mask (1 × 512 × 512)
```

**Why U-Net?**
- Specifically designed for pixel-level segmentation
- Skip connections preserve fine spatial details (exact oil boundaries)
- Works well with small datasets (our 811 training images)

**Why ResNet34 encoder?**
- Pre-trained on ImageNet (1.2 million images) → already knows edges, textures, patterns (**transfer learning**)
- Lighter than ResNet50/101 → good accuracy-speed balance for our dataset size
- Residual connections prevent the vanishing gradient problem in deep networks

**Model size:** ~24.4 million parameters

#### Forward Pass Test
The script includes a self-test that:
1. Creates the model
2. Passes a dummy batch through it
3. Verifies output shape matches expected `[B, 1, 512, 512]`
4. Tests with real dataset samples
5. Computes a sanity-check BCE loss

---

### Stage 8-10: Training — [`train.py`](file:///c:/Users/vedhi/.gemini/antigravity-ide/scratch/SpillSense-AI/train.py)

**What it does:** The complete training pipeline.

#### Loss Function: BCE + Dice (Combined)

```python
total_loss = 0.5 × BCE_loss + 0.5 × Dice_loss
```

| Loss | What it measures | Why needed |
|------|-----------------|-----------|
| **BCE** (Binary Cross-Entropy) | Per-pixel prediction error | Good for overall pixel accuracy |
| **Dice Loss** (1 - Dice Score) | Region overlap | Handles class imbalance |

**Why combine them?** Oil might cover only 5% of an image. A lazy model could predict "no oil everywhere" and be 95% accurate by BCE alone! Dice Loss specifically penalizes the model for missing oil regions, forcing it to actually detect the oil.

#### Training Loop (per epoch)

```
For each batch of 8 images:
    1. FORWARD PASS:   Feed images → model → get predictions
    2. COMPUTE LOSS:   Compare predictions with true masks → get error
    3. BACKWARD PASS:  Calculate gradients (how to adjust weights)
    4. OPTIMIZER STEP: Update weights using Adam optimizer
    5. TRACK METRICS:  Record Dice score and IoU
```

#### Key Training Components

| Component | Setting | Purpose |
|-----------|---------|---------|
| **Optimizer** | Adam, LR=0.0001 | Adaptive learning rate per parameter |
| **LR Scheduler** | ReduceLROnPlateau (patience=5, factor=0.5) | Halves LR when validation loss plateaus for 5 epochs |
| **Early Stopping** | patience=10 | Stops training if no improvement for 10 consecutive epochs |
| **Batch Size** | 8 | Balance between stable gradients and GPU memory |
| **Max Epochs** | 150 | Upper limit (early stopping usually kicks in before this) |
| **Debug Mode** | 5 epochs | Quick sanity check before committing to full training |

#### What Gets Saved
- **Best model checkpoint** (`models/best_model.pth`) — saved whenever validation loss improves
- **Training curves** (`outputs/training_curves.png`) — Loss, Dice, IoU plots
- **Training log** (`logs/training_log.csv`) — epoch-by-epoch metrics in CSV format

#### Checkpoint Contents
```python
{
    "epoch": 42,                    # Which epoch this was saved at
    "model_state_dict": {...},      # All model weights
    "optimizer_state_dict": {...},  # Optimizer state (for resume)
    "val_loss": 0.1234,             # Validation loss at save time
    "val_dice": 0.8765,             # Validation Dice at save time
    "val_iou": 0.7890,              # Validation IoU at save time
}
```

---

### Stage 11: Evaluation — [`evaluate.py`](file:///c:/Users/vedhi/.gemini/antigravity-ide/scratch/SpillSense-AI/evaluate.py)

**What it does:** Final, honest evaluation on the **test set** (254 images never seen during training).

#### Metrics Computed

| Metric | Formula | What It Tells Us |
|--------|---------|-----------------|
| **Dice Score** | 2×|A∩B| / (|A|+|B|) | How well prediction overlaps with truth (0–1) |
| **IoU** | |A∩B| / |A∪B| | Strictest overlap measure, always ≤ Dice |
| **Precision** | TP / (TP + FP) | Of pixels we labeled oil, how many are truly oil? |
| **Recall** | TP / (TP + FN) | Of all real oil pixels, how many did we find? |
| **F1 Score** | 2 × (P×R)/(P+R) | Harmonic mean of precision and recall |

> **TP** = True Positive (correctly detected oil), **FP** = False Positive (predicted oil where there's none), **FN** = False Negative (missed real oil)

#### Visual Predictions
Generates 20 test sample visualizations, each showing:
```
[Original Image] [Ground Truth Mask] [Model Prediction] [Color-coded Overlay]
```

The overlay uses:
- 🟢 **Green** = True Positive (correctly detected oil)
- 🔴 **Red** = False Positive (predicted oil but wasn't)
- 🔵 **Blue** = False Negative (missed oil)

Each visualization is categorized as **GOOD** (Dice > 0.7) or **POOR** (Dice ≤ 0.7).

---

### Stage 12: Inference — [`predict.py`](file:///c:/Users/vedhi/.gemini/antigravity-ide/scratch/SpillSense-AI/predict.py)

**What it does:** Run the trained model on **any new image** (not from the dataset).

```bash
python predict.py --image path/to/drone_photo.jpg --threshold 0.5
```

**Pipeline:**
1. Load the trained model from `models/best_model.pth`
2. Load and preprocess the input image (resize to 512×512, normalize)
3. Forward pass through the model → probability map (0–1 for each pixel)
4. Apply threshold (default 0.5) → binary mask
5. Generate outputs:
   - **Binary mask** (oil regions in white)
   - **Probability heatmap** (color-coded confidence levels)
   - **Overlay visualization** (original + probability + mask + overlay)
   - **Statistics** (oil pixel count, coverage percentage)

---

## 5. Supporting Components

### Configuration — [`config.py`](file:///c:/Users/vedhi/.gemini/antigravity-ide/scratch/SpillSense-AI/config.py)

**The single source of truth** for every setting. Instead of hard-coding values across files:

| Category | Key Settings |
|----------|-------------|
| **Paths** | Dataset dirs, model save path, output dirs |
| **Mask Colors** | Oil = (255, 0, 124), Background, Water, Others |
| **Preprocessing** | IMAGE_SIZE = 512, ImageNet mean/std |
| **Model** | encoder = "resnet34", weights = "imagenet", classes = 1 |
| **Loss** | BCE_WEIGHT = 0.5, DICE_WEIGHT = 0.5 |
| **Training** | LR = 1e-4, batch = 8, max_epochs = 150, early_stop = 10 |
| **Device** | Auto-detect: CUDA GPU if available, else CPU |
| **Seed** | 42 (for reproducibility) |

**Why a central config?** Change any setting in ONE place and it updates everywhere. Without this, you'd need to search every file to change a value like image size.

### Utility Functions — [`utils.py`](file:///c:/Users/vedhi/.gemini/antigravity-ide/scratch/SpillSense-AI/utils.py)

| Function | Used By | Purpose |
|----------|---------|---------|
| `rgb_mask_to_binary()` | dataset.py | Convert colored annotation mask → binary (oil/not oil) |
| `set_seed()` | train.py | Fix all random seeds for reproducible experiments |
| `dice_score()` | train.py, evaluate.py | Compute Dice coefficient between prediction and truth |
| `iou_score()` | train.py, evaluate.py | Compute Intersection over Union |

### Interactive Dashboard — [`dashboard.html`](file:///c:/Users/vedhi/.gemini/antigravity-ide/scratch/SpillSense-AI/dashboard.html)

A **standalone HTML/CSS/JS** dashboard (no backend needed) with:
- Project overview with animated UI
- Architecture visualization
- Dataset statistics with Chart.js graphs
- Training metrics display
- Dark-themed, modern glassmorphism design
- Fully responsive layout

### Google Colab Notebook — [`SpillSense_AI_Colab_Training.ipynb`](file:///c:/Users/vedhi/.gemini/antigravity-ide/scratch/SpillSense-AI/SpillSense_AI_Colab_Training.ipynb)

For training on **Google's free T4 GPU** (since local machines may not have CUDA):
1. Check GPU availability
2. Upload project from Google Drive
3. Install dependencies
4. Verify dataset
5. Run full training (~30-60 minutes on T4)
6. Evaluate on test set
7. View training curves and predictions
8. Download the trained model

---

## 6. Key Design Decisions & Rationale

### Why U-Net + ResNet34?

| Decision | Alternatives Considered | Why This Choice |
|----------|------------------------|----------------|
| U-Net architecture | DeepLab, FCN, PSPNet | U-Net excels with small datasets due to skip connections; widely proven for medical/remote sensing segmentation |
| ResNet34 encoder | EfficientNet, VGG, ResNet50 | Good balance of accuracy vs. speed; lighter than ResNet50/101 but deep enough for our task |
| ImageNet pretrained | Training from scratch | Transfer learning gives a massive head-start — the encoder already knows edges, textures, and patterns |

### Why Binary Segmentation?

The dataset has 4 classes (Oil, Water, Others, Background), but we simplified to **binary** (Oil vs. Everything Else) because:
- Our goal is specifically **oil detection**, not classifying water vs. background
- Binary is simpler, faster to train, and more robust with limited data
- Makes deployment easier (single threshold decision)

### Why Combined BCE + Dice Loss?

The dataset has **severe class imbalance** — oil often covers only 5-15% of an image. Using only BCE, a model could achieve 85-95% accuracy by predicting "no oil" everywhere! Dice Loss specifically penalizes this behavior by measuring overlap with the oil region.

### Why 512×512 Image Size?

- **Too small** (256×256): Loses fine oil boundary details
- **Too large** (1024×1024): Requires too much GPU memory, slower training
- **512×512**: Good balance — preserves enough detail while fitting in most GPU memory budgets

### Why Adam Optimizer?

Adam adapts the learning rate individually for each parameter, which works well out-of-the-box without extensive hyperparameter tuning. Combined with ReduceLROnPlateau scheduler, it starts learning fast and slows down for fine-tuning as the model converges.

---

## 7. Data Flow Diagram

```
                          TRAINING PHASE
                          ═══════════════

  dataset/train/images/    dataset/train/masks/
         │                        │
         └──────────┬─────────────┘
                    │
           ┌────────▼────────┐
           │  OilSpillDataset │  ← Match by filename, load, augment
           │  (dataset.py)    │
           └────────┬────────┘
                    │
            [3,512,512]  [1,512,512]     ← Normalized image + binary mask
                    │
           ┌────────▼────────┐
           │    DataLoader    │  ← Batch 8 images, shuffle, parallel load
           └────────┬────────┘
                    │
         ┌──────────▼──────────┐
         │   U-Net + ResNet34   │  ← Forward pass → predictions
         │   (model.py)         │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  CombinedLoss        │  ← 0.5×BCE + 0.5×Dice
         │  (train.py)          │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  Backward + Adam     │  ← Update weights
         │  Optimizer Step      │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  Validate on Val Set │  ← Monitor Dice/IoU (no weight updates)
         └──────────┬──────────┘
                    │
              ┌─────▼─────┐
              │ Improved?  │
              └─────┬──┬───┘
              Yes   │  │  No
         ┌──────────┘  └─────────────────────┐
    Save best model                  patience_counter += 1
    (best_model.pth)                 if counter >= 10 → STOP
```

```
                          INFERENCE PHASE
                          ════════════════

   New Drone Image (any size)
          │
   ┌──────▼──────┐
   │  Resize 512  │
   │  Normalize   │
   └──────┬──────┘
          │
   ┌──────▼──────────┐
   │  Load best_model │
   │  Forward pass    │
   └──────┬──────────┘
          │
   ┌──────▼──────┐
   │  Sigmoid     │  ← Probability map (0.0 to 1.0)
   └──────┬──────┘
          │
   ┌──────▼──────┐
   │  Threshold   │  ← > 0.5 = oil, ≤ 0.5 = not oil
   │  (default 0.5)
   └──────┬──────┘
          │
   ┌──────▼──────────────┐
   │  Binary Mask         │
   │  Probability Heatmap │
   │  Overlay Viz          │
   │  Oil Coverage %       │
   └─────────────────────┘
```

---

## 8. Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.8+ | Programming language |
| **PyTorch** | ≥ 2.0 | Deep learning framework (tensors, autograd, GPU acceleration) |
| **torchvision** | ≥ 0.15 | Image utilities, pretrained models |
| **segmentation-models-pytorch** | ≥ 0.3.3 | Provides U-Net architecture with ResNet34 encoder |
| **Albumentations** | ≥ 1.3.1 | Fast, flexible image augmentation library |
| **OpenCV** | ≥ 4.8 | Image reading (BGR→RGB), processing |
| **NumPy** | ≥ 1.24 | Numerical arrays and math operations |
| **Matplotlib** | ≥ 3.7 | Plotting training curves and visualizations |
| **scikit-learn** | ≥ 1.3 | Evaluation metrics (Precision, Recall, F1) |
| **Pillow** | ≥ 10.0 | Image I/O and format handling |
| **tqdm** | ≥ 4.65 | Progress bars for training loops |
| **Chart.js** | 4.4 | Dashboard data visualizations (in dashboard.html) |
| **Google Colab** | — | Free T4 GPU for training |

---

## 9. Reproducibility

The project ensures reproducible results through:

```python
# From utils.py → set_seed()
random.seed(42)           # Python's random module
np.random.seed(42)        # NumPy's random
torch.manual_seed(42)     # PyTorch CPU
torch.cuda.manual_seed(42)        # PyTorch GPU
torch.cuda.manual_seed_all(42)    # All GPUs
torch.backends.cudnn.deterministic = True   # Deterministic CUDA ops
torch.backends.cudnn.benchmark = False      # Disable auto-tuning
```

**Why?** With fixed seeds, running the training twice gives the **exact same results**. This is essential for scientific reproducibility — your guide can verify your results independently.

---

## 10. Project Execution Order

```
Step 1:  pip install -r requirements.txt         → Install dependencies
Step 2:  python verify_environment.py             → Pre-flight checks
Step 3:  python inspect_dataset.py                → Validate dataset integrity
Step 4:  python visualize_dataset.py              → Visual sanity check
Step 5:  python model.py                          → Forward-pass test
Step 6:  python train.py --debug                  → Quick 5-epoch test run
Step 7:  python train.py                          → Full training (or use Colab notebook)
Step 8:  python evaluate.py                       → Test set evaluation + visualizations
Step 9:  python predict.py --image photo.jpg      → Inference on new images
Step 10: Open dashboard.html in browser           → View interactive dashboard
```

---

## 11. Summary

| Aspect | Detail |
|--------|--------|
| **Task** | Binary semantic segmentation of oil spills in drone imagery |
| **Dataset** | 1,268 drone images, pixel-level annotated, 3-way split |
| **Model** | U-Net with ResNet34 encoder (ImageNet pretrained) |
| **Loss** | 0.5 × BCE + 0.5 × Dice (handles class imbalance) |
| **Optimizer** | Adam (LR=1e-4) + ReduceLROnPlateau scheduler |
| **Training** | Up to 150 epochs, early stopping (patience=10) |
| **Evaluation** | Dice, IoU, Precision, Recall, F1 on held-out test set |
| **Output** | Binary mask + probability heatmap + overlay + coverage % |
| **Infrastructure** | Local Python + Google Colab (free T4 GPU) |
| **Total Files** | 12 Python scripts + 1 HTML dashboard + 1 Colab notebook |

> **In one sentence:** SpillSense AI takes drone images as input, processes them through a U-Net deep learning model with a ResNet34 encoder, and outputs pixel-level oil spill masks with coverage statistics — enabling rapid environmental response to marine oil spills.
