"""
config.py — Central Configuration for SpillSense AI
=====================================================

PURPOSE:
    This file is the "single source of truth" for every setting in the project.
    Instead of hard-coding numbers and paths in each script, we define them all
    here. Any script can then do:

        from config import IMAGE_SIZE, DEVICE, LEARNING_RATE

    and use those values directly.

WHY THIS MATTERS:
    Imagine you want to change the image size from 512 to 256. Without a config
    file, you'd have to search through every file and change it everywhere.
    With a config file, you change it in ONE place and it automatically updates
    everywhere.

WHAT THIS FILE CONTAINS:
    1. File paths        — Where the dataset, models, and outputs live
    2. Mask colors       — The RGB colors used in the annotation masks
    3. Preprocessing     — Image size, normalization values
    4. Model settings    — Architecture choices (U-Net + ResNet34)
    5. Loss weights      — How to balance BCE and Dice losses
    6. Training settings — Learning rate, batch size, epochs, etc.
    7. Device selection  — GPU or CPU
    8. Random seed       — For reproducibility
"""

import os
import torch

# ══════════════════════════════════════════════════════════════════
# 1. PROJECT ROOT
# ══════════════════════════════════════════════════════════════════
# __file__ is a special Python variable that holds the path to the
# current file (config.py). We use it to figure out where the project
# folder is, so all other paths are relative to it.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════
# 2. DATASET PATHS
# ══════════════════════════════════════════════════════════════════
# The Oil Spill Drone Dataset comes pre-split into:
#   train/ — used to teach the model (the model learns from these)
#   val/   — used to monitor learning progress (NOT used for learning)
#   test/  — used ONLY at the end to measure final performance
#
# Each split has:
#   images/ — the actual drone photographs (RGB)
#   masks/  — the annotations showing where oil is (colored images)
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, "train", "images")
TRAIN_MASK_DIR  = os.path.join(DATASET_DIR, "train", "masks")

VAL_IMAGE_DIR = os.path.join(DATASET_DIR, "val", "images")
VAL_MASK_DIR  = os.path.join(DATASET_DIR, "val", "masks")

TEST_IMAGE_DIR = os.path.join(DATASET_DIR, "test", "images")
TEST_MASK_DIR  = os.path.join(DATASET_DIR, "test", "masks")

# ══════════════════════════════════════════════════════════════════
# 3. OUTPUT PATHS
# ══════════════════════════════════════════════════════════════════
# Where we save everything the project produces.
MODELS_DIR           = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR          = os.path.join(PROJECT_ROOT, "outputs")
LOGS_DIR             = os.path.join(PROJECT_ROOT, "logs")

VISUALIZATION_DIR    = os.path.join(OUTPUTS_DIR, "dataset_visualization")
TEST_PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "test_predictions")
PREDICTIONS_DIR      = os.path.join(OUTPUTS_DIR, "predictions")

BEST_MODEL_PATH      = os.path.join(MODELS_DIR, "best_model.pth")
TRAINING_CURVES_PATH = os.path.join(OUTPUTS_DIR, "training_curves.png")
TRAINING_LOG_PATH    = os.path.join(LOGS_DIR, "training_log.csv")

# ══════════════════════════════════════════════════════════════════
# 4. MASK COLOR DEFINITIONS (RGB)
# ══════════════════════════════════════════════════════════════════
# The dataset annotations use specific RGB colors for each class:
#
#   Color                 RGB Value        Class       → Binary Label
#   ─────────────────────────────────────────────────────────────────
#   Black                 (0, 0, 0)        Background  → 0
#   Pink/Magenta          (255, 0, 124)    Oil         → 1  ← what we detect
#   Yellow                (255, 204, 51)   Others      → 0
#   Cyan/Light blue       (51, 221, 255)   Water       → 0
#
# For our BINARY segmentation task:
#   Oil pixels  → 1 (white in the binary mask)
#   Everything else → 0 (black in the binary mask)
MASK_COLORS = {
    "background": (0, 0, 0),
    "oil":        (255, 0, 124),
    "others":     (255, 204, 51),
    "water":      (51, 221, 255),
}

OIL_COLOR = MASK_COLORS["oil"]  # (255, 0, 124) — the color we look for

# ══════════════════════════════════════════════════════════════════
# 5. IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════════════════
# All images and masks are resized to this square size.
# 512×512 is a good balance between:
#   - Detail (larger = more detail, but needs more GPU memory)
#   - Speed (smaller = faster training, but may lose fine details)
IMAGE_SIZE = 512

# ImageNet normalization constants.
# WHY THESE SPECIFIC NUMBERS?
#   Our encoder (ResNet34) was pre-trained on the ImageNet dataset,
#   which used these mean and standard deviation values to normalize
#   pixel values. We MUST use the same normalization so the pre-trained
#   weights work correctly with our images.
#
#   Think of it like unit conversion: if a recipe is in grams, you
#   must measure in grams too, not ounces. Same idea here.
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # Mean for [R, G, B] channels
IMAGENET_STD  = [0.229, 0.224, 0.225]  # Std  for [R, G, B] channels

# ══════════════════════════════════════════════════════════════════
# 6. MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════
# We use U-Net with a ResNet34 encoder from segmentation_models_pytorch.
#
# U-Net: A neural network designed for pixel-level segmentation.
#   It has an "encoder" (compresses the image to understand features)
#   and a "decoder" (expands back to full size to classify each pixel).
#
# ResNet34: A proven image classification network that we use as
#   the encoder part of U-Net. Using pretrained ImageNet weights
#   gives it a head-start — it already knows basic visual features
#   like edges, textures, and shapes.
ENCODER_NAME    = "resnet34"
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS     = 3   # RGB input → 3 color channels
NUM_CLASSES     = 1   # Binary output → 1 channel (oil probability)

# ══════════════════════════════════════════════════════════════════
# 7. LOSS FUNCTION WEIGHTS
# ══════════════════════════════════════════════════════════════════
# We combine TWO loss functions to train the model:
#
# BCE (Binary Cross-Entropy):
#   Measures how wrong each pixel prediction is. Works well for
#   overall pixel accuracy.
#
# Dice Loss:
#   Measures how much the predicted oil region overlaps with the
#   true oil region. Works well for imbalanced data (where oil
#   may only cover a small part of the image).
#
# WHY BOTH?
#   If oil covers only 5% of an image, a lazy model could predict
#   "no oil everywhere" and be 95% accurate! BCE alone might allow
#   this. Dice Loss specifically penalizes the model for missing
#   oil regions, forcing it to actually find the oil.
#
# total_loss = BCE_WEIGHT × BCE_loss + DICE_WEIGHT × Dice_loss
BCE_WEIGHT  = 0.5
DICE_WEIGHT = 0.5

# ══════════════════════════════════════════════════════════════════
# 8. TRAINING HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════
# Learning Rate: Controls how much the model adjusts its weights
#   after each batch. Too high = unstable learning. Too low = very
#   slow learning. 1e-4 (0.0001) is a common starting point.
LEARNING_RATE = 1e-4

# Batch Size: How many images the model processes at once before
#   updating its weights. Larger batch = more stable but needs more
#   GPU memory. 8 is reasonable for a 512×512 input on most GPUs.
BATCH_SIZE = 8

# Number of worker threads for loading data in parallel.
# More workers = faster data loading, but uses more CPU/RAM.
# Set to 0 if you get multiprocessing errors on Windows.
NUM_WORKERS = 4

# --- Debug Mode ---
# Run a quick 5-epoch test to verify everything works before
# committing to a long training run.
DEBUG_EPOCHS = 5

# --- Full Training ---
MAX_EPOCHS              = 150  # Maximum training epochs
EARLY_STOPPING_PATIENCE = 10   # Stop if no improvement for N epochs

# --- Learning Rate Scheduler ---
# Automatically reduces the learning rate when validation loss stops
# improving. WHY: A large LR helps at the start for big improvements,
# but as the model gets better, smaller steps help it fine-tune.
LR_SCHEDULER_PATIENCE = 5    # Wait N epochs before reducing LR
LR_SCHEDULER_FACTOR   = 0.5  # Multiply LR by 0.5 when reducing

# ══════════════════════════════════════════════════════════════════
# 9. DEVICE SELECTION
# ══════════════════════════════════════════════════════════════════
# GPU (CUDA) is dramatically faster for deep learning.
# This auto-detects: if you have an NVIDIA GPU with CUDA, it uses it.
# Otherwise, it falls back to CPU (much slower but still works).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════
# 10. RANDOM SEED
# ══════════════════════════════════════════════════════════════════
# A fixed seed makes experiments reproducible. With the same seed,
# you get the same "random" numbers every run → same results.
# This is important for scientific reproducibility in your project.
SEED = 42
