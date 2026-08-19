"""
utils.py -- Utility Functions for SpillSense AI
=================================================

PURPOSE:
    Common helper functions used by multiple scripts.
    Keeps the code DRY (Don't Repeat Yourself).

WHAT'S IN HERE:
    1. rgb_mask_to_binary()  — Convert RGB mask to binary (oil=1, else=0)
    2. set_seed()            — Set random seeds for reproducibility
    3. dice_score()          — Calculate Dice coefficient
    4. iou_score()           — Calculate IoU (Intersection over Union)

HOW IT CONNECTS:
    - dataset.py uses rgb_mask_to_binary() to preprocess masks
    - train.py uses dice_score() and iou_score() for monitoring
    - evaluate.py uses all metrics for final evaluation
"""

import random

import numpy as np
import torch

import config


# ══════════════════════════════════════════════════════════════════
# 1. MASK CONVERSION
# ══════════════════════════════════════════════════════════════════

def rgb_mask_to_binary(mask_rgb):
    """
    Convert an RGB annotation mask to a binary oil/non-oil mask.

    How it works:
        Input mask has colored pixels:
            Pink (255, 0, 124) = Oil
            Everything else    = Not oil

        We compare each pixel to the oil color.
        If it matches -> 1 (oil)
        If it doesn't -> 0 (not oil)

    Args:
        mask_rgb: numpy array, shape (H, W, 3), dtype uint8

    Returns:
        binary_mask: numpy array, shape (H, W), dtype uint8
                     Values are 0 or 1
    """
    oil_color = np.array(config.OIL_COLOR, dtype=np.uint8)
    binary_mask = np.all(mask_rgb == oil_color, axis=-1).astype(np.uint8)
    return binary_mask


# ══════════════════════════════════════════════════════════════════
# 2. REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════

def set_seed(seed=None):
    """
    Set random seeds for reproducibility.

    WHY: Deep learning involves many random operations (weight init,
    data shuffling, augmentation). Setting a fixed seed ensures you
    get the same results every time you run the code.

    Args:
        seed: integer seed value (default: from config.py)
    """
    if seed is None:
        seed = config.SEED

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # These settings make CUDA operations deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════
# 3. EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════

def dice_score(pred, target, smooth=1e-6):
    """
    Calculate the Dice coefficient between prediction and ground truth.

    WHAT IS DICE SCORE?
        Imagine two circles overlapping (like a Venn diagram).
        Dice measures how much they overlap:
            - Dice = 1.0 -> perfect overlap (prediction matches truth exactly)
            - Dice = 0.0 -> no overlap at all

        Formula:
            Dice = (2 * |A intersection B|) / (|A| + |B|)

        In pixel terms:
            A = predicted oil pixels
            B = true oil pixels
            intersection = pixels that are oil in BOTH

    Args:
        pred:   torch tensor, predicted probabilities or binary mask
        target: torch tensor, ground truth binary mask
        smooth: small number to avoid division by zero

    Returns:
        dice: float, the Dice score (0 to 1, higher is better)
    """
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice.item()


def iou_score(pred, target, smooth=1e-6):
    """
    Calculate IoU (Intersection over Union), also called Jaccard Index.

    WHAT IS IoU?
        Similar to Dice but stricter.
        IoU = intersection / union
        Where union = everything in A OR B (not double-counted)

        IoU is always <= Dice for the same prediction.

        - IoU = 1.0 -> perfect (prediction = truth)
        - IoU = 0.0 -> no overlap

    Args:
        pred:   torch tensor
        target: torch tensor
        smooth: small number to avoid division by zero

    Returns:
        iou: float (0 to 1, higher is better)
    """
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()
