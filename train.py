"""
train.py -- Stage 8-10: Training Pipeline
===========================================

PURPOSE:
    Trains the U-Net + ResNet34 model on the oil spill dataset.

HOW TRAINING WORKS (simplified):
    Training is a loop that repeats many times (epochs):

    For each epoch:
        1. FORWARD PASS:  Feed images through model -> get predictions
        2. LOSS:          Compare predictions with true masks -> get error
        3. BACKWARD PASS: Calculate how to adjust each weight to reduce error
        4. UPDATE:        Adjust weights using the optimizer (Adam)

    After each epoch, we check validation performance:
        - If it improved -> save this model as "best"
        - If it hasn't improved for N epochs -> stop early (no point continuing)
        - If it plateaus -> reduce learning rate for finer adjustments

    Think of it like archery practice:
        - Forward pass = shooting an arrow
        - Loss = measuring how far from the bullseye
        - Backward pass = figuring out how to adjust aim
        - Optimizer = making the adjustment
        - LR scheduler = making smaller adjustments as you get closer

HOW TO RUN:
    Debug mode (5 epochs):    python train.py --debug
    Full training:            python train.py
    Resume training:          python train.py --resume
"""

import argparse
import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from dataset import get_dataloaders
from model import create_model
from utils import set_seed, dice_score, iou_score


# ══════════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ══════════════════════════════════════════════════════════════════

class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice Loss.

    WHY TWO LOSSES?
        BCE (Binary Cross-Entropy):
            Treats each pixel independently. Good for overall accuracy
            but can be fooled by class imbalance (model predicts "no oil
            everywhere" and still gets ~95% accuracy).

        Dice Loss:
            Measures overlap between predicted and true oil regions.
            Specifically penalizes the model for missing oil pixels.
            Handles class imbalance naturally.

        By combining them, we get the benefits of both.

    total_loss = bce_weight * BCE + dice_weight * DiceLoss
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()  # Includes sigmoid internally

    def dice_loss(self, pred, target, smooth=1e-6):
        """Calculate Dice Loss (1 - Dice Score)."""
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + smooth) / \
               (pred_flat.sum() + target_flat.sum() + smooth)
        return 1.0 - dice

    def forward(self, pred, target):
        """
        Calculate combined loss.

        Args:
            pred:   model output (logits, before sigmoid), [B, 1, H, W]
            target: ground truth binary mask, [B, 1, H, W]

        Returns:
            combined loss value
        """
        bce = self.bce(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice


# ══════════════════════════════════════════════════════════════════
# TRAINING AND VALIDATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one full pass through the training data.

    For each batch:
        1. Load images and masks
        2. Forward pass -> predictions
        3. Calculate loss
        4. Backward pass -> gradients
        5. Optimizer step -> update weights
        6. Track metrics

    Returns:
        avg_loss, avg_dice, avg_iou for this epoch
    """
    model.train()  # Set model to training mode (enables dropout, etc.)
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="  Train", ncols=80, leave=False)
    for images, masks in pbar:
        # Move data to device (CPU or GPU)
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        predictions = model(images)

        # Calculate loss
        loss = criterion(predictions, masks)

        # Backward pass (calculate gradients)
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Calculate new gradients
        optimizer.step()       # Update weights

        # Track metrics (use sigmoid for predictions)
        with torch.no_grad():
            pred_binary = (torch.sigmoid(predictions) > 0.5).float()
            batch_dice = dice_score(pred_binary, masks)
            batch_iou = iou_score(pred_binary, masks)

        running_loss += loss.item()
        running_dice += batch_dice
        running_iou += batch_iou
        num_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         dice=f"{batch_dice:.4f}")

    avg_loss = running_loss / num_batches
    avg_dice = running_dice / num_batches
    avg_iou = running_iou / num_batches

    return avg_loss, avg_dice, avg_iou


def validate(model, loader, criterion, device):
    """
    Evaluate the model on validation data (no weight updates).

    Same as training but:
    - model.eval() disables dropout
    - torch.no_grad() saves memory (no gradient tracking needed)
    - No optimizer.step() (we're just measuring, not learning)

    Returns:
        avg_loss, avg_dice, avg_iou
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="  Val  ", ncols=80, leave=False)
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            predictions = model(images)
            loss = criterion(predictions, masks)

            pred_binary = (torch.sigmoid(predictions) > 0.5).float()
            batch_dice = dice_score(pred_binary, masks)
            batch_iou = iou_score(pred_binary, masks)

            running_loss += loss.item()
            running_dice += batch_dice
            running_iou += batch_iou
            num_batches += 1

    avg_loss = running_loss / num_batches
    avg_dice = running_dice / num_batches
    avg_iou = running_iou / num_batches

    return avg_loss, avg_dice, avg_iou


# ══════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════

def plot_training_curves(history, save_path):
    """Plot training vs validation loss, Dice, and IoU curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val")
    axes[0].set_title("Loss (BCE + Dice)", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dice
    axes[1].plot(epochs, history["train_dice"], "b-", label="Train")
    axes[1].plot(epochs, history["val_dice"], "r-", label="Val")
    axes[1].set_title("Dice Score", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # IoU
    axes[2].plot(epochs, history["train_iou"], "b-", label="Train")
    axes[2].plot(epochs, history["val_iou"], "r-", label="Val")
    axes[2].set_title("IoU Score", fontsize=14)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("SpillSense AI - Training Curves", fontsize=16,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

def train(debug=False, resume=False):
    """
    Main training function.

    Args:
        debug:  If True, run only 5 epochs for testing
        resume: If True, resume from the saved checkpoint
    """
    # Setup
    set_seed()
    device = config.DEVICE
    max_epochs = config.DEBUG_EPOCHS if debug else config.MAX_EPOCHS
    mode_str = "DEBUG (5 epochs)" if debug else f"FULL ({max_epochs} epochs max)"

    print("\n" + "=" * 60)
    print(f"  SPILLSENSE AI -- TRAINING")
    print(f"  Mode: {mode_str}")
    if resume:
        print(f"  Resuming from checkpoint: {config.BEST_MODEL_PATH}")
    print(f"  Device: {device}")
    print("=" * 60)

    # Create output directories
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    # Create data loaders
    print(f"\n  Loading datasets...")
    train_loader, val_loader, _ = get_dataloaders()
    print(f"  Train: {len(train_loader.dataset)} samples, "
          f"{len(train_loader)} batches")
    print(f"  Val:   {len(val_loader.dataset)} samples, "
          f"{len(val_loader)} batches")

    # Create model
    print(f"\n  Creating U-Net + ResNet34...")
    model = create_model().to(device)

    # Loss function
    criterion = CombinedLoss(
        bce_weight=config.BCE_WEIGHT,
        dice_weight=config.DICE_WEIGHT
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",                            # Monitor loss (lower = better)
        patience=config.LR_SCHEDULER_PATIENCE,
        factor=config.LR_SCHEDULER_FACTOR,
    )

    # Training history
    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "train_iou": [],  "val_iou": [],
        "lr": [],
    }

    # Early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    start_epoch = 1  # Default start epoch

    # ── Resume from checkpoint ────────────────────────────────────
    if resume:
        if os.path.exists(config.BEST_MODEL_PATH):
            print(f"\n  Loading checkpoint...")
            checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["val_loss"]
            best_epoch = checkpoint["epoch"]
            print(f"  Resumed from epoch {checkpoint['epoch']}")
            print(f"  Best val loss so far: {best_val_loss:.4f}")
            print(f"  Best val Dice so far: {checkpoint['val_dice']:.4f}")
            print(f"  Best val IoU so far:  {checkpoint['val_iou']:.4f}")

            # Load existing training log to continue history
            if os.path.exists(config.TRAINING_LOG_PATH):
                import csv as csv_reader
                with open(config.TRAINING_LOG_PATH, "r") as f:
                    reader = csv_reader.DictReader(f)
                    for row in reader:
                        history["train_loss"].append(float(row["train_loss"]))
                        history["val_loss"].append(float(row["val_loss"]))
                        history["train_dice"].append(float(row["train_dice"]))
                        history["val_dice"].append(float(row["val_dice"]))
                        history["train_iou"].append(float(row["train_iou"]))
                        history["val_iou"].append(float(row["val_iou"]))
                        history["lr"].append(float(row["lr"]))
                print(f"  Loaded {len(history['train_loss'])} epochs of history")
        else:
            print(f"\n  [WARN] No checkpoint found at {config.BEST_MODEL_PATH}")
            print(f"  Starting fresh training...")
            resume = False

    # CSV log
    log_path = config.TRAINING_LOG_PATH
    if not resume:
        # Only create a fresh log if NOT resuming
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss",
                             "train_dice", "val_dice",
                             "train_iou", "val_iou", "lr"])

    remaining = max_epochs - start_epoch + 1
    print(f"\n  Training epochs {start_epoch} to {max_epochs} ({remaining} remaining)...")
    print(f"  Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.1e}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Loss: {config.BCE_WEIGHT}*BCE + {config.DICE_WEIGHT}*Dice")
    print(f"\n  {'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | "
          f"{'Train Dice':>10} | {'Val Dice':>10} | "
          f"{'Train IoU':>10} | {'Val IoU':>10} | {'LR':>10}")
    print(f"  {'-'*95}")

    start_time = time.time()

    for epoch in range(start_epoch, max_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, device
        )

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)
        history["lr"].append(current_lr)

        # Log to CSV
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                             f"{train_dice:.6f}", f"{val_dice:.6f}",
                             f"{train_iou:.6f}", f"{val_iou:.6f}",
                             f"{current_lr:.6f}"])

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        marker = ""
        if val_loss < best_val_loss:
            marker = " *BEST*"
        print(f"  {epoch:>5} | {train_loss:>10.4f} | {val_loss:>10.4f} | "
              f"{train_dice:>10.4f} | {val_dice:>10.4f} | "
              f"{train_iou:>10.4f} | {val_iou:>10.4f} | "
              f"{current_lr:.1e}{marker}  ({epoch_time:.0f}s)")

        # Learning rate scheduler step
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
            }, config.BEST_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE and not debug:
                print(f"\n  Early stopping at epoch {epoch}! "
                      f"No improvement for {config.EARLY_STOPPING_PATIENCE} epochs.")
                break

    total_time = time.time() - start_time

    # Plot training curves
    plot_training_curves(history, config.TRAINING_CURVES_PATH)

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total time:     {total_time/60:.1f} minutes")
    print(f"  Best epoch:     {best_epoch}")
    print(f"  Best val loss:  {best_val_loss:.4f}")
    print(f"  Best val dice:  {history['val_dice'][best_epoch-1]:.4f}")
    print(f"  Best val IoU:   {history['val_iou'][best_epoch-1]:.4f}")
    print(f"\n  Model saved:    {config.BEST_MODEL_PATH}")
    print(f"  Curves saved:   {config.TRAINING_CURVES_PATH}")
    print(f"  Log saved:      {log_path}")
    print(f"{'=' * 60}\n")


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpillSense AI Training")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode (5 epochs only)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    args = parser.parse_args()

    train(debug=args.debug, resume=args.resume)
