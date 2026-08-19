"""
evaluate.py -- Stage 11: Test Set Evaluation
==============================================

PURPOSE:
    Evaluate the trained model on the TEST set ONLY.
    The test set was NEVER seen during training — this gives us
    an honest measure of how well the model generalizes.

WHAT IT CALCULATES:
    - Dice Score:  Overlap between prediction and truth
    - IoU:         Strictest overlap measure
    - Precision:   Of predicted oil, how much is truly oil?
    - Recall:      Of all real oil, how much did we find?
    - F1 Score:    Balance of precision and recall

WHAT IT GENERATES:
    - Printed metrics report
    - Visual predictions saved to outputs/test_predictions/
    - Good predictions, poor predictions, false positives, false negatives

HOW TO RUN:
    python evaluate.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

import config
from dataset import get_dataloaders, OilSpillDataset, get_val_transforms
from model import create_model
from utils import dice_score, iou_score


def evaluate_model():
    """Evaluate the trained model on the test set."""
    print("\n" + "=" * 60)
    print("  SPILLSENSE AI -- TEST SET EVALUATION")
    print("=" * 60)

    device = config.DEVICE

    # Load model
    print(f"\n  Loading model from: {config.BEST_MODEL_PATH}")
    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"  [FAIL] Model file not found! Train the model first.")
        print(f"  Run: python train.py --debug")
        return 1

    model = create_model().to(device)
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"  Model loaded (trained for {checkpoint['epoch']} epochs)")
    print(f"  Val loss at save: {checkpoint['val_loss']:.4f}")

    # Get test data
    _, _, test_loader = get_dataloaders(batch_size=4, num_workers=0)
    print(f"  Test set: {len(test_loader.dataset)} images")

    # Evaluate
    print(f"\n  Running evaluation...")
    all_preds = []
    all_targets = []
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="  Evaluating",
                                   ncols=70):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            pred_binary = (torch.sigmoid(outputs) > 0.5).float()

            total_dice += dice_score(pred_binary, masks)
            total_iou += iou_score(pred_binary, masks)
            num_batches += 1

            # Flatten for sklearn metrics
            all_preds.append(pred_binary.cpu().numpy().flatten())
            all_targets.append(masks.cpu().numpy().flatten())

    # Calculate metrics
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    # Print report
    print(f"\n{'=' * 60}")
    print(f"  TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"  Dice Score:  {avg_dice:.4f}")
    print(f"  IoU:         {avg_iou:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  F1 Score:    {f1:.4f}")
    print(f"{'=' * 60}")

    # Generate visual predictions
    print(f"\n  Generating visual predictions...")
    generate_test_visualizations(model, device)

    return 0


def generate_test_visualizations(model, device, num_samples=20):
    """Generate side-by-side visualizations of test predictions."""
    save_dir = config.TEST_PREDICTIONS_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Load test dataset directly for access to individual samples
    test_dataset = OilSpillDataset(
        config.TEST_IMAGE_DIR, config.TEST_MASK_DIR,
        transform=get_val_transforms()
    )

    # Also load raw images for display (without normalization)
    from PIL import Image
    from pathlib import Path

    raw_image_files = sorted(os.listdir(config.TEST_IMAGE_DIR))
    raw_mask_files = sorted(os.listdir(config.TEST_MASK_DIR))

    # Sample indices
    np.random.seed(config.SEED)
    indices = np.random.choice(len(test_dataset),
                                min(num_samples, len(test_dataset)),
                                replace=False)

    # Track metrics per sample for categorization
    sample_metrics = []

    for i, idx in enumerate(tqdm(indices, desc="  Visualizing", ncols=70)):
        # Get preprocessed data
        image_tensor, mask_tensor = test_dataset[idx]

        # Predict
        with torch.no_grad():
            pred = model(image_tensor.unsqueeze(0).to(device))
            pred_prob = torch.sigmoid(pred).squeeze().cpu().numpy()
            pred_binary = (pred_prob > 0.5).astype(np.uint8)

        # Ground truth
        gt_mask = mask_tensor.squeeze().numpy()

        # Load raw image for display
        raw_img_path = os.path.join(config.TEST_IMAGE_DIR,
                                     raw_image_files[idx])
        raw_img = np.array(Image.open(raw_img_path).convert("RGB"))

        # Calculate per-sample dice
        pred_t = torch.from_numpy(pred_binary.astype(np.float32))
        gt_t = torch.from_numpy(gt_mask.astype(np.float32))
        sample_dice = dice_score(pred_t, gt_t)

        sample_metrics.append({
            "idx": idx,
            "dice": sample_dice,
            "file": raw_image_files[idx],
        })

        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(raw_img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred_binary, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title(f"Prediction (Dice: {sample_dice:.3f})")
        axes[2].axis("off")

        # Overlay: Green=correct, Red=false positive, Blue=false negative
        overlay = raw_img.copy().astype(np.float32)
        # Resize overlay to match prediction size
        from PIL import Image as PILImage
        overlay = np.array(PILImage.fromarray(raw_img).resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE)))
        overlay = overlay.astype(np.float32)

        # Color coding
        tp = (pred_binary == 1) & (gt_mask == 1)  # True positive (green)
        fp = (pred_binary == 1) & (gt_mask == 0)  # False positive (red)
        fn = (pred_binary == 0) & (gt_mask == 1)  # False negative (blue)

        overlay[tp] = overlay[tp] * 0.5 + np.array([0, 255, 0]) * 0.5
        overlay[fp] = overlay[fp] * 0.5 + np.array([255, 0, 0]) * 0.5
        overlay[fn] = overlay[fn] * 0.5 + np.array([0, 0, 255]) * 0.5

        axes[3].imshow(overlay.astype(np.uint8))
        axes[3].set_title("Overlay (G=TP, R=FP, B=FN)")
        axes[3].axis("off")

        quality = "GOOD" if sample_dice > 0.7 else "POOR"
        fig.suptitle(f"Test Sample {i+1} - {quality} (Dice: {sample_dice:.3f})",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = os.path.join(save_dir,
                                  f"test_{i+1:02d}_{quality}_dice{sample_dice:.3f}.png")
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    # Summary of predictions
    dices = [s["dice"] for s in sample_metrics]
    good = sum(1 for d in dices if d > 0.7)
    poor = sum(1 for d in dices if d <= 0.7)

    print(f"\n  Saved {len(indices)} visualizations to: {save_dir}")
    print(f"  Good predictions (Dice > 0.7): {good}")
    print(f"  Poor predictions (Dice <= 0.7): {poor}")
    print(f"  Average Dice: {np.mean(dices):.4f}")


if __name__ == "__main__":
    sys.exit(evaluate_model())
