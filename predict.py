"""
predict.py -- Stage 12: Inference on New Images
=================================================

PURPOSE:
    Run the trained model on a NEW image (not from the dataset)
    to predict where oil is present.

WHAT IT DOES:
    1. Loads the trained model
    2. Preprocesses the input image (resize, normalize)
    3. Runs the model to get a prediction
    4. Generates:
       - Binary oil mask
       - Overlay visualization
       - Oil pixel count and coverage percentage

HOW TO RUN:
    python predict.py --image path/to/image.jpg

    Optional:
    python predict.py --image path/to/image.jpg --threshold 0.3
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from PIL import Image

import config
from model import create_model
from dataset import get_val_transforms


def predict(image_path, threshold=0.5):
    """
    Run inference on a single image.

    Args:
        image_path: Path to the input image
        threshold: Probability threshold for oil detection (0-1)
    """
    print("\n" + "=" * 60)
    print("  SPILLSENSE AI -- INFERENCE")
    print("=" * 60)

    device = config.DEVICE

    # Validate input
    if not os.path.exists(image_path):
        print(f"  [FAIL] Image not found: {image_path}")
        return 1

    # Load model
    print(f"\n  Loading model: {config.BEST_MODEL_PATH}")
    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"  [FAIL] Model not found! Train first: python train.py")
        return 1

    model = create_model().to(device)
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Model loaded (epoch {checkpoint['epoch']})")

    # Load and preprocess image
    print(f"  Input image: {image_path}")
    raw_image = cv2.imread(image_path)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = raw_image.shape[:2]
    print(f"  Original size: {orig_w} x {orig_h}")

    # Apply preprocessing (same as validation: resize + normalize)
    transform = get_val_transforms()
    transformed = transform(image=raw_image)
    image_tensor = transformed["image"].unsqueeze(0).to(device)  # [1, 3, 512, 512]

    # Run inference
    print(f"\n  Running inference...")
    with torch.no_grad():
        output = model(image_tensor)
        probability_map = torch.sigmoid(output).squeeze().cpu().numpy()  # [512, 512]
        binary_mask = (probability_map > threshold).astype(np.uint8)

    # Calculate statistics
    oil_pixels = np.sum(binary_mask == 1)
    total_pixels = binary_mask.size
    oil_percentage = (oil_pixels / total_pixels) * 100

    print(f"\n  {'=' * 40}")
    print(f"  RESULTS")
    print(f"  {'=' * 40}")
    print(f"  Oil Pixels:   {oil_pixels:,} / {total_pixels:,}")
    print(f"  Oil Coverage: {oil_percentage:.2f}%")
    print(f"  Threshold:    {threshold}")
    print(f"  {'=' * 40}")

    # Save outputs
    save_dir = config.PREDICTIONS_DIR
    os.makedirs(save_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(image_path))[0]

    # 1. Binary mask
    mask_save = os.path.join(save_dir, f"{basename}_mask.png")
    cv2.imwrite(mask_save, binary_mask * 255)
    print(f"\n  Saved mask:    {mask_save}")

    # 2. Probability heatmap
    heatmap = (probability_map * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_save = os.path.join(save_dir, f"{basename}_heatmap.png")
    cv2.imwrite(heatmap_save, heatmap_color)
    print(f"  Saved heatmap: {heatmap_save}")

    # 3. Overlay visualization
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Original
    axes[0].imshow(raw_image)
    axes[0].set_title(f"Original ({orig_w}x{orig_h})", fontsize=12)
    axes[0].axis("off")

    # Probability map
    axes[1].imshow(probability_map, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Oil Probability", fontsize=12)
    axes[1].axis("off")

    # Binary mask
    axes[2].imshow(binary_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Binary Mask (t={threshold})", fontsize=12)
    axes[2].axis("off")

    # Overlay on resized image
    resized = cv2.resize(raw_image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    overlay = resized.copy().astype(np.float32)
    oil_region = binary_mask == 1
    overlay[oil_region] = overlay[oil_region] * 0.5 + \
                          np.array([255, 0, 0], dtype=np.float32) * 0.5
    axes[3].imshow(overlay.astype(np.uint8))
    axes[3].set_title(f"Overlay (Oil: {oil_percentage:.1f}%)", fontsize=12)
    axes[3].axis("off")

    fig.suptitle(f"SpillSense AI Prediction: {basename}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    overlay_save = os.path.join(save_dir, f"{basename}_prediction.png")
    fig.savefig(overlay_save, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overlay: {overlay_save}")

    print(f"\n  [OK] Inference complete!")
    print(f"{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SpillSense AI - Oil Spill Detection Inference"
    )
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input RGB image")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold (default: 0.5)")
    args = parser.parse_args()

    sys.exit(predict(args.image, args.threshold))
