"""
visualize_dataset.py -- Stage 3 & 4: Mask Conversion + Visualization
======================================================================

PURPOSE:
    BEFORE training, we need to visually confirm that our mask conversion
    is working correctly. This script:
    1. Takes random samples from the dataset
    2. Shows the original image, original RGB mask, and binary mask side-by-side
    3. Saves the visualization so you can verify:
       - Oil regions appear as WHITE (1) in the binary mask
       - Everything else appears as BLACK (0)

WHY THIS MATTERS:
    If our mask conversion is wrong, the model will learn wrong labels.
    Visual inspection catches bugs that code alone cannot.

    For example:
    - If oil appears as BLACK instead of WHITE -> model learns the opposite!
    - If water appears as WHITE -> model confuses water with oil!

WHAT GOES IN:
    RGB images + RGB annotation masks from the dataset

WHAT COMES OUT:
    Visualization images saved to outputs/dataset_visualization/
    Each shows: [Original Image] [RGB Mask] [Binary Mask]

HOW IT CONNECTS:
    This validates Stage 3 (mask conversion logic).
    If the binary masks look correct -> move to Stage 5 (PyTorch Dataset)
    If they look wrong -> fix the conversion function first!

HOW TO RUN:
    python visualize_dataset.py
"""

import os
import sys
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (saves to file, no window needed)
import matplotlib.pyplot as plt
from PIL import Image

import config


def rgb_mask_to_binary(mask_rgb):
    """
    Convert an RGB annotation mask to a binary oil/non-oil mask.

    HOW IT WORKS:
        The annotation mask uses specific colors:
            Oil:        (255, 0, 124) -> pink/magenta
            Background: (0, 0, 0)     -> black
            Water:      (51, 221, 255) -> cyan
            Others:     (255, 204, 51) -> yellow

        We check each pixel: if it matches the OIL color -> 1, else -> 0.

    Args:
        mask_rgb: numpy array of shape (H, W, 3), dtype uint8

    Returns:
        binary_mask: numpy array of shape (H, W), dtype uint8
                     1 = oil, 0 = non-oil
    """
    # config.OIL_COLOR is (255, 0, 124)
    oil_color = np.array(config.OIL_COLOR, dtype=np.uint8)

    # Compare each pixel with the oil color
    # np.all(..., axis=-1) checks if ALL 3 channels (R, G, B) match
    binary_mask = np.all(mask_rgb == oil_color, axis=-1).astype(np.uint8)

    return binary_mask


def visualize_samples(split_name, image_dir, mask_dir, num_samples=5,
                      save_dir=None):
    """
    Randomly select samples and create side-by-side visualizations.

    For each sample, shows:
        [Original Image] [RGB Mask (annotation)] [Binary Mask (oil=white)]

    Args:
        split_name: 'train', 'val', or 'test'
        image_dir: path to images folder
        mask_dir: path to masks folder
        num_samples: how many random samples to visualize
        save_dir: where to save the output images
    """
    # Get matched pairs
    from pathlib import Path

    image_files = {Path(f).stem: f for f in os.listdir(image_dir)
                   if os.path.isfile(os.path.join(image_dir, f))}
    mask_files = {Path(f).stem: f for f in os.listdir(mask_dir)
                  if os.path.isfile(os.path.join(mask_dir, f))}

    matched_stems = sorted(set(image_files) & set(mask_files))

    if len(matched_stems) == 0:
        print(f"  [FAIL] No matched pairs in {split_name}!")
        return

    # Random sample
    random.seed(config.SEED)
    samples = random.sample(matched_stems,
                            min(num_samples, len(matched_stems)))

    print(f"\n  Visualizing {len(samples)} samples from {split_name}...")

    for i, stem in enumerate(samples):
        img_path = os.path.join(image_dir, image_files[stem])
        mask_path = os.path.join(mask_dir, mask_files[stem])

        # Load image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))

        # Convert to binary
        binary_mask = rgb_mask_to_binary(mask_rgb)

        # Calculate oil coverage
        oil_pixels = np.sum(binary_mask == 1)
        total_pixels = binary_mask.size
        oil_pct = (oil_pixels / total_pixels) * 100

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title(f"Original Image\n{image.shape[1]}x{image.shape[0]}",
                          fontsize=12)
        axes[0].axis("off")

        # RGB mask (original annotation)
        axes[1].imshow(mask_rgb)
        axes[1].set_title("RGB Annotation Mask\n(Pink=Oil, Cyan=Water, "
                          "Yellow=Other, Black=BG)", fontsize=10)
        axes[1].axis("off")

        # Binary mask (what the model will learn)
        axes[2].imshow(binary_mask, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title(f"Binary Mask (Oil=White)\n"
                          f"Oil: {oil_pixels:,} px ({oil_pct:.1f}%)",
                          fontsize=12)
        axes[2].axis("off")

        fig.suptitle(f"{split_name.upper()} - {stem}",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir,
                                     f"{split_name}_sample_{i+1}_{stem}.png")
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
            print(f"    Saved: {os.path.basename(save_path)}")

        plt.close(fig)


def create_color_legend(save_dir):
    """Create a color legend showing the mask color mapping."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    colors = [
        ("Background", config.MASK_COLORS["background"], "-> 0 (Black)"),
        ("Oil", config.MASK_COLORS["oil"], "-> 1 (WHITE)"),
        ("Others", config.MASK_COLORS["others"], "-> 0 (Black)"),
        ("Water", config.MASK_COLORS["water"], "-> 0 (Black)"),
    ]

    for i, (name, rgb, mapping) in enumerate(colors):
        color_norm = [c / 255.0 for c in rgb]
        ax.barh(i, 1, color=color_norm, edgecolor="black", height=0.6)
        text_color = "white" if name == "Background" else "black"
        ax.text(0.5, i, f"  {name}  RGB{rgb}  {mapping}",
                va="center", ha="center", fontsize=12,
                fontweight="bold", color=text_color)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(colors) - 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Mask Color Mapping: RGB -> Binary",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "color_legend.png")
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: color_legend.png")


def main():
    print("\n" + "=" * 60)
    print("  SPILLSENSE AI -- DATASET VISUALIZATION")
    print("  Stage 3: Mask Conversion + Stage 4: Visualization")
    print("=" * 60)

    save_dir = config.VISUALIZATION_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Create color legend
    print("\n  Creating color legend...")
    create_color_legend(save_dir)

    # Visualize samples from each split
    splits = [
        ("train", config.TRAIN_IMAGE_DIR, config.TRAIN_MASK_DIR, 6),
        ("val",   config.VAL_IMAGE_DIR,   config.VAL_MASK_DIR,   4),
        ("test",  config.TEST_IMAGE_DIR,  config.TEST_MASK_DIR,  4),
    ]

    for split_name, img_dir, mask_dir, n in splits:
        visualize_samples(split_name, img_dir, mask_dir,
                          num_samples=n, save_dir=save_dir)

    # Summary
    saved_files = [f for f in os.listdir(save_dir) if f.endswith(".png")]
    print(f"\n{'=' * 60}")
    print(f"  DONE! Saved {len(saved_files)} visualizations to:")
    print(f"  {save_dir}")
    print(f"\n  Please open these images and verify:")
    print(f"  1. Binary masks show oil as WHITE and everything else as BLACK")
    print(f"  2. White regions in binary mask match pink regions in RGB mask")
    print(f"  3. No unexpected artifacts or color conversion errors")
    print(f"\n  If everything looks correct -> proceed to Stage 5")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
