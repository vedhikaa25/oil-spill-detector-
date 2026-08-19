"""
dataset.py -- Stage 5: PyTorch Dataset and DataLoader
=======================================================

PURPOSE:
    Creates a PyTorch Dataset class that feeds images and masks to the model.

HOW DEEP LEARNING DATA LOADING WORKS (simplified):
    1. Dataset: Knows how to load ONE image-mask pair and preprocess it
    2. DataLoader: Wraps the Dataset and handles:
       - Batching (grouping multiple images together)
       - Shuffling (randomizing order each epoch)
       - Parallel loading (using multiple CPU threads)

    Think of it like a restaurant:
    - Dataset = the recipe (how to prepare one dish)
    - DataLoader = the kitchen staff (prepares many dishes in parallel)

WHAT GOES IN:
    Raw image (any size, JPEG) + Raw mask (any size, PNG with RGB colors)

WHAT COMES OUT:
    Preprocessed tensors ready for the model:
    - Image: shape [3, 512, 512], normalized float32
    - Mask:  shape [1, 512, 512], binary float32 (0 or 1)

DATA AUGMENTATION:
    Training set ONLY gets random augmentations (flips, rotations, etc.)
    to help the model generalize. Validation and test sets are processed
    deterministically (no randomness) for consistent evaluation.
"""

import os
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config
from utils import rgb_mask_to_binary


class OilSpillDataset(Dataset):
    """
    PyTorch Dataset for the Oil Spill Drone Dataset.

    For each index, returns a preprocessed (image, mask) pair.

    HOW IT WORKS:
        1. __init__: Finds all matched image-mask pairs by filename
        2. __len__:  Returns total number of pairs
        3. __getitem__: Loads one pair, preprocesses, and returns tensors
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir: Path to folder containing images (.jpg)
            mask_dir:  Path to folder containing masks (.png)
            transform: Albumentations transform pipeline
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Match images to masks by filename stem
        image_files = {Path(f).stem: f for f in os.listdir(image_dir)
                       if os.path.isfile(os.path.join(image_dir, f))}
        mask_files = {Path(f).stem: f for f in os.listdir(mask_dir)
                      if os.path.isfile(os.path.join(mask_dir, f))}

        # Only keep stems that exist in BOTH directories
        matched_stems = sorted(set(image_files) & set(mask_files))

        if len(matched_stems) == 0:
            raise ValueError(
                f"No matched image-mask pairs found!\n"
                f"  Image dir: {image_dir}\n"
                f"  Mask dir:  {mask_dir}"
            )

        # Store the full paths
        self.image_paths = [os.path.join(image_dir, image_files[s])
                            for s in matched_stems]
        self.mask_paths = [os.path.join(mask_dir, mask_files[s])
                           for s in matched_stems]

        # Verify counts match
        assert len(self.image_paths) == len(self.mask_paths), \
            "Image and mask counts don't match!"

    def __len__(self):
        """Return the total number of image-mask pairs."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load and preprocess one image-mask pair.

        Steps:
        1. Read the image (BGR -> RGB)
        2. Read the mask (BGR -> RGB)
        3. Convert RGB mask to binary (oil=1, else=0)
        4. Apply augmentation/preprocessing transforms
        5. Return tensors

        Args:
            idx: Index of the pair to load

        Returns:
            image: tensor [3, H, W], normalized float32
            mask:  tensor [1, H, W], binary float32
        """
        # Load image with OpenCV (it reads as BGR, we convert to RGB)
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask with OpenCV
        mask = cv2.imread(self.mask_paths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Convert RGB mask to binary (oil=1, everything else=0)
        binary_mask = rgb_mask_to_binary(mask)

        # Apply transforms (augmentation + preprocessing)
        if self.transform:
            transformed = self.transform(image=image, mask=binary_mask)
            image = transformed["image"]           # Already a tensor [3, H, W]
            binary_mask = transformed["mask"]       # numpy or tensor [H, W]

        # Ensure mask is a float tensor with shape [1, H, W]
        if isinstance(binary_mask, np.ndarray):
            binary_mask = torch.from_numpy(binary_mask)
        binary_mask = binary_mask.float().unsqueeze(0)  # Add channel dim

        return image, binary_mask


# ══════════════════════════════════════════════════════════════════
# TRANSFORM PIPELINES
# ══════════════════════════════════════════════════════════════════

def get_train_transforms():
    """
    Augmentation + preprocessing for TRAINING images.

    WHY AUGMENTATION?
        We only have 811 training images. By randomly flipping, rotating,
        and adjusting brightness/contrast, we effectively create many
        "new" variations of each image. This helps the model:
        - Not memorize specific images (overfitting)
        - Learn to recognize oil in different orientations/lighting

    IMPORTANT: The same transform is applied to BOTH the image AND mask
    so they stay aligned. If we flip the image, we flip the mask too.

    MASK RESIZING: Uses NEAREST interpolation (not bilinear!) because
    bilinear creates averaged pixel values (like 0.5) which would corrupt
    our binary labels (should only be 0 or 1).
    """
    return A.Compose([
        # Resize to 512x512 (mask uses nearest-neighbor)
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE,
                 interpolation=cv2.INTER_LINEAR,        # Smooth for images
                 mask_interpolation=cv2.INTER_NEAREST),  # Exact for masks

        # Random augmentations (training only)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                 scale=(0.95, 1.05), rotate=(-15, 15), p=0.5,
                 border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                    contrast_limit=0.2, p=0.5),

        # Normalize with ImageNet stats (required for pretrained ResNet34)
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),

        # Convert numpy array to PyTorch tensor
        ToTensorV2(),
    ])


def get_val_transforms():
    """
    Preprocessing for VALIDATION and TEST images.

    NO random augmentation — we want deterministic, consistent results.
    Only resize and normalize.
    """
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE,
                 interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2(),
    ])


# ══════════════════════════════════════════════════════════════════
# DATALOADER FACTORY
# ══════════════════════════════════════════════════════════════════

def get_dataloaders(batch_size=None, num_workers=None):
    """
    Create DataLoaders for train, validation, and test sets.

    A DataLoader wraps a Dataset and provides:
    - Automatic batching (groups images into batches)
    - Shuffling (randomizes order each epoch for training)
    - Parallel data loading (uses multiple CPU threads)

    Args:
        batch_size: Override config.BATCH_SIZE if needed
        num_workers: Override config.NUM_WORKERS if needed

    Returns:
        train_loader, val_loader, test_loader
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if num_workers is None:
        num_workers = config.NUM_WORKERS

    # Create datasets
    train_dataset = OilSpillDataset(
        config.TRAIN_IMAGE_DIR, config.TRAIN_MASK_DIR,
        transform=get_train_transforms()
    )
    val_dataset = OilSpillDataset(
        config.VAL_IMAGE_DIR, config.VAL_MASK_DIR,
        transform=get_val_transforms()
    )
    test_dataset = OilSpillDataset(
        config.TEST_IMAGE_DIR, config.TEST_MASK_DIR,
        transform=get_val_transforms()
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # Randomize order each epoch (training only!)
        num_workers=num_workers,
        pin_memory=True,       # Faster GPU transfer
        drop_last=True,        # Drop incomplete last batch for stable training
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,         # No shuffling for evaluation
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """Quick test to verify the dataset and dataloaders work."""
    print("\n" + "=" * 60)
    print("  DATASET QUICK TEST")
    print("=" * 60)

    # Use small batch and 0 workers for testing
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=2, num_workers=0
    )

    print(f"\n  Train: {len(train_loader.dataset)} samples, "
          f"{len(train_loader)} batches")
    print(f"  Val:   {len(val_loader.dataset)} samples, "
          f"{len(val_loader)} batches")
    print(f"  Test:  {len(test_loader.dataset)} samples, "
          f"{len(test_loader)} batches")

    # Load one batch
    print(f"\n  Loading one training batch...")
    images, masks = next(iter(train_loader))

    print(f"  Image batch shape: {images.shape}")     # [B, 3, 512, 512]
    print(f"  Image dtype:       {images.dtype}")      # float32
    print(f"  Image value range: [{images.min():.2f}, {images.max():.2f}]")

    print(f"  Mask batch shape:  {masks.shape}")       # [B, 1, 512, 512]
    print(f"  Mask dtype:        {masks.dtype}")        # float32
    print(f"  Mask unique vals:  {torch.unique(masks).tolist()}")  # [0.0, 1.0]

    # Verify mask is binary
    mask_vals = torch.unique(masks)
    is_binary = all(v in [0.0, 1.0] for v in mask_vals.tolist())
    print(f"\n  Mask is binary:    {is_binary} {'[OK]' if is_binary else '[FAIL]'}")

    # Verify shapes
    expected_img = (2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    expected_mask = (2, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)
    print(f"  Image shape OK:    {images.shape == expected_img} "
          f"{'[OK]' if images.shape == expected_img else '[FAIL]'}")
    print(f"  Mask shape OK:     {masks.shape == expected_mask} "
          f"{'[OK]' if masks.shape == expected_mask else '[FAIL]'}")

    print(f"\n  [OK] Dataset and DataLoader working correctly!")
    print(f"{'=' * 60}\n")
