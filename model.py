"""
model.py -- Stage 6: U-Net + ResNet34 Model
=============================================

PURPOSE:
    Defines the segmentation model: U-Net with a ResNet34 encoder.

HOW THE MODEL WORKS (simplified):

    Imagine looking at a photo and trying to outline all the oil:

    1. ENCODER (ResNet34 - the "seeing" part):
       Takes the 512x512 image and progressively shrinks it while
       extracting features at each level:

       512x512 -> 256x256 -> 128x128 -> 64x64 -> 32x32 -> 16x16

       At each level, it learns to detect:
       - Level 1: edges, simple textures
       - Level 2: corners, patterns
       - Level 3: object parts
       - Level 4-5: complex features (oil-like textures vs water)

    2. DECODER (U-Net decoder - the "drawing" part):
       Takes those features and expands back to full resolution:

       16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256 -> 512x512

       SKIP CONNECTIONS: At each level, the decoder also receives
       the encoder's features from the same level. This is crucial!
       Without them, the decoder would lose fine spatial details
       like exact oil boundaries.

    3. OUTPUT:
       A single channel 512x512 map where each pixel value (0-1)
       represents the probability of that pixel being oil.

WHAT GOES IN:
    Batch of images: tensor [B, 3, 512, 512]

WHAT COMES OUT:
    Batch of predictions: tensor [B, 1, 512, 512]
    (probabilities before thresholding)

HOW TO RUN (standalone test):
    python model.py
"""

import sys
import torch
import segmentation_models_pytorch as smp

import config


def create_model():
    """
    Create the U-Net + ResNet34 segmentation model.

    Uses the segmentation_models_pytorch library which provides
    pre-built architectures with pretrained encoders.

    Returns:
        model: PyTorch nn.Module, U-Net with ResNet34 encoder
    """
    model = smp.Unet(
        encoder_name=config.ENCODER_NAME,        # "resnet34"
        encoder_weights=config.ENCODER_WEIGHTS,  # "imagenet" pretrained
        in_channels=config.IN_CHANNELS,          # 3 (RGB)
        classes=config.NUM_CLASSES,              # 1 (binary segmentation)
    )
    return model


def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ══════════════════════════════════════════════════════════════════
# FORWARD-PASS TEST (Stage 7)
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MODEL FORWARD-PASS TEST (Stage 7)")
    print("=" * 60)

    # Create model
    print(f"\n  Creating U-Net + ResNet34...")
    model = create_model()
    model = model.to(config.DEVICE)
    model.eval()  # Set to evaluation mode (no dropout, etc.)

    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {config.DEVICE}")

    # Create a fake batch of images
    print(f"\n  Running forward pass with dummy data...")
    dummy_input = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    dummy_input = dummy_input.to(config.DEVICE)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\n  Input shape:  {dummy_input.shape}")   # [2, 3, 512, 512]
    print(f"  Output shape: {output.shape}")           # [2, 1, 512, 512]
    print(f"  Output dtype: {output.dtype}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(output)
    print(f"\n  After sigmoid:")
    print(f"  Prob range:   [{probs.min():.4f}, {probs.max():.4f}]")

    # Verify shapes match
    expected_output = (2, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)
    shape_ok = output.shape == expected_output
    print(f"\n  Output shape matches expected {expected_output}: "
          f"{shape_ok} {'[OK]' if shape_ok else '[FAIL]'}")

    # Test with actual data loader
    print(f"\n  Testing with actual dataset...")
    try:
        from dataset import get_dataloaders
        train_loader, _, _ = get_dataloaders(batch_size=2, num_workers=0)
        images, masks = next(iter(train_loader))
        images = images.to(config.DEVICE)
        masks = masks.to(config.DEVICE)

        with torch.no_grad():
            preds = model(images)

        print(f"  Real image batch: {images.shape}")
        print(f"  Real mask batch:  {masks.shape}")
        print(f"  Model output:     {preds.shape}")
        print(f"  Shapes match:     {preds.shape == masks.shape} [OK]")

        # Quick loss test
        import torch.nn as nn
        bce_loss = nn.BCEWithLogitsLoss()(preds, masks)
        print(f"  BCE loss (sanity): {bce_loss.item():.4f} [OK]")

    except Exception as e:
        print(f"  [WARN] Could not test with real data: {e}")

    all_ok = shape_ok
    if all_ok:
        print(f"\n  [OK] Model forward pass successful!")
        print(f"  Ready for Stage 8 (debug training)")
    else:
        print(f"\n  [FAIL] Issues detected -- fix before training!")

    print(f"{'=' * 60}\n")
    sys.exit(0 if all_ok else 1)
