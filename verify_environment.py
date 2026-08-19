"""
verify_environment.py — Stage 1: Environment Verification
==========================================================

PURPOSE:
    This is the FIRST script you run after setting up the project.
    It checks that everything is installed correctly before you start
    any actual work.

WHAT IT CHECKS:
    1. Python version (need 3.8+)
    2. All required packages are installed
    3. GPU availability (CUDA)
    4. Dataset directory structure exists
    5. Dataset files are present in each split
    6. Output directories can be created
    7. Quick PyTorch tensor test

WHY RUN THIS FIRST:
    It's much better to catch missing packages or wrong paths NOW
    than to discover them halfway through training. Think of it as
    a pre-flight checklist before takeoff.

HOW TO RUN:
    python verify_environment.py

WHAT TO EXPECT:
    A list of checks with ✓ (pass) or ✗ (fail) next to each one.
    If everything passes, you're ready for Stage 2.
"""

import sys
import os

# ── Add project root to path ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def print_header(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def check_pass(msg):
    """Print a passing check."""
    print(f"  [PASS]  {msg}")
    return True


def check_fail(msg):
    """Print a failing check."""
    print(f"  [FAIL]  {msg}")
    return False


def check_warn(msg):
    """Print a warning (not a failure)."""
    print(f"  [WARN]  {msg}")


def main():
    all_passed = True
    total_checks = 0
    passed_checks = 0

    # ──────────────────────────────────────────────────────
    # CHECK 1: Python Version
    # ──────────────────────────────────────────────────────
    print_header("1. PYTHON VERSION")

    py_version = sys.version_info
    total_checks += 1
    if py_version >= (3, 8):
        passed_checks += 1
        check_pass(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        all_passed = False
        check_fail(f"Python {py_version.major}.{py_version.minor}.{py_version.micro} "
                    f"(need 3.8+)")

    # ──────────────────────────────────────────────────────
    # CHECK 2: Required Packages
    # ──────────────────────────────────────────────────────
    print_header("2. REQUIRED PACKAGES")

    packages = {
        "torch":                      "PyTorch (deep learning framework)",
        "torchvision":                "TorchVision (image utilities)",
        "segmentation_models_pytorch": "Segmentation Models PyTorch (U-Net)",
        "albumentations":             "Albumentations (image augmentation)",
        "cv2":                        "OpenCV (image processing)",
        "numpy":                      "NumPy (numerical computing)",
        "matplotlib":                 "Matplotlib (plotting)",
        "sklearn":                    "Scikit-learn (evaluation metrics)",
        "PIL":                        "Pillow (image I/O)",
        "tqdm":                       "tqdm (progress bars)",
    }

    for import_name, description in packages.items():
        total_checks += 1
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            passed_checks += 1
            check_pass(f"{description}: v{version}")
        except ImportError:
            all_passed = False
            check_fail(f"{description}: NOT INSTALLED")

    # ──────────────────────────────────────────────────────
    # CHECK 3: GPU / CUDA Availability
    # ──────────────────────────────────────────────────────
    print_header("3. GPU / CUDA")

    total_checks += 1
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            passed_checks += 1
            check_pass(f"CUDA available: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            passed_checks += 1  # Not a failure, just a warning
            check_warn("CUDA NOT available — training will use CPU (much slower)")
            check_warn("This is OK for testing, but GPU is recommended for full training")
    except Exception as e:
        all_passed = False
        check_fail(f"Error checking CUDA: {e}")

    # ──────────────────────────────────────────────────────
    # CHECK 4: Dataset Directory Structure
    # ──────────────────────────────────────────────────────
    print_header("4. DATASET STRUCTURE")

    required_dirs = [
        ("dataset",              "Dataset root"),
        ("dataset/train",        "Training split"),
        ("dataset/train/images", "Training images"),
        ("dataset/train/masks",  "Training masks"),
        ("dataset/val",          "Validation split"),
        ("dataset/val/images",   "Validation images"),
        ("dataset/val/masks",    "Validation masks"),
        ("dataset/test",         "Test split"),
        ("dataset/test/images",  "Test images"),
        ("dataset/test/masks",   "Test masks"),
    ]

    for rel_path, description in required_dirs:
        total_checks += 1
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if os.path.isdir(full_path):
            passed_checks += 1
            check_pass(f"{description}: {rel_path}/")
        else:
            all_passed = False
            check_fail(f"{description}: {rel_path}/ NOT FOUND")

    # ──────────────────────────────────────────────────────
    # CHECK 5: Dataset File Counts
    # ──────────────────────────────────────────────────────
    print_header("5. DATASET FILE COUNTS")

    splits = {
        "train": ("dataset/train/images", "dataset/train/masks"),
        "val":   ("dataset/val/images",   "dataset/val/masks"),
        "test":  ("dataset/test/images",  "dataset/test/masks"),
    }

    total_images = 0
    for split_name, (img_rel, mask_rel) in splits.items():
        img_dir = os.path.join(PROJECT_ROOT, img_rel)
        mask_dir = os.path.join(PROJECT_ROOT, mask_rel)

        total_checks += 1
        if os.path.isdir(img_dir) and os.path.isdir(mask_dir):
            img_count = len([f for f in os.listdir(img_dir)
                            if os.path.isfile(os.path.join(img_dir, f))])
            mask_count = len([f for f in os.listdir(mask_dir)
                             if os.path.isfile(os.path.join(mask_dir, f))])
            total_images += img_count

            if img_count > 0 and img_count == mask_count:
                passed_checks += 1
                check_pass(f"{split_name}: {img_count} images, {mask_count} masks")
            elif img_count == 0:
                all_passed = False
                check_fail(f"{split_name}: NO FILES FOUND (dataset may not be downloaded)")
            else:
                all_passed = False
                check_fail(f"{split_name}: MISMATCH — {img_count} images vs {mask_count} masks")
        else:
            all_passed = False
            check_fail(f"{split_name}: directories not found")

    if total_images > 0:
        print(f"\n  Total images across all splits: {total_images}")

    # ──────────────────────────────────────────────────────
    # CHECK 6: Output Directories
    # ──────────────────────────────────────────────────────
    print_header("6. OUTPUT DIRECTORIES")

    output_dirs = [
        "models",
        "outputs",
        "outputs/dataset_visualization",
        "outputs/test_predictions",
        "outputs/predictions",
        "logs",
    ]

    for rel_path in output_dirs:
        total_checks += 1
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        try:
            os.makedirs(full_path, exist_ok=True)
            passed_checks += 1
            check_pass(f"Created/verified: {rel_path}/")
        except Exception as e:
            all_passed = False
            check_fail(f"Cannot create {rel_path}/: {e}")

    # ──────────────────────────────────────────────────────
    # CHECK 7: Quick PyTorch Test
    # ──────────────────────────────────────────────────────
    print_header("7. PYTORCH QUICK TEST")

    total_checks += 1
    try:
        import torch
        # Create a small random tensor and do a basic operation
        x = torch.randn(2, 3, 512, 512)  # Simulate a batch of 2 RGB 512x512 images
        y = torch.sigmoid(x)             # Apply sigmoid (like our model output)
        assert y.shape == (2, 3, 512, 512), "Shape mismatch"
        assert y.min() >= 0 and y.max() <= 1, "Sigmoid output out of range"
        passed_checks += 1
        check_pass(f"Tensor operations working (tested shape: {x.shape})")
    except Exception as e:
        all_passed = False
        check_fail(f"PyTorch tensor test failed: {e}")

    # ──────────────────────────────────────────────────────
    # CHECK 8: Config Import Test
    # ──────────────────────────────────────────────────────
    print_header("8. CONFIG IMPORT TEST")

    total_checks += 1
    try:
        import config
        # Verify key settings exist
        attrs = [
            "IMAGE_SIZE", "DEVICE", "LEARNING_RATE", "BATCH_SIZE",
            "OIL_COLOR", "ENCODER_NAME", "BEST_MODEL_PATH",
        ]
        missing = [a for a in attrs if not hasattr(config, a)]
        if missing:
            all_passed = False
            check_fail(f"config.py missing attributes: {missing}")
        else:
            passed_checks += 1
            check_pass(f"config.py loaded successfully")
            print(f"       IMAGE_SIZE    = {config.IMAGE_SIZE}")
            print(f"       DEVICE        = {config.DEVICE}")
            print(f"       ENCODER_NAME  = {config.ENCODER_NAME}")
            print(f"       LEARNING_RATE = {config.LEARNING_RATE}")
            print(f"       BATCH_SIZE    = {config.BATCH_SIZE}")
            print(f"       OIL_COLOR     = {config.OIL_COLOR}")
    except Exception as e:
        all_passed = False
        check_fail(f"Failed to import config.py: {e}")

    # ──────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ──────────────────────────────────────────────────────
    print_header("SUMMARY")
    print(f"  Checks passed: {passed_checks}/{total_checks}")

    if all_passed:
        print(f"\n  [OK] ALL CHECKS PASSED -- Ready for Stage 2!")
        print(f"  Next step: Run 'python inspect_dataset.py'")
    else:
        print(f"\n  [!!] SOME CHECKS FAILED -- Please fix the issues above.")
        print(f"  If packages are missing, run: pip install -r requirements.txt")

    print(f"\n{'=' * 60}\n")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
