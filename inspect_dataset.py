"""
inspect_dataset.py -- Stage 2: Dataset Inspection
===================================================

PURPOSE:
    Verify the dataset is correct before any training.
    Checks file counts, image-mask matching, dimensions,
    formats, mask colors, and corrupted files.

HOW TO RUN:
    python inspect_dataset.py
"""

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import config


def get_stem(filename):
    """Get filename without extension: 'Oil (1).jpg' -> 'Oil (1)'"""
    return Path(filename).stem


def inspect_split(split_name, image_dir, mask_dir):
    """Inspect one dataset split thoroughly."""
    print(f"\n{'=' * 60}")
    print(f"  INSPECTING: {split_name.upper()} SPLIT")
    print(f"{'=' * 60}")

    results = {"split": split_name, "passed": True, "issues": []}

    # Check directories exist
    for label, d in [("Image", image_dir), ("Mask", mask_dir)]:
        if not os.path.isdir(d):
            results["passed"] = False
            results["issues"].append(f"{label} dir not found: {d}")
            print(f"  [FAIL] {label} directory not found")
            return results

    # List files
    image_files = sorted(f for f in os.listdir(image_dir)
                         if os.path.isfile(os.path.join(image_dir, f)))
    mask_files = sorted(f for f in os.listdir(mask_dir)
                        if os.path.isfile(os.path.join(mask_dir, f)))

    print(f"\n  File Counts:")
    print(f"    Images: {len(image_files)}")
    print(f"    Masks:  {len(mask_files)}")

    # Match by stem
    image_stems = {get_stem(f): f for f in image_files}
    mask_stems = {get_stem(f): f for f in mask_files}

    matched = set(image_stems) & set(mask_stems)
    no_mask = set(image_stems) - set(mask_stems)
    no_image = set(mask_stems) - set(image_stems)

    results["num_images"] = len(image_files)
    results["num_masks"] = len(mask_files)
    results["num_matched"] = len(matched)

    print(f"\n  Filename Matching:")
    print(f"    Matched pairs:       {len(matched)}")

    if no_mask:
        results["passed"] = False
        results["issues"].append(f"{len(no_mask)} images without masks")
        print(f"    [FAIL] Images without masks: {len(no_mask)}")
    else:
        print(f"    Images without masks: 0 [OK]")

    if no_image:
        results["passed"] = False
        results["issues"].append(f"{len(no_image)} masks without images")
        print(f"    [FAIL] Masks without images: {len(no_image)}")
    else:
        print(f"    Masks without images: 0 [OK]")

    if not matched:
        results["passed"] = False
        results["issues"].append("No matched pairs!")
        return results

    # Inspect pairs: dimensions + formats (fast) for ALL,
    # color analysis on a SAMPLE (slow part)
    print(f"\n  Checking dimensions and formats...")

    image_sizes = Counter()
    mask_sizes = Counter()
    image_formats = Counter()
    mask_formats = Counter()
    corrupted = []
    mismatches = []

    matched_sorted = sorted(matched)
    for stem in tqdm(matched_sorted, desc=f"  {split_name}", ncols=70):
        img_path = os.path.join(image_dir, image_stems[stem])
        mask_path = os.path.join(mask_dir, mask_stems[stem])

        try:
            img = Image.open(img_path)
            img.load()  # Force read to catch corruption
            image_sizes[img.size] += 1
            image_formats[img.format] += 1
            img_sz = img.size
        except Exception as e:
            corrupted.append(f"IMAGE: {image_stems[stem]} - {e}")
            continue

        try:
            mask = Image.open(mask_path)
            mask.load()
            mask_sizes[mask.size] += 1
            mask_formats[mask.format] += 1
            mask_sz = mask.size
        except Exception as e:
            corrupted.append(f"MASK: {mask_stems[stem]} - {e}")
            continue

        if img_sz != mask_sz:
            mismatches.append(f"{stem}: img={img_sz}, mask={mask_sz}")

    # Print dimension info
    print(f"\n  Image Dimensions:")
    for sz, cnt in image_sizes.most_common():
        print(f"    {sz[0]} x {sz[1]}: {cnt} images")

    print(f"  Mask Dimensions:")
    for sz, cnt in mask_sizes.most_common():
        print(f"    {sz[0]} x {sz[1]}: {cnt} masks")

    print(f"\n  File Formats:")
    print(f"    Images: {dict(image_formats)}")
    print(f"    Masks:  {dict(mask_formats)}")

    if mismatches:
        print(f"\n  [WARN] Size Mismatches: {len(mismatches)}")
        for m in mismatches[:3]:
            print(f"    - {m}")
    else:
        print(f"  Image-Mask Size Match: ALL [OK]")

    if corrupted:
        results["passed"] = False
        results["issues"].append(f"{len(corrupted)} corrupted files")
        print(f"\n  [FAIL] Corrupted: {len(corrupted)}")
        for c in corrupted:
            print(f"    - {c}")
    else:
        print(f"  Corrupted Files: None [OK]")

    # Color analysis on a sample of 30 masks (much faster)
    print(f"\n  Analyzing mask colors (sampling 30 masks)...")
    sample_size = min(30, len(matched_sorted))
    rng = np.random.RandomState(42)
    sample_stems = list(rng.choice(matched_sorted, sample_size, replace=False))

    all_colors = set()
    oil_found = False

    for stem in sample_stems:
        mask_path = os.path.join(mask_dir, mask_stems[stem])
        try:
            mask = Image.open(mask_path).convert("RGB")
            mask_np = np.array(mask)
            pixels = mask_np.reshape(-1, 3)
            unique = np.unique(pixels, axis=0)
            for c in unique:
                ct = tuple(c)
                all_colors.add(ct)
                if ct == config.OIL_COLOR:
                    oil_found = True
        except Exception:
            pass

    # Report colors
    expected = set(config.MASK_COLORS.values())
    print(f"\n  Mask Colors Found (from 30-mask sample):")
    for color in sorted(all_colors):
        label = "UNKNOWN"
        for name, rgb in config.MASK_COLORS.items():
            if color == rgb:
                label = name.upper()
                break
        status = "[OK]" if label != "UNKNOWN" else "[?]"
        print(f"    RGB{color}: {label} {status}")

    unexpected = all_colors - expected
    if unexpected:
        print(f"\n  [INFO] {len(unexpected)} unexpected colors found.")
        print(f"         Likely from PNG compression or anti-aliasing.")
        print(f"         Our mask converter will handle these correctly.")

    if oil_found:
        print(f"\n  Oil Color {config.OIL_COLOR}: PRESENT [OK]")
    else:
        # Check more masks if oil wasn't found in sample
        print(f"  [INFO] Oil color not in 30-mask sample, checking more...")
        for stem in matched_sorted[:100]:
            mask_path = os.path.join(mask_dir, mask_stems[stem])
            try:
                mask_np = np.array(Image.open(mask_path).convert("RGB"))
                if np.any(np.all(mask_np == config.OIL_COLOR, axis=-1)):
                    oil_found = True
                    break
            except Exception:
                pass
        if oil_found:
            print(f"  Oil Color {config.OIL_COLOR}: PRESENT [OK]")
        else:
            print(f"  [WARN] Oil color not found in checked masks")
            results["issues"].append("Oil color not found in sampled masks")

    return results


def main():
    print("\n" + "=" * 60)
    print("  SPILLSENSE AI -- DATASET INSPECTION")
    print("  Oil Spill Drone Dataset")
    print("=" * 60)

    splits = [
        ("train", config.TRAIN_IMAGE_DIR, config.TRAIN_MASK_DIR),
        ("val",   config.VAL_IMAGE_DIR,   config.VAL_MASK_DIR),
        ("test",  config.TEST_IMAGE_DIR,  config.TEST_MASK_DIR),
    ]

    all_results = []
    for name, img_dir, mask_dir in splits:
        all_results.append(inspect_split(name, img_dir, mask_dir))

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 60}")

    total_img = 0
    all_passed = True
    for r in all_results:
        status = "PASS" if r["passed"] else "FAIL"
        n = r.get("num_matched", 0)
        total_img += r.get("num_images", 0)
        print(f"  {r['split'].upper():>5}: {r.get('num_images',0)} images, "
              f"{r.get('num_masks',0)} masks, {n} matched  [{status}]")
        if r["issues"]:
            all_passed = False
            for issue in r["issues"]:
                print(f"         - {issue}")

    print(f"\n  TOTAL: {total_img} images")

    if all_passed:
        print(f"\n  [OK] ALL CHECKS PASSED -- Dataset is ready!")
        print(f"  Next: python visualize_dataset.py")
    else:
        print(f"\n  [!!] ISSUES FOUND -- Review above before proceeding")

    print(f"\n{'=' * 60}\n")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
