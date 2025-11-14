# Dataset Preparation Guide

## Overview

For training MMPose models, you need to split your dataset into three sets:
- **Train (70%)**: Used to train the model
- **Val (15%)**: Used for monitoring during training (prevents overfitting)
- **Test (15%)**: Used for final evaluation (unseen data)

## Quick Start

### 1. Prepare Your Dataset

First, ensure you have:
```
data/
├── annotations/
│   └── horse_all.json    # All annotations in COCO format
└── images/               # All images
    ├── IMG_001.jpg
    ├── IMG_002.jpg
    └── ...
```

### 2. Split the Dataset

**Default 70/15/15 split:**
```bash
make split-dataset
```

**Custom split ratios:**
```bash
make split-custom TRAIN=0.8 VAL=0.1 TEST=0.1
```

**Use symlinks (saves disk space):**
```bash
make split-symlink
```

### 3. Result Structure

After splitting, you'll have:
```
data/
├── annotations/
│   ├── horse_all.json      # Original
│   ├── horse_train.json    # Training set
│   ├── horse_val.json      # Validation set
│   └── horse_test.json     # Test set
├── images/                 # Original (can delete after split)
├── train/                  # Training images
├── val/                    # Validation images
└── test/                   # Test images
```

## Detailed Usage

### split_dataset.py Options

```bash
python tools/split_dataset.py \
    --ann-file data/annotations/horse_all.json \
    --img-dir data/images \
    --out-dir data \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42
```

**Parameters:**
- `--ann-file`: Path to COCO format annotation file
- `--img-dir`: Directory containing images
- `--out-dir`: Output directory for splits
- `--train-ratio`: Training set ratio (default: 0.7)
- `--val-ratio`: Validation set ratio (default: 0.15)
- `--test-ratio`: Test set ratio (default: 0.15)
- `--seed`: Random seed for reproducibility (default: 42)
- `--symlink`: Create symlinks instead of copying images
- `--no-copy-images`: Only split annotations, don't copy images

### Examples

**1. Standard split with copied images:**
```bash
python tools/split_dataset.py \
    --ann-file data/annotations/horse_all.json \
    --img-dir data/images \
    --out-dir data
```

**2. Split with symlinks (recommended for large datasets):**
```bash
python tools/split_dataset.py \
    --ann-file data/annotations/horse_all.json \
    --img-dir data/images \
    --out-dir data \
    --symlink
```

**3. Custom 80/10/10 split:**
```bash
python tools/split_dataset.py \
    --ann-file data/annotations/horse_all.json \
    --img-dir data/images \
    --out-dir data \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

**4. Split annotations only (images already organized):**
```bash
python tools/split_dataset.py \
    --ann-file data/annotations/horse_all.json \
    --out-dir data \
    --no-copy-images
```

## Understanding the Splits

### Why Split?

**Training Set (70-80%)**
- Largest portion used to train the model
- Model learns patterns from this data
- More data = better learning (usually)

**Validation Set (10-15%)**
- Used during training to monitor performance
- Helps detect overfitting
- Used for hyperparameter tuning
- NOT used for training

**Test Set (10-15%)**
- Final evaluation on completely unseen data
- Gives true measure of model performance
- Should only be used once at the end

### Split Ratios by Dataset Size

**Small Dataset (< 1000 images)**
- Train: 70%, Val: 15%, Test: 15%
- Ensures enough validation/test data

**Medium Dataset (1000-5000 images)**
- Train: 75%, Val: 15%, Test: 10%
- Can slightly increase training data

**Large Dataset (> 5000 images)**
- Train: 80%, Val: 10%, Test: 10%
- Plenty of data for all sets

### For Your 800 Image Dataset

Recommended split: **70/15/15**
- Train: 560 images
- Val: 120 images
- Test: 120 images

This gives enough validation data to monitor training and enough test data for reliable evaluation.

## Features

### 1. Stratified Splitting

The script uses stratified splitting to ensure:
- Balanced distribution across splits
- Similar category representation in each set
- Random but reproducible splits (with seed)

### 2. Category Balance

For horse datasets with single category:
- Ensures random distribution
- Maintains annotation density

For multi-category datasets:
- Balances categories across splits
- Prevents train/val/test bias

### 3. Statistics Output

After splitting, you'll see:
```
==============================================================
DATASET SPLIT STATISTICS
==============================================================

Split      Images     %        Annotations     Avg/Image
--------------------------------------------------------------
Train      560        70.0%    560             1.00
Val        120        15.0%    120             1.00
Test       120        15.0%    120             1.00
--------------------------------------------------------------
Total      800        100.0%   800
```

### 4. Reproducibility

Using `--seed 42` ensures:
- Same split every time
- Reproducible experiments
- Consistent results across runs

## Common Scenarios

### Scenario 1: First Time Setup

```bash
# 1. Put all images in one folder
data/images/

# 2. Create single annotation file
data/annotations/horse_all.json

# 3. Split the dataset
make split-dataset

# 4. Start training
make train-rtm
```

### Scenario 2: Already Have Train/Val Split

If you already have `train.json` and `val.json`:
```bash
# Just rename them
mv data/annotations/train.json data/annotations/horse_train.json
mv data/annotations/val.json data/annotations/horse_val.json

# Create test set from validation if needed
# Or manually split validation into val+test
```

### Scenario 3: Large Dataset (Save Space)

```bash
# Use symlinks instead of copying
make split-symlink

# This creates:
# data/train/IMG_001.jpg -> ../images/IMG_001.jpg
# data/val/IMG_005.jpg -> ../images/IMG_005.jpg
# etc.
```

### Scenario 4: Adding More Data Later

```bash
# 1. Add new images to original folder
data/images/

# 2. Update horse_all.json with new annotations

# 3. Re-split (use same seed for consistency)
make split-dataset

# 4. Resume training from checkpoint
make resume-rtm
```

## Verification

After splitting, verify your dataset:

```bash
# Check annotations
python tools/verify_dataset.py --ann-file data/annotations/horse_train.json
python tools/verify_dataset.py --ann-file data/annotations/horse_val.json
python tools/verify_dataset.py --ann-file data/annotations/horse_test.json

# Check image counts
ls data/train/ | wc -l    # Should show ~560 for 800 total
ls data/val/ | wc -l      # Should show ~120
ls data/test/ | wc -l     # Should show ~120
```

## Troubleshooting

### Issue: Ratios don't sum to 1.0

```bash
# Error: Ratios must sum to 1.0, got 0.95
```

**Solution:** Ensure train + val + test = 1.0
```bash
make split-custom TRAIN=0.7 VAL=0.15 TEST=0.15  # ✓ 1.0
```

### Issue: Image not found warnings

```bash
# Warning: Source image not found: data/images/IMG_123.jpg
```

**Solution:** 
- Check `file_name` in annotations matches actual filenames
- Ensure images are in specified `--img-dir`
- Check for case sensitivity (IMG.jpg vs img.jpg)

### Issue: No annotations for some images

This is normal! Some images might not have annotations.
The script handles this automatically.

### Issue: Imbalanced splits

If you see very different annotation counts:
```
Train: 600, Val: 50, Test: 150  # Imbalanced!
```

**Solution:** Use a different random seed or check your dataset:
```bash
python tools/split_dataset.py ... --seed 123
```

## Best Practices

### 1. Always Use Seed

```bash
# Always specify seed for reproducibility
--seed 42
```

### 2. Split Before Training

Never split mid-training. Always split first, then train.

### 3. Keep Original Data

```bash
# Keep horse_all.json and images/ as backup
# Don't delete originals until training is successful
```

### 4. Validate After Splitting

```bash
# Always verify splits
make verify-data
```

### 5. Document Your Split

```bash
# Save split command for reproducibility
echo "make split-dataset" > DATASET_NOTES.txt
echo "Seed: 42, Ratio: 70/15/15" >> DATASET_NOTES.txt
```

## Integration with Training

The dataset config (`configs/_base_/datasets/horse_ap10k.py`) expects:
```python
ann_file='annotations/horse_train.json'
ann_file='annotations/horse_val.json'
```

After running `make split-dataset`, these files will be created automatically.

## Summary

**Quick Commands:**
```bash
# Standard split
make split-dataset

# Custom split
make split-custom TRAIN=0.8 VAL=0.1 TEST=0.1

# Space-saving split
make split-symlink

# Verify
make verify-data
```

**Dataset Flow:**
1. Annotate images → `horse_all.json`
2. Split dataset → `horse_train/val/test.json`
3. Verify splits → `make verify-data`
4. Train model → `make train-rtm`

---

**Ready to split your dataset? Run `make split-dataset`! 🚀**
