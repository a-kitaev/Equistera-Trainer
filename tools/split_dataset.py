#!/usr/bin/env python3
"""
Split COCO format dataset into train/val/test sets.

This script splits a COCO format annotation file into multiple subsets while:
- Preserving all annotation metadata (categories, keypoints, etc.)
- Maintaining class balance across splits
- Supporting stratified splitting by category
- Handling images without annotations
- Creating proper train/val/test directory structure
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import random
from collections import defaultdict


def load_coco_annotations(ann_file: Path) -> Dict:
    """Load COCO format annotations from JSON file."""
    print(f"Loading annotations from {ann_file}...")
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    print(f"  Images: {len(data.get('images', []))}")
    print(f"  Annotations: {len(data.get('annotations', []))}")
    print(f"  Categories: {len(data.get('categories', []))}")
    
    return data


def get_image_annotations_map(annotations: List[Dict]) -> Dict[int, List[Dict]]:
    """Create a mapping from image_id to list of annotations."""
    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann['image_id']].append(ann)
    return dict(img_to_anns)


def get_image_categories_map(annotations: List[Dict]) -> Dict[int, set]:
    """Create a mapping from image_id to set of category_ids."""
    img_to_cats = defaultdict(set)
    for ann in annotations:
        img_to_cats[ann['image_id']].add(ann['category_id'])
    return dict(img_to_cats)


def stratified_split(
    images: List[Dict],
    img_to_cats: Dict[int, set],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split images into train/val/test sets with stratification by category.
    
    For horse pose estimation with single category, this ensures balanced distribution.
    For multi-category datasets, it tries to balance category representation.
    """
    random.seed(seed)
    
    # Group images by their categories (for stratification)
    # For single-category datasets, this still helps ensure randomness
    category_groups = defaultdict(list)
    for img in images:
        img_id = img['id']
        # Use frozenset of categories as key (handles multi-label)
        cats = frozenset(img_to_cats.get(img_id, set()))
        category_groups[cats].append(img)
    
    train_imgs, val_imgs, test_imgs = [], [], []
    
    # Split each category group
    for cat_group, imgs in category_groups.items():
        random.shuffle(imgs)
        n_total = len(imgs)
        
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_imgs.extend(imgs[:n_train])
        val_imgs.extend(imgs[n_train:n_train + n_val])
        test_imgs.extend(imgs[n_train + n_val:])
    
    # Shuffle again to mix category groups
    random.shuffle(train_imgs)
    random.shuffle(val_imgs)
    random.shuffle(test_imgs)
    
    return train_imgs, val_imgs, test_imgs


def create_split_annotations(
    original_data: Dict,
    split_images: List[Dict],
    img_to_anns: Dict[int, List[Dict]],
    split_name: str
) -> Dict:
    """Create COCO annotations dict for a specific split."""
    split_img_ids = {img['id'] for img in split_images}
    
    # Filter annotations for this split
    split_annotations = []
    for img_id in split_img_ids:
        split_annotations.extend(img_to_anns.get(img_id, []))
    
    # Create new annotation dict
    split_data = {
        'info': original_data.get('info', {}),
        'licenses': original_data.get('licenses', []),
        'categories': original_data.get('categories', []),
        'images': split_images,
        'annotations': split_annotations
    }
    
    # Update info
    if 'info' in split_data:
        split_data['info']['description'] = f"{split_data['info'].get('description', '')} - {split_name} split"
    
    return split_data


def copy_images(
    images: List[Dict],
    src_img_dir: Path,
    dst_img_dir: Path,
    symlink: bool = False
) -> None:
    """Copy or symlink images to destination directory."""
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    
    for img in images:
        src_path = src_img_dir / img['file_name']
        dst_path = dst_img_dir / img['file_name']
        
        if not src_path.exists():
            print(f"Warning: Source image not found: {src_path}")
            continue
        
        if dst_path.exists():
            continue
        
        if symlink:
            dst_path.symlink_to(src_path.absolute())
        else:
            shutil.copy2(src_path, dst_path)


def print_split_statistics(
    train_data: Dict,
    val_data: Dict,
    test_data: Dict
) -> None:
    """Print statistics about the dataset splits."""
    print("\n" + "="*60)
    print("DATASET SPLIT STATISTICS")
    print("="*60)
    
    splits = [
        ('Train', train_data),
        ('Val', val_data),
        ('Test', test_data)
    ]
    
    total_imgs = sum(len(data['images']) for _, data in splits)
    total_anns = sum(len(data['annotations']) for _, data in splits)
    
    print(f"\n{'Split':<10} {'Images':<10} {'%':<8} {'Annotations':<15} {'Avg/Image':<10}")
    print("-" * 60)
    
    for split_name, data in splits:
        n_imgs = len(data['images'])
        n_anns = len(data['annotations'])
        pct = (n_imgs / total_imgs * 100) if total_imgs > 0 else 0
        avg = (n_anns / n_imgs) if n_imgs > 0 else 0
        
        print(f"{split_name:<10} {n_imgs:<10} {pct:>6.1f}%  {n_anns:<15} {avg:>8.2f}")
    
    print("-" * 60)
    print(f"{'Total':<10} {total_imgs:<10} {'100.0%':<8} {total_anns:<15}")
    
    # Category distribution
    print("\n" + "="*60)
    print("CATEGORY DISTRIBUTION")
    print("="*60)
    
    categories = train_data['categories']
    print(f"\n{'Category':<20} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("-" * 60)
    
    for cat in categories:
        cat_id = cat['id']
        cat_name = cat['name']
        
        train_count = sum(1 for ann in train_data['annotations'] if ann['category_id'] == cat_id)
        val_count = sum(1 for ann in val_data['annotations'] if ann['category_id'] == cat_id)
        test_count = sum(1 for ann in test_data['annotations'] if ann['category_id'] == cat_id)
        
        print(f"{cat_name:<20} {train_count:<10} {val_count:<10} {test_count:<10}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Split COCO format dataset into train/val/test sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default 70/15/15 split
  python tools/split_dataset.py \\
    --ann-file data/annotations/horse_all.json \\
    --img-dir data/images \\
    --out-dir data

  # Custom 80/10/10 split
  python tools/split_dataset.py \\
    --ann-file data/annotations/horse_all.json \\
    --img-dir data/images \\
    --out-dir data \\
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

  # Use symlinks instead of copying images (faster, saves space)
  python tools/split_dataset.py \\
    --ann-file data/annotations/horse_all.json \\
    --img-dir data/images \\
    --out-dir data \\
    --symlink

  # Annotations only, no image copying
  python tools/split_dataset.py \\
    --ann-file data/annotations/horse_all.json \\
    --out-dir data \\
    --no-copy-images
        """
    )
    
    parser.add_argument(
        '--ann-file',
        type=Path,
        required=True,
        help='Path to COCO format annotation file'
    )
    parser.add_argument(
        '--img-dir',
        type=Path,
        help='Directory containing images (required unless --no-copy-images)'
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        required=True,
        help='Output directory for splits'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Ratio of training data (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Ratio of validation data (default: 0.15)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Ratio of test data (default: 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--symlink',
        action='store_true',
        help='Create symlinks instead of copying images (saves space)'
    )
    parser.add_argument(
        '--no-copy-images',
        action='store_true',
        help='Only create annotation splits, do not copy images'
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not 0.99 <= total_ratio <= 1.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Validate image directory
    if not args.no_copy_images and not args.img_dir:
        raise ValueError("--img-dir is required unless --no-copy-images is set")
    
    # Load original annotations
    original_data = load_coco_annotations(args.ann_file)
    
    # Create mappings
    img_to_anns = get_image_annotations_map(original_data.get('annotations', []))
    img_to_cats = get_image_categories_map(original_data.get('annotations', []))
    
    # Split dataset
    print(f"\nSplitting dataset ({args.train_ratio:.0%}/{args.val_ratio:.0%}/{args.test_ratio:.0%})...")
    train_imgs, val_imgs, test_imgs = stratified_split(
        original_data['images'],
        img_to_cats,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
    
    print(f"  Train: {len(train_imgs)} images")
    print(f"  Val: {len(val_imgs)} images")
    print(f"  Test: {len(test_imgs)} images")
    
    # Create split annotations
    print("\nCreating annotation files...")
    train_data = create_split_annotations(original_data, train_imgs, img_to_anns, 'train')
    val_data = create_split_annotations(original_data, val_imgs, img_to_anns, 'val')
    test_data = create_split_annotations(original_data, test_imgs, img_to_anns, 'test')
    
    # Create output directories
    ann_out_dir = args.out_dir / 'annotations'
    ann_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save annotation files
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        out_file = ann_out_dir / f'horse_{split_name}.json'
        print(f"  Saving {out_file}")
        with open(out_file, 'w') as f:
            json.dump(split_data, f, indent=2)
    
    # Copy/symlink images
    if not args.no_copy_images:
        print(f"\n{'Symlinking' if args.symlink else 'Copying'} images...")
        for split_name, split_imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            print(f"  {split_name}: {len(split_imgs)} images")
            img_out_dir = args.out_dir / split_name
            copy_images(split_imgs, args.img_dir, img_out_dir, args.symlink)
    
    # Print statistics
    print_split_statistics(train_data, val_data, test_data)
    
    print("✅ Dataset split complete!")
    print(f"\nOutput structure:")
    print(f"  {args.out_dir}/")
    print(f"  ├── annotations/")
    print(f"  │   ├── horse_train.json")
    print(f"  │   ├── horse_val.json")
    print(f"  │   └── horse_test.json")
    if not args.no_copy_images:
        print(f"  ├── train/")
        print(f"  ├── val/")
        print(f"  └── test/")


if __name__ == '__main__':
    main()
