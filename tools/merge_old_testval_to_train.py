#!/usr/bin/env python3
"""
Merge old test/val datasets into new training dataset.

Since you're retraining with a new larger dataset, the old test/val sets
can be added to the new training data to maximize training samples.

This script:
1. Loads the new train/val/test COCO annotations
2. Loads the old test/val COCO annotations
3. Merges old test/val into new train set
4. Updates image IDs and annotation IDs to avoid conflicts
5. Saves updated COCO files

Usage:
    python tools/merge_old_testval_to_train.py \
        --old-test data/annotations/old_horse_test.json \
        --old-val data/annotations/old_horse_val.json \
        --new-train data/annotations/horse_train.json \
        --output data/annotations/horse_train_merged.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge old test/val datasets into new training set')
    parser.add_argument(
        '--old-test',
        required=True,
        help='Old test annotations (COCO format)')
    parser.add_argument(
        '--old-val',
        required=True,
        help='Old val annotations (COCO format)')
    parser.add_argument(
        '--new-train',
        required=True,
        help='New train annotations (COCO format)')
    parser.add_argument(
        '--output',
        required=True,
        help='Output merged train annotations')
    parser.add_argument(
        '--check-duplicates',
        action='store_true',
        help='Check for duplicate images and skip them')
    return parser.parse_args()


def load_coco(file_path):
    """Load COCO format annotations."""
    print(f"Loading: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Check if this is COCO format (dict) or Label Studio format (list)
    if isinstance(data, list):
        print(f"  ❌ ERROR: This file appears to be in Label Studio format (list), not COCO format (dict)")
        print(f"  Please run convert_labelstudio_to_coco_v2.py first to create COCO format files.")
        sys.exit(1)
    
    if 'images' not in data or 'annotations' not in data:
        print(f"  ❌ ERROR: Invalid COCO format - missing 'images' or 'annotations' keys")
        sys.exit(1)
    
    print(f"  Images: {len(data['images'])}")
    print(f"  Annotations: {len(data['annotations'])}")
    
    return data


def merge_datasets(new_train, old_test, old_val, check_duplicates=False):
    """
    Merge old test/val into new train dataset.
    
    Returns merged dataset with updated IDs.
    """
    print("\nMerging datasets...")
    
    # Start with new_train as base
    merged = {
        'info': new_train['info'],
        'licenses': new_train['licenses'],
        'categories': new_train['categories'],
        'images': new_train['images'].copy(),
        'annotations': new_train['annotations'].copy()
    }
    
    # Track existing filenames if checking duplicates
    existing_filenames = set()
    if check_duplicates:
        existing_filenames = {img['file_name'] for img in merged['images']}
        print(f"  Existing images in new train: {len(existing_filenames)}")
    
    # Get current max IDs
    max_image_id = max((img['id'] for img in merged['images']), default=0)
    max_ann_id = max((ann['id'] for ann in merged['annotations']), default=0)
    
    print(f"  Starting image ID: {max_image_id + 1}")
    print(f"  Starting annotation ID: {max_ann_id + 1}")
    
    # Merge old test and val
    datasets_to_merge = [
        ('old_test', old_test),
        ('old_val', old_val)
    ]
    
    total_added_images = 0
    total_added_annotations = 0
    total_skipped = 0
    
    for dataset_name, dataset in datasets_to_merge:
        print(f"\n  Processing {dataset_name}...")
        
        # Create ID mapping for this dataset
        image_id_map = {}
        
        added_images = 0
        added_annotations = 0
        skipped = 0
        
        # Add images with new IDs
        for old_img in dataset['images']:
            old_id = old_img['id']
            filename = old_img['file_name']
            
            # Check for duplicates
            if check_duplicates and filename in existing_filenames:
                skipped += 1
                continue
            
            # Assign new ID
            max_image_id += 1
            new_id = max_image_id
            image_id_map[old_id] = new_id
            
            # Add image with new ID
            new_img = old_img.copy()
            new_img['id'] = new_id
            merged['images'].append(new_img)
            
            if check_duplicates:
                existing_filenames.add(filename)
            
            added_images += 1
        
        # Add annotations with new IDs
        for old_ann in dataset['annotations']:
            old_image_id = old_ann['image_id']
            
            # Skip if image was skipped (duplicate)
            if old_image_id not in image_id_map:
                continue
            
            # Assign new IDs
            max_ann_id += 1
            new_ann_id = max_ann_id
            new_image_id = image_id_map[old_image_id]
            
            # Add annotation with new IDs
            new_ann = old_ann.copy()
            new_ann['id'] = new_ann_id
            new_ann['image_id'] = new_image_id
            merged['annotations'].append(new_ann)
            
            added_annotations += 1
        
        print(f"    Added images: {added_images}")
        print(f"    Added annotations: {added_annotations}")
        if check_duplicates:
            print(f"    Skipped (duplicates): {skipped}")
        
        total_added_images += added_images
        total_added_annotations += added_annotations
        total_skipped += skipped
    
    print(f"\n{'='*70}")
    print("Merge Summary")
    print(f"{'='*70}")
    print(f"Original new train:")
    print(f"  Images: {len(new_train['images'])}")
    print(f"  Annotations: {len(new_train['annotations'])}")
    print(f"\nAdded from old test/val:")
    print(f"  Images: {total_added_images}")
    print(f"  Annotations: {total_added_annotations}")
    if check_duplicates:
        print(f"  Skipped duplicates: {total_skipped}")
    print(f"\nFinal merged dataset:")
    print(f"  Images: {len(merged['images'])}")
    print(f"  Annotations: {len(merged['annotations'])}")
    print(f"  Increase: +{total_added_images} images (+{total_added_images/len(new_train['images'])*100:.1f}%)")
    
    return merged


def validate_merged_dataset(merged):
    """Validate the merged dataset has no ID conflicts."""
    print("\nValidating merged dataset...")
    
    # Check image IDs are unique
    image_ids = [img['id'] for img in merged['images']]
    if len(image_ids) != len(set(image_ids)):
        print("  ❌ ERROR: Duplicate image IDs found!")
        return False
    
    # Check annotation IDs are unique
    ann_ids = [ann['id'] for ann in merged['annotations']]
    if len(ann_ids) != len(set(ann_ids)):
        print("  ❌ ERROR: Duplicate annotation IDs found!")
        return False
    
    # Check all annotation image_ids reference valid images
    valid_image_ids = set(image_ids)
    for ann in merged['annotations']:
        if ann['image_id'] not in valid_image_ids:
            print(f"  ❌ ERROR: Annotation {ann['id']} references invalid image_id {ann['image_id']}")
            return False
    
    # Check each image has at least one annotation
    images_with_anns = set(ann['image_id'] for ann in merged['annotations'])
    images_without_anns = valid_image_ids - images_with_anns
    if images_without_anns:
        print(f"  ⚠️  WARNING: {len(images_without_anns)} images have no annotations")
    
    print("  ✓ All image IDs unique")
    print("  ✓ All annotation IDs unique")
    print("  ✓ All annotation image_ids valid")
    
    return True


def main():
    args = parse_args()
    
    print("="*70)
    print("🐴 Merge Old Test/Val into New Training Set")
    print("="*70)
    print()
    
    # Load datasets
    old_test = load_coco(args.old_test)
    old_val = load_coco(args.old_val)
    new_train = load_coco(args.new_train)
    print()
    
    # Merge datasets
    merged = merge_datasets(new_train, old_test, old_val, args.check_duplicates)
    
    # Validate
    if not validate_merged_dataset(merged):
        print("\n❌ Validation failed! Not saving output.")
        sys.exit(1)
    
    # Save merged dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving merged dataset to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print()
    print(f"{'='*70}")
    print("✅ Merge Complete!")
    print(f"{'='*70}")
    print()
    print("Next steps:")
    print(f"  1. Update your config to use the merged training file:")
    print(f"     ann_file = '{args.output}'")
    print()
    print(f"  2. Make sure all images are accessible:")
    print(f"     - Old test/val images should be in data/test/ or data/val/")
    print(f"     - New images should be in data/images/")
    print(f"     - Or configure data_root in your config appropriately")
    print()
    print(f"  3. Start training:")
    print(f"     python tools/train.py configs/rtmpose_m_ap10k.py")
    print()
    print(f"  💡 Tip: You now have {len(merged['images'])} training images!")
    print(f"     This should significantly improve model performance.")


if __name__ == '__main__':
    main()
