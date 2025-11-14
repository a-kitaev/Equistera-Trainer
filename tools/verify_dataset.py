#!/usr/bin/env python
"""
Verify dataset annotations for common issues

Usage:
    python tools/verify_dataset.py --ann-file data/annotations/train.json
"""

import argparse
import json
from collections import defaultdict

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Verify COCO format dataset')
    parser.add_argument('--ann-file', required=True, help='annotation file')
    return parser.parse_args()


def verify_dataset(ann_file):
    """Verify dataset for common issues."""
    
    print(f'Loading annotations from {ann_file}...')
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    print('\n' + '='*70)
    print('DATASET VERIFICATION REPORT')
    print('='*70)
    
    # Basic statistics
    print('\n1. BASIC STATISTICS')
    print('-' * 70)
    print(f"Total images: {len(data['images'])}")
    print(f"Total annotations: {len(data['annotations'])}")
    print(f"Categories: {len(data['categories'])}")
    
    # Category info
    print('\n2. CATEGORY INFORMATION')
    print('-' * 70)
    for cat in data['categories']:
        print(f"Category ID: {cat['id']}, Name: {cat['name']}")
        print(f"Number of keypoints: {len(cat['keypoints'])}")
        print(f"Keypoints: {', '.join(cat['keypoints'][:5])}...")
        print(f"Skeleton connections: {len(cat['skeleton'])}")
    
    # Image statistics
    print('\n3. IMAGE STATISTICS')
    print('-' * 70)
    widths = [img['width'] for img in data['images']]
    heights = [img['height'] for img in data['images']]
    print(f"Width range: {min(widths)} - {max(widths)} (avg: {np.mean(widths):.1f})")
    print(f"Height range: {min(heights)} - {max(heights)} (avg: {np.mean(heights):.1f})")
    
    # Annotation statistics
    print('\n4. ANNOTATION STATISTICS')
    print('-' * 70)
    
    num_keypoints_list = []
    bbox_areas = []
    keypoint_visibility = defaultdict(int)
    
    for ann in data['annotations']:
        num_keypoints_list.append(ann['num_keypoints'])
        bbox_areas.append(ann['area'])
        
        # Count visibility
        keypoints = ann['keypoints']
        for i in range(2, len(keypoints), 3):
            visibility = keypoints[i]
            keypoint_visibility[visibility] += 1
    
    print(f"Keypoints per annotation - Min: {min(num_keypoints_list)}, "
          f"Max: {max(num_keypoints_list)}, Avg: {np.mean(num_keypoints_list):.1f}")
    print(f"Bbox area - Min: {min(bbox_areas):.0f}, "
          f"Max: {max(bbox_areas):.0f}, Avg: {np.mean(bbox_areas):.0f}")
    
    print('\nKeypoint visibility distribution:')
    total_kpts = sum(keypoint_visibility.values())
    for vis, count in sorted(keypoint_visibility.items()):
        pct = count / total_kpts * 100
        vis_label = ['Not labeled', 'Occluded', 'Visible'][int(vis)]
        print(f"  {vis_label} ({vis}): {count} ({pct:.1f}%)")
    
    # Per-keypoint statistics
    print('\n5. PER-KEYPOINT VISIBILITY')
    print('-' * 70)
    
    num_keypoints = len(data['categories'][0]['keypoints'])
    keypoint_names = data['categories'][0]['keypoints']
    keypoint_stats = defaultdict(lambda: {'visible': 0, 'occluded': 0, 'missing': 0})
    
    for ann in data['annotations']:
        keypoints = ann['keypoints']
        for i in range(num_keypoints):
            vis = keypoints[i * 3 + 2]
            name = keypoint_names[i]
            
            if vis == 0:
                keypoint_stats[name]['missing'] += 1
            elif vis == 1:
                keypoint_stats[name]['occluded'] += 1
            elif vis == 2:
                keypoint_stats[name]['visible'] += 1
    
    print(f"{'Keypoint':<15} {'Visible':<10} {'Occluded':<10} {'Missing':<10} {'Visible %':<10}")
    print('-' * 70)
    
    for name in keypoint_names:
        stats = keypoint_stats[name]
        total = stats['visible'] + stats['occluded'] + stats['missing']
        vis_pct = stats['visible'] / total * 100 if total > 0 else 0
        print(f"{name:<15} {stats['visible']:<10} {stats['occluded']:<10} "
              f"{stats['missing']:<10} {vis_pct:<10.1f}")
    
    # Data quality checks
    print('\n6. DATA QUALITY CHECKS')
    print('-' * 70)
    
    issues = []
    
    # Check for images without annotations
    image_ids = {img['id'] for img in data['images']}
    annotated_image_ids = {ann['image_id'] for ann in data['annotations']}
    images_without_ann = image_ids - annotated_image_ids
    
    if images_without_ann:
        issues.append(f"⚠️  {len(images_without_ann)} images without annotations")
    
    # Check for very small bboxes
    small_bbox_count = sum(1 for area in bbox_areas if area < 1000)
    if small_bbox_count > 0:
        issues.append(f"⚠️  {small_bbox_count} annotations with very small bboxes (area < 1000)")
    
    # Check for annotations with too few keypoints
    low_keypoint_count = sum(1 for n in num_keypoints_list if n < 10)
    if low_keypoint_count > 0:
        issues.append(f"⚠️  {low_keypoint_count} annotations with < 10 visible keypoints")
    
    # Check for duplicate image IDs
    img_ids = [img['id'] for img in data['images']]
    if len(img_ids) != len(set(img_ids)):
        issues.append(f"❌ Duplicate image IDs found!")
    
    # Check for duplicate annotation IDs
    ann_ids = [ann['id'] for ann in data['annotations']]
    if len(ann_ids) != len(set(ann_ids)):
        issues.append(f"❌ Duplicate annotation IDs found!")
    
    if issues:
        print('\n'.join(issues))
    else:
        print('✓ No major issues found!')
    
    print('\n' + '='*70)
    print('VERIFICATION COMPLETE')
    print('='*70 + '\n')


def main():
    args = parse_args()
    verify_dataset(args.ann_file)


if __name__ == '__main__':
    main()
