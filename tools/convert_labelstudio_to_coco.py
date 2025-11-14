#!/usr/bin/env python3
"""
Convert Label Studio JSON exports to COCO format and combine multiple files.

This script:
1. Reads multiple Label Studio export JSON files
2. Extracts keypoint annotations
3. Combines all annotations into one COCO format file
4. Handles duplicate images across files

Usage:
    python tools/convert_labelstudio_to_coco.py \
        --input data/*.json \
        --output data/annotations/horse_all.json \
        --images data/images
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
from tqdm import tqdm


# Define horse keypoint schema (26 keypoints)
KEYPOINT_NAMES = [
    'nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear',
    'neck', 'withers', 'back', 'tail_base', 'tail_end',
    'l_elbow', 'r_elbow', 'l_knee', 'r_knee',
    'lf_fetlock', 'rf_fetlock', 'lf_hoof', 'rf_hoof',
    'l_hip', 'r_hip', 'l_hock', 'r_hock',
    'lh_fetlock', 'rh_fetlock', 'lh_hoof', 'rh_hoof'
]

# Define skeleton connections
SKELETON = [
    [0, 1], [0, 2],  # nose to eyes
    [1, 3], [2, 4],  # eyes to ears
    [0, 5],  # nose to neck
    [5, 6],  # neck to withers
    [6, 7],  # withers to back
    [7, 8],  # back to tail_base
    [8, 9],  # tail_base to tail_end
    [6, 10], [6, 11],  # withers to elbows
    [10, 12], [11, 13],  # elbows to knees
    [12, 14], [13, 15],  # knees to front fetlocks
    [14, 16], [15, 17],  # front fetlocks to hooves
    [7, 18], [7, 19],  # back to hips
    [18, 20], [19, 21],  # hips to hocks
    [20, 22], [21, 23],  # hocks to hind fetlocks
    [22, 24], [23, 25],  # hind fetlocks to hooves
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Label Studio exports to COCO format')
    parser.add_argument(
        '--input',
        nargs='+',
        required=True,
        help='Input Label Studio JSON files (can specify multiple)')
    parser.add_argument(
        '--output',
        required=True,
        help='Output COCO JSON file')
    parser.add_argument(
        '--images',
        required=True,
        help='Directory containing images')
    parser.add_argument(
        '--min-keypoints',
        type=int,
        default=5,
        help='Minimum number of visible keypoints required (default: 5)')
    return parser.parse_args()


def load_labelstudio_files(file_paths):
    """Load and combine multiple Label Studio export files."""
    all_tasks = []
    seen_task_ids = set()
    
    print(f"\nLoading {len(file_paths)} Label Studio files...")
    for file_path in file_paths:
        print(f"  Loading: {file_path}")
        with open(file_path, 'r') as f:
            tasks = json.load(f)
        
        # Deduplicate by task ID
        for task in tasks:
            task_id = task.get('id')
            if task_id not in seen_task_ids:
                all_tasks.append(task)
                seen_task_ids.add(task_id)
            else:
                print(f"    ⚠️  Skipping duplicate task ID: {task_id}")
        
        print(f"    ✓ Loaded {len(tasks)} tasks ({len(seen_task_ids)} unique)")
    
    print(f"\n✓ Total unique tasks: {len(all_tasks)}")
    return all_tasks


def parse_labelstudio_keypoints(task, image_width, image_height):
    """
    Parse keypoints from Label Studio annotation format.
    
    Label Studio format has keypoints as:
    {
        "value": {
            "x": <percentage>,
            "y": <percentage>,
            "keypointlabels": ["keypoint_name"]
        }
    }
    """
    if 'annotations' not in task or len(task['annotations']) == 0:
        return None
    
    # Get the first completed annotation
    annotation = task['annotations'][0]
    if 'result' not in annotation:
        return None
    
    # Parse keypoints
    keypoints_dict = {}
    for item in annotation['result']:
        if item.get('type') == 'keypointlabels':
            value = item['value']
            kpt_labels = value.get('keypointlabels', [])
            if len(kpt_labels) > 0:
                kpt_name = kpt_labels[0]
                
                # Convert percentage to pixels
                x = value['x'] / 100.0 * image_width
                y = value['y'] / 100.0 * image_height
                
                keypoints_dict[kpt_name] = [x, y, 2]  # visibility=2 (visible)
    
    # Create keypoints array in correct order
    keypoints = []
    for kpt_name in KEYPOINT_NAMES:
        if kpt_name in keypoints_dict:
            keypoints.extend(keypoints_dict[kpt_name])
        else:
            keypoints.extend([0, 0, 0])  # Not annotated
    
    return keypoints


def calculate_bbox(keypoints):
    """Calculate bounding box from keypoints."""
    # Reshape to (26, 3)
    kpts = np.array(keypoints).reshape(-1, 3)
    
    # Get visible keypoints
    visible = kpts[kpts[:, 2] > 0]
    
    if len(visible) == 0:
        return [0, 0, 0, 0], 0
    
    x_coords = visible[:, 0]
    y_coords = visible[:, 1]
    
    x_min = x_coords.min()
    y_min = y_coords.min()
    x_max = x_coords.max()
    y_max = y_coords.max()
    
    # Add padding
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    width = x_max - x_min + 2 * padding
    height = y_max - y_min + 2 * padding
    
    bbox = [x_min, y_min, width, height]
    area = width * height
    
    return bbox, area


def get_image_info(image_path):
    """Get image dimensions."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None
    return img.shape[1], img.shape[0]  # width, height


def convert_to_coco(tasks, image_dir, min_keypoints=5):
    """Convert Label Studio tasks to COCO format."""
    
    image_dir = Path(image_dir)
    
    # Initialize COCO structure
    coco_data = {
        'info': {
            'description': 'Horse Pose Dataset - 26 Keypoints',
            'version': '2.0',
            'year': datetime.now().year,
            'contributor': 'Equistera Trainer',
            'date_created': datetime.now().strftime('%Y-%m-%d')
        },
        'licenses': [{
            'id': 1,
            'name': 'Custom License',
            'url': ''
        }],
        'categories': [{
            'id': 1,
            'name': 'horse',
            'supercategory': 'animal',
            'keypoints': KEYPOINT_NAMES,
            'skeleton': SKELETON
        }],
        'images': [],
        'annotations': []
    }
    
    image_id = 1
    ann_id = 1
    skipped_tasks = []
    
    print(f"\nConverting {len(tasks)} tasks to COCO format...")
    print(f"Minimum keypoints required: {min_keypoints}")
    
    for task in tqdm(tasks):
        # Get image filename
        if 'data' not in task or 'image' not in task['data']:
            skipped_tasks.append((task.get('id'), 'No image data'))
            continue
        
        image_url = task['data']['image']
        # Extract filename from URL (e.g., /data/upload/1/image.jpg -> image.jpg)
        image_filename = Path(image_url).name
        
        # Find image file
        image_path = image_dir / image_filename
        if not image_path.exists():
            skipped_tasks.append((task.get('id'), f'Image not found: {image_filename}'))
            continue
        
        # Get image dimensions
        width, height = get_image_info(image_path)
        if width is None:
            skipped_tasks.append((task.get('id'), f'Failed to read image: {image_filename}'))
            continue
        
        # Parse keypoints
        keypoints = parse_labelstudio_keypoints(task, width, height)
        if keypoints is None:
            skipped_tasks.append((task.get('id'), 'No annotations'))
            continue
        
        # Count visible keypoints
        num_keypoints = sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0)
        
        if num_keypoints < min_keypoints:
            skipped_tasks.append((task.get('id'), f'Too few keypoints: {num_keypoints}'))
            continue
        
        # Calculate bbox
        bbox, area = calculate_bbox(keypoints)
        
        # Add image entry
        coco_data['images'].append({
            'id': image_id,
            'file_name': image_filename,
            'width': width,
            'height': height,
            'license': 1
        })
        
        # Add annotation entry
        coco_data['annotations'].append({
            'id': ann_id,
            'image_id': image_id,
            'category_id': 1,
            'keypoints': keypoints,
            'num_keypoints': num_keypoints,
            'bbox': bbox,
            'area': area,
            'iscrowd': 0
        })
        
        image_id += 1
        ann_id += 1
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Conversion Statistics")
    print(f"{'='*60}")
    print(f"✓ Total images converted: {len(coco_data['images'])}")
    print(f"✓ Total annotations: {len(coco_data['annotations'])}")
    print(f"⚠️  Skipped tasks: {len(skipped_tasks)}")
    
    if len(skipped_tasks) > 0:
        print(f"\nSkipped task details:")
        for task_id, reason in skipped_tasks[:10]:  # Show first 10
            print(f"  Task {task_id}: {reason}")
        if len(skipped_tasks) > 10:
            print(f"  ... and {len(skipped_tasks) - 10} more")
    
    return coco_data


def main():
    args = parse_args()
    
    print("="*60)
    print("🐴 Label Studio to COCO Converter")
    print("="*60)
    
    # Load all Label Studio files
    tasks = load_labelstudio_files(args.input)
    
    # Convert to COCO format
    coco_data = convert_to_coco(tasks, args.images, args.min_keypoints)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save COCO format file
    print(f"\nSaving to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ Conversion Complete!")
    print(f"{'='*60}")
    print(f"Output file: {args.output}")
    print(f"Images: {len(coco_data['images'])}")
    print(f"Annotations: {len(coco_data['annotations'])}")
    print(f"Keypoints per annotation: 26")
    
    # Calculate average keypoints per annotation
    if len(coco_data['annotations']) > 0:
        avg_keypoints = sum(ann['num_keypoints'] for ann in coco_data['annotations']) / len(coco_data['annotations'])
        print(f"Average visible keypoints: {avg_keypoints:.1f}")


if __name__ == '__main__':
    main()
