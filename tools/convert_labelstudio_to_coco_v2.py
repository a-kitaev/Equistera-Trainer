#!/usr/bin/env python3
"""
Convert Label Studio JSON exports (v2) to COCO format with Google Cloud Storage support.

This script:
1. Reads Label Studio export JSON with keypoints format
2. Downloads images from Google Cloud Storage
3. Converts annotations to COCO format for MMPose
4. Handles duplicate images and creates train/val/test splits

Key differences from v1:
- Images stored in GCS (gs://bucket/path format)
- New keypoint format (direct list instead of nested annotations)
- Uses Google Cloud SDK for image download
- Automatic train/val/test split

Usage:
    python tools/convert_labelstudio_to_coco_v2.py \
        --input data/project-1-at-2025-11-15-17-13-6ae276bb.json \
        --output-dir data/annotations \
        --images-dir data/images \
        --gcs-key i-destiny.json \
        --train-ratio 0.7 \
        --val-ratio 0.15 \
        --test-ratio 0.15
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import hashlib
import random

import cv2
import numpy as np
from tqdm import tqdm
from google.cloud import storage
from google.oauth2 import service_account


# Define horse keypoint schema (26 keypoints)
KEYPOINT_NAMES = [
    'nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear',
    'neck', 'withers', 'back', 'tail_base', 'tail_end',
    'l_elbow', 'r_elbow', 'l_knee', 'r_knee',
    'lf_fetlock', 'rf_fetlock', 'lf_hoof', 'rf_hoof',
    'l_hip', 'r_hip', 'l_hock', 'r_hock',
    'lh_fetlock', 'rh_fetlock', 'lh_hoof', 'rh_hoof'
]

# Define skeleton connections (for visualization)
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
        description='Convert Label Studio exports (v2) to COCO format')
    parser.add_argument(
        '--input',
        required=True,
        help='Input Label Studio JSON file')
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for COCO annotations')
    parser.add_argument(
        '--images-dir',
        required=True,
        help='Directory to download/store images')
    parser.add_argument(
        '--gcs-key',
        required=True,
        help='Google Cloud Storage service account key JSON file')
    parser.add_argument(
        '--min-keypoints',
        type=int,
        default=5,
        help='Minimum number of visible keypoints required (default: 5)')
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Ratio of training data (default: 0.7)')
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Ratio of validation data (default: 0.15)')
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Ratio of test data (default: 0.15)')
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible splits (default: 42)')
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip image download (use if images already downloaded)')
    return parser.parse_args()


def init_gcs_client(key_path):
    """Initialize Google Cloud Storage client."""
    credentials = service_account.Credentials.from_service_account_file(key_path)
    client = storage.Client(credentials=credentials)
    return client


def parse_gcs_path(gcs_url):
    """
    Parse GCS URL to extract bucket and blob path.
    
    Formats:
    - gs://bucket/path/to/file.jpg
    - gs://bucket/folder/file.jpg
    """
    if not gcs_url.startswith('gs://'):
        return None, None
    
    # Remove gs:// prefix and split
    path = gcs_url[5:]
    parts = path.split('/', 1)
    
    if len(parts) != 2:
        return parts[0], ''
    
    bucket_name = parts[0]
    blob_path = parts[1]
    
    return bucket_name, blob_path


def download_image_from_gcs(gcs_client, gcs_url, output_path):
    """Download image from Google Cloud Storage."""
    bucket_name, blob_path = parse_gcs_path(gcs_url)
    
    if bucket_name is None:
        print(f"    ⚠️  Invalid GCS URL: {gcs_url}")
        return False
    
    try:
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(output_path)
        return True
    except Exception as e:
        print(f"    ⚠️  Failed to download {gcs_url}: {e}")
        return False


def parse_labelstudio_keypoints_v2(task):
    """
    Parse keypoints from Label Studio annotation format (v2).
    
    Format:
    {
        "keypoints": [
            {
                "x": <percentage>,
                "y": <percentage>,
                "keypointlabels": ["keypoint_name"],
                "original_width": <int>,
                "original_height": <int>
            },
            ...
        ]
    }
    """
    if 'keypoints' not in task:
        return None, None, None
    
    keypoints_data = task['keypoints']
    
    if len(keypoints_data) == 0:
        return None, None, None
    
    # Get image dimensions from first keypoint
    original_width = keypoints_data[0].get('original_width')
    original_height = keypoints_data[0].get('original_height')
    
    if not original_width or not original_height:
        return None, None, None
    
    # Parse keypoints into dict
    keypoints_dict = {}
    for kpt in keypoints_data:
        kpt_labels = kpt.get('keypointlabels', [])
        if len(kpt_labels) > 0:
            kpt_name = kpt_labels[0]
            # Convert percentage to absolute coordinates
            x = (kpt['x'] / 100.0) * original_width
            y = (kpt['y'] / 100.0) * original_height
            # Visibility: 2 = visible
            keypoints_dict[kpt_name] = [x, y, 2]
    
    # Create keypoints array in correct order (26 keypoints × 3)
    keypoints = []
    for kpt_name in KEYPOINT_NAMES:
        if kpt_name in keypoints_dict:
            keypoints.extend(keypoints_dict[kpt_name])
        else:
            keypoints.extend([0, 0, 0])  # Not annotated
    
    return keypoints, original_width, original_height


def calculate_bbox(keypoints, width, height):
    """Calculate bounding box from keypoints."""
    # Reshape to (26, 3)
    kpts = np.array(keypoints).reshape(-1, 3)
    
    # Get visible keypoints
    visible = kpts[kpts[:, 2] > 0]
    
    if len(visible) == 0:
        return [0, 0, width, height], width * height
    
    x_coords = visible[:, 0]
    y_coords = visible[:, 1]
    
    x_min = max(0, x_coords.min())
    y_min = max(0, y_coords.min())
    x_max = min(width, x_coords.max())
    y_max = min(height, y_coords.max())
    
    # Add padding (10% of bbox size, max 50px)
    w = x_max - x_min
    h = y_max - y_min
    padding = min(max(w, h) * 0.1, 50)
    
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)
    
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    
    bbox = [x_min, y_min, bbox_width, bbox_height]
    area = bbox_width * bbox_height
    
    return bbox, area


def create_coco_structure():
    """Create base COCO format structure."""
    return {
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


def split_dataset(annotations, train_ratio, val_ratio, test_ratio, seed):
    """Split annotations into train/val/test sets."""
    random.seed(seed)
    
    # Shuffle annotations
    indices = list(range(len(annotations)))
    random.shuffle(indices)
    
    # Calculate split points
    n = len(indices)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_indices = set(indices[:train_end])
    val_indices = set(indices[train_end:val_end])
    test_indices = set(indices[val_end:])
    
    return train_indices, val_indices, test_indices


def get_image_filename(gcs_url):
    """Extract filename from GCS URL."""
    # gs://bucket/path/to/file.jpg -> file.jpg
    return Path(gcs_url).name.strip()


def main():
    args = parse_args()
    
    print("="*70)
    print("🐴 Label Studio to COCO Converter (v2)")
    print("="*70)
    print()
    
    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"❌ Split ratios must sum to 1.0 (got {total_ratio})")
        sys.exit(1)
    
    # Create output directories
    output_dir = Path(args.output_dir)
    images_dir = Path(args.images_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize GCS client
    print("Initializing Google Cloud Storage client...")
    try:
        gcs_client = init_gcs_client(args.gcs_key)
        print("  ✓ GCS client initialized")
    except Exception as e:
        print(f"  ❌ Failed to initialize GCS client: {e}")
        sys.exit(1)
    print()
    
    # Load Label Studio export
    print(f"Loading Label Studio export: {args.input}")
    with open(args.input, 'r') as f:
        tasks = json.load(f)
    print(f"  ✓ Loaded {len(tasks)} tasks")
    print()
    
    # Process tasks and download images
    print("Processing tasks and downloading images...")
    print(f"Minimum keypoints required: {args.min_keypoints}")
    print()
    
    valid_tasks = []
    skipped_tasks = []
    downloaded_images = set()
    
    for task in tqdm(tasks, desc="Processing"):
        # Parse keypoints
        keypoints, width, height = parse_labelstudio_keypoints_v2(task)
        
        if keypoints is None:
            skipped_tasks.append((task.get('id'), 'No keypoints'))
            continue
        
        # Count visible keypoints
        num_keypoints = sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0)
        
        if num_keypoints < args.min_keypoints:
            skipped_tasks.append((task.get('id'), f'Only {num_keypoints} keypoints'))
            continue
        
        # Get image URL
        gcs_url = task.get('image', '')
        if not gcs_url:
            skipped_tasks.append((task.get('id'), 'No image URL'))
            continue
        
        # Get filename
        filename = get_image_filename(gcs_url)
        if not filename:
            skipped_tasks.append((task.get('id'), 'Invalid filename'))
            continue
        
        image_path = images_dir / filename
        
        # Download image if not exists
        if not args.skip_download:
            if filename not in downloaded_images:
                if not image_path.exists():
                    success = download_image_from_gcs(gcs_client, gcs_url, str(image_path))
                    if not success:
                        skipped_tasks.append((task.get('id'), f'Download failed: {filename}'))
                        continue
                downloaded_images.add(filename)
        else:
            # Check if image exists locally
            if not image_path.exists():
                skipped_tasks.append((task.get('id'), f'Image not found: {filename}'))
                continue
        
        # Verify image dimensions match
        img = cv2.imread(str(image_path))
        if img is None:
            skipped_tasks.append((task.get('id'), f'Cannot read image: {filename}'))
            continue
        
        actual_height, actual_width = img.shape[:2]
        
        # If dimensions don't match, scale keypoints
        if actual_width != width or actual_height != height:
            scale_x = actual_width / width
            scale_y = actual_height / height
            
            # Scale keypoints
            scaled_keypoints = []
            for i in range(0, len(keypoints), 3):
                x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
                if v > 0:
                    scaled_keypoints.extend([x * scale_x, y * scale_y, v])
                else:
                    scaled_keypoints.extend([0, 0, 0])
            keypoints = scaled_keypoints
            width, height = actual_width, actual_height
        
        # Calculate bbox
        bbox, area = calculate_bbox(keypoints, width, height)
        
        # Store valid task with processed data
        valid_tasks.append({
            'task_id': task.get('id'),
            'filename': filename,
            'width': width,
            'height': height,
            'keypoints': keypoints,
            'num_keypoints': num_keypoints,
            'bbox': bbox,
            'area': area
        })
    
    print()
    print(f"{'='*70}")
    print("Processing Statistics")
    print(f"{'='*70}")
    print(f"✓ Valid tasks: {len(valid_tasks)}")
    print(f"✓ Downloaded images: {len(downloaded_images)}")
    print(f"⚠️  Skipped tasks: {len(skipped_tasks)}")
    
    if len(skipped_tasks) > 0 and len(skipped_tasks) <= 10:
        print(f"\nSkipped task details:")
        for task_id, reason in skipped_tasks:
            print(f"  - Task {task_id}: {reason}")
    elif len(skipped_tasks) > 10:
        print(f"\nFirst 10 skipped tasks:")
        for task_id, reason in skipped_tasks[:10]:
            print(f"  - Task {task_id}: {reason}")
    print()
    
    if len(valid_tasks) == 0:
        print("❌ No valid tasks to convert!")
        sys.exit(1)
    
    # Split into train/val/test
    print("Splitting dataset...")
    train_indices, val_indices, test_indices = split_dataset(
        valid_tasks, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    
    print(f"  Train: {len(train_indices)} ({args.train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_indices)} ({args.val_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_indices)} ({args.test_ratio*100:.0f}%)")
    print()
    
    # Create COCO structures for each split
    train_coco = create_coco_structure()
    val_coco = create_coco_structure()
    test_coco = create_coco_structure()
    
    # Populate COCO structures
    print("Creating COCO format annotations...")
    
    image_id = 1
    train_ann_id = 1
    val_ann_id = 1
    test_ann_id = 1
    
    for idx, task_data in enumerate(tqdm(valid_tasks, desc="Converting")):
        # Determine split
        if idx in train_indices:
            coco_data = train_coco
            ann_id = train_ann_id
            train_ann_id += 1
        elif idx in val_indices:
            coco_data = val_coco
            ann_id = val_ann_id
            val_ann_id += 1
        else:
            coco_data = test_coco
            ann_id = test_ann_id
            test_ann_id += 1
        
        # Add image
        coco_data['images'].append({
            'id': image_id,
            'file_name': task_data['filename'],
            'width': task_data['width'],
            'height': task_data['height'],
            'license': 1
        })
        
        # Add annotation
        coco_data['annotations'].append({
            'id': ann_id,
            'image_id': image_id,
            'category_id': 1,
            'keypoints': task_data['keypoints'],
            'num_keypoints': task_data['num_keypoints'],
            'bbox': task_data['bbox'],
            'area': task_data['area'],
            'iscrowd': 0
        })
        
        image_id += 1
    
    # Save COCO format files
    print()
    print("Saving COCO format files...")
    
    splits = [
        ('train', train_coco),
        ('val', val_coco),
        ('test', test_coco)
    ]
    
    for split_name, coco_data in splits:
        output_file = output_dir / f'horse_{split_name}.json'
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        print(f"  ✓ {split_name}: {output_file}")
        print(f"      Images: {len(coco_data['images'])}")
        print(f"      Annotations: {len(coco_data['annotations'])}")
        if len(coco_data['annotations']) > 0:
            avg_kpts = sum(ann['num_keypoints'] for ann in coco_data['annotations']) / len(coco_data['annotations'])
            print(f"      Avg keypoints: {avg_kpts:.1f}")
    
    print()
    print(f"{'='*70}")
    print("✅ Conversion Complete!")
    print(f"{'='*70}")
    print()
    print("Next steps:")
    print(f"  1. Update your config to use these annotations:")
    print(f"     - Train: {output_dir}/horse_train.json")
    print(f"     - Val:   {output_dir}/horse_val.json")
    print(f"     - Test:  {output_dir}/horse_test.json")
    print()
    print(f"  2. Images are stored in: {images_dir}")
    print()
    print(f"  3. Start training:")
    print(f"     python tools/train.py configs/rtmpose_m_ap10k.py")


if __name__ == '__main__':
    main()
