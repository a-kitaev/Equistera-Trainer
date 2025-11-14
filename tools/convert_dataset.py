#!/usr/bin/env python
"""
Convert custom dataset to COCO format for MMPose

This script converts your custom horse pose dataset to COCO format.
Modify the conversion logic based on your dataset structure.

Usage:
    python tools/convert_dataset.py --input data/raw --output data/annotations
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert dataset to COCO format')
    parser.add_argument(
        '--input', 
        required=True,
        help='input directory with raw dataset')
    parser.add_argument(
        '--output',
        required=True,
        help='output directory for COCO annotations')
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='ratio of training data')
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='ratio of validation data')
    return parser.parse_args()


def load_keypoint_schema(schema_path='horse_keypoint_schema.json'):
    """Load keypoint schema."""
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    return schema


def create_coco_format(schema):
    """Create base COCO format structure."""
    
    coco_format = {
        'info': {
            'description': 'Horse Pose Dataset - 26 Keypoints',
            'version': '2.0',
            'year': datetime.now().year,
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
            'keypoints': [kpt['name'] for kpt in schema['keypoints']],
            'skeleton': schema['skeleton']
        }],
        'images': [],
        'annotations': []
    }
    
    return coco_format


def convert_annotation(annotation_data, image_id, ann_id):
    """
    Convert your annotation format to COCO format.
    
    Modify this function based on your annotation format.
    
    Expected annotation_data format (example):
    {
        'keypoints': [[x1, y1, v1], [x2, y2, v2], ...],  # 26 keypoints
        'bbox': [x, y, w, h],  # optional
        'area': float,  # optional
    }
    """
    
    keypoints = []
    for kpt in annotation_data['keypoints']:
        keypoints.extend([kpt[0], kpt[1], kpt[2]])  # x, y, visibility
    
    # Calculate bbox if not provided
    if 'bbox' not in annotation_data:
        kpts_array = np.array(annotation_data['keypoints'])
        visible_kpts = kpts_array[kpts_array[:, 2] > 0]
        
        if len(visible_kpts) > 0:
            x_min = visible_kpts[:, 0].min()
            y_min = visible_kpts[:, 1].min()
            x_max = visible_kpts[:, 0].max()
            y_max = visible_kpts[:, 1].max()
            
            # Add padding
            padding = 20
            bbox = [
                max(0, x_min - padding),
                max(0, y_min - padding),
                x_max - x_min + 2 * padding,
                y_max - y_min + 2 * padding
            ]
        else:
            bbox = [0, 0, 0, 0]
    else:
        bbox = annotation_data['bbox']
    
    # Calculate area
    area = bbox[2] * bbox[3] if 'area' not in annotation_data else annotation_data['area']
    
    coco_annotation = {
        'id': ann_id,
        'image_id': image_id,
        'category_id': 1,
        'keypoints': keypoints,
        'num_keypoints': sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0),
        'bbox': bbox,
        'area': area,
        'iscrowd': 0
    }
    
    return coco_annotation


def convert_dataset(args):
    """Main conversion function."""
    
    # Load schema
    schema = load_keypoint_schema()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # TODO: Modify this section based on your dataset structure
    # Example assumes:
    # - input/images/ contains image files
    # - input/annotations.json contains all annotations
    
    input_path = Path(args.input)
    
    # Load your annotations (modify based on your format)
    # This is a placeholder - replace with your actual loading logic
    print("Loading annotations...")
    # annotations_file = input_path / 'annotations.json'
    # with open(annotations_file, 'r') as f:
    #     raw_annotations = json.load(f)
    
    # For demonstration, create sample structure
    print("⚠️  WARNING: This is a template conversion script.")
    print("Please modify the convert_dataset() function based on your dataset format.")
    print("\nCreating sample COCO format files...")
    
    # Create train and val splits
    train_coco = create_coco_format(schema)
    val_coco = create_coco_format(schema)
    
    # Example: Add sample image and annotation
    # Replace this with your actual conversion logic
    sample_image = {
        'id': 1,
        'file_name': 'sample_image.jpg',
        'height': 480,
        'width': 640,
        'license': 1,
    }
    train_coco['images'].append(sample_image)
    
    # TODO: Iterate through your images and annotations
    # for img_idx, (image_path, annotations) in enumerate(your_dataset):
    #     # Add to train or val based on split ratio
    #     ...
    
    # Save annotations
    train_file = os.path.join(args.output, 'train.json')
    val_file = os.path.join(args.output, 'val.json')
    
    with open(train_file, 'w') as f:
        json.dump(train_coco, f, indent=2)
    print(f"Saved train annotations to {train_file}")
    
    with open(val_file, 'w') as f:
        json.dump(val_coco, f, indent=2)
    print(f"Saved val annotations to {val_file}")
    
    print("\n" + "="*50)
    print("Conversion template created!")
    print("="*50)
    print(f"Train images: {len(train_coco['images'])}")
    print(f"Train annotations: {len(train_coco['annotations'])}")
    print(f"Val images: {len(val_coco['images'])}")
    print(f"Val annotations: {len(val_coco['annotations'])}")
    print("\n⚠️  Please modify convert_dataset() function to match your data format!")


def main():
    args = parse_args()
    convert_dataset(args)


if __name__ == '__main__':
    main()
