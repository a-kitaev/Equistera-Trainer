#!/usr/bin/env python3
"""
Renumber image IDs and annotation IDs in COCO format to be sequential.

This fixes issues when testing a model on a dataset with non-sequential IDs.

Usage:
    python tools/renumber_coco_ids.py \
        --input data/annotations/horse_test.json \
        --output data/annotations/horse_test_renumbered.json
"""

import argparse
import json
from pathlib import Path


def renumber_coco_ids(input_file, output_file):
    """Renumber image and annotation IDs to be sequential starting from 1."""
    
    print(f"Loading: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"  Original images: {len(data['images'])}")
    print(f"  Original annotations: {len(data['annotations'])}")
    
    # Create ID mapping
    old_to_new_image_id = {}
    
    # Renumber images
    for new_id, img in enumerate(data['images'], start=1):
        old_id = img['id']
        old_to_new_image_id[old_id] = new_id
        img['id'] = new_id
    
    # Renumber annotations and update image_id references
    for new_id, ann in enumerate(data['annotations'], start=1):
        old_image_id = ann['image_id']
        ann['id'] = new_id
        ann['image_id'] = old_to_new_image_id[old_image_id]
    
    # Save renumbered dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved to: {output_file}")
    print(f"  New image IDs: 1 to {len(data['images'])}")
    print(f"  New annotation IDs: 1 to {len(data['annotations'])}")
    print("\n✅ Done! Use this file for testing.")


def main():
    parser = argparse.ArgumentParser(description='Renumber COCO IDs to be sequential')
    parser.add_argument('--input', required=True, help='Input COCO JSON file')
    parser.add_argument('--output', required=True, help='Output COCO JSON file')
    args = parser.parse_args()
    
    renumber_coco_ids(args.input, args.output)


if __name__ == '__main__':
    main()
