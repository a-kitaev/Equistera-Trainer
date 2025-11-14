#!/bin/bash
# Generate visualization predictions for all 3 models on test images

ssh -i MLCompute_key.pem azureuser@52.159.248.236 << 'EOF'
cd ~/equistera-trainer
source venv/bin/activate

echo "=========================================="
echo "GENERATING VISUALIZATIONS FOR ALL 3 MODELS"
echo "Test Dataset: 10 randomly selected images"
echo "=========================================="
echo ""

# Create output directories
mkdir -p visualizations/rtmpose_m
mkdir -p visualizations/hrnet_ap10k
mkdir -p visualizations/hrnet_animalpose

# Create visualization script
cat > tools/visualize_models.py << 'PYEOF'
import json
import random
import os
import numpy as np
from pathlib import Path

# Load test annotations
with open('data/annotations/horse_test.json', 'r') as f:
    data = json.load(f)

# Get 10 random images
image_ids = list(set([ann['image_id'] for ann in data['annotations']]))
random.seed(42)
selected_ids = random.sample(image_ids, min(10, len(image_ids)))

# Get full image info
selected_images = [img for img in data['images'] if img['id'] in selected_ids]

print(f"Selected {len(selected_images)} images:")
for img in selected_images:
    print(f"  - {img['file_name']}")
    
# Create subset annotation file for visualization
subset_data = {
    'images': selected_images,
    'annotations': [ann for ann in data['annotations'] if ann['image_id'] in selected_ids],
    'categories': data['categories']
}

with open('data/annotations/horse_test_viz.json', 'w') as f:
    json.dump(subset_data, f)

print(f"\nCreated visualization subset: {len(subset_data['images'])} images")
PYEOF

python tools/visualize_models.py

echo ""
echo "1/3 Generating RTMPose-M visualizations..."
echo "----------------------------------------"
python tools/test.py \
    configs/rtmpose_m_test_final.py \
    work_dirs/rtmpose_m_669imgs/best_coco_AP_epoch_210.pth \
    --cfg-options test_dataloader.dataset.ann_file=annotations/horse_test_viz.json \
    --show-dir visualizations/rtmpose_m \
    2>&1 | tail -5

echo ""
echo "2/3 Generating HRNet-AP10K visualizations..."
echo "----------------------------------------"
python tools/test.py \
    configs/hrnet_ap10k_test_final.py \
    work_dirs/hrnet_ap10k/best_coco_AP_epoch_280.pth \
    --cfg-options test_dataloader.dataset.ann_file=annotations/horse_test_viz.json \
    --show-dir visualizations/hrnet_ap10k \
    2>&1 | tail -5

echo ""
echo "3/3 Generating HRNet-AnimalPose visualizations..."
echo "----------------------------------------"
python tools/test.py \
    configs/hrnet_animalpose_test_final.py \
    work_dirs/hrnet_ap/best_coco_AP_epoch_270.pth \
    --cfg-options test_dataloader.dataset.ann_file=annotations/horse_test_viz.json \
    --show-dir visualizations/hrnet_animalpose \
    2>&1 | tail -5

echo ""
echo "=========================================="
echo "VISUALIZATIONS COMPLETE!"
echo "=========================================="
echo ""
ls -lh visualizations/*/

EOF
