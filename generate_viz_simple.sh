#!/bin/bash
# Simple visualization script - generate predictions for sample test images

ssh -i MLCompute_key.pem azureuser@52.159.248.236 << 'EOF'
cd ~/equistera-trainer
source venv/bin/activate

echo "=========================================="
echo "GENERATING VISUALIZATIONS"
echo "=========================================="
echo ""

# Create visualization directory
mkdir -p visualizations

# Create a simple demo script
cat > demo_visualize.py << 'PYEOF'
from mmpose.apis import init_model, inference_topdown, visualize
from mmengine.config import Config
import cv2
import os
import json
import random
from pathlib import Path

# Load test images
with open('data/annotations/horse_test.json') as f:
    data = json.load(f)

# Select 5 random test images
random.seed(42)
sample_images = random.sample(data['images'], 5)

print(f"Selected {len(sample_images)} images for visualization\n")

# Test each model
models = [
    ('RTMPose-M', 'configs/rtmpose_m_test_final.py', 'work_dirs/rtmpose_m_669imgs/best_coco_AP_epoch_210.pth'),
    ('HRNet-AP10K', 'configs/hrnet_ap10k_test_final.py', 'work_dirs/hrnet_ap10k/best_coco_AP_epoch_280.pth'),
    ('HRNet-AnimalPose', 'configs/hrnet_animalpose_test_final.py', 'work_dirs/hrnet_ap/best_coco_AP_epoch_270.pth'),
]

for model_name, config, checkpoint in models:
    print(f"\n{'='*50}")
    print(f"Processing {model_name}")
    print(f"{'='*50}")
    
    # Initialize model
    print(f"Loading model: {checkpoint}")
    model = init_model(config, checkpoint, device='cuda:0')
    
    # Create output directory
    out_dir = f'visualizations/{model_name.lower().replace(" ", "_").replace("-", "_")}'
    os.makedirs(out_dir, exist_ok=True)
    
    # Process each image
    for img_info in sample_images:
        img_path = os.path.join('data/test', img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"  ⚠️  Image not found: {img_path}")
            continue
            
        print(f"  Processing: {img_info['file_name']}")
        
        # Read image
        img = cv2.imread(img_path)
        
        # Inference
        results = inference_topdown(model, img)
        
        # Visualize
        vis_img = visualize(img, results, model.cfg, show=False)
        
        # Save
        out_file = os.path.join(out_dir, img_info['file_name'])
        cv2.imwrite(out_file, vis_img)
        print(f"    ✓ Saved to: {out_file}")
    
    print(f"\n{model_name} complete!")

print(f"\n{'='*50}")
print("ALL VISUALIZATIONS COMPLETE!")
print(f"{'='*50}")
print("\nGenerated files:")
os.system('ls -lh visualizations/*/*.jpg visualizations/*/*.png visualizations/*/*.jpeg 2>/dev/null')
PYEOF

echo "Running visualization script..."
python demo_visualize.py

echo ""
echo "Listing generated files..."
ls -lh visualizations/*/

EOF
