#!/usr/bin/env python3
"""
Local visualization script - generates predictions for all 3 models
Runs on your local machine with downloaded checkpoints
"""

import sys
import os
from pathlib import Path

# Check if we're running locally
print("Running LOCAL visualization script")
print(f"Working directory: {os.getcwd()}")
print()

# Import required libraries
try:
    from mmpose.apis import init_model, inference_topdown
    from mmpose.structures import merge_data_samples
    import cv2
    import json
    import random
    import numpy as np
    from mmengine.visualization import Visualizer
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("\nPlease install MMPose dependencies:")
    print("  conda activate equistera-inference")
    print("  pip install mmpose mmcv mmengine opencv-python")
    sys.exit(1)

# Configuration
MODELS = [
    {
        'name': 'RTMPose-M-Baseline',
        'config': 'configs/rtmpose_m_ap10k.py',
        'checkpoint': 'checkpoints/best_coco_AP_epoch_210.pth',
        'output_dir': 'visualizations_local/rtmpose_m_baseline'
    },
    {
        'name': 'RTMPose-M-V2-TextFusion',
        'config': 'configs/rtmpose_m_ap10k.py',  # V2 uses same config for inference (neck is bypassed)
        'checkpoint': 'checkpoints/v2_best.pth/best_coco_AP_epoch_145.pth',  # Download this from VM when ready
        'output_dir': 'visualizations_local/rtmpose_m_v2'
    },
    {
        'name': 'HRNet-AP10K',
        'config': 'configs/hrnet_w32_ap10k.py',
        'checkpoint': 'checkpoints/hrnet_ap10k_best_epoch_280.pth',
        'output_dir': 'visualizations_local/hrnet_ap10k'
    },
]

def select_test_images(num_images=5):
    """Select random test images"""
    test_json = 'data/annotations/horse_test.json'
    
    if not os.path.exists(test_json):
        print(f"❌ Test annotations not found: {test_json}")
        print("\nPlease download test data from VM:")
        print("  scp -r -i MLCompute_key.pem azureuser@52.159.248.236:~/equistera-trainer/data .")
        sys.exit(1)
    
    with open(test_json) as f:
        data = json.load(f)
    
    random.seed(42)
    selected = random.sample(data['images'], min(num_images, len(data['images'])))
    
    print(f"Selected {len(selected)} test images:")
    for img in selected:
        print(f"  - {img['file_name']}")
    print()
    
    return selected

def visualize_predictions(model, img_path, output_path):
    """Run inference and save visualization"""
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        print(f"    ⚠️  Could not read image: {img_path}")
        return False
    
    # Run inference
    results = inference_topdown(model, img)
    
    # Draw predictions
    img_vis = img.copy()
    
    # Extract keypoints from results
    if len(results) > 0:
        pred_instances = results[0].pred_instances
        keypoints = pred_instances.keypoints[0]  # First instance
        scores = pred_instances.keypoint_scores[0]
        
        # Draw keypoints
        for i, (kpt, score) in enumerate(zip(keypoints, scores)):
            if score > 0.3:  # Confidence threshold
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(img_vis, (x, y), 3, (0, 255, 0), -1)
                cv2.putText(img_vis, f'{i}', (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw skeleton connections
        skeleton = model.dataset_meta.get('skeleton_info', {})
        for link_id, link_info in skeleton.items():
            link = link_info.get('link', [])
            if len(link) == 2:
                # Find keypoint indices
                kpt_names = [kpt['name'] for kpt in model.dataset_meta['keypoint_info'].values()]
                try:
                    idx1 = kpt_names.index(link[0])
                    idx2 = kpt_names.index(link[1])
                    
                    if scores[idx1] > 0.3 and scores[idx2] > 0.3:
                        pt1 = tuple(map(int, keypoints[idx1]))
                        pt2 = tuple(map(int, keypoints[idx2]))
                        cv2.line(img_vis, pt1, pt2, (0, 255, 255), 2)
                except (ValueError, IndexError):
                    pass
    
    # Save visualization
    cv2.imwrite(output_path, img_vis)
    return True

def main():
    print("="*60)
    print("LOCAL MODEL VISUALIZATION")
    print("="*60)
    print()
    
    # Select test images
    test_images = select_test_images(5)
    
    # Check all models exist
    print("Checking models...")
    for model_info in MODELS:
        if not os.path.exists(model_info['checkpoint']):
            print(f"❌ Checkpoint not found: {model_info['checkpoint']}")
            print(f"\nPlease download from VM:")
            print(f"  Already downloaded: {os.path.basename(model_info['checkpoint'])}")
            sys.exit(1)
        print(f"  ✓ {model_info['name']}: {model_info['checkpoint']}")
    print()
    
    # Process each model
    for model_info in MODELS:
        print("="*60)
        print(f"Processing {model_info['name']}")
        print("="*60)
        print()
        
        # Create output directory
        os.makedirs(model_info['output_dir'], exist_ok=True)
        
        # Initialize model
        print(f"Loading model...")
        try:
            model = init_model(
                model_info['config'],
                model_info['checkpoint'],
                device='cpu'  # Use CPU for Mac
            )
            print(f"  ✓ Model loaded")
        except Exception as e:
            print(f"  ❌ Failed to load model: {e}")
            continue
        
        # Process each image
        print(f"\nGenerating visualizations...")
        for img_info in test_images:
            img_path = os.path.join('data/test', img_info['file_name'])
            
            if not os.path.exists(img_path):
                print(f"  ⚠️  Image not found: {img_path}")
                continue
            
            output_path = os.path.join(model_info['output_dir'], img_info['file_name'])
            
            print(f"  Processing: {img_info['file_name']}", end='')
            success = visualize_predictions(model, img_path, output_path)
            
            if success:
                print(f" ✓")
            else:
                print(f" ✗")
        
        print(f"\n{model_info['name']} complete!")
        print()
    
    print("="*60)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*60)
    print()
    print("Results saved to:")
    for model_info in MODELS:
        print(f"  - {model_info['output_dir']}/")
    print()

if __name__ == '__main__':
    main()
