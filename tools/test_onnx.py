#!/usr/bin/env python
"""
Test ONNX exported models on test datasets

This script loads an ONNX model and evaluates it on a COCO-format test dataset,
calculating metrics like AP, AR, and per-keypoint accuracy.

Usage:
    # Test ONNX model on test dataset
    python tools/test_onnx.py --onnx work_dirs/rtmpose_m_horse_opset17.onnx \
                               --ann data/annotations/horse_test.json \
                               --img-dir data/test

    # Test with visualization
    python tools/test_onnx.py --onnx work_dirs/rtmpose_m_horse_opset17.onnx \
                               --ann data/annotations/horse_test.json \
                               --img-dir data/test \
                               --show-dir visualizations/onnx_test

    # Test with custom input size
    python tools/test_onnx.py --onnx work_dirs/model.onnx \
                               --ann data/annotations/horse_test.json \
                               --img-dir data/test \
                               --input-size 256 256
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed. Please run: pip install onnxruntime")
    exit(1)

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    print("ERROR: pycocotools not installed. Please run: pip install pycocotools")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Test ONNX model on test dataset')
    parser.add_argument('--onnx', required=True, help='Path to ONNX model file')
    parser.add_argument('--ann', required=True, help='Path to COCO annotation file')
    parser.add_argument('--img-dir', required=True, help='Directory containing test images')
    parser.add_argument(
        '--input-size',
        type=int,
        nargs=2,
        default=[256, 256],
        help='Input size (height width), default: 256 256')
    parser.add_argument(
        '--show-dir',
        help='Directory to save visualizations')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Keypoint score threshold for visualization')
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run inference on')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for inference (default: 1)')
    
    return parser.parse_args()


def load_onnx_model(onnx_path, device='cpu'):
    """Load ONNX model and create inference session"""
    print(f"\n{'='*60}")
    print(f"  Loading ONNX Model")
    print(f"{'='*60}\n")
    
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"📦 Model: {onnx_path}")
    print(f"📏 Size: {file_size_mb:.2f} MB")
    
    # Set up providers based on device
    if device == 'cuda':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    
    print(f"🔧 Device: {device}")
    print(f"🔧 Providers: {providers}")
    
    # Create session
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Get input/output info
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_names = [output.name for output in session.get_outputs()]
    
    print(f"\n✅ Model loaded successfully")
    print(f"   Input: {input_name} {input_shape}")
    print(f"   Outputs: {output_names}")
    
    return session, input_name, output_names


def preprocess_image(img, input_size, bbox=None):
    """
    Preprocess image for ONNX model inference
    
    Args:
        img: Input image (BGR)
        input_size: Tuple of (height, width)
        bbox: Bounding box [x, y, w, h] in COCO format
    
    Returns:
        Preprocessed image tensor and transformation info
    """
    h, w = img.shape[:2]
    
    # If bbox is provided, crop the region
    if bbox is not None:
        x, y, bw, bh = bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + bw), int(y + bh)
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        img_crop = img[y1:y2, x1:x2]
        center = np.array([x + bw/2, y + bh/2])
        scale = np.array([bw, bh])
    else:
        # Use full image
        img_crop = img
        center = np.array([w/2, h/2])
        scale = np.array([w, h])
    
    # Resize to input size
    target_h, target_w = input_size
    img_resized = cv2.resize(img_crop, (target_w, target_h))
    
    # Convert to RGB and normalize
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # Standardize (ImageNet mean and std)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_normalized - mean) / std
    
    # Transpose to CHW format and add batch dimension
    img_tensor = img_normalized.transpose(2, 0, 1)
    img_tensor = np.expand_dims(img_tensor, 0)
    
    return img_tensor, center, scale


def decode_simcc_output(simcc_x, simcc_y, input_size, center, scale):
    """
    Decode SimCC outputs to keypoint coordinates
    
    Args:
        simcc_x: X-axis SimCC predictions [1, num_keypoints, simcc_split_ratio * input_w]
        simcc_y: Y-axis SimCC predictions [1, num_keypoints, simcc_split_ratio * input_h]
        input_size: Tuple of (height, width)
        center: Image center [x, y]
        scale: Image scale [w, h]
    
    Returns:
        keypoints: [num_keypoints, 2] in original image coordinates
        scores: [num_keypoints] confidence scores
    """
    # Remove batch dimension
    simcc_x = simcc_x[0]  # [num_keypoints, W]
    simcc_y = simcc_y[0]  # [num_keypoints, H]
    
    num_keypoints = simcc_x.shape[0]
    
    # Get max values and indices
    x_scores = np.max(simcc_x, axis=1)  # [num_keypoints]
    x_indices = np.argmax(simcc_x, axis=1)  # [num_keypoints]
    y_scores = np.max(simcc_y, axis=1)  # [num_keypoints]
    y_indices = np.argmax(simcc_y, axis=1)  # [num_keypoints]
    
    # Combine scores (geometric mean)
    scores = np.sqrt(x_scores * y_scores)
    
    # Convert indices to coordinates in input image space
    simcc_split_ratio = simcc_x.shape[1] / input_size[1]
    x_coords = x_indices / simcc_split_ratio
    y_coords = y_indices / simcc_split_ratio
    
    # Transform from input space to original image space
    keypoints = np.stack([x_coords, y_coords], axis=1)  # [num_keypoints, 2]
    
    # Scale back to original image coordinates
    keypoints[:, 0] = keypoints[:, 0] * scale[0] / input_size[1] + center[0] - scale[0] / 2
    keypoints[:, 1] = keypoints[:, 1] * scale[1] / input_size[0] + center[1] - scale[1] / 2
    
    return keypoints, scores


def run_inference(session, input_name, img_tensor):
    """Run ONNX inference"""
    outputs = session.run(None, {input_name: img_tensor.astype(np.float32)})
    return outputs


def test_onnx_model(args):
    """Main testing function"""
    print(f"\n{'='*60}")
    print(f"  ONNX Model Testing")
    print(f"{'='*60}\n")
    
    # Load ONNX model
    session, input_name, output_names = load_onnx_model(args.onnx, args.device)
    
    # Load COCO annotations
    print(f"\n{'='*60}")
    print(f"  Loading Test Dataset")
    print(f"{'='*60}\n")
    
    print(f"📋 Annotations: {args.ann}")
    print(f"📁 Images: {args.img_dir}")
    
    coco = COCO(args.ann)
    img_ids = coco.getImgIds()
    
    print(f"\n✅ Dataset loaded")
    print(f"   Images: {len(img_ids)}")
    print(f"   Categories: {len(coco.getCatIds())}")
    
    # Prepare output directory for visualizations
    if args.show_dir:
        os.makedirs(args.show_dir, exist_ok=True)
        print(f"   Visualizations will be saved to: {args.show_dir}")
    
    # Run inference on all images
    print(f"\n{'='*60}")
    print(f"  Running Inference")
    print(f"{'='*60}\n")
    
    results = []
    inference_times = []
    
    for img_id in tqdm(img_ids, desc="Testing"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(args.img_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"⚠️  Image not found: {img_path}")
            continue
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  Failed to load image: {img_path}")
            continue
        
        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        for ann in anns:
            # Get bbox
            bbox = ann['bbox']  # [x, y, w, h]
            
            # Preprocess
            img_tensor, center, scale = preprocess_image(
                img, tuple(args.input_size), bbox)
            
            # Run inference
            start_time = time.time()
            outputs = run_inference(session, input_name, img_tensor)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Decode outputs
            # Assuming outputs are [simcc_x, simcc_y]
            if len(outputs) == 2:
                simcc_x, simcc_y = outputs
                keypoints, scores = decode_simcc_output(
                    simcc_x, simcc_y, tuple(args.input_size), center, scale)
            else:
                print(f"⚠️  Unexpected number of outputs: {len(outputs)}")
                continue
            
            # Format result for COCO evaluation
            result = {
                'image_id': img_id,
                'category_id': ann['category_id'],
                'keypoints': [],
                'score': float(np.mean(scores))
            }
            
            # Format keypoints as COCO format: [x, y, visibility] for each keypoint
            for kpt, score in zip(keypoints, scores):
                result['keypoints'].extend([float(kpt[0]), float(kpt[1]), 2.0])
            
            results.append(result)
        
        # Visualize if requested
        if args.show_dir and len(anns) > 0:
            vis_img = img.copy()
            
            for ann in anns:
                bbox = ann['bbox']
                img_tensor, center, scale = preprocess_image(
                    img, tuple(args.input_size), bbox)
                outputs = run_inference(session, input_name, img_tensor)
                
                if len(outputs) == 2:
                    simcc_x, simcc_y = outputs
                    keypoints, scores = decode_simcc_output(
                        simcc_x, simcc_y, tuple(args.input_size), center, scale)
                    
                    # Draw keypoints
                    for kpt, score in zip(keypoints, scores):
                        if score > args.kpt_thr:
                            x, y = int(kpt[0]), int(kpt[1])
                            cv2.circle(vis_img, (x, y), 3, (0, 255, 0), -1)
                            cv2.circle(vis_img, (x, y), 4, (255, 255, 255), 1)
                    
                    # Draw bbox
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(vis_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Save visualization
            vis_path = os.path.join(args.show_dir, f"vis_{img_info['file_name']}")
            cv2.imwrite(vis_path, vis_img)
    
    # Print inference statistics
    print(f"\n{'='*60}")
    print(f"  Inference Statistics")
    print(f"{'='*60}\n")
    
    if inference_times:
        avg_time = np.mean(inference_times) * 1000  # Convert to ms
        std_time = np.std(inference_times) * 1000
        fps = 1.0 / np.mean(inference_times)
        print(f"   Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"   FPS: {fps:.2f}")
    
    # Save results to temporary file
    results_file = '/tmp/onnx_test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f)
    
    print(f"\n   Predictions saved to: {results_file}")
    
    # Evaluate using COCO metrics
    print(f"\n{'='*60}")
    print(f"  Evaluation Results")
    print(f"{'='*60}\n")
    
    if len(results) == 0:
        print("❌ No predictions to evaluate")
        return
    
    # Load results
    coco_dt = coco.loadRes(results_file)
    
    # Run evaluation
    coco_eval = COCOeval(coco, coco_dt, 'keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Print detailed metrics
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}\n")
    
    print(f"✅ Testing complete!")
    print(f"   Total images: {len(img_ids)}")
    print(f"   Total predictions: {len(results)}")
    print(f"   AP: {coco_eval.stats[0]:.4f}")
    print(f"   AP@0.5: {coco_eval.stats[1]:.4f}")
    print(f"   AP@0.75: {coco_eval.stats[2]:.4f}")
    print(f"   AR: {coco_eval.stats[5]:.4f}")
    
    if args.show_dir:
        print(f"   Visualizations saved to: {args.show_dir}")


def main():
    args = parse_args()
    
    # Validate inputs
    if not os.path.exists(args.onnx):
        print(f"❌ Error: ONNX model not found: {args.onnx}")
        return 1
    
    if not os.path.exists(args.ann):
        print(f"❌ Error: Annotation file not found: {args.ann}")
        return 1
    
    if not os.path.exists(args.img_dir):
        print(f"❌ Error: Image directory not found: {args.img_dir}")
        return 1
    
    try:
        test_onnx_model(args)
        return 0
    except Exception as e:
        print(f"\n❌ Error during testing:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
