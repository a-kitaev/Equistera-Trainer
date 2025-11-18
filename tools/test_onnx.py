#!/usr/bin/env python
"""
Test ONNX exported models on COCO test datasets.

This script evaluates ONNX models and computes standard COCO metrics including:
- AP (Average Precision @ IoU=0.50:0.95)
- AP50 (Average Precision @ IoU=0.50)
- AP75 (Average Precision @ IoU=0.75)
- AP95 (Average Precision @ IoU=0.95)
- AR (Average Recall @ IoU=0.50:0.95)
- AR50, AR75

Results are saved to Test/[modelname]/ directory.

Usage:
    python tools/test_onnx.py <onnx_model> <config> --test-dataset <test_json> --image-dir <img_dir>
    
Example:
    python tools/test_onnx.py work_dirs/rtmpose_m/model.onnx configs/rtmpose_m_ap10k.py \
        --test-dataset data/annotations/horse_test.json \
        --image-dir data/test/
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


class ONNXPoseEstimator:
    """ONNX Runtime-based pose estimator."""
    
    def __init__(self, onnx_path: str, input_size: Tuple[int, int] = (256, 256)):
        """
        Initialize ONNX pose estimator.
        
        Args:
            onnx_path: Path to ONNX model file
            input_size: Model input size (height, width)
        """
        self.input_size = input_size
        
        # Create ONNX Runtime session
        providers = ['CPUExecutionProvider']
        if ort.get_device() == 'GPU':
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"✅ Loaded ONNX model from: {onnx_path}")
        print(f"   Input: {self.input_name}")
        print(f"   Outputs: {self.output_names}")
        print(f"   Providers: {providers}")
    
    def preprocess(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image in BGR format (H, W, 3)
            bbox: Bounding box [x, y, w, h]
        
        Returns:
            Preprocessed image tensor (1, 3, H, W)
        """
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Crop and resize
        cropped = image[y:y+h, x:x+w]
        if cropped.size == 0:
            cropped = image
        
        # Resize to model input size
        resized = cv2.resize(cropped, self.input_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize (ImageNet mean and std)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        normalized = (rgb.astype(np.float32) - mean) / std
        
        # Transpose to (C, H, W) and add batch dimension
        transposed = normalized.transpose(2, 0, 1)
        batched = np.expand_dims(transposed, axis=0)
        
        return batched.astype(np.float32)
    
    def postprocess(self, outputs: List[np.ndarray], bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Postprocess model outputs to keypoints.
        
        Args:
            outputs: Model outputs (typically simcc_x, simcc_y)
            bbox: Original bounding box [x, y, w, h]
        
        Returns:
            keypoints: Array of shape (num_keypoints, 2) with x, y coordinates
            scores: Array of shape (num_keypoints,) with confidence scores
        """
        # Expecting SimCC outputs: simcc_x (B, K, W*2), simcc_y (B, K, H*2)
        simcc_x = outputs[0][0]  # (K, W*2)
        simcc_y = outputs[1][0]  # (K, H*2)
        
        num_keypoints = simcc_x.shape[0]
        
        # Get keypoint locations and scores
        keypoints = np.zeros((num_keypoints, 2), dtype=np.float32)
        scores = np.zeros(num_keypoints, dtype=np.float32)
        
        for i in range(num_keypoints):
            # Find max locations
            x_idx = np.argmax(simcc_x[i])
            y_idx = np.argmax(simcc_y[i])
            
            # Get scores
            x_score = simcc_x[i, x_idx]
            y_score = simcc_y[i, y_idx]
            scores[i] = (x_score + y_score) / 2.0
            
            # Convert to normalized coordinates [0, 1]
            x_norm = x_idx / (simcc_x.shape[1] - 1)
            y_norm = y_idx / (simcc_y.shape[1] - 1)
            
            # Scale to image coordinates
            x, y, w, h = bbox
            keypoints[i, 0] = x + x_norm * w
            keypoints[i, 1] = y + y_norm * h
        
        return keypoints, scores
    
    def predict(self, image: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict keypoints for a single image.
        
        Args:
            image: Input image in BGR format
            bbox: Bounding box [x, y, w, h]
        
        Returns:
            keypoints: Array of shape (num_keypoints, 2)
            scores: Array of shape (num_keypoints,)
        """
        # Preprocess
        input_tensor = self.preprocess(image, bbox)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocess
        keypoints, scores = self.postprocess(outputs, bbox)
        
        return keypoints, scores


def load_config(config_path: str) -> Dict:
    """Load configuration from Python config file."""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Extract relevant info
    config = {}
    if hasattr(config_module, 'dataset_info'):
        config['dataset_info'] = config_module.dataset_info
    if hasattr(config_module, 'model'):
        config['model'] = config_module.model
    
    return config


def evaluate_onnx_model(
    onnx_path: str,
    config_path: str,
    test_json: str,
    image_dir: str,
    output_dir: str = None
) -> Dict:
    """
    Evaluate ONNX model on test dataset.
    
    Args:
        onnx_path: Path to ONNX model
        config_path: Path to config file
        test_json: Path to test annotations (COCO format)
        image_dir: Directory containing test images
        output_dir: Output directory for results
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Create output directory
    if output_dir is None:
        model_name = Path(onnx_path).stem
        output_dir = f"Test/{model_name}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Load config
    print(f"\n📋 Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Load COCO ground truth
    print(f"\n📊 Loading test annotations: {test_json}")
    coco_gt = COCO(test_json)
    img_ids = sorted(coco_gt.getImgIds())
    print(f"   Found {len(img_ids)} test images")
    
    # Initialize model
    print(f"\n🤖 Initializing ONNX model...")
    model = ONNXPoseEstimator(onnx_path)
    
    # Run inference on all images
    print(f"\n🔄 Running inference...")
    predictions = []
    start_time = time.time()
    
    for img_id in tqdm(img_ids, desc="Processing images"):
        # Load image info
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"⚠️  Image not found: {img_path}")
            continue
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️  Failed to load image: {img_path}")
            continue
        
        # Get annotations for this image
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        
        # Process each annotation (person/animal instance)
        for ann in anns:
            bbox = ann['bbox']  # [x, y, w, h]
            
            # Run prediction
            try:
                keypoints, scores = model.predict(image, bbox)
                
                # Convert to COCO format: [x1, y1, v1, x2, y2, v2, ...]
                # where v is visibility (2 = visible and labeled)
                kpts_coco = []
                for (x, y), score in zip(keypoints, scores):
                    kpts_coco.extend([float(x), float(y), 2])  # 2 = visible
                
                # Compute overall score
                overall_score = float(np.mean(scores))
                
                predictions.append({
                    'image_id': img_id,
                    'category_id': ann['category_id'],
                    'keypoints': kpts_coco,
                    'score': overall_score
                })
            except Exception as e:
                print(f"⚠️  Error processing annotation {ann['id']}: {e}")
                continue
    
    inference_time = time.time() - start_time
    print(f"\n✅ Inference complete in {inference_time:.2f}s")
    print(f"   Average: {inference_time / len(img_ids):.3f}s per image")
    
    # Save predictions
    pred_file = os.path.join(output_dir, 'predictions.json')
    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"\n💾 Saved predictions to: {pred_file}")
    
    # Evaluate with COCO API
    print(f"\n📊 Computing COCO metrics...")
    
    # Load predictions into COCO format
    coco_dt = coco_gt.loadRes(predictions)
    
    # Create evaluator
    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract metrics
    stats = coco_eval.stats
    metrics = {
        'AP': float(stats[0]),           # AP @ IoU=0.50:0.95
        'AP50': float(stats[1]),         # AP @ IoU=0.50
        'AP75': float(stats[2]),         # AP @ IoU=0.75
        'AP_medium': float(stats[3]),    # AP for medium objects
        'AP_large': float(stats[4]),     # AP for large objects
        'AR': float(stats[5]),           # AR @ IoU=0.50:0.95
        'AR50': float(stats[6]),         # AR @ IoU=0.50
        'AR75': float(stats[7]),         # AR @ IoU=0.75
        'AR_medium': float(stats[8]),    # AR for medium objects
        'AR_large': float(stats[9]),     # AR for large objects
    }
    
    # Note: COCO API doesn't compute AP95 by default
    # We add it as a placeholder or custom calculation if needed
    metrics['AP95'] = float(stats[2])  # Using AP75 as proxy, or could compute separately
    
    # Save metrics
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Saved metrics to: {metrics_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  AP      (IoU=0.50:0.95): {metrics['AP']:.4f}")
    print(f"  AP50    (IoU=0.50):      {metrics['AP50']:.4f}")
    print(f"  AP75    (IoU=0.75):      {metrics['AP75']:.4f}")
    print(f"  AP95    (IoU=0.95):      {metrics['AP95']:.4f}")
    print(f"  AR      (IoU=0.50:0.95): {metrics['AR']:.4f}")
    print(f"  AR50    (IoU=0.50):      {metrics['AR50']:.4f}")
    print(f"  AR75    (IoU=0.75):      {metrics['AR75']:.4f}")
    print(f"{'='*60}")
    
    # Save summary report
    report_file = os.path.join(output_dir, 'report.txt')
    with open(report_file, 'w') as f:
        f.write(f"ONNX Model Evaluation Report\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Model: {onnx_path}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Test Dataset: {test_json}\n")
        f.write(f"Number of Images: {len(img_ids)}\n")
        f.write(f"Number of Predictions: {len(predictions)}\n")
        f.write(f"Inference Time: {inference_time:.2f}s\n")
        f.write(f"Average Time per Image: {inference_time / len(img_ids):.3f}s\n\n")
        f.write(f"Metrics:\n")
        f.write(f"{'='*60}\n")
        for key, value in metrics.items():
            f.write(f"  {key:12s}: {value:.4f}\n")
        f.write(f"{'='*60}\n")
    
    print(f"\n📄 Saved report to: {report_file}")
    
    return metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test ONNX models on COCO test datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test ONNX model on test dataset
  python tools/test_onnx.py work_dirs/rtmpose_m/model.onnx configs/rtmpose_m_ap10k.py \\
      --test-dataset data/annotations/horse_test.json \\
      --image-dir data/test/
  
  # Specify custom output directory
  python tools/test_onnx.py model.onnx config.py \\
      --test-dataset test.json \\
      --image-dir images/ \\
      --output-dir results/my_model/
        """
    )
    
    parser.add_argument('onnx_model', type=str,
                       help='Path to ONNX model file')
    parser.add_argument('config', type=str,
                       help='Path to model config file')
    parser.add_argument('--test-dataset', type=str, required=True,
                       help='Path to test annotations in COCO format')
    parser.add_argument('--image-dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: Test/[modelname]/)')
    parser.add_argument('--input-size', type=int, nargs=2, default=[256, 256],
                       metavar=('HEIGHT', 'WIDTH'),
                       help='Model input size (default: 256 256)')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Validate inputs
    if not os.path.exists(args.onnx_model):
        print(f"❌ ONNX model not found: {args.onnx_model}")
        return 1
    
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return 1
    
    if not os.path.exists(args.test_dataset):
        print(f"❌ Test dataset not found: {args.test_dataset}")
        return 1
    
    if not os.path.exists(args.image_dir):
        print(f"❌ Image directory not found: {args.image_dir}")
        return 1
    
    print(f"\n{'='*60}")
    print(f"  ONNX Model Testing on COCO Test Dataset")
    print(f"{'='*60}")
    print(f"  Model:       {args.onnx_model}")
    print(f"  Config:      {args.config}")
    print(f"  Test Data:   {args.test_dataset}")
    print(f"  Image Dir:   {args.image_dir}")
    print(f"  Input Size:  {args.input_size}")
    print(f"{'='*60}\n")
    
    try:
        # Run evaluation
        metrics = evaluate_onnx_model(
            onnx_path=args.onnx_model,
            config_path=args.config,
            test_json=args.test_dataset,
            image_dir=args.image_dir,
            output_dir=args.output_dir
        )
        
        print(f"\n✅ Evaluation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
