#!/usr/bin/env python
"""
Visualize predictions on images

Usage:
    python tools/visualize.py --config configs/rtmpose_m_ap10k.py \
                              --checkpoint work_dirs/rtmpose_m/best.pth \
                              --img data/images/test_image.jpg \
                              --out-file output.jpg
    
    # Visualize on multiple images
    python tools/visualize.py --config configs/rtmpose_m_ap10k.py \
                              --checkpoint work_dirs/rtmpose_m/best.pth \
                              --img-dir data/images/test/ \
                              --out-dir visualizations/
"""

import argparse
import os
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import PoseDataSample, merge_data_samples
from mmpose.utils import register_all_modules

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize pose predictions')
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint file')
    parser.add_argument('--img', help='single image file')
    parser.add_argument('--img-dir', help='directory of images')
    parser.add_argument('--out-file', help='output file for single image')
    parser.add_argument('--out-dir', help='output directory for multiple images')
    parser.add_argument(
        '--device', 
        default='cuda:0', 
        help='device used for inference')
    parser.add_argument(
        '--det-config',
        help='detector config file path (if using detector)')
    parser.add_argument(
        '--det-checkpoint',
        help='detector checkpoint file (if using detector)')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.5,
        help='detector score threshold')
    parser.add_argument(
        '--draw-bbox',
        action='store_true',
        help='draw bounding boxes')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='link thickness for visualization')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        choices=['mmpose', 'openpose'],
        help='skeleton style')
    args = parser.parse_args()
    return args


def process_one_image(args, model, detector, img_path, out_path):
    """Process a single image."""
    
    # Read image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get bounding boxes
    if detector is not None:
        # Use detector
        det_result = inference_detector(detector, img_path)
        pred_instances = det_result.pred_instances
        
        # Filter by score
        bboxes = pred_instances.bboxes[
            pred_instances.scores > args.det_score_thr].cpu().numpy()
    else:
        # Use whole image
        h, w = img.shape[:2]
        bboxes = np.array([[0, 0, w, h]], dtype=np.float32)
    
    # Inference
    pose_results = inference_topdown(model, img_path, bboxes)
    
    # Visualize
    vis_img = img.copy()
    
    # Draw bounding boxes
    if args.draw_bbox:
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[:4].astype(int)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw keypoints and skeleton
    for pose_result in pose_results:
        keypoints = pose_result.pred_instances.keypoints[0]
        scores = pose_result.pred_instances.keypoint_scores[0]
        
        # Get skeleton connections
        skeleton = model.dataset_meta.get('skeleton_info', {})
        
        # Draw skeleton
        for sk_id, sk in skeleton.items():
            link = sk['link']
            kpt1_name, kpt2_name = link
            
            # Find keypoint indices
            keypoint_info = model.dataset_meta.get('keypoint_info', {})
            kpt1_idx = None
            kpt2_idx = None
            
            for idx, info in keypoint_info.items():
                if info['name'] == kpt1_name:
                    kpt1_idx = idx
                if info['name'] == kpt2_name:
                    kpt2_idx = idx
            
            if kpt1_idx is not None and kpt2_idx is not None:
                if scores[kpt1_idx] > args.kpt_thr and scores[kpt2_idx] > args.kpt_thr:
                    pt1 = keypoints[kpt1_idx].astype(int)
                    pt2 = keypoints[kpt2_idx].astype(int)
                    color = sk.get('color', [255, 255, 255])
                    cv2.line(vis_img, tuple(pt1), tuple(pt2), 
                            color, args.thickness)
        
        # Draw keypoints
        for idx, (kpt, score) in enumerate(zip(keypoints, scores)):
            if score > args.kpt_thr:
                color = keypoint_info.get(idx, {}).get('color', [255, 0, 0])
                pt = kpt.astype(int)
                cv2.circle(vis_img, tuple(pt), args.radius, color, -1)
                cv2.circle(vis_img, tuple(pt), args.radius + 1, 
                          (255, 255, 255), 1)
    
    # Save
    cv2.imwrite(out_path, vis_img)
    print(f'Saved visualization to {out_path}')
    
    return vis_img


def main():
    args = parse_args()
    
    # Register all modules
    register_all_modules()
    
    # Initialize pose model
    model = init_model(args.config, args.checkpoint, device=args.device)
    
    # Initialize detector if provided
    detector = None
    if args.det_config and args.det_checkpoint:
        if has_mmdet:
            detector = init_detector(
                args.det_config, 
                args.det_checkpoint, 
                device=args.device)
        else:
            print('MMDetection is not installed. Using whole image as bbox.')
    
    # Process single image
    if args.img:
        if not args.out_file:
            args.out_file = 'output_' + osp.basename(args.img)
        
        process_one_image(args, model, detector, args.img, args.out_file)
    
    # Process directory of images
    elif args.img_dir:
        if not args.out_dir:
            args.out_dir = 'visualizations'
        os.makedirs(args.out_dir, exist_ok=True)
        
        img_dir = Path(args.img_dir)
        img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + \
                   list(img_dir.glob('*.jpeg'))
        
        for img_file in img_files:
            out_file = osp.join(args.out_dir, 'vis_' + img_file.name)
            process_one_image(args, model, detector, str(img_file), out_file)
    
    else:
        print('Please provide --img or --img-dir')


if __name__ == '__main__':
    main()
