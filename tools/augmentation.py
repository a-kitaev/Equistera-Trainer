"""
Advanced data augmentation pipelines for horse pose estimation

This module contains custom augmentation strategies optimized for 
small datasets (800 images) with aggressive augmentation techniques.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomOcclusion(BaseTransform):
    """
    Randomly occlude parts of the image to simulate real-world scenarios
    (e.g., horses partially hidden by obstacles).
    """
    
    def __init__(self,
                 max_patches: int = 3,
                 patch_ratio_range: Tuple[float, float] = (0.05, 0.15),
                 prob: float = 0.3):
        self.max_patches = max_patches
        self.patch_ratio_range = patch_ratio_range
        self.prob = prob
    
    def transform(self, results: Dict) -> Dict:
        if np.random.random() > self.prob:
            return results
        
        img = results['img']
        h, w = img.shape[:2]
        
        num_patches = np.random.randint(1, self.max_patches + 1)
        
        for _ in range(num_patches):
            # Random patch size
            ratio = np.random.uniform(*self.patch_ratio_range)
            patch_h = int(h * ratio)
            patch_w = int(w * ratio)
            
            # Random position
            y = np.random.randint(0, h - patch_h + 1)
            x = np.random.randint(0, w - patch_w + 1)
            
            # Random color (gray or random)
            if np.random.random() < 0.5:
                color = np.random.randint(0, 256, 3)
            else:
                color = np.array([127, 127, 127])
            
            img[y:y+patch_h, x:x+patch_w] = color
        
        results['img'] = img
        return results


@TRANSFORMS.register_module()
class RandomBackground(BaseTransform):
    """
    Randomly replace background to improve model robustness.
    Requires segmentation mask.
    """
    
    def __init__(self, prob: float = 0.2):
        self.prob = prob
    
    def transform(self, results: Dict) -> Dict:
        if np.random.random() > self.prob:
            return results
        
        # This requires a segmentation mask
        # Skip if mask not available
        if 'mask' not in results:
            return results
        
        img = results['img']
        mask = results['mask']
        
        # Generate random background
        bg = np.random.randint(0, 256, img.shape, dtype=np.uint8)
        
        # Blend with mask
        img = np.where(mask[..., None] > 0, img, bg)
        
        results['img'] = img
        return results


@TRANSFORMS.register_module()
class AdaptiveAugmentation(BaseTransform):
    """
    Adaptive augmentation that increases intensity based on training progress.
    Can be used with a callback to adjust augmentation strength.
    """
    
    def __init__(self,
                 aug_strength: float = 1.0,
                 brightness_range: Tuple[float, float] = (0.7, 1.3),
                 contrast_range: Tuple[float, float] = (0.7, 1.3)):
        self.aug_strength = aug_strength
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def transform(self, results: Dict) -> Dict:
        img = results['img'].astype(np.float32)
        
        # Brightness adjustment
        brightness = np.random.uniform(
            1.0 - (1.0 - self.brightness_range[0]) * self.aug_strength,
            1.0 + (self.brightness_range[1] - 1.0) * self.aug_strength
        )
        img = img * brightness
        
        # Contrast adjustment
        contrast = np.random.uniform(
            1.0 - (1.0 - self.contrast_range[0]) * self.aug_strength,
            1.0 + (self.contrast_range[1] - 1.0) * self.aug_strength
        )
        mean = img.mean()
        img = (img - mean) * contrast + mean
        
        # Clip values
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        results['img'] = img
        return results


# Augmentation presets for different training phases

def get_light_augmentation_pipeline():
    """Light augmentation for initial training phase."""
    return [
        dict(type='RandomFlip', direction='horizontal', prob=0.5),
        dict(
            type='RandomBBoxTransform',
            shift_factor=0.1,
            scale_factor=[0.8, 1.2],
            rotate_factor=20),
        dict(
            type='PhotometricDistortion',
            brightness_delta=16,
            contrast_range=(0.7, 1.3),
            saturation_range=(0.7, 1.3),
            hue_delta=9),
    ]


def get_medium_augmentation_pipeline():
    """Medium augmentation for mid training."""
    return [
        dict(type='RandomFlip', direction='horizontal', prob=0.5),
        dict(type='RandomHalfBody', min_total_keypoints=13, min_upper_keypoints=5),
        dict(
            type='RandomBBoxTransform',
            shift_factor=0.14,
            scale_factor=[0.75, 1.25],
            rotate_factor=30),
        dict(
            type='PhotometricDistortion',
            brightness_delta=24,
            contrast_range=(0.6, 1.4),
            saturation_range=(0.6, 1.4),
            hue_delta=14),
        dict(
            type='Albumentation',
            transforms=[
                dict(type='Blur', blur_limit=3, p=0.1),
                dict(type='MedianBlur', blur_limit=3, p=0.1),
            ]),
    ]


def get_aggressive_augmentation_pipeline():
    """Aggressive augmentation for maximum regularization."""
    return [
        dict(type='RandomFlip', direction='horizontal', prob=0.5),
        dict(type='RandomHalfBody', min_total_keypoints=13, min_upper_keypoints=5),
        dict(
            type='RandomBBoxTransform',
            shift_factor=0.16,
            scale_factor=[0.7, 1.3],
            rotate_factor=40),
        dict(
            type='PhotometricDistortion',
            brightness_delta=32,
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            hue_delta=18),
        dict(
            type='Albumentation',
            transforms=[
                dict(type='Blur', blur_limit=5, p=0.15),
                dict(type='MedianBlur', blur_limit=5, p=0.15),
                dict(type='GaussNoise', var_limit=(10.0, 50.0), p=0.15),
                dict(type='CoarseDropout', 
                     max_holes=8, 
                     max_height=32, 
                     max_width=32, 
                     p=0.15),
                dict(type='RandomBrightnessContrast', p=0.2),
                dict(type='HueSaturationValue', p=0.2),
            ]),
        dict(type='RandomOcclusion', max_patches=3, prob=0.2),
    ]


# Export augmentation configs
AUGMENTATION_PRESETS = {
    'light': get_light_augmentation_pipeline(),
    'medium': get_medium_augmentation_pipeline(),
    'aggressive': get_aggressive_augmentation_pipeline(),
}
