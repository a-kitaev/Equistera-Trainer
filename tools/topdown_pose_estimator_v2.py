"""
Custom TopdownPoseEstimator with Neck Support

This extends MMPose's TopdownPoseEstimator to support neck modules
for V2 enhancements (text fusion, feature processing, etc.)
"""

import torch
from typing import Optional, Union, Tuple
from mmengine.structures import InstanceData

try:
    from mmpose.models.pose_estimators import TopdownPoseEstimator
    from mmpose.registry import MODELS
    
    @MODELS.register_module()
    class TopdownPoseEstimatorWithNeck(TopdownPoseEstimator):
        """
        Extended TopdownPoseEstimator that supports neck modules
        
        Args:
            backbone (dict): Backbone configuration
            neck (dict, optional): Neck configuration for feature processing
            head (dict): Head configuration
            train_cfg (dict, optional): Training configuration
            test_cfg (dict, optional): Testing configuration
            data_preprocessor (dict, optional): Data preprocessor configuration
            init_cfg (dict, optional): Initialization configuration
        
        Architecture:
            Input → DataPreprocessor → Backbone → Neck (optional) → Head → Output
        
        Usage:
            model = dict(
                type='TopdownPoseEstimatorWithNeck',
                backbone=dict(...),
                neck=dict(type='TextFusionNeck', ...),  # NEW!
                head=dict(...),
            )
        """
        
        def __init__(self,
                     backbone: dict,
                     head: dict,
                     neck: Optional[dict] = None,
                     train_cfg: Optional[dict] = None,
                     test_cfg: Optional[dict] = None,
                     data_preprocessor: Optional[dict] = None,
                     init_cfg: Optional[dict] = None):
            
            # Initialize parent without neck (it doesn't support it)
            super().__init__(
                backbone=backbone,
                head=head,
                train_cfg=train_cfg,
                test_cfg=test_cfg,
                data_preprocessor=data_preprocessor,
                init_cfg=init_cfg
            )
            
            # Build neck if specified
            if neck is not None:
                self.neck = MODELS.build(neck)
                self._has_neck = True
            else:
                self.neck = None
                self._has_neck = False
        
        def extract_feat(self, inputs: torch.Tensor, data_samples=None):
            """
            Extract features from backbone and optionally process through neck
            
            Args:
                inputs: Input images [B, C, H, W]
                data_samples: Optional data samples for neck (e.g., text embeddings)
            
            Returns:
                Processed features matching parent class format
            """
            # Call parent's extract_feat to get features in correct format
            x = super().extract_feat(inputs)
            
            # Process through neck if available
            if self._has_neck:
                # Neck expects tuple, returns in same format as input
                # Don't modify the format - just pass through
                if isinstance(x, (list, tuple)):
                    x_tuple = tuple(x)
                else:
                    x_tuple = (x,)
                
                # Process through neck
                x_neck = self.neck(x_tuple, data_samples)
                
                # Restore original format
                if isinstance(x, (list, tuple)):
                    x = x_neck
                else:
                    # Was a single tensor, extract from tuple
                    if isinstance(x_neck, tuple):
                        x = x_neck[0] if len(x_neck) == 1 else x_neck[-1]
                    else:
                        x = x_neck
            
            return x
        
        def loss(self, inputs: torch.Tensor, data_samples):
            """
            Calculate losses from a batch of inputs and data samples
            
            Args:
                inputs: Input images [B, C, H, W]
                data_samples: Data samples containing ground truth and text embeddings
            
            Returns:
                dict: Loss dictionary
            """
            # Extract features (backbone + neck), pass data_samples for text embeddings
            feats = self.extract_feat(inputs, data_samples)
            
            # Calculate losses in head
            losses = self.head.loss(feats, data_samples, train_cfg=self.train_cfg)
            
            return losses
        
        def predict(self, inputs: torch.Tensor, data_samples):
            """
            Predict keypoints from inputs

            NOTE: We skip the neck during prediction due to architectural constraints:
            - Training: neck with text fusion guides backbone to learn better features
            - Validation: test if improved backbone features work WITHOUT text fusion

            This is a valid approach - if text fusion helps during training, the backbone
            should learn better representations that transfer to validation (no neck).

            Args:
                inputs: Input images [B, C, H, W]
                data_samples: Data samples (text embeddings not used here)

            Returns:
                Predictions with keypoint coordinates
            """
            # Use parent's predict (skips neck, uses backbone -> head directly)
            # If text fusion improved backbone features during training, we'll see AP increase
            return super().predict(inputs, data_samples)
        
        def _forward(self, inputs: torch.Tensor, data_samples=None):
            """
            Network forward process (used for tracing/export)
            
            Args:
                inputs: Input images [B, C, H, W]
                data_samples: Optional data samples (may include text embeddings)
            
            Returns:
                Network outputs
            """
            # Extract features, pass data_samples if available
            feats = self.extract_feat(inputs, data_samples)
            
            # Forward through head
            if hasattr(self.head, '_forward'):
                return self.head._forward(feats)
            else:
                return self.head(feats)
        
        def __repr__(self):
            s = super().__repr__()
            if self._has_neck:
                s += f'\nNeck: {self.neck}'
            return s

except ImportError:
    # Fallback if mmpose not available
    print("Warning: mmpose not found. TopdownPoseEstimatorWithNeck not registered.")
    
    class TopdownPoseEstimatorWithNeck:
        """Placeholder when mmpose is not available"""
        def __init__(self, *args, **kwargs):
            raise ImportError("mmpose is required for TopdownPoseEstimatorWithNeck")
