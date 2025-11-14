"""
Enhanced RTMCCHead with optional diffusion refinement

This is a wrapper around the standard RTMCCHead that adds:
1. Optional diffusion refinement during inference
2. Optional refinement loss during training
3. Backward compatible - works as standard RTMCCHead if refiner=None
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    from mmpose.models.heads import RTMCCHead
    from mmpose.registry import MODELS
    
    @MODELS.register_module()
    class RTMCCHeadWithRefinement(RTMCCHead):
        """
        RTMCCHead with conditional diffusion refinement
        
        Args:
            refiner (dict, optional): Config for SimCCDiffusionRefiner
            use_refiner_in_training (bool): Whether to use refiner during training
            *args, **kwargs: Arguments for RTMCCHead
        
        Usage:
            # Fast inference mode (3 steps)
            head = RTMCCHeadWithRefinement(..., refiner=dict(
                type='SimCCDiffusionRefiner',
                num_steps=3,
                confidence_threshold=0.7
            ))
            
            # Precise inference mode (5 steps)
            head.refiner.num_steps = 5
        """
        
        def __init__(self, *args, refiner=None, use_refiner_in_training=False, **kwargs):
            super().__init__(*args, **kwargs)
            
            # Build refiner if specified
            self.use_refiner_in_training = use_refiner_in_training
            if refiner is not None:
                self.refiner = MODELS.build(refiner)
                self.use_refiner = True
            else:
                self.refiner = None
                self.use_refiner = False
        
        def forward(self, feats: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass with optional refinement
            
            Args:
                feats: Features from backbone/neck
            
            Returns:
                simcc_x: [B, K, D] x-coordinate distributions
                simcc_y: [B, K, D] y-coordinate distributions
            """
            # Standard RTMCCHead forward
            simcc_x, simcc_y = super().forward(feats)
            
            # Apply refinement if enabled
            if self.use_refiner:
                # Only refine during inference OR if explicitly enabled for training
                if not self.training or self.use_refiner_in_training:
                    simcc_x, simcc_y, _ = self.refiner(
                        simcc_x, 
                        simcc_y, 
                        training=self.training
                    )
            
            return simcc_x, simcc_y
        
        def predict(self, feats, batch_data_samples, test_cfg={}):
            """
            Prediction with refinement
            
            Override to support different refinement modes:
            - test_cfg['refine_mode'] = 'fast' (3 steps, default)
            - test_cfg['refine_mode'] = 'precise' (5 steps)
            - test_cfg['refine_mode'] = 'off' (no refinement)
            """
            # Set refinement mode if specified
            refine_mode = test_cfg.get('refine_mode', 'fast')
            
            if self.use_refiner and refine_mode != 'off':
                original_steps = self.refiner.num_steps
                
                if refine_mode == 'fast':
                    self.refiner.num_steps = 3
                elif refine_mode == 'precise':
                    self.refiner.num_steps = 5
                
                # Call parent predict
                results = super().predict(feats, batch_data_samples, test_cfg)
                
                # Restore original steps
                self.refiner.num_steps = original_steps
            else:
                # No refinement
                if self.use_refiner:
                    # Temporarily disable refiner
                    self.use_refiner = False
                    results = super().predict(feats, batch_data_samples, test_cfg)
                    self.use_refiner = True
                else:
                    results = super().predict(feats, batch_data_samples, test_cfg)
            
            return results
        
        def loss(self, feats, batch_data_samples, train_cfg={}):
            """
            Compute loss with optional refinement loss
            
            Standard RTMCCHead loss + optional refinement consistency loss
            """
            # Standard loss from RTMCCHead
            losses = super().loss(feats, batch_data_samples, train_cfg)
            
            # Optional: Add refinement consistency loss
            # This encourages the refiner to maintain consistency
            if self.use_refiner and train_cfg.get('use_refine_loss', False):
                # Get predictions
                simcc_x, simcc_y = super().forward(feats)
                
                # Refine
                refined_x, refined_y, uncertain_mask = self.refiner(
                    simcc_x.detach(), 
                    simcc_y.detach(), 
                    training=True
                )
                
                # Compute refinement consistency loss
                # (Only for uncertain keypoints to focus learning)
                mask = uncertain_mask.unsqueeze(-1).float()
                
                refine_loss_x = torch.nn.functional.mse_loss(
                    refined_x * mask, 
                    simcc_x.detach() * mask,
                    reduction='sum'
                ) / (mask.sum() + 1e-6)
                
                refine_loss_y = torch.nn.functional.mse_loss(
                    refined_y * mask, 
                    simcc_y.detach() * mask,
                    reduction='sum'
                ) / (mask.sum() + 1e-6)
                
                losses['loss_refine'] = (refine_loss_x + refine_loss_y) * 0.1
            
            return losses
        
        def __repr__(self):
            s = super().__repr__()
            if self.use_refiner:
                s += f'\nRefiner: {self.refiner}'
            return s

except ImportError:
    # Fallback if mmpose not available
    print("Warning: mmpose not found. RTMCCHeadWithRefinement not registered.")
    
    class RTMCCHeadWithRefinement:
        """Placeholder when mmpose is not available"""
        def __init__(self, *args, **kwargs):
            raise ImportError("mmpose is required for RTMCCHeadWithRefinement")
