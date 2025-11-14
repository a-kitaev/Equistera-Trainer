"""
Text Fusion Neck for integrating anatomical priors into visual features
Uses Feature-wise Linear Modulation (FiLM) for efficient fusion

This neck sits between backbone and head, modulating visual features
with species-level and keypoint-level text embeddings.
"""

import torch
import torch.nn as nn
from mmengine.model import BaseModule

try:
    from mmpose.registry import MODELS
except ImportError:
    # Fallback for standalone testing
    class MODELS:
        @staticmethod
        def register_module():
            def decorator(cls):
                return cls
            return decorator


@MODELS.register_module()
class TextFusionNeck(BaseModule):
    """
    Fuses visual features with text embeddings using FiLM
    (Feature-wise Linear Modulation)
    
    FiLM: https://arxiv.org/abs/1709.07871
    gamma * x + beta (learned from text embeddings)
    
    Args:
        in_channels (int): Input feature channels from backbone
        out_channels (int): Output feature channels
        global_text_dim (int): Global text embedding dimension (384 for MiniLM)
        local_text_dim (int): Local text embedding dimension (384 for MiniLM)
        fusion_type (str): 'film' (fast) or 'attention' (richer but slower)
        num_keypoints (int): Number of keypoints (26 for horse)
        dropout (float): Dropout rate for regularization
    
    Overhead:
        - FiLM mode: ~0.5ms on GPU
        - Attention mode: ~2ms on GPU
    """
    
    def __init__(self,
                 in_channels=768,
                 out_channels=768,
                 global_text_dim=384,
                 local_text_dim=384,
                 fusion_type='film',
                 num_keypoints=26,
                 dropout=0.1,
                 init_cfg=None):
        super().__init__(init_cfg)
        
        self.fusion_type = fusion_type
        self.num_keypoints = num_keypoints
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Global text → feature modulation parameters (gamma, beta)
        self.global_mlp = nn.Sequential(
            nn.Linear(global_text_dim, in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Local text → keypoint-specific features
        self.local_mlp = nn.Sequential(
            nn.Linear(local_text_dim, in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Learnable gating parameter for residual connection
        # Initialized to 0.1 so model starts with 90% original features, 10% FiLM
        self.film_gate = nn.Parameter(torch.tensor(0.1))
        
        # Optional: Cross-attention for richer fusion
        if fusion_type == 'attention':
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=in_channels,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.norm = nn.LayerNorm(in_channels)
            self.ffn = nn.Sequential(
                nn.Linear(in_channels, in_channels * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(in_channels * 2, in_channels),
                nn.Dropout(dropout)
            )
            self.norm2 = nn.LayerNorm(in_channels)
        
        # Output projection if channels differ
        if in_channels != out_channels:
            self.out_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.out_proj = nn.Identity()
        
        # Initialize FiLM to identity transformation (gamma=1, beta=0)
        self._init_film_parameters()
    
    def _init_film_parameters(self):
        """Initialize FiLM MLP to output identity transformation initially"""
        # For global_mlp: output should be [gamma=1, beta=0] for all channels
        # This prevents destroying features during early training
        with torch.no_grad():
            # Zero out weights (so output is only bias)
            nn.init.zeros_(self.global_mlp[0].weight)
            
            # Set bias to [1, 1, ..., 1, 0, 0, ..., 0]
            # First half (gamma) = 1, second half (beta) = 0
            bias = self.global_mlp[0].bias
            half_dim = len(bias) // 2
            bias[:half_dim].fill_(1.0)  # gamma = 1 (identity scaling)
            bias[half_dim:].fill_(0.0)   # beta = 0 (no shift)
            
        # For local_mlp: initialize normally (used in attention mode)
        if self.fusion_type == 'attention':
            nn.init.xavier_uniform_(self.local_mlp[0].weight)
            nn.init.zeros_(self.local_mlp[0].bias)
    
    def forward(self, visual_feats, data_samples=None):
        """
        Args:
            visual_feats: Tuple of features from backbone, or single tensor [B, C, H, W]
            data_samples: Optional list of data samples containing text embeddings
                Each sample should have:
                - global_text: [384] species-level embedding
                - local_text: [26, 384] keypoint-level embeddings
        
        Returns:
            Modulated features (same format as input - tuple or tensor)
        """
        # Handle tuple input (multi-scale features)
        input_is_tuple = isinstance(visual_feats, tuple)
        if input_is_tuple:
            visual_feats = visual_feats[-1]  # Use highest level features
        
        B, C, H, W = visual_feats.shape
        
        # Extract text embeddings if available
        global_text = None
        local_text = None
        
        if data_samples is not None and len(data_samples) > 0:
            # data_samples is a list during training/validation
            # Extract text embeddings from the first sample
            sample = data_samples[0] if isinstance(data_samples, list) else data_samples
            
            # Try different locations
            if hasattr(sample, 'metainfo'):
                global_text = sample.metainfo.get('global_text', None)
                local_text = sample.metainfo.get('local_text', None)
            elif isinstance(sample, dict):
                global_text = sample.get('global_text', None)
                local_text = sample.get('local_text', None)
            
            # Convert numpy arrays to tensors and replicate for batch
            if global_text is not None:
                if not isinstance(global_text, torch.Tensor):
                    # Use torch.from_numpy for numpy arrays (zero-copy, more efficient)
                    import numpy as np
                    if isinstance(global_text, np.ndarray):
                        global_text = torch.from_numpy(global_text).to(device=visual_feats.device, dtype=torch.float32)
                    else:
                        global_text = torch.tensor(global_text, device=visual_feats.device, dtype=torch.float32)
                if global_text.dim() == 1:
                    global_text = global_text.unsqueeze(0).expand(B, -1)
                    
            if local_text is not None:
                if not isinstance(local_text, torch.Tensor):
                    # Use torch.from_numpy for numpy arrays (zero-copy, more efficient)
                    import numpy as np
                    if isinstance(local_text, np.ndarray):
                        local_text = torch.from_numpy(local_text).to(device=visual_feats.device, dtype=torch.float32)
                    else:
                        local_text = torch.tensor(local_text, device=visual_feats.device, dtype=torch.float32)
                if local_text.dim() == 2:
                    local_text = local_text.unsqueeze(0).expand(B, -1, -1)
        
        # If text embeddings not provided, skip fusion (inference without text)
        if global_text is None or local_text is None:
            # DEBUG: Log why fusion is being skipped
            import warnings
            if not hasattr(self, '_warned_no_text'):
                warnings.warn(
                    f"TextFusionNeck: Skipping fusion! "
                    f"global_text={'None' if global_text is None else 'OK'}, "
                    f"local_text={'None' if local_text is None else 'OK'}, "
                    f"data_samples={'None' if data_samples is None else type(data_samples).__name__}"
                )
                self._warned_no_text = True

            # Direct pass-through - don't even apply out_proj
            # Return in same format as input
            if input_is_tuple:
                return (visual_feats,)
            else:
                return visual_feats
        
        # DEBUG: Log successful fusion (once)
        if not hasattr(self, '_logged_fusion_success'):
            import warnings
            warnings.warn(
                f"TextFusionNeck: Fusion ACTIVE! "
                f"global_text.shape={global_text.shape}, "
                f"local_text.shape={local_text.shape}, "
                f"fusion_type={self.fusion_type}"
            )
            self._logged_fusion_success = True

        if self.fusion_type == 'film':
            # FiLM: gamma * x + beta
            film_params = self.global_mlp(global_text)  # [B, C*2]
            gamma = film_params[:, :C].unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            beta = film_params[:, C:].unsqueeze(-1).unsqueeze(-1)   # [B, C, 1, 1]
            
            # Apply FiLM modulation with residual gating
            # This allows the network to learn to blend FiLM with original features
            film_output = gamma * visual_feats + beta
            modulated_feats = self.film_gate * film_output + (1 - self.film_gate) * visual_feats
            
        elif self.fusion_type == 'attention':
            # Cross-attention between visual and text features
            feat_tokens = visual_feats.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            text_tokens = self.local_mlp(local_text)  # [B, 26, C]
            
            # Multi-head cross-attention
            attended_feats, attn_weights = self.cross_attn(
                query=feat_tokens,
                key=text_tokens,
                value=text_tokens
            )
            
            # Residual connection + LayerNorm
            attended_feats = self.norm(feat_tokens + attended_feats)
            
            # Feed-forward network with residual
            ffn_out = self.ffn(attended_feats)
            attended_feats = self.norm2(attended_feats + ffn_out)
            
            # Reshape back to spatial
            modulated_feats = attended_feats.permute(0, 2, 1).view(B, C, H, W)
        
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
        
        # Apply output projection
        output = self.out_proj(modulated_feats)
        
        # Return in same format as input (tuple or tensor)
        if input_is_tuple:
            return (output,)
        else:
            return output
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'fusion_type={self.fusion_type}, '
                f'num_keypoints={self.num_keypoints})')
