"""
Data loading transform for text embeddings

Loads pre-computed text embeddings during data loading pipeline
"""

import numpy as np
import os
from typing import Dict

try:
    from mmpose.registry import TRANSFORMS
    
    @TRANSFORMS.register_module()
    class LoadTextEmbeddings:
        """
        Load pre-computed text embeddings for anatomical priors
        
        Args:
            global_path (str): Path to global species embedding (.npy)
            local_path (str): Path to local keypoint embeddings (.npy)
            required (bool): If True, raise error if embeddings not found
        
        Example:
            dict(type='LoadTextEmbeddings',
                 global_path='embeddings/horse_global.npy',
                 local_path='embeddings/horse_local.npy')
        """
        
        def __init__(self, 
                     global_path='embeddings/horse_global.npy',
                     local_path='embeddings/horse_local.npy',
                     required=False):
            self.global_path = global_path
            self.local_path = local_path
            self.required = required
            
            # Load embeddings once (shared across dataset)
            if os.path.exists(global_path) and os.path.exists(local_path):
                self.global_embed = np.load(global_path)  # Shape: [384]
                self.local_embeds = np.load(local_path)   # Shape: [26, 384]
                self.embeddings_loaded = True
                
                print(f"✓ Text embeddings loaded:")
                print(f"  Global: {global_path} - {self.global_embed.shape}")
                print(f"  Local: {local_path} - {self.local_embeds.shape}")
            else:
                self.embeddings_loaded = False
                self.global_embed = None
                self.local_embeds = None
                
                if self.required:
                    raise FileNotFoundError(
                        f"Text embeddings not found!\n"
                        f"  Global: {global_path}\n"
                        f"  Local: {local_path}\n"
                        f"Please run: python tools/generate_text_embeddings.py"
                    )
                else:
                    print(f"⚠ Text embeddings not found (optional):")
                    print(f"  Global: {global_path}")
                    print(f"  Local: {local_path}")
                    print(f"  Training will proceed without text embeddings.")
        
        def transform(self, results: Dict) -> Dict:
            """
            Args:
                results (dict): Result dict from previous transforms
            
            Returns:
                dict: Updated results with text embeddings
            """
            if self.embeddings_loaded:
                # CRITICAL: Add at TOP LEVEL of results dict
                # PackPoseInputs will look for these keys here and copy to metainfo
                results['global_text'] = self.global_embed.copy()
                results['local_text'] = self.local_embeds.copy()
            
            return results
        
        def __call__(self, results: Dict) -> Dict:
            """
            Required interface for MMEngine transforms.
            Calls transform() internally.
            """
            return self.transform(results)
        
        def __repr__(self):
            return (f'{self.__class__.__name__}('
                    f'global_path={self.global_path}, '
                    f'local_path={self.local_path}, '
                    f'loaded={self.embeddings_loaded})')

except ImportError:
    # Fallback if mmpose not available
    print("Warning: mmpose not found. LoadTextEmbeddings not registered.")
    
    class LoadTextEmbeddings:
        """Placeholder when mmpose is not available"""
        def __init__(self, *args, **kwargs):
            raise ImportError("mmpose is required for LoadTextEmbeddings")
