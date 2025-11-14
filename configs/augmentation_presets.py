"""
Configuration for different augmentation presets.
Switch between light, medium, and aggressive augmentation strategies.
"""

# Import augmentation presets
from tools.augmentation import (
    AUGMENTATION_PRESETS,
    get_aggressive_augmentation_pipeline,
    get_light_augmentation_pipeline,
    get_medium_augmentation_pipeline,
)

# Export all presets
__all__ = [
    'AUGMENTATION_PRESETS',
    'get_light_augmentation_pipeline',
    'get_medium_augmentation_pipeline', 
    'get_aggressive_augmentation_pipeline',
]


# Example usage in configs:
# from configs.augmentation_presets import get_aggressive_augmentation_pipeline
# train_pipeline = get_aggressive_augmentation_pipeline()
