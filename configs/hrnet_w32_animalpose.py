"""
HRNet-W32 configuration for horse pose estimation (26 keypoints)
Fine-tuning on AnimalPose dataset with layer-wise learning rates

This config uses AnimalPose pretrained checkpoint (20 keypoints)
and fine-tunes for 26 horse keypoints.
"""

_base_ = ['./hrnet_w32_ap10k.py']

# Override to use AnimalPose pretrained checkpoint
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            # Animal-pretrained checkpoint (AnimalPose - 20 keypoints)
            checkpoint='checkpoints/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth')))

# Use horse dataset (not AnimalPose dataset)
# We're just using AnimalPose pretrained weights
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='annotations/horse_train.json',
        data_prefix=dict(img='train/'),
    ))

val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='annotations/horse_val.json',
        data_prefix=dict(img='val/'),
    ))

test_dataloader = val_dataloader

# Update evaluator
val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/annotations/horse_val.json')

test_evaluator = val_evaluator

# Random seed
randomness = dict(seed=42)
