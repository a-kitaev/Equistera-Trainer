"""
HRNet-W32 configuration for horse pose estimation (26 keypoints)
Fine-tuning on AP-10K dataset with layer-wise learning rates

Training strategy:
- Freeze: Stage 1 (early features)
- Fine-tune (LR=0.0001): Stages 2-3 (part detectors)
- Train (LR=0.001): New 26-keypoint head (rebuilt from scratch)
"""

_base_ = [
    '_base_/datasets/horse_ap10k.py',
    '_base_/default_runtime.py'
]

# Model architecture
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        init_cfg=dict(
            type='Pretrained',
            # Animal-pretrained checkpoint (AP-10K)
            checkpoint='checkpoints/hrnet_w32_ap10k_256x256-18aac840_20211029.pth'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=26,  # 26 keypoints for horse
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=dict(
            type='MSRAHeatmap',
            input_size=(256, 256),
            heatmap_size=(64, 64),
            sigma=2)),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

# Base learning rate for batch size 256
base_lr = 5e-4

# Simplified optimizer - NO layer-wise LR (that's causing the collapse!)
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=base_lr, weight_decay=0.0))

# Learning rate scheduler - USE COSINE LIKE RTMPOSE (NOT MULTISTEP!)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),  # Warmup
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=1,
        end=300,
        T_max=299,
        by_epoch=True,
        convert_to_iter_based=True)
]

# Training settings
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=10)

# Codec for HRNet (Heatmap encoding)
codec = dict(
    type='MSRAHeatmap',
    input_size=(256, 256),
    heatmap_size=(64, 64),
    sigma=2)

# Override train pipeline to add heatmap label generation
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal', prob=0.5),
    dict(type='RandomHalfBody', min_total_keypoints=13, min_upper_keypoints=5),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.16,
        scale_factor=[0.7, 1.3],
        rotate_factor=40),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(
        type='PhotometricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', blur_limit=3, p=0.1),
            dict(type='MedianBlur', blur_limit=3, p=0.1),
            dict(type='GaussNoise', var_limit=(10.0, 50.0), p=0.1),
            dict(type='CoarseDropout', 
                 max_holes=8, 
                 max_height=32, 
                 max_width=32, 
                 p=0.1),
        ]),
    dict(type='GenerateTarget', encoder=codec),  # Generate heatmap labels
    dict(type='PackPoseInputs')
]

# Adjust batch size for small dataset (800 images)
train_dataloader = dict(
    batch_size=16,  # Reduced for small dataset
    num_workers=4,
    dataset=dict(
        pipeline=train_pipeline)  # Use our heatmap pipeline!
)

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
)

# Auto scaling learning rate
auto_scale_lr = dict(base_batch_size=256)

# Resume training
resume = False

# Random seed
randomness = dict(seed=21)
