"""
RTMPose-M V2 with Text Fusion - Fine-tuning from your best checkpoint

This config loads your best trained model (0.83 AP) and adds the text fusion neck
on top. This is better than starting from generic AP-10K weights because:
- Backbone already optimized for YOUR specific horse dataset
- Only neck needs to learn (smaller optimization problem)
- Lower risk of regression (baseline is already 0.83)

Training strategy:
- Load full model from your checkpoint (backbone + head)
- Insert text fusion neck between backbone and head
- Freeze backbone (already optimal for your data)
- Train ONLY the neck with high LR
- Fine-tune head with low LR to adapt to neck outputs
"""

_base_ = [
    '_base_/datasets/horse_ap10k.py',
    '_base_/default_runtime.py'
]

# Register V2 custom modules
custom_imports = dict(
    imports=[
        'tools.topdown_pose_estimator_v2',
        'tools.text_fusion_neck',
        'tools.rtmcc_head_v2',
        'tools.simcc_diffusion_refiner',
        'tools.loading_text'
    ],
    allow_failed_imports=False
)

# Model architecture
model = dict(
    type='TopdownPoseEstimatorWithNeck',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=None),  # Don't load here, use load_from instead
    neck=dict(
        type='TextFusionNeck',
        in_channels=768,
        out_channels=768,
        global_text_dim=384,
        local_text_dim=384,
        fusion_type='attention',  # Changed from 'film' - richer fusion for occlusions
        num_keypoints=26,
        dropout=0.1
    ),
    head=dict(
        type='RTMCCHead',  # Use base head (no refiner) to match checkpoint
        in_channels=768,
        out_channels=26,
        input_size=(256, 256),
        in_featuremap_size=(8, 8),
        simcc_split_ratio=2.0,
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=dict(
            type='SimCCLabel',
            input_size=(256, 256),
            sigma=(5.66, 5.66),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False),
        init_cfg=None),  # Don't load here, use load_from instead
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=False,
    ))

# Base learning rate - lowered for stable joint optimization
base_lr = 1e-3

# Optimizer with differential learning rates (paper's approach)
# Key fix: DON'T freeze backbone - allow gradient flow from text-fused features
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            # DON'T FREEZE - use low LR for gradient flow from text-fused features
            # Paper approach: joint optimization with differential learning rates
            'backbone': dict(lr_mult=0.1),  # Effective LR: 1e-4 (allows gradient flow)

            # Train neck with full LR (new component learning text fusion)
            'neck': dict(lr_mult=1.0),  # Effective LR: 1e-3

            # Head adapts to neck outputs with medium LR
            'head.gau': dict(lr_mult=0.3),  # Effective LR: 3e-4
            'head.cls_x': dict(lr_mult=0.3),
            'head.cls_y': dict(lr_mult=0.3),
            'head.refiner': dict(lr_mult=0.5),  # Effective LR: 5e-4
        },
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        bypass_duplicate=True))

# Learning rate scheduler with warmup
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=800),  # Warmup (800 iterations ~10 epochs)
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=1,
        end=200,  # Extended training for text fusion learning
        T_max=199,
        by_epoch=True,
        convert_to_iter_based=True)
]

# Training settings - Extended to allow text fusion neck to learn
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=5)

# Codec for RTMPose (SimCC encoding)
codec = dict(
    type='SimCCLabel',
    input_size=(256, 256),
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# Override train pipeline to add text embeddings
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='LoadTextEmbeddings',
         global_path='embeddings/horse_global.npy',
         local_path='embeddings/horse_local.npy'),
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
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs',
         meta_keys=('id', 'img_id', 'img_path', 'category_id', 'crowd_index', 
                    'ori_shape', 'img_shape', 'input_size', 'input_center', 
                    'input_scale', 'flip', 'flip_direction', 'flip_indices', 
                    'raw_ann_info', 'dataset_name',
                    'global_text', 'local_text'))
]

# Override val pipeline to add text embeddings (CRITICAL FIX!)
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='LoadTextEmbeddings',
         global_path='embeddings/horse_global.npy',
         local_path='embeddings/horse_local.npy'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs',
         meta_keys=('id', 'img_id', 'img_path', 'category_id', 'crowd_index',
                    'ori_shape', 'img_shape', 'input_size', 'input_center',
                    'input_scale', 'flip', 'flip_direction', 'flip_indices',
                    'raw_ann_info', 'dataset_name',
                    'global_text', 'local_text'))
]

# Batch size
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/',
        data_mode='topdown',
        ann_file='annotations/horse_val.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=val_pipeline)
)

# Auto scaling learning rate
auto_scale_lr = dict(base_batch_size=256)

# Resume training
resume = False

# Random seed
randomness = dict(seed=21)

# Load your best checkpoint (backbone + head)
# Neck will be randomly initialized (new component)
# MMPose will automatically skip missing keys (neck) with strict=False
load_from = '/home/azureuser/equistera-trainer/work_dirs/rtmpose_m_669imgs/best_coco_AP_epoch_210.pth'

# Ensure checkpoint loading ignores missing neck weights
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,  # Save every 5 epochs
        save_best='coco/AP',
        rule='greater',
        max_keep_ckpts=3)
)
