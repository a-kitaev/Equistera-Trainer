"""
RTMPose-M configuration for horse pose estimation (26 keypoints)
Fine-tuning on AP-10K dataset with layer-wise learning rates

Training strategy:
- Freeze: First 1-2 blocks (early features)
- Fine-tune (LR=0.0001): Middle blocks (part detectors)
- Train (LR=0.001): New 26-keypoint head (rebuilt from scratch)
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
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            # Animal-pretrained checkpoint (AP-10K)
            checkpoint='checkpoints/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.pth'
        )),
    neck=dict(
        type='TextFusionNeck',
        in_channels=768,
        out_channels=768,
        global_text_dim=384,
        local_text_dim=384,
        fusion_type='film',
        num_keypoints=26,
        dropout=0.1
    ),
    head=dict(
        type='RTMCCHeadWithRefinement',  # V2: Head with diffusion refinement
        in_channels=768,
        out_channels=26,  # 26 keypoints for horse
        input_size=(256, 256),
        in_featuremap_size=(8, 8),
        simcc_split_ratio=2.0,
        final_layer_kernel_size=7,
        refiner=dict(
            type='SimCCDiffusionRefiner',
            coord_dim=512,
            hidden_dim=256,
            num_keypoints=26,
            num_steps=3,
            confidence_threshold=0.7
        ),
        use_refiner_in_training=False,
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
            use_dark=False)),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=False,
    ))

# Base learning rate for batch size 256
base_lr = 4e-3

# Optimizer with layer-wise learning rates
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            # Freeze early blocks (backbone stages 0-1)
            'backbone.stem': dict(lr_mult=0.0, decay_mult=0.0),
            'backbone.stage1': dict(lr_mult=0.0, decay_mult=0.0),
            
            # Fine-tune middle blocks with lower LR (backbone stages 2-3)
            'backbone.stage2': dict(lr_mult=0.025),  # 0.0001 / 0.004 = 0.025
            'backbone.stage3': dict(lr_mult=0.025),
            
            # Train final stage and head with full LR
            'backbone.stage4': dict(lr_mult=0.25),   # 0.001 / 0.004 = 0.25
            'neck': dict(lr_mult=1.25),              # 5x higher for random init (0.005 LR)
            'head': dict(lr_mult=0.25),              # New head with higher LR
            'head.refiner': dict(lr_mult=0.5),       # Refiner with higher LR
        },
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        bypass_duplicate=True))

# Learning rate scheduler
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

# Codec for RTMPose (SimCC encoding)
codec = dict(
    type='SimCCLabel',
    input_size=(256, 256),
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# Override train pipeline to add text embeddings and SimCC label generation
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='LoadTextEmbeddings',  # V2: Load pre-computed text embeddings
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
    dict(type='GenerateTarget', encoder=codec),  # Generate SimCC labels
    dict(type='PackPoseInputs',
         meta_keys=('id', 'img_id', 'img_path', 'category_id', 'crowd_index', 
                    'ori_shape', 'img_shape', 'input_size', 'input_center', 
                    'input_scale', 'flip', 'flip_direction', 'flip_indices', 
                    'raw_ann_info', 'dataset_name',
                    'global_text', 'local_text'))  # Add text embedding keys!
]

# Adjust batch size for small dataset (800 images)
train_dataloader = dict(
    batch_size=16,  # Reduced for small dataset
    num_workers=4,
    dataset=dict(
        pipeline=train_pipeline)  # Use our SimCC pipeline!
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
