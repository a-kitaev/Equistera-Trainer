# Dataset settings for horse pose estimation (26 keypoints)

dataset_name = 'horse_ap10k'
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/'

# Dataset info - 26 keypoints for horse
dataset_info = dict(
    dataset_name='horse_pose',
    paper_info=dict(
        author='Equistera',
        title='Horse Pose Estimation with 26 Keypoints',
        year='2025',
    ),
    keypoint_info={
        0: dict(name='nose', id=1, color=[255, 107, 107], type='upper', swap=''),
        1: dict(name='l_eye', id=2, color=[78, 205, 196], type='upper', swap='r_eye'),
        2: dict(name='r_eye', id=3, color=[69, 183, 209], type='upper', swap='l_eye'),
        3: dict(name='l_ear', id=4, color=[150, 206, 180], type='upper', swap='r_ear'),
        4: dict(name='r_ear', id=5, color=[255, 234, 167], type='upper', swap='l_ear'),
        5: dict(name='neck', id=6, color=[223, 230, 233], type='upper', swap=''),
        6: dict(name='withers', id=7, color=[162, 155, 254], type='upper', swap=''),
        7: dict(name='back', id=8, color=[253, 121, 168], type='upper', swap=''),
        8: dict(name='tail_base', id=9, color=[253, 203, 110], type='upper', swap=''),
        9: dict(name='tail_end', id=10, color=[225, 112, 85], type='upper', swap=''),
        10: dict(name='l_elbow', id=11, color=[116, 185, 255], type='upper', swap='r_elbow'),
        11: dict(name='r_elbow', id=12, color=[9, 132, 227], type='upper', swap='l_elbow'),
        12: dict(name='l_knee', id=13, color=[85, 239, 196], type='lower', swap='r_knee'),
        13: dict(name='r_knee', id=14, color=[0, 184, 148], type='lower', swap='l_knee'),
        14: dict(name='lf_fetlock', id=15, color=[250, 177, 160], type='lower', swap='rf_fetlock'),
        15: dict(name='rf_fetlock', id=16, color=[225, 112, 85], type='lower', swap='lf_fetlock'),
        16: dict(name='lf_hoof', id=17, color=[250, 177, 160], type='lower', swap='rf_hoof'),
        17: dict(name='rf_hoof', id=18, color=[225, 112, 85], type='lower', swap='lf_hoof'),
        18: dict(name='l_hip', id=19, color=[162, 155, 254], type='lower', swap='r_hip'),
        19: dict(name='r_hip', id=20, color=[108, 92, 231], type='lower', swap='l_hip'),
        20: dict(name='l_hock', id=21, color=[85, 239, 196], type='lower', swap='r_hock'),
        21: dict(name='r_hock', id=22, color=[0, 184, 148], type='lower', swap='l_hock'),
        22: dict(name='lh_fetlock', id=23, color=[250, 177, 160], type='lower', swap='rh_fetlock'),
        23: dict(name='rh_fetlock', id=24, color=[225, 119, 85], type='lower', swap='lh_fetlock'),
        24: dict(name='lh_hoof', id=25, color=[253, 121, 168], type='lower', swap='rh_hoof'),
        25: dict(name='rh_hoof', id=26, color=[214, 48, 49], type='lower', swap='lh_hoof'),
    },
    skeleton_info={
        0: dict(link=('nose', 'l_eye'), id=0, color=[255, 107, 107]),
        1: dict(link=('nose', 'r_eye'), id=1, color=[255, 107, 107]),
        2: dict(link=('l_eye', 'l_ear'), id=2, color=[78, 205, 196]),
        3: dict(link=('r_eye', 'r_ear'), id=3, color=[69, 183, 209]),
        4: dict(link=('nose', 'neck'), id=4, color=[255, 107, 107]),
        5: dict(link=('neck', 'withers'), id=5, color=[223, 230, 233]),
        6: dict(link=('withers', 'back'), id=6, color=[162, 155, 254]),
        7: dict(link=('back', 'tail_base'), id=7, color=[253, 121, 168]),
        8: dict(link=('tail_base', 'tail_end'), id=8, color=[253, 203, 110]),
        9: dict(link=('withers', 'l_elbow'), id=9, color=[162, 155, 254]),
        10: dict(link=('withers', 'r_elbow'), id=10, color=[162, 155, 254]),
        11: dict(link=('l_elbow', 'l_knee'), id=11, color=[116, 185, 255]),
        12: dict(link=('r_elbow', 'r_knee'), id=12, color=[9, 132, 227]),
        13: dict(link=('l_knee', 'lf_fetlock'), id=13, color=[85, 239, 196]),
        14: dict(link=('r_knee', 'rf_fetlock'), id=14, color=[0, 184, 148]),
        15: dict(link=('lf_fetlock', 'lf_hoof'), id=15, color=[250, 177, 160]),
        16: dict(link=('rf_fetlock', 'rf_hoof'), id=16, color=[225, 112, 85]),
        17: dict(link=('back', 'l_hip'), id=17, color=[253, 121, 168]),
        18: dict(link=('back', 'r_hip'), id=18, color=[253, 121, 168]),
        19: dict(link=('l_hip', 'l_hock'), id=19, color=[162, 155, 254]),
        20: dict(link=('r_hip', 'r_hock'), id=20, color=[108, 92, 231]),
        21: dict(link=('l_hock', 'lh_fetlock'), id=21, color=[85, 239, 196]),
        22: dict(link=('r_hock', 'rh_fetlock'), id=22, color=[0, 184, 148]),
        23: dict(link=('lh_fetlock', 'lh_hoof'), id=23, color=[250, 177, 160]),
        24: dict(link=('rh_fetlock', 'rh_hoof'), id=24, color=[225, 119, 85]),
    },
    joint_weights=[1.0, 1.0, 1.0, 0.8, 0.8, 1.2, 1.5, 1.2, 1.0, 0.7, 
                   1.3, 1.3, 1.1, 1.1, 1.0, 1.0, 1.0, 1.0, 1.3, 1.3, 
                   1.1, 1.1, 1.0, 1.0, 1.0, 1.0],
    sigmas=[0.025, 0.025, 0.025, 0.035, 0.035, 0.079, 0.072, 0.079, 0.079, 0.089,
            0.072, 0.072, 0.062, 0.062, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087,
            0.062, 0.062, 0.062, 0.062, 0.107, 0.107]
)

# Train pipeline with aggressive augmentation
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal', prob=0.5),
    dict(type='RandomHalfBody', min_total_keypoints=13, min_upper_keypoints=5),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.16,
        scale_factor=[0.7, 1.3],  # Aggressive scaling
        rotate_factor=40),  # Aggressive rotation
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(
        type='PhotometricDistortion',  # Color jittering
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
    dict(type='PackPoseInputs')
]

# Val pipeline - no augmentation
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='PackPoseInputs')
]

# Test pipeline
test_pipeline = val_pipeline

# Data loaders
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/horse_train.json',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline,
        metainfo=dataset_info,
    ))

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/horse_val.json',
        data_prefix=dict(img='val/'),
        pipeline=val_pipeline,
        test_mode=True,
        metainfo=dataset_info,
    ))

test_dataloader = val_dataloader

# Evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/horse_val.json')

test_evaluator = val_evaluator
