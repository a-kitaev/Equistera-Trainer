# Test dataset configuration (uses real test split)
_base_ = ['./horse_ap10k.py']

# Override to use test dataset instead of val
test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/',
        data_mode='topdown',
        ann_file='annotations/horse_test.json',  # TEST dataset
        data_prefix=dict(img='test/'),  # TEST images
        test_mode=True,
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(256, 256)),
            dict(type='PackPoseInputs')
        ],
    ))

test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/annotations/horse_test.json')  # TEST annotations
