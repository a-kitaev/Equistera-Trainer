#!/usr/bin/env python3
"""
Test RTMPose model on test dataset locally
"""
import os
import sys
from mmengine.config import Config
from mmengine.runner import Runner

def main():
    # Load config
    config_file = 'configs/rtmpose_m_ap10k.py'
    cfg = Config.fromfile(config_file)
    
    # Override test dataset to use our test set
    cfg.test_dataloader = dict(
        batch_size=16,
        num_workers=4,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
        dataset=dict(
            type='CocoDataset',
            data_root='data/',
            data_mode='topdown',
            ann_file='annotations/horse_test.json',  # Use TEST dataset
            data_prefix=dict(img='test/'),
            test_mode=True,
            pipeline=[
                dict(type='LoadImage'),
                dict(type='GetBBoxCenterScale'),
                dict(type='TopdownAffine', input_size=(256, 256)),
                dict(type='PackPoseInputs'),
            ],
        )
    )
    
    # Test evaluator for test set
    cfg.test_evaluator = dict(
        type='CocoMetric',
        ann_file='data/annotations/horse_test.json'  # Use TEST annotations
    )
    
    # Set checkpoint path
    cfg.load_from = 'checkpoints/best_coco_AP_epoch_210.pth'
    
    # Work directory for results
    cfg.work_dir = 'work_dirs/test_results_local'
    
    # Build the runner
    runner = Runner.from_cfg(cfg)
    
    # Run test
    runner.test()

if __name__ == '__main__':
    main()
