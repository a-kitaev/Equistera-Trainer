#!/usr/bin/env python
"""
Training script for MMPose models

Usage:
    python tools/train.py configs/rtmpose_m_ap10k.py --work-dir work_dirs/rtmpose_m
    
    # Resume training
    python tools/train.py configs/rtmpose_m_ap10k.py --work-dir work_dirs/rtmpose_m --resume
    
    # Multi-GPU training
    python -m torch.distributed.launch --nproc_per_node=2 tools/train.py configs/rtmpose_m_ap10k.py
"""

import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to automatically scale lr with the number of gpus')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    return args


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    
    # Merge cli arguments into config
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Work directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # Use config filename as default work_dir if not specified
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # Enable automatic-mixed-precision training
    if args.amp:
        optim_wrapper = cfg.optim_wrapper.get('type', 'OptimWrapper')
        assert optim_wrapper in ['OptimWrapper', 'AmpOptimWrapper'], \
            '`--amp` is not supported with `{}`.'.format(optim_wrapper)
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # Resume training
    if args.resume:
        cfg.resume = True
        cfg.load_from = None

    # Auto scale learning rate
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    # Disable validation
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    # Build the runner
    runner = Runner.from_cfg(cfg)

    # Start training
    runner.train()


if __name__ == '__main__':
    main()
