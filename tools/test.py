#!/usr/bin/env python
"""
Testing script for MMPose models

Usage:
    python tools/test.py configs/rtmpose_m_ap10k.py work_dirs/rtmpose_m/best.pth
    
    # Save predictions
    python tools/test.py configs/rtmpose_m_ap10k.py work_dirs/rtmpose_m/best.pth --out results.pkl
"""

import argparse
import os

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Test a pose model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out', 
        help='output result file in pickle format')
    parser.add_argument(
        '--dump',
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the predictions')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize every N images')
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
        cfg.work_dir = './work_dirs'

    cfg.load_from = args.checkpoint

    # Show or save visualizations
    if args.show or args.show_dir:
        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = args.show
        cfg.default_hooks.visualization.interval = args.interval
        if args.show_dir:
            cfg.default_hooks.visualization.out_dir = args.show_dir

    # Build the runner
    runner = Runner.from_cfg(cfg)

    # Start testing
    metrics = runner.test()
    
    # Print results
    print('\n' + '='*50)
    print('Evaluation Results:')
    print('='*50)
    for key, value in metrics.items():
        print(f'{key}: {value:.4f}')
    
    return metrics


if __name__ == '__main__':
    main()
