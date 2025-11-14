#!/usr/bin/env python
"""
Quick experiment launcher for hyperparameter tuning

Usage:
    python tools/run_experiments.py --config configs/rtmpose_m_ap10k.py
"""

import argparse
import os
import subprocess
import time
from itertools import product


def parse_args():
    parser = argparse.ArgumentParser(description='Run hyperparameter experiments')
    parser.add_argument('--config', required=True, help='base config file')
    parser.add_argument('--work-dir-prefix', default='work_dirs/exp', 
                       help='prefix for experiment directories')
    parser.add_argument('--dry-run', action='store_true',
                       help='print commands without running')
    return parser.parse_args()


def run_experiment(config, work_dir, lr, batch_size, dry_run=False):
    """Run a single experiment."""
    
    cmd = [
        'python', 'tools/train.py',
        config,
        '--work-dir', work_dir,
        '--cfg-options',
        f'train_dataloader.batch_size={batch_size}',
        f'optim_wrapper.optimizer.lr={lr}',
    ]
    
    print(f'\n{"="*70}')
    print(f'Running experiment: {work_dir}')
    print(f'LR: {lr}, Batch size: {batch_size}')
    print(f'{"="*70}\n')
    
    if dry_run:
        print('Command:', ' '.join(cmd))
        return 0
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f'Error running experiment: {e}')
        return e.returncode


def main():
    args = parse_args()
    
    # Define hyperparameter grid
    # Modify these based on your needs
    learning_rates = [0.002, 0.004, 0.008]
    batch_sizes = [16, 32]
    
    print(f'Running {len(learning_rates) * len(batch_sizes)} experiments...')
    
    # Run experiments
    experiment_id = 0
    for lr, batch_size in product(learning_rates, batch_sizes):
        experiment_id += 1
        work_dir = f'{args.work_dir_prefix}_{experiment_id}_lr{lr}_bs{batch_size}'
        
        run_experiment(
            config=args.config,
            work_dir=work_dir,
            lr=lr,
            batch_size=batch_size,
            dry_run=args.dry_run
        )
        
        if not args.dry_run:
            time.sleep(5)  # Brief pause between experiments
    
    print('\n' + '='*70)
    print('All experiments completed!')
    print('='*70)


if __name__ == '__main__':
    main()
