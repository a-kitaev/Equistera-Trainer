#!/usr/bin/env python
"""
Training monitoring and analysis tools

Usage:
    # Monitor training progress
    python tools/monitor_training.py --work-dir work_dirs/rtmpose_m
    
    # Compare multiple experiments
    python tools/monitor_training.py --compare work_dirs/rtmpose_m work_dirs/hrnet_ap10k
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument('--work-dir', help='work directory to monitor')
    parser.add_argument(
        '--compare',
        nargs='+',
        help='compare multiple work directories')
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=['loss', 'AP'],
        help='metrics to plot')
    parser.add_argument(
        '--out-dir',
        default='analysis',
        help='output directory for plots')
    return parser.parse_args()


def parse_log_file(log_file):
    """Parse training log file."""
    
    metrics = {
        'epoch': [],
        'iter': [],
        'loss': [],
        'lr': [],
        'time': [],
    }
    
    val_metrics = {
        'epoch': [],
        'AP': [],
        'AR': [],
    }
    
    if not os.path.exists(log_file):
        print(f'Log file not found: {log_file}')
        return metrics, val_metrics
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'Epoch' in line and 'loss' in line:
                # Parse training metrics
                # Format: Epoch [1][10/50] loss: 0.123, lr: 0.001
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'loss:':
                            metrics['loss'].append(float(parts[i+1].rstrip(',')))
                        elif part == 'lr:':
                            metrics['lr'].append(float(parts[i+1].rstrip(',')))
                except:
                    pass
            
            elif 'Evaluating' in line or 'coco/AP' in line:
                # Parse validation metrics
                try:
                    if 'coco/AP' in line:
                        parts = line.split('coco/AP:')
                        if len(parts) > 1:
                            ap_value = float(parts[1].strip().split()[0])
                            val_metrics['AP'].append(ap_value)
                except:
                    pass
    
    return metrics, val_metrics


def plot_training_curves(work_dirs, metrics, out_dir):
    """Plot training curves."""
    
    os.makedirs(out_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16)
    
    for work_dir in work_dirs:
        name = Path(work_dir).name
        log_file = os.path.join(work_dir, 'vis_data', 'scalars.json')
        
        # Try alternative log locations
        if not os.path.exists(log_file):
            log_file = os.path.join(work_dir, 'tf_logs', 'events.out.tfevents.*')
        
        train_metrics, val_metrics = parse_log_file(log_file)
        
        # Plot loss
        if train_metrics['loss']:
            axes[0, 0].plot(train_metrics['loss'], label=name, alpha=0.7)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot learning rate
        if train_metrics['lr']:
            axes[0, 1].plot(train_metrics['lr'], label=name, alpha=0.7)
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('LR')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # Plot validation AP
        if val_metrics['AP']:
            axes[1, 0].plot(val_metrics['AP'], label=name, marker='o', alpha=0.7)
        axes[1, 0].set_title('Validation AP')
        axes[1, 0].set_xlabel('Validation Step')
        axes[1, 0].set_ylabel('AP')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics
        if val_metrics['AP']:
            best_ap = max(val_metrics['AP'])
            latest_ap = val_metrics['AP'][-1]
            axes[1, 1].bar([name], [latest_ap], alpha=0.7, label=f'Latest: {latest_ap:.3f}')
            axes[1, 1].axhline(y=best_ap, linestyle='--', alpha=0.5)
    
    axes[1, 1].set_title('Latest Validation AP')
    axes[1, 1].set_ylabel('AP')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_file = os.path.join(out_dir, 'training_curves.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f'Saved training curves to {out_file}')
    
    plt.close()


def print_summary(work_dirs):
    """Print training summary."""
    
    print('\n' + '='*70)
    print('TRAINING SUMMARY')
    print('='*70)
    
    for work_dir in work_dirs:
        name = Path(work_dir).name
        print(f'\n{name}:')
        print('-' * 70)
        
        # Check for checkpoints
        ckpt_dir = Path(work_dir)
        if ckpt_dir.exists():
            ckpts = list(ckpt_dir.glob('*.pth'))
            print(f'  Checkpoints: {len(ckpts)}')
            
            if ckpts:
                latest_ckpt = max(ckpts, key=os.path.getmtime)
                print(f'  Latest: {latest_ckpt.name}')
        
        # Check for logs
        log_file = ckpt_dir / 'vis_data' / 'scalars.json'
        if log_file.exists():
            train_metrics, val_metrics = parse_log_file(str(log_file))
            
            if val_metrics['AP']:
                best_ap = max(val_metrics['AP'])
                latest_ap = val_metrics['AP'][-1]
                print(f'  Best AP: {best_ap:.4f}')
                print(f'  Latest AP: {latest_ap:.4f}')
    
    print('\n' + '='*70 + '\n')


def main():
    args = parse_args()
    
    if args.compare:
        work_dirs = args.compare
    elif args.work_dir:
        work_dirs = [args.work_dir]
    else:
        # Auto-detect work directories
        work_dirs = [str(p) for p in Path('work_dirs').glob('*') if p.is_dir()]
    
    if not work_dirs:
        print('No work directories found!')
        return
    
    print(f'Monitoring {len(work_dirs)} experiment(s)...')
    
    # Print summary
    print_summary(work_dirs)
    
    # Plot curves
    plot_training_curves(work_dirs, args.metrics, args.out_dir)


if __name__ == '__main__':
    main()
