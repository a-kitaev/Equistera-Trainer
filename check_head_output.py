"""
Check raw model head output (before post-processing) for PTH model.
This shows the actual SimCC outputs (simcc_x, simcc_y) from the head.

Usage:
    python check_head_output.py --pth checkpoints/best_coco_AP_epoch_210.pth --config configs/rtmpose_m_ap10k.py
"""
import torch
import argparse
import os


def check_head_output(config_file, checkpoint_file, input_shape=(1, 3, 256, 256)):
    """Check raw head output from PTH model"""
    print(f"\n{'='*60}")
    print(f"  Checking Raw Model Head Output")
    print(f"{'='*60}\n")
    
    print(f"📋 Config: {config_file}")
    print(f"📦 Checkpoint: {checkpoint_file}")
    
    try:
        # Load model
        print(f"\n🔄 Loading PyTorch model...")
        from mmpose.apis import init_model
        model = init_model(config_file, checkpoint_file, device='cpu')
        model.eval()
        print("✅ Model loaded successfully")
        
        # Create dummy input
        print(f"\n🔄 Creating dummy input with shape: {input_shape}")
        dummy_input = torch.randn(*input_shape)
        
        # Run backbone + head (no post-processing)
        print(f"\n🔄 Running forward pass (backbone + head)...")
        with torch.no_grad():
            # Extract features
            feats = model.backbone(dummy_input)
            print(f"\n📊 Backbone output:")
            if isinstance(feats, (tuple, list)):
                for i, feat in enumerate(feats):
                    print(f"   Feature {i}: {feat.shape}")
            else:
                print(f"   Features: {feats.shape}")
            
            # Run head forward
            head_output = model.head.forward(feats)
            
        print(f"\n✅ Forward pass complete")
        
        # Analyze head output
        print(f"\n📊 Head Output Analysis:")
        print(f"   Output type: {type(head_output)}")
        
        if isinstance(head_output, (tuple, list)):
            print(f"   Number of outputs: {len(head_output)}")
            for i, out in enumerate(head_output):
                if isinstance(out, torch.Tensor):
                    print(f"\n   Output {i} (likely simcc_{'x' if i == 0 else 'y'}):")
                    print(f"      Shape: {out.shape}")
                    print(f"      Dtype: {out.dtype}")
                    print(f"      Range: [{out.min().item():.4f}, {out.max().item():.4f}]")
                    print(f"      Mean: {out.mean().item():.4f}")
                    print(f"      Std: {out.std().item():.4f}")
        elif isinstance(out, torch.Tensor):
            print(f"   Shape: {head_output.shape}")
            print(f"   Dtype: {head_output.dtype}")
            print(f"   Range: [{head_output.min().item():.4f}, {head_output.max().item():.4f}]")
        
        # Model info
        print(f"\n📋 Model Info:")
        print(f"   Number of keypoints: {model.head.out_channels}")
        print(f"   Input size: {model.head.input_size}")
        print(f"   Feature map size: {model.head.in_featuremap_size}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Check raw model head output shapes')
    parser.add_argument('--pth', type=str, required=True, help='Path to PTH checkpoint file')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--input-shape', type=str, default='1,3,256,256', 
                       help='Input shape as comma-separated values (default: 1,3,256,256)')
    
    args = parser.parse_args()
    
    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # Check model
    success = check_head_output(args.config, args.pth, input_shape)
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
