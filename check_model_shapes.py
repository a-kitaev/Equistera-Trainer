"""
Check and compare output shapes of PTH and ONNX models.

Usage:
    python check_model_shapes.py --pth checkpoints/best_coco_AP_epoch_210.pth --config configs/rtmpose_m_ap10k.py
    python check_model_shapes.py --onnx work_dirs/model.onnx
    python check_model_shapes.py --pth checkpoints/best_coco_AP_epoch_210.pth --config configs/rtmpose_m_ap10k.py --onnx work_dirs/model.onnx --compare
"""
import torch
import numpy as np
import argparse
import os
from pathlib import Path


def check_pth_model(config_file, checkpoint_file, input_shape=(1, 3, 256, 256)):
    """Check PTH model output shape"""
    print(f"\n{'='*60}")
    print(f"  Checking PTH Model")
    print(f"{'='*60}\n")
    
    print(f"📋 Config: {config_file}")
    print(f"📦 Checkpoint: {checkpoint_file}")
    
    # Check if files exist
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return None
    if not os.path.exists(checkpoint_file):
        print(f"❌ Checkpoint file not found: {checkpoint_file}")
        return None
    
    try:
        # Load model
        print(f"\n🔄 Loading PyTorch model...")
        from mmpose.apis import init_model
        from mmpose.structures import PoseDataSample
        import numpy as np
        
        model = init_model(config_file, checkpoint_file, device='cpu')
        model.eval()
        print("✅ Model loaded successfully")
        
        # Create dummy input
        print(f"\n🔄 Creating dummy input with shape: {input_shape}")
        dummy_input = torch.randn(*input_shape)
        
        # Get model config for flip_indices
        num_keypoints = model.head.out_channels if hasattr(model.head, 'out_channels') else 26
        
        # Create dummy data_samples for MMPose with required metainfo
        from mmengine.structures import InstanceData
        
        data_samples = []
        for _ in range(input_shape[0]):
            sample = PoseDataSample()
            # Add required metainfo
            sample.set_metainfo({
                'flip_indices': list(range(num_keypoints)),  # Identity mapping (no flip)
                'input_size': (input_shape[2], input_shape[3]),
                'input_center': np.array([input_shape[3] / 2, input_shape[2] / 2]),
                'input_scale': np.array([input_shape[3], input_shape[2]]),
            })
            # Add gt_instances with required fields
            gt_instances = InstanceData()
            gt_instances.bboxes = np.array([[0, 0, input_shape[3], input_shape[2]]])  # Full image bbox
            gt_instances.bbox_scores = np.array([1.0])
            sample.gt_instances = gt_instances
            data_samples.append(sample)
        
        # Run inference in test mode
        print(f"\n🔄 Running inference...")
        with torch.no_grad():
            # Use test_step which handles the full inference pipeline
            output = model.test_step({'inputs': dummy_input, 'data_samples': data_samples})
        
        print(f"✅ Inference complete")
        
        # Analyze output - MMPose returns list of PoseDataSample
        print(f"\n📊 Output Analysis:")
        print(f"   Output type: {type(output)}")
        
        results = {}
        
        if isinstance(output, (tuple, list)):
            print(f"   Number of samples: {len(output)}")
            
            # Check if it's PoseDataSample objects
            if len(output) > 0 and hasattr(output[0], 'pred_instances'):
                print(f"   Output format: PoseDataSample objects")
                
                # Extract predictions from first sample
                pred = output[0].pred_instances
                print(f"\n   Available tensor fields:")
                
                # Check specific known fields
                for field in ['keypoints', 'keypoint_scores', 'keypoints_visible', 'bboxes', 'bbox_scores']:
                    try:
                        val = getattr(pred, field, None)
                        if val is not None and isinstance(val, (torch.Tensor, np.ndarray)):
                            if isinstance(val, np.ndarray):
                                val = torch.from_numpy(val)
                            print(f"\n   Field '{field}':")
                            print(f"      Shape: {val.shape}")
                            print(f"      Dtype: {val.dtype}")
                            if val.numel() > 0:
                                print(f"      Range: [{val.min().item():.4f}, {val.max().item():.4f}]")
                            results[field] = {
                                'shape': list(val.shape),
                                'dtype': str(val.dtype),
                                'min': float(val.min().item()) if val.numel() > 0 else 0,
                                'max': float(val.max().item()) if val.numel() > 0 else 0
                            }
                    except Exception as e:
                        pass
            else:
                # Handle as list of tensors
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        print(f"\n   Output {i}:")
                        print(f"      Shape: {out.shape}")
                        print(f"      Dtype: {out.dtype}")
                        print(f"      Range: [{out.min().item():.4f}, {out.max().item():.4f}]")
                        results[f'output_{i}'] = {
                            'shape': list(out.shape),
                            'dtype': str(out.dtype),
                            'min': float(out.min().item()),
                            'max': float(out.max().item())
                        }
                    else:
                        print(f"\n   Output {i}: {type(out)}")
        elif isinstance(output, torch.Tensor):
            print(f"   Output shape: {output.shape}")
            print(f"   Output dtype: {output.dtype}")
            print(f"   Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            results['output'] = {
                'shape': list(output.shape),
                'dtype': str(output.dtype),
                'min': float(output.min().item()),
                'max': float(output.max().item())
            }
        elif isinstance(output, dict):
            print(f"   Number of outputs: {len(output)}")
            for key, out in output.items():
                if isinstance(out, torch.Tensor):
                    print(f"\n   Output '{key}':")
                    print(f"      Shape: {out.shape}")
                    print(f"      Dtype: {out.dtype}")
                    print(f"      Range: [{out.min().item():.4f}, {out.max().item():.4f}]")
                    results[key] = {
                        'shape': list(out.shape),
                        'dtype': str(out.dtype),
                        'min': float(out.min().item()),
                        'max': float(out.max().item())
                    }
                else:
                    print(f"\n   Output '{key}': {type(out)}")
        
        return {
            'input_shape': input_shape,
            'outputs': results,
            'model_type': 'pth'
        }
        
    except Exception as e:
        print(f"\n❌ Error checking PTH model:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_onnx_model(onnx_file, input_shape=(1, 3, 256, 256)):
    """Check ONNX model output shape"""
    print(f"\n{'='*60}")
    print(f"  Checking ONNX Model")
    print(f"{'='*60}\n")
    
    print(f"📦 ONNX Model: {onnx_file}")
    
    # Check if file exists
    if not os.path.exists(onnx_file):
        print(f"❌ ONNX file not found: {onnx_file}")
        return None
    
    file_size_mb = os.path.getsize(onnx_file) / (1024 * 1024)
    print(f"📏 File size: {file_size_mb:.2f} MB")
    
    try:
        import onnx
        import onnxruntime as ort
        
        # Load ONNX model
        print(f"\n🔄 Loading ONNX model...")
        onnx_model = onnx.load(onnx_file)
        print("✅ Model loaded successfully")
        
        # Check opset version
        opset_version = onnx_model.opset_import[0].version
        print(f"📌 Opset version: {opset_version}")
        
        # Validate model
        print(f"\n🔄 Validating ONNX model...")
        try:
            onnx.checker.check_model(onnx_model)
            print("✅ Validation passed")
        except Exception as e:
            print(f"⚠️  Validation warning: {e}")
        
        # Print input/output info from graph
        print(f"\n📊 Model Graph Information:")
        print(f"\n   Inputs:")
        for input_tensor in onnx_model.graph.input:
            shape = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in input_tensor.type.tensor_type.shape.dim]
            print(f"      {input_tensor.name}: {shape}")
        
        print(f"\n   Outputs:")
        output_info = {}
        for output_tensor in onnx_model.graph.output:
            shape = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in output_tensor.type.tensor_type.shape.dim]
            print(f"      {output_tensor.name}: {shape}")
            output_info[output_tensor.name] = shape
        
        # Create ONNX Runtime session
        print(f"\n🔄 Creating ONNX Runtime session...")
        session = ort.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
        print("✅ Session created successfully")
        
        # Get input/output names and shapes
        input_name = session.get_inputs()[0].name
        input_metadata_shape = session.get_inputs()[0].shape
        print(f"\n📋 Runtime Input Info:")
        print(f"   Name: {input_name}")
        print(f"   Shape: {input_metadata_shape}")
        
        # Create dummy input
        print(f"\n🔄 Creating dummy input with shape: {input_shape}")
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Run inference
        print(f"\n🔄 Running inference...")
        outputs = session.run(None, {input_name: dummy_input})
        print(f"✅ Inference complete")
        
        # Analyze outputs
        print(f"\n📊 Runtime Output Analysis:")
        print(f"   Number of outputs: {len(outputs)}")
        
        results = {}
        for i, output in enumerate(outputs):
            output_name = session.get_outputs()[i].name
            print(f"\n   Output {i} ('{output_name}'):")
            print(f"      Shape: {output.shape}")
            print(f"      Dtype: {output.dtype}")
            print(f"      Range: [{output.min():.4f}, {output.max():.4f}]")
            results[output_name] = {
                'shape': list(output.shape),
                'dtype': str(output.dtype),
                'min': float(output.min()),
                'max': float(output.max())
            }
        
        return {
            'input_shape': input_shape,
            'opset_version': opset_version,
            'outputs': results,
            'model_type': 'onnx'
        }
        
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print(f"   Please install: pip install onnx onnxruntime")
        return None
    except Exception as e:
        print(f"\n❌ Error checking ONNX model:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_models(pth_result, onnx_result):
    """Compare PTH and ONNX model outputs"""
    print(f"\n{'='*60}")
    print(f"  Comparing Models")
    print(f"{'='*60}\n")
    
    if pth_result is None or onnx_result is None:
        print("❌ Cannot compare - one or both models failed to load")
        return
    
    print(f"📊 Comparison:")
    
    # Compare input shapes
    print(f"\n   Input Shapes:")
    print(f"      PTH:  {pth_result['input_shape']}")
    print(f"      ONNX: {onnx_result['input_shape']}")
    if pth_result['input_shape'] == onnx_result['input_shape']:
        print(f"      ✅ Input shapes match")
    else:
        print(f"      ❌ Input shapes differ!")
    
    # Compare number of outputs
    pth_outputs = pth_result['outputs']
    onnx_outputs = onnx_result['outputs']
    
    print(f"\n   Number of Outputs:")
    print(f"      PTH:  {len(pth_outputs)}")
    print(f"      ONNX: {len(onnx_outputs)}")
    if len(pth_outputs) == len(onnx_outputs):
        print(f"      ✅ Output count matches")
    else:
        print(f"      ⚠️  Output count differs")
    
    # Compare output shapes
    print(f"\n   Output Shapes:")
    
    # Try to match outputs by shape or name
    for pth_key, pth_info in pth_outputs.items():
        print(f"\n      PTH '{pth_key}':")
        print(f"         Shape: {pth_info['shape']}")
        print(f"         Range: [{pth_info['min']:.4f}, {pth_info['max']:.4f}]")
        
        # Try to find matching ONNX output
        matched = False
        for onnx_key, onnx_info in onnx_outputs.items():
            if pth_info['shape'] == onnx_info['shape']:
                print(f"      ONNX '{onnx_key}':")
                print(f"         Shape: {onnx_info['shape']}")
                print(f"         Range: [{onnx_info['min']:.4f}, {onnx_info['max']:.4f}]")
                print(f"         ✅ Shapes match")
                matched = True
                break
        
        if not matched:
            print(f"         ⚠️  No matching ONNX output found")
    
    # List any unmatched ONNX outputs
    for onnx_key, onnx_info in onnx_outputs.items():
        already_matched = False
        for pth_info in pth_outputs.values():
            if pth_info['shape'] == onnx_info['shape']:
                already_matched = True
                break
        if not already_matched:
            print(f"\n      ONNX '{onnx_key}' (unmatched):")
            print(f"         Shape: {onnx_info['shape']}")
            print(f"         Range: [{onnx_info['min']:.4f}, {onnx_info['max']:.4f}]")


def main():
    parser = argparse.ArgumentParser(description='Check and compare PTH and ONNX model output shapes')
    parser.add_argument('--pth', type=str, help='Path to PTH checkpoint file')
    parser.add_argument('--config', type=str, help='Path to config file (required with --pth)')
    parser.add_argument('--onnx', type=str, help='Path to ONNX model file')
    parser.add_argument('--compare', action='store_true', help='Compare PTH and ONNX outputs')
    parser.add_argument('--input-shape', type=str, default='1,3,256,256', 
                       help='Input shape as comma-separated values (default: 1,3,256,256)')
    
    args = parser.parse_args()
    
    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # Check if at least one model is specified
    if not args.pth and not args.onnx:
        print("❌ Error: Please specify at least one model (--pth or --onnx)")
        parser.print_help()
        return 1
    
    # Check PTH model requirements
    if args.pth and not args.config:
        print("❌ Error: --config is required when using --pth")
        parser.print_help()
        return 1
    
    pth_result = None
    onnx_result = None
    
    # Check PTH model
    if args.pth:
        pth_result = check_pth_model(args.config, args.pth, input_shape)
    
    # Check ONNX model
    if args.onnx:
        onnx_result = check_onnx_model(args.onnx, input_shape)
    
    # Compare if requested
    if args.compare and args.pth and args.onnx:
        compare_models(pth_result, onnx_result)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}\n")
    
    if pth_result:
        print(f"✅ PTH model check complete")
        print(f"   Input: {pth_result['input_shape']}")
        print(f"   Outputs: {len(pth_result['outputs'])}")
    
    if onnx_result:
        print(f"✅ ONNX model check complete")
        print(f"   Input: {onnx_result['input_shape']}")
        print(f"   Outputs: {len(onnx_result['outputs'])}")
        print(f"   Opset: {onnx_result['opset_version']}")
    
    return 0


if __name__ == '__main__':
    exit(main())
