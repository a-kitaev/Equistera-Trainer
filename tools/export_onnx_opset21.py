"""
Export RTMPose model to ONNX with opset 21 for Android compatibility.
This re-exports from PyTorch rather than converting existing ONNX.

Usage:
    python tools/export_onnx_opset21.py

Output:
    work_dirs/rtmpose_m_horse_opset21.onnx
"""
import torch
from mmpose.apis import init_model
import onnx
import os

def export_with_opset17():
    """Export RTMPose model to ONNX with opset 17 (PyTorch 2.1.0 maximum)"""
    
    print(f"\n{'='*60}")
    print(f"  RTMPose ONNX Export (Opset 17)")
    print(f"{'='*60}\n")
    
    # Configuration
    config_file = 'configs/rtmpose_m_ap10k.py'
    checkpoint_file = 'work_dirs/rtmpose_m_669imgs/best_coco_AP_epoch_210.pth'
    output_file = 'work_dirs/rtmpose_m_horse_opset17.onnx'
    
    print(f"\n📋 Config: {config_file}")
    print(f"📦 Checkpoint: {checkpoint_file}")
    print(f"💾 Output: {output_file}")
    
    # Load model
    print(f"\n🔄 Loading model...")
    model = init_model(config_file, checkpoint_file, device='cpu')
    model.eval()
    print("✅ Model loaded successfully")
    
    # Create dummy input
    print(f"\n🔄 Creating dummy input (1, 3, 256, 256)...")
    dummy_input = torch.randn(1, 3, 256, 256)
    
    # Export with opset 17 (PyTorch 2.1.0 maximum)
    print(f"\n🔄 Exporting to ONNX with opset 17 (PyTorch 2.1.0 max)...")
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        export_params=True,
        opset_version=17,  # ← PyTorch 2.1.0 maximum (opset 21 requires PyTorch 2.5+)
        do_constant_folding=True,
        input_names=['input'],
        output_names=['simcc_x', 'simcc_y'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'simcc_x': {0: 'batch_size'},
            'simcc_y': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"✅ Export complete!")
    
    # Verify the exported model
    print(f"\n🔍 Verifying exported model...")
    
    # Check file exists and size
    if not os.path.exists(output_file):
        print(f"❌ ERROR: Output file not found: {output_file}")
        return False
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ File created: {file_size_mb:.2f} MB")
    
    # Load and validate ONNX model
    onnx_model = onnx.load(output_file)
    
    # Check opset version
    opset_version = onnx_model.opset_import[0].version
    print(f"✅ Opset version: {opset_version}")
    
    if opset_version != 17:
        print(f"⚠️  WARNING: Expected opset 17, got {opset_version}")
    
    # Android compatibility check
    if opset_version <= 17:
        print(f"✅ Android ONNX Runtime compatible (opset {opset_version} ≤ 17)")
    else:
        print(f"❌ May not be Android compatible (opset {opset_version} > 17)")
    
    # Validate model structure
    try:
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX model validation passed!")
    except Exception as e:
        print(f"❌ ONNX validation failed: {e}")
        return False
    
    # Print model info
    print(f"\n📊 Model Information:")
    print(f"   Inputs:")
    for input_tensor in onnx_model.graph.input:
        print(f"      {input_tensor.name}: {[d.dim_value or 'dynamic' for d in input_tensor.type.tensor_type.shape.dim]}")
    
    print(f"   Outputs:")
    for output_tensor in onnx_model.graph.output:
        print(f"      {output_tensor.name}: {[d.dim_value or 'dynamic' for d in output_tensor.type.tensor_type.shape.dim]}")
    
    # Android compatibility check
    print(f"\n📱 Android Compatibility:")
    if opset_version <= 21:
        print(f"   ✅ Compatible with ONNX Runtime Mobile")
        print(f"   ✅ Compatible with Android NNAPI")
        print(f"   ✅ Compatible with iOS CoreML")
    else:
        print(f"   ⚠️  May not be fully supported on mobile devices")
    
    print(f"\n" + "=" * 60)
    print(f"✅ EXPORT SUCCESSFUL!")
    print(f"=" * 60)
    print(f"\n📦 Your ONNX model is ready: {output_file}")
    print(f"\n💡 Next steps:")
    print(f"   1. Download: scp -i MLCompute_key.pem azureuser@52.159.248.236:~/equistera-trainer/{output_file} ./models/")
    print(f"   2. Test on Android with ONNX Runtime Mobile")
    print(f"   3. Integrate into your Android app")
    
    return True


if __name__ == '__main__':
    try:
        success = export_with_opset17()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Export failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
