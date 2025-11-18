# ONNX Model Testing Guide

This guide explains how to test ONNX exported models on test datasets using the `test_onnx.py` script.

## Overview

The `test_onnx.py` script enables you to:
- Test ONNX models on COCO-format test datasets
- Calculate standard pose estimation metrics (AP, AR)
- Visualize predictions
- Measure inference performance

## Prerequisites

Install required dependencies:
```bash
pip install onnx onnxruntime pycocotools
```

For GPU support, use `onnxruntime-gpu` instead:
```bash
pip install onnx onnxruntime-gpu pycocotools
```

## Quick Start

### 1. Export Your Model to ONNX

First, export a trained PyTorch model to ONNX format:
```bash
python tools/export_onnx_opset21.py
```

This creates an ONNX model at `work_dirs/rtmpose_m_horse_opset17.onnx`.

### 2. Test the ONNX Model

Run basic testing:
```bash
python tools/test_onnx.py \
    --onnx work_dirs/rtmpose_m_horse_opset17.onnx \
    --ann data/annotations/horse_test.json \
    --img-dir data/test
```

## Usage Examples

### Basic Testing
```bash
python tools/test_onnx.py \
    --onnx work_dirs/rtmpose_m_horse_opset17.onnx \
    --ann data/annotations/horse_test.json \
    --img-dir data/test
```

### With Visualizations
```bash
python tools/test_onnx.py \
    --onnx work_dirs/rtmpose_m_horse_opset17.onnx \
    --ann data/annotations/horse_test.json \
    --img-dir data/test \
    --show-dir visualizations/onnx_test
```

### Custom Input Size
```bash
python tools/test_onnx.py \
    --onnx work_dirs/model.onnx \
    --ann data/annotations/horse_test.json \
    --img-dir data/test \
    --input-size 384 384
```

### GPU Inference
```bash
python tools/test_onnx.py \
    --onnx work_dirs/model.onnx \
    --ann data/annotations/horse_test.json \
    --img-dir data/test \
    --device cuda
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--onnx` | Path to ONNX model file | Required |
| `--ann` | Path to COCO annotation file | Required |
| `--img-dir` | Directory containing test images | Required |
| `--input-size` | Input size (height width) | 256 256 |
| `--show-dir` | Directory to save visualizations | None |
| `--kpt-thr` | Keypoint score threshold for visualization | 0.3 |
| `--device` | Device to run inference (cpu/cuda) | cpu |
| `--batch-size` | Batch size for inference | 1 |

## Output Metrics

The script calculates and reports:
- **AP (Average Precision)**: Overall keypoint detection accuracy
- **AP@0.5**: AP at IoU threshold 0.5
- **AP@0.75**: AP at IoU threshold 0.75
- **AR (Average Recall)**: Maximum recall given 20 detections
- **Inference time**: Average time per image in milliseconds
- **FPS**: Frames per second throughput

Example output:
```
======================================================================
  Evaluation Results
======================================================================

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.732
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.891
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.798
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.687
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.776
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.781
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.912
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.834
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.723
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.819

======================================================================
  Summary
======================================================================

✅ Testing complete!
   Total images: 120
   Total predictions: 120
   AP: 0.7320
   AP@0.5: 0.8910
   AP@0.75: 0.7980
   AR: 0.7810
   Average inference time: 15.34 ± 2.11 ms
   FPS: 65.23
```

## Using Makefile Commands

For convenience, you can use the Makefile commands:

```bash
# Export model to ONNX
make export-onnx

# Test ONNX model
make test-onnx

# Test with visualizations
make test-onnx-vis
```

## Dataset Format

The script expects:
1. **Annotation file**: COCO-format JSON with keypoint annotations
2. **Image directory**: Directory containing test images referenced in annotations

Example annotation structure:
```json
{
  "images": [
    {"id": 1, "file_name": "horse001.jpg", "width": 640, "height": 480}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 400, 300],
      "keypoints": [x1, y1, v1, x2, y2, v2, ...],
      "num_keypoints": 26
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "horse",
      "keypoints": ["nose", "l_eye", "r_eye", ...],
      "skeleton": [[0, 1], [0, 2], ...]
    }
  ]
}
```

## Troubleshooting

### Model Loading Errors
```
Error: ONNX model not found
```
**Solution**: Run `make export-onnx` or `python tools/export_onnx_opset21.py` first.

### Missing Dependencies
```
ModuleNotFoundError: No module named 'onnxruntime'
```
**Solution**: Install dependencies: `pip install onnx onnxruntime`

### Image Not Found
```
⚠️  Image not found: data/test/horse001.jpg
```
**Solution**: Ensure the image directory contains all images referenced in the annotation file.

### Unexpected Output Format
```
⚠️  Unexpected number of outputs: 1
```
**Solution**: The script expects SimCC outputs (2 tensors). If your model uses a different format, you may need to modify the `decode_simcc_output` function.

## Comparing PyTorch vs ONNX Performance

You can compare PyTorch and ONNX model performance:

```bash
# Test PyTorch model
python tools/test.py configs/rtmpose_m_ap10k.py work_dirs/rtmpose_m/best.pth

# Test ONNX model
python tools/test_onnx.py --onnx work_dirs/rtmpose_m_horse_opset17.onnx \
    --ann data/annotations/horse_test.json --img-dir data/test
```

Both should produce similar metrics (within 1-2% due to minor numerical differences).

## Advanced Usage

### Custom Preprocessing

If you need to modify the preprocessing pipeline, edit the `preprocess_image` function in `test_onnx.py`:

```python
def preprocess_image(img, input_size, bbox=None):
    # Your custom preprocessing here
    ...
```

### Custom Output Decoding

For models with different output formats, modify the `decode_simcc_output` function or the output handling in the main testing loop.

## Performance Tips

1. **Use GPU**: Add `--device cuda` for faster inference (requires `onnxruntime-gpu`)
2. **Optimize model**: Use ONNX optimization tools like `onnxoptimizer`
3. **Quantization**: Consider INT8 quantization for faster inference on edge devices
4. **Batch inference**: Modify the script to process multiple images per batch

## Related Tools

- `tools/export_onnx_opset21.py` - Export PyTorch models to ONNX
- `tools/test.py` - Test PyTorch models
- `tools/visualize.py` - Visualize predictions
- `check_model_shapes.py` - Compare PyTorch and ONNX model outputs
