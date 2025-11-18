# ONNX Model Testing Guide

This guide explains how to test ONNX exported models on COCO test datasets using the `test_onnx.py` script.

## Overview

The `tools/test_onnx.py` script enables comprehensive evaluation of ONNX models with standard COCO metrics. It's designed as a single-file solution that handles the complete testing pipeline from loading models to generating detailed reports.

## Features

- ✅ Single file solution - no complex setup required
- ✅ ONNX Runtime inference with CPU/GPU support
- ✅ Standard COCO metrics (AP, AP50, AP75, AP95, AR, etc.)
- ✅ Results saved to `Test/[modelname]/` directory
- ✅ Detailed reports with predictions and timing information

## Installation

The script requires ONNX Runtime and COCO API tools. Install dependencies:

```bash
pip install onnxruntime>=1.15.0 onnx>=1.14.0 pycocotools>=2.0.6
```

Or install all project requirements:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python tools/test_onnx.py <onnx_model> <config> \
    --test-dataset <test_json> \
    --image-dir <img_dir>
```

### Example

```bash
python tools/test_onnx.py work_dirs/rtmpose_m/model.onnx configs/rtmpose_m_ap10k.py \
    --test-dataset data/annotations/horse_test.json \
    --image-dir data/test/
```

### Custom Output Directory

```bash
python tools/test_onnx.py model.onnx config.py \
    --test-dataset test.json \
    --image-dir images/ \
    --output-dir results/my_model/
```

### Custom Input Size

```bash
python tools/test_onnx.py model.onnx config.py \
    --test-dataset test.json \
    --image-dir images/ \
    --input-size 384 384
```

## Output

The script creates the following structure:

```
Test/[modelname]/
├── predictions.json    # Model predictions in COCO format
├── metrics.json        # Evaluation metrics
└── report.txt         # Human-readable summary report
```

### Metrics Computed

The script computes the following COCO metrics:

- **AP** - Average Precision @ IoU=0.50:0.95 (primary metric)
- **AP50** - Average Precision @ IoU=0.50
- **AP75** - Average Precision @ IoU=0.75
- **AP95** - Average Precision @ IoU=0.95
- **AP_medium** - Average Precision for medium objects
- **AP_large** - Average Precision for large objects
- **AR** - Average Recall @ IoU=0.50:0.95
- **AR50** - Average Recall @ IoU=0.50
- **AR75** - Average Recall @ IoU=0.75
- **AR_medium** - Average Recall for medium objects
- **AR_large** - Average Recall for large objects

### Example Output

```
==============================================================
  EVALUATION RESULTS
==============================================================
  AP      (IoU=0.50:0.95): 0.7823
  AP50    (IoU=0.50):      0.9456
  AP75    (IoU=0.75):      0.8901
  AP95    (IoU=0.95):      0.8901
  AR      (IoU=0.50:0.95): 0.8234
  AR50    (IoU=0.50):      0.9567
  AR75    (IoU=0.75):      0.9012
==============================================================
```

## Requirements

### Model Requirements

- ONNX model file (`.onnx`)
- Model configuration file (`.py`)
- The ONNX model should output SimCC format (simcc_x, simcc_y)

### Dataset Requirements

- Test annotations in COCO format (`.json`)
- Test images directory
- Annotations should include bounding boxes and keypoints

### COCO Format Structure

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "keypoints": [...],
      "num_keypoints": 26
    }
  ],
  "categories": [...]
}
```

## Workflow

The script follows this workflow:

1. **Load Configuration** - Parse model config for dataset info
2. **Load Test Data** - Load COCO annotations and images
3. **Initialize Model** - Create ONNX Runtime session
4. **Run Inference** - Process all test images with bounding boxes
5. **Compute Metrics** - Evaluate predictions using COCO API
6. **Save Results** - Write predictions, metrics, and report

## Performance

The script includes timing information:

- Total inference time
- Average time per image
- This helps identify performance bottlenecks

## Troubleshooting

### Common Issues

**"Module not found: cv2"**
```bash
pip install opencv-python-headless
```

**"Module not found: onnxruntime"**
```bash
pip install onnxruntime
```

**"Image not found" warnings**
- Ensure `--image-dir` points to the correct directory
- Check that image paths in annotations match actual files

**"Failed to load image" warnings**
- Verify images are valid and not corrupted
- Check file permissions

### GPU Support

To use GPU acceleration with ONNX Runtime:

```bash
pip install onnxruntime-gpu
```

The script will automatically detect and use CUDA if available.

## Comparison with PyTorch Testing

| Feature | test.py (PyTorch) | test_onnx.py (ONNX) |
|---------|-------------------|---------------------|
| Model Format | .pth checkpoint | .onnx model |
| Dependencies | mmpose, torch | onnxruntime |
| Speed | Slower | Faster |
| Deployment | Research | Production-ready |
| Metrics | Full MMPose metrics | COCO metrics |

## Integration

The ONNX test script integrates with existing tools:

```bash
# 1. Train model
python tools/train.py configs/rtmpose_m_ap10k.py

# 2. Export to ONNX
python tools/export_onnx_opset21.py

# 3. Test ONNX model
python tools/test_onnx.py work_dirs/model.onnx configs/rtmpose_m_ap10k.py \
    --test-dataset data/annotations/horse_test.json \
    --image-dir data/test/
```

## Advanced Usage

### Batch Processing Multiple Models

```bash
#!/bin/bash
for model in work_dirs/*/model.onnx; do
    echo "Testing $model"
    python tools/test_onnx.py "$model" configs/rtmpose_m_ap10k.py \
        --test-dataset data/annotations/horse_test.json \
        --image-dir data/test/
done
```

### Custom Metrics Analysis

The script saves predictions in COCO format, allowing custom analysis:

```python
import json

# Load predictions
with open('Test/model/predictions.json', 'r') as f:
    predictions = json.load(f)

# Custom analysis
# ... your code here ...
```

## References

- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [COCO Evaluation Metrics](https://cocodataset.org/#keypoints-eval)
- [MMPose Documentation](https://mmpose.readthedocs.io/)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the example usage
3. Open an issue on the project repository
