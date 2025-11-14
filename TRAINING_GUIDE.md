# Equistera Trainer - Complete Training Guide

## Table of Contents
1. [Setup](#setup)
2. [Dataset Preparation](#dataset-preparation)
3. [Training Strategy](#training-strategy)
4. [Model Configurations](#model-configurations)
5. [Monitoring](#monitoring)
6. [Troubleshooting](#troubleshooting)

---

## Setup

### 1. System Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM recommended
- 50GB+ free disk space

### 2. Quick Setup
```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

### 3. Manual Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR: venv\Scripts\activate  # On Windows

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install MMPose from source
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -e .
cd ..

# Download pretrained weights
python tools/download_checkpoints.py
```

---

## Dataset Preparation

### 1. Dataset Structure
Your dataset should be in COCO format. See `data/README.md` for details.

```
data/
├── annotations/
│   ├── train.json
│   └── val.json
└── images/
    ├── train/
    └── val/
```

### 2. Convert Your Dataset
```bash
# Modify tools/convert_dataset.py to match your data format
python tools/convert_dataset.py --input data/raw --output data/annotations
```

### 3. Verify Dataset
```bash
python tools/verify_dataset.py --ann-file data/annotations/train.json
```

---

## Training Strategy

### Overview
We implement a **layer-wise learning rate strategy** optimized for small datasets (800 images):

1. **Freeze**: Early feature extraction layers (blocks 1-2)
2. **Fine-tune** (LR=0.0001): Middle layers (part detectors)
3. **Train** (LR=0.001): New 26-keypoint head

### Phase 1: Initial Training (Epochs 1-100)
- Focus on head adaptation
- Frozen backbone stages 1-2
- Aggressive data augmentation

### Phase 2: Fine-tuning (Epochs 100-200)
- Gradually unfreeze middle layers
- Balanced augmentation
- Monitor for overfitting

### Phase 3: Full Training (Epochs 200-300)
- All layers trainable (except frozen early blocks)
- Reduced learning rate
- Focus on convergence

---

## Model Configurations

### 1. RTMPose-M (Recommended for Speed)

**Characteristics:**
- Fast inference (~50 FPS)
- Good accuracy
- Suitable for real-time applications

**Training:**
```bash
python tools/train.py configs/rtmpose_m_ap10k.py \
    --work-dir work_dirs/rtmpose_m \
    --amp  # Enable mixed precision
```

**Key Parameters:**
- Base LR: 0.004
- Batch size: 16
- Input size: 256x256
- Training epochs: 300

### 2. HRNet-W32 on AP-10K (Best Accuracy)

**Characteristics:**
- High accuracy
- Slower inference (~30 FPS)
- Better keypoint localization

**Training:**
```bash
python tools/train.py configs/hrnet_w32_ap10k.py \
    --work-dir work_dirs/hrnet_ap10k
```

**Key Parameters:**
- Base LR: 0.0005
- Batch size: 16
- Input size: 256x256
- Training epochs: 300

### 3. HRNet-W32 on AnimalPose

**Training:**
```bash
python tools/train.py configs/hrnet_w32_animalpose.py \
    --work-dir work_dirs/hrnet_animalpose
```

---

## Advanced Training Options

### Multi-GPU Training
```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    tools/train.py configs/rtmpose_m_ap10k.py \
    --launcher pytorch
```

### Resume Training
```bash
python tools/train.py configs/rtmpose_m_ap10k.py \
    --work-dir work_dirs/rtmpose_m \
    --resume
```

### Custom Configuration
```bash
python tools/train.py configs/rtmpose_m_ap10k.py \
    --work-dir work_dirs/rtmpose_m \
    --cfg-options train_dataloader.batch_size=32 \
                  optim_wrapper.optimizer.lr=0.002
```

---

## Monitoring

### TensorBoard
```bash
tensorboard --logdir work_dirs/
```
Open http://localhost:6006

### Training Analysis
```bash
# Monitor single experiment
python tools/monitor_training.py --work-dir work_dirs/rtmpose_m

# Compare multiple experiments
python tools/monitor_training.py --compare work_dirs/rtmpose_m work_dirs/hrnet_ap10k
```

### Weights & Biases (Optional)
Uncomment in `configs/_base_/default_runtime.py`:
```python
dict(type='WandbVisBackend', 
     init_kwargs=dict(project='horse-pose', name='experiment'))
```

---

## Evaluation

### Test Model
```bash
python tools/test.py configs/rtmpose_m_ap10k.py \
    work_dirs/rtmpose_m/best.pth
```

### Visualize Predictions
```bash
# Single image
python tools/visualize.py \
    --config configs/rtmpose_m_ap10k.py \
    --checkpoint work_dirs/rtmpose_m/best.pth \
    --img data/images/test/sample.jpg \
    --out-file output.jpg

# Multiple images
python tools/visualize.py \
    --config configs/rtmpose_m_ap10k.py \
    --checkpoint work_dirs/rtmpose_m/best.pth \
    --img-dir data/images/test/ \
    --out-dir visualizations/
```

---

## Data Augmentation

The training pipeline uses **aggressive augmentation** optimized for small datasets:

### Geometric Transformations
- Random rotation: ±40°
- Random scaling: 0.7-1.3x
- Random shift: ±16% of image size
- Random horizontal flip: 50%

### Photometric Transformations
- Brightness: ±32 (on 0-255 scale)
- Contrast: 0.5-1.5x
- Saturation: 0.5-1.5x
- Hue: ±18°

### Advanced Augmentations
- Gaussian blur (p=0.1)
- Gaussian noise (p=0.1)
- Coarse dropout (p=0.1)
- Random occlusion (p=0.2)

See `tools/augmentation.py` for implementation details.

---

## Expected Performance

### With 800 Images
| Model | AP | AP@50 | AP@75 | Training Time |
|-------|-----|-------|-------|---------------|
| RTMPose-M | 0.75-0.80 | 0.92-0.95 | 0.80-0.85 | ~8 hours (V100) |
| HRNet-W32 | 0.78-0.83 | 0.94-0.96 | 0.82-0.87 | ~12 hours (V100) |

### With 5000 Images (Target)
| Model | AP | AP@50 | AP@75 | Training Time |
|-------|-----|-------|-------|---------------|
| RTMPose-M | 0.83-0.87 | 0.95-0.97 | 0.87-0.90 | ~24 hours (V100) |
| HRNet-W32 | 0.85-0.89 | 0.96-0.98 | 0.89-0.92 | ~36 hours (V100) |

*Note: Performance varies based on data quality and diversity*

---

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python tools/train.py configs/rtmpose_m_ap10k.py \
    --cfg-options train_dataloader.batch_size=8
```

### Slow Training
```bash
# Enable mixed precision
python tools/train.py configs/rtmpose_m_ap10k.py --amp

# Reduce num_workers if CPU is bottleneck
--cfg-options train_dataloader.num_workers=2
```

### Model Not Converging
1. Check learning rate (try reducing by 10x)
2. Verify dataset quality
3. Reduce augmentation strength
4. Increase warmup epochs

### Overfitting (High train AP, low val AP)
1. Increase augmentation strength
2. Add dropout/regularization
3. Collect more diverse data
4. Use stronger weight decay

---

## Tips for Best Results

### 1. Data Quality
- Ensure consistent keypoint annotation
- Cover diverse poses and viewing angles
- Include challenging examples (occlusion, etc.)

### 2. Training
- Start with pretrained weights
- Use cosine annealing LR schedule
- Monitor both training and validation metrics
- Save checkpoints frequently

### 3. Augmentation
- Start with aggressive augmentation
- Adjust based on validation performance
- Ensure augmentation preserves keypoint relationships

### 4. Evaluation
- Evaluate on diverse test set
- Check per-keypoint accuracy
- Identify problematic keypoints for improvement

---

## Next Steps

Once you expand to 5000 images:

1. **Retrain from scratch** or fine-tune existing models
2. **Increase batch size** to 32-64 for better gradient estimates
3. **Extend training** to 500 epochs
4. **Experiment with larger models** (RTMPose-L, HRNet-W48)
5. **Implement ensemble methods** for production

---

## Citation

If you use this code, please cite:

```bibtex
@misc{equistera2025,
  title={Horse Pose Estimation with 26 Keypoints},
  author={Equistera Trainer},
  year={2025}
}
```

## License

See LICENSE file for details.

## Support

For issues and questions:
- Check the [Troubleshooting](#troubleshooting) section
- Review MMPose documentation: https://mmpose.readthedocs.io/
- Open an issue on GitHub
