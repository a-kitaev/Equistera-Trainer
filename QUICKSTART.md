# 🐴 Equistera Trainer - Quick Start Guide

Welcome! This guide will get you training MMPose models for horse pose estimation in minutes.

## 📋 What You Have

- **26-keypoint schema** for horses (detailed leg anatomy)
- **3 model configs**: RTMPose-M, HRNet-W32 (2 variants)
- **Layer-wise learning rates** (freeze early layers, fine-tune middle, train head)
- **Aggressive augmentation** optimized for small datasets (800 images)
- **Complete training pipeline** with monitoring and evaluation tools

---

## 🚀 5-Minute Setup

### For Azure VM (Ubuntu 22.04) - **Recommended**

```bash
# On your local machine - deploy to Azure
./deploy_to_azure.sh <vm-ip> azureuser

# SSH into VM
ssh azureuser@<vm-ip>
cd ~/equistera-trainer

# Run Azure setup
./setup_azure.sh

# Upload dataset (from local machine in another terminal)
scp -r data/annotations data/images azureuser@<vm-ip>:~/equistera-trainer/data/
```

See **[AZURE_DEPLOYMENT.md](AZURE_DEPLOYMENT.md)** for complete guide.

### For Local Development (macOS/Linux)

```bash
# Option A: Automated setup
chmod +x setup.sh
./setup.sh

# Option B: Using Make
make setup

# Option C: Manual setup
python -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Step 2: Prepare Your Dataset (variable time)

Your dataset needs to be in **COCO format**:

```json
{
  "images": [{"id": 1, "file_name": "horse_001.jpg", "width": 640, "height": 480}],
  "annotations": [{
    "id": 1, 
    "image_id": 1,
    "keypoints": [x1, y1, v1, x2, y2, v2, ...],  // 26 keypoints × 3 = 78 values
    "bbox": [x, y, width, height]
  }],
  "categories": [{
    "id": 1,
    "name": "horse",
    "keypoints": ["nose", "l_eye", "r_eye", ...]
  }]
}
```

**Convert your data:**
```bash
python tools/convert_dataset.py --input data/raw --output data/annotations
```

**Verify quality:**
```bash
make verify-data
# or
python tools/verify_dataset.py --ann-file data/annotations/train.json
```

### Step 3: Download Pretrained Weights (1 minute)

```bash
make download-ckpts
# or
python tools/download_checkpoints.py
```

---

## 🎯 Start Training (Choose Your Model)

### Option 1: RTMPose-M (Recommended - Fast & Accurate)

```bash
# Simple command
make train-rtm

# Full command
python tools/train.py configs/rtmpose_m_ap10k.py --work-dir work_dirs/rtmpose_m

# With mixed precision (faster)
make train-rtm-amp
```

**Expected time:** ~8 hours on V100 GPU

### Option 2: HRNet-W32 (Higher Accuracy)

```bash
# For AP-10K dataset
make train-hrnet-ap10k

# For AnimalPose dataset
make train-hrnet-animal
```

**Expected time:** ~12 hours on V100 GPU

### Multi-GPU Training

```bash
make train-rtm-multi
# or
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py configs/rtmpose_m_ap10k.py
```

---

## 📊 Monitor Training

### TensorBoard (Real-time Monitoring)

```bash
make tensorboard
# or
tensorboard --logdir work_dirs/
```

Open http://localhost:6006 in your browser

### Training Analysis

```bash
# Compare all experiments
make monitor

# Specific experiment
python tools/monitor_training.py --work-dir work_dirs/rtmpose_m
```

---

## 🧪 Evaluate & Visualize

### Test Your Model

```bash
make test-rtm
# or
python tools/test.py configs/rtmpose_m_ap10k.py work_dirs/rtmpose_m/best.pth
```

### Export to ONNX

Export your trained model to ONNX format for deployment:

```bash
python tools/export_onnx_opset21.py
```

### Test ONNX Model

Test the exported ONNX model on your test dataset:

```bash
# Basic testing
python tools/test_onnx.py \
    --onnx work_dirs/rtmpose_m_horse_opset17.onnx \
    --ann data/annotations/horse_test.json \
    --img-dir data/test

# With visualizations
python tools/test_onnx.py \
    --onnx work_dirs/rtmpose_m_horse_opset17.onnx \
    --ann data/annotations/horse_test.json \
    --img-dir data/test \
    --show-dir visualizations/onnx_test
```

### Visualize Predictions

```bash
# Single image
python tools/visualize.py \
    --config configs/rtmpose_m_ap10k.py \
    --checkpoint work_dirs/rtmpose_m/best.pth \
    --img path/to/horse.jpg \
    --out-file result.jpg

# Batch processing
make visualize
# or
python tools/visualize.py \
    --config configs/rtmpose_m_ap10k.py \
    --checkpoint work_dirs/rtmpose_m/best.pth \
    --img-dir data/images/test \
    --out-dir visualizations/
```

---

## 🎓 Understanding the Training Strategy

### Layer-wise Learning Rates

```
Backbone:
├── Stage 1-2 (Early features) ──> FROZEN (LR = 0)
├── Stage 2-3 (Part detectors)  ──> FINE-TUNE (LR = 0.0001)  
└── Stage 4 + Head (26 keypoints) ──> TRAIN (LR = 0.001)
```

**Why?** 
- Early layers learn generic features (edges, textures) - no need to retrain
- Middle layers detect body parts - needs adaptation to horses
- Head is completely new (26 keypoints) - needs full training

### Aggressive Augmentation

With only 800 images, we use **aggressive augmentation**:

- ✅ Rotation: ±40°
- ✅ Scaling: 0.7-1.3×
- ✅ Color jittering: Brightness, contrast, saturation
- ✅ Flip: Horizontal (50%)
- ✅ Noise & blur: 10-15%
- ✅ Random occlusion: 20%

This effectively multiplies your dataset size!

---

## 📈 Expected Performance

### With 800 Images (Current)

| Model | AP | AP@50 | AP@75 | Speed |
|-------|-----|-------|-------|-------|
| RTMPose-M | 0.75-0.80 | 0.92-0.95 | 0.80-0.85 | ~50 FPS |
| HRNet-W32 | 0.78-0.83 | 0.94-0.96 | 0.82-0.87 | ~30 FPS |

### With 5000 Images (Target)

| Model | AP | AP@50 | AP@75 | Speed |
|-------|-----|-------|-------|-------|
| RTMPose-M | 0.83-0.87 | 0.95-0.97 | 0.87-0.90 | ~50 FPS |
| HRNet-W32 | 0.85-0.89 | 0.96-0.98 | 0.89-0.92 | ~30 FPS |

---

## 🛠️ Common Tasks

```bash
# View all available commands
make help

# Verify dataset
make verify-data

# Train with custom batch size
python tools/train.py configs/rtmpose_m_ap10k.py \
    --cfg-options train_dataloader.batch_size=32

# Resume training
make resume-rtm

# Clean temporary files
make clean

# Project statistics
make stats
```

---

## ❓ Troubleshooting

### Out of Memory?
```bash
# Reduce batch size
python tools/train.py configs/rtmpose_m_ap10k.py \
    --cfg-options train_dataloader.batch_size=8
```

### Training too slow?
```bash
# Enable mixed precision
make train-rtm-amp

# Reduce data loading workers
--cfg-options train_dataloader.num_workers=2
```

### Model not converging?
1. Check learning rate (try reducing by 10×)
2. Verify dataset quality
3. Reduce augmentation strength
4. Increase warmup epochs

### Overfitting?
1. Increase augmentation strength
2. Add more dropout
3. Collect more diverse data
4. Use stronger weight decay

---

## 📚 Key Files

| File | Purpose |
|------|---------|
| `configs/rtmpose_m_ap10k.py` | RTMPose-M configuration |
| `configs/hrnet_w32_ap10k.py` | HRNet-W32 configuration |
| `configs/_base_/datasets/horse_ap10k.py` | Dataset & augmentation |
| `tools/train.py` | Training script |
| `tools/test.py` | Evaluation script |
| `tools/visualize.py` | Visualization tool |
| `TRAINING_GUIDE.md` | Comprehensive guide |
| `Makefile` | Convenience commands |

---

## 🎯 Your Training Checklist

- [ ] ✅ Setup complete (`make setup`)
- [ ] 📂 Dataset in COCO format
- [ ] ✓ Dataset verified (`make verify-data`)
- [ ] ⬇️ Checkpoints downloaded (`make download-ckpts`)
- [ ] 🚀 Training started (`make train-rtm`)
- [ ] 📊 TensorBoard running (`make tensorboard`)
- [ ] 🧪 Model tested (`make test-rtm`)
- [ ] 🎨 Results visualized (`make visualize`)

---

## 💡 Pro Tips

1. **Start with RTMPose-M** - It's faster and gives good results
2. **Monitor early** - Check TensorBoard after 10 epochs
3. **Save checkpoints** - Use `--cfg-options default_hooks.checkpoint.interval=5`
4. **Test augmentation** - Visualize augmented samples first
5. **Compare models** - Train both and compare with `make monitor`

---

## 🚀 Next Steps

1. **Train baseline model** (RTMPose-M)
2. **Evaluate performance** on validation set
3. **Identify weak keypoints** (per-keypoint analysis)
4. **Collect more data** for challenging cases
5. **Retrain with 5k images** for production model

---

## 📞 Need Help?

- 📖 Read the full [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- 📊 Check [PROJECT_STATUS.md](PROJECT_STATUS.md) for progress
- 🔍 Review MMPose docs: https://mmpose.readthedocs.io/
- 🐛 Common issues in [TRAINING_GUIDE.md#troubleshooting](TRAINING_GUIDE.md#troubleshooting)

---

## 🎉 Ready to Start!

```bash
# Quick start in 3 commands
make setup
make verify-data
make train-rtm
```

**Good luck with your training!** 🐴✨

---

*Generated by Equistera Trainer v2.0 - January 2025*
