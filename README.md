# Equistera Trainer - Horse Pose Estimation Fine-tuning

This project fine-tunes MMPose models for 26-keypoint horse pose estimation.

## Models
- **RTMPose-M** on AP-10K dataset
- **HRNet-W32** on AP-10K dataset  
- **HRNet-W32** on AnimalPose dataset

## Dataset
- **Initial**: 800 images with 26 keypoints per horse
- **Target**: 5,000 images
- Custom keypoint schema with detailed leg anatomy

## Project Structure
```
.
├── 📄 README.md                    # Project overview
├── 📄 QUICKSTART.md               # 5-minute getting started guide
├── 📄 TRAINING_GUIDE.md           # Comprehensive training manual
├── 📄 DATASET_PREPARATION.md      # Dataset splitting guide
├── 📄 AZURE_DEPLOYMENT.md         # Complete Azure VM deployment guide
├── 📄 RTMPOSE_V2_GUIDE.md         # V2 enhancements (text + diffusion)
├── 📄 PROJECT_STATUS.md           # Current status and TODO list
├── 📄 requirements.txt            # Python dependencies
├── 📄 Makefile                    # Convenient command shortcuts
├── 🔧 setup.sh                    # Local setup script
├── 🔧 setup_azure.sh              # Azure VM setup script
├── 🔧 deploy_to_azure.sh          # Azure deployment script
├── 📄 .gitignore                  # Git ignore rules
├── 📄 horse_keypoint_schema.json  # Your 26-keypoint definition
│
```
.
├── data/                      # Dataset directory
│   ├── annotations/           # COCO format annotations
│   ├── images/               # Training images
│   └── README.md             # Dataset documentation
├── configs/                   # Training configurations
│   ├── _base_/               # Base config modules
│   ├── rtmpose_m_ap10k.py   # RTMPose-M config
│   ├── hrnet_w32_ap10k.py   # HRNet on AP-10K
│   └── hrnet_w32_animalpose.py  # HRNet on AnimalPose
├── tools/                     # Training and utility scripts
│   ├── train.py              # Training script
│   ├── test.py               # Testing script
│   ├── convert_dataset.py    # Dataset conversion
│   └── visualize.py          # Visualization tools
├── work_dirs/                 # Training outputs
├── checkpoints/              # Pretrained model weights
└── horse_keypoint_schema.json # Keypoint definitions

```

## Setup

### Local Development (macOS/Windows)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Install MMPose from source (recommended for latest features)
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -e .
cd ..
```

### Azure VM (Ubuntu 22.04) - **Recommended for Training**
```bash
# 1. Deploy to Azure VM (from local machine)
./deploy_to_azure.sh <vm-ip> azureuser

# With SSH key:
./deploy_to_azure.sh <vm-ip> azureuser MLCompute_key.pem

# 2. SSH into VM
ssh azureuser@<vm-ip>
cd ~/equistera-trainer

# 3. Run setup script on VM
./setup_azure.sh

# 4. Upload dataset (from local machine)
scp -r data/annotations data/images azureuser@<vm-ip>:~/equistera-trainer/data/
```

See **[AZURE_DEPLOYMENT.md](AZURE_DEPLOYMENT.md)** for complete Azure deployment guide.

### 2. Download Pretrained Weights
```bash
# Download pretrained checkpoints
python tools/download_checkpoints.py
```

### 3. Prepare Dataset
```bash
# Convert your dataset to COCO format
python tools/convert_dataset.py --input data/raw --output data/annotations
```

## Training

### RTMPose-M on AP-10K
```bash
python tools/train.py configs/rtmpose_m_ap10k.py --work-dir work_dirs/rtmpose_m
```

### HRNet-W32 on AP-10K
```bash
python tools/train.py configs/hrnet_w32_ap10k.py --work-dir work_dirs/hrnet_ap10k
```

### HRNet-W32 on AnimalPose
```bash
python tools/train.py configs/hrnet_w32_animalpose.py --work-dir work_dirs/hrnet_animalpose
```

## Export to ONNX
Export trained models to ONNX format for deployment:
```bash
python tools/export_onnx_opset21.py
```

## Testing

### Test PyTorch Model
```bash
python tools/test.py configs/rtmpose_m_ap10k.py work_dirs/rtmpose_m/best.pth
```

### Test ONNX Model
Test ONNX exported models on test datasets with COCO metrics:
```bash
# Basic testing
python tools/test_onnx.py --onnx work_dirs/rtmpose_m_horse_opset17.onnx \
                           --ann data/annotations/horse_test.json \
                           --img-dir data/test

# With visualizations
python tools/test_onnx.py --onnx work_dirs/rtmpose_m_horse_opset17.onnx \
                           --ann data/annotations/horse_test.json \
                           --img-dir data/test \
                           --show-dir visualizations/onnx_test
```

## Visualization
```bash
python tools/visualize.py --config configs/rtmpose_m_ap10k.py \
                          --checkpoint work_dirs/rtmpose_m/best.pth \
                          --img data/images/test_image.jpg
```

## Training Strategy

### Layer-wise Learning Rates
- **Frozen**: First 1-2 blocks (early feature extraction)
- **Fine-tune (LR=0.0001)**: Middle blocks (part detectors)
- **Train (LR=0.001)**: New 26-keypoint head (rebuilt from scratch)

### Data Augmentation
- Random rotation (±40°)
- Random scaling (0.7-1.3)
- Color jittering (brightness, contrast, saturation)
- Random horizontal flip
- Random crop and resize

## Monitoring
Training logs and metrics are saved to `work_dirs/`. Use TensorBoard to monitor:
```bash
tensorboard --logdir work_dirs/
```

## License
[Your License]
