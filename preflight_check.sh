#!/bin/bash
# Pre-flight checklist before starting training

echo "=============================================="
echo "🐴 Equistera Trainer - Pre-Flight Checklist"
echo "=============================================="
echo ""

# Activate venv
source venv/bin/activate

# Function to print check result
check() {
    if [ $1 -eq 0 ]; then
        echo "✅ $2"
    else
        echo "❌ $2"
        FAILED=1
    fi
}

FAILED=0

# 1. Check Python environment
echo "1. Checking Python environment..."
python -c "import torch, mmcv, mmengine, mmpose" 2>/dev/null
check $? "All Python packages installed"

python -c "import cv2, numpy; assert cv2.__version__ == '4.8.1.78'; assert numpy.__version__ == '1.26.4'" 2>/dev/null
check $? "NumPy/OpenCV versions correct (1.26.4 / 4.8.1.78)"

python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null
check $? "CUDA available"

# 2. Check dataset
echo ""
echo "2. Checking dataset..."
if [ -f "data/annotations/horse_train.json" ]; then
    check 0 "Training annotations exist"
else
    check 1 "Training annotations missing"
fi

if [ -f "data/annotations/horse_val.json" ]; then
    check 0 "Validation annotations exist"
else
    check 1 "Validation annotations missing"
fi

if [ -d "data/train" ] && [ "$(ls -A data/train 2>/dev/null)" ]; then
    img_count=$(ls data/train | wc -l)
    check 0 "Training images exist ($img_count files)"
else
    check 1 "Training images missing"
fi

if [ -d "data/val" ] && [ "$(ls -A data/val 2>/dev/null)" ]; then
    img_count=$(ls data/val | wc -l)
    check 0 "Validation images exist ($img_count files)"
else
    check 1 "Validation images missing"
fi

# 3. Check config files
echo ""
echo "3. Checking configuration..."
python -c "from mmengine.config import Config; Config.fromfile('configs/rtmpose_m_ap10k.py')" 2>/dev/null
check $? "RTMPose config loads"

python -c "from mmengine.config import Config; Config.fromfile('configs/hrnet_w32_ap10k.py')" 2>/dev/null
check $? "HRNet AP10K config loads"

# 4. Check checkpoints
echo ""
echo "4. Checking pretrained checkpoints..."
if [ -f "checkpoints/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.pth" ]; then
    check 0 "RTMPose-M checkpoint exists"
else
    check 1 "RTMPose-M checkpoint missing (run: make download-ckpts)"
fi

if [ -f "checkpoints/hrnet_w32_ap10k_256x256-18aac840_20211029.pth" ]; then
    check 0 "HRNet AP10K checkpoint exists"
else
    check 1 "HRNet AP10K checkpoint missing (run: make download-ckpts)"
fi

# 5. Check disk space
echo ""
echo "5. Checking disk space..."
available=$(df -h . | awk 'NR==2 {print $4}')
echo "   Available space: $available"
if [ $(df . | awk 'NR==2 {print $4}') -gt 10485760 ]; then  # 10GB in KB
    check 0 "Sufficient disk space (>10GB)"
else
    check 1 "Low disk space (<10GB)"
fi

# 6. Check GPU
echo ""
echo "6. Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.free --format=csv,noheader | while read line; do
        echo "   GPU: $line"
    done
    check 0 "GPU detected"
else
    check 1 "nvidia-smi not available"
fi

# Summary
echo ""
echo "=============================================="
if [ $FAILED -eq 0 ]; then
    echo "✅ All checks passed! Ready to train!"
    echo "=============================================="
    echo ""
    echo "Start training with:"
    echo "  make train-rtm          # RTMPose-M"
    echo "  make train-hrnet-ap10k  # HRNet-W32"
    echo ""
    echo "Monitor training with:"
    echo "  make tensorboard"
    echo "  make monitor"
else
    echo "❌ Some checks failed. Please fix issues above."
    echo "=============================================="
    echo ""
    echo "Common fixes:"
    echo "  make setup              # Setup environment"
    echo "  make download-ckpts     # Download checkpoints"
    echo "  ./fix_dataset_and_split.sh  # Fix and split dataset"
fi

echo ""
