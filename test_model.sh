#!/bin/bash
# Test RTMPose model on test dataset

# Activate environment
source venv/bin/activate

# Run test
python tools/test.py \
    configs/rtmpose_m_ap10k.py \
    checkpoints/best_coco_AP_epoch_210.pth \
    --work-dir work_dirs/test_results \
    --out results/test_predictions.pkl

echo ""
echo "Test complete! Results saved to:"
echo "  - work_dirs/test_results/ (logs and metrics)"
echo "  - results/test_predictions.pkl (predictions)"
