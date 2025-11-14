# Azure VM Deployment Guide

Complete guide for deploying and training on **Azure NC4as T4 v3** with **Ubuntu 22.04**.

---

## 📋 VM Specifications

- **GPU**: 1× NVIDIA T4 (16 GB VRAM)
- **vCPUs**: 4 AMD EPYC 7V12
- **RAM**: 28 GB
- **OS**: Ubuntu 22.04 LTS
- **Storage**: 128 GB Premium SSD (minimum)
- **Cost**: ~$0.526/hour (~$5-6 for 10-hour training)

---

## 🚀 Quick Start (3 Steps)

### Step 1: Deploy from Your Local Machine

```bash
# Make deployment script executable
chmod +x deploy_to_azure.sh

# Deploy to Azure VM
./deploy_to_azure.sh <your-vm-ip> azureuser

# Example:
./deploy_to_azure.sh 20.123.45.67 azureuser

# With SSH key:
./deploy_to_azure.sh 20.123.45.67 azureuser MLCompute_key.pem
```

**What it does:**
- ✅ Tests SSH connection
- ✅ Creates optimized deployment package (excludes venv, large files)
- ✅ Uploads project to VM (~5 minutes, ~30 MB)
- ✅ Sets correct permissions

### Step 2: Setup Environment on Azure VM

```bash
# SSH into VM
ssh azureuser@<vm-ip>

# Or with SSH key:
ssh -i MLCompute_key.pem azureuser@<vm-ip>

# Navigate to project
cd ~/equistera-trainer

# Run setup script
./setup_azure.sh
```

**What it installs:**
- NVIDIA drivers (if needed) + CUDA 11.8
- Python 3.10 with conda environment
- PyTorch 2.1.0 (CUDA 11.8 compatible)
- MMCV 2.1.0 (prebuilt binary)
- MMPose (from source)
- MMDetection (required dependency)
- All training dependencies

**Setup time:** ~10-15 minutes  
**Reboot required:** Only if NVIDIA drivers are installed (script will prompt)

**If prompted to reboot:**
```bash
sudo reboot
# SSH back in and run setup again:
cd ~/equistera-trainer
./setup_azure.sh
```

### Step 3: Upload Dataset

**From your local machine:**

```bash
# Upload annotations
scp -r data/annotations azureuser@<vm-ip>:~/equistera-trainer/data/

# Upload images  
scp -r data/images azureuser@<vm-ip>:~/equistera-trainer/data/

# Faster with rsync:
rsync -avz --progress data/ azureuser@<vm-ip>:~/equistera-trainer/data/

# With SSH key:
scp -i MLCompute_key.pem -r data/annotations azureuser@<vm-ip>:~/equistera-trainer/data/
```

---

## 🎓 Training

### Activate Environment

```bash
# Activate conda environment
conda activate rtmpose

# Verify GPU
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# Verify dataset
python tools/verify_dataset.py
```

### Start Training (IMPORTANT: Use tmux!)

```bash
# Start tmux session (keeps training running if SSH disconnects)
tmux new -s training

# Set PYTHONPATH for V2 config
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Train with base config
python tools/train.py configs/rtmpose_m_ap10k.py

# OR train with V2 config (diffusion refinement)
python tools/train.py configs/rtmpose_m_ap10k_v2.py

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

**Why tmux?**
- Keeps training running if SSH connection drops
- Can reconnect from anywhere
- Essential for long training sessions

### Recommended Settings for T4 GPU

**RTMPose-M (Optimal for T4):**
```bash
python tools/train.py configs/rtmpose_m_ap10k.py \
    --work-dir work_dirs/rtmpose_m \
    --amp \
    --cfg-options \
        train_dataloader.batch_size=32 \
        train_dataloader.num_workers=4
```

**Performance:**
- Training time: 8-10 hours (800 images, 300 epochs)
- GPU utilization: ~90%
- VRAM usage: ~12-14 GB
- Throughput: ~3-4 images/sec

**RTMPose-M V2 (with diffusion refinement):**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python tools/train.py configs/rtmpose_m_ap10k_v2.py \
    --work-dir work_dirs/rtmpose_m_v2 \
    --amp \
    --cfg-options \
        train_dataloader.batch_size=16 \
        train_dataloader.num_workers=4
```

**Performance:**
- Training time: 10-12 hours (800 images, 300 epochs)
- GPU utilization: ~85%
- VRAM usage: ~14-15 GB
- +10-15% training time vs base config

**HRNet-W32 (if you need highest accuracy):**
```bash
python tools/train.py configs/hrnet_w32_ap10k.py \
    --work-dir work_dirs/hrnet_ap10k \
    --cfg-options \
        train_dataloader.batch_size=16 \
        train_dataloader.num_workers=4
```

**Performance:**
- Training time: 12-15 hours (800 images, 300 epochs)
- GPU utilization: ~85%
- VRAM usage: ~14-15 GB

---

## 📊 Monitor Training

### Option 1: TensorBoard (Recommended)

```bash
# In a new tmux session
tmux new -s tensorboard
conda activate rtmpose
tensorboard --logdir work_dirs/ --bind_all

# Detach: Ctrl+B, D
```

**Access from browser:**
```
http://<vm-ip>:6006
```

**⚠️ Important:** Open port 6006 in Azure Network Security Group:
- Go to Azure Portal → VM → Networking → Add inbound port rule
- Port: 6006, Protocol: TCP, Source: Your IP

### Option 2: Check Logs

```bash
# View training logs
tail -f work_dirs/rtmpose_m/*.log

# Or for V2:
tail -f work_dirs/rtmpose_m_v2/*.log

# Check last 100 lines
tail -n 100 work_dirs/rtmpose_m/*.log
```

### Option 3: Reconnect to Training Session

```bash
# SSH back into VM
ssh azureuser@<vm-ip>

# Reattach to training
tmux attach -t training

# List all sessions
tmux ls
```

---

## 💾 Download Results

### From Azure VM to Local Machine

```bash
# Download best checkpoint
scp azureuser@<vm-ip>:~/equistera-trainer/work_dirs/rtmpose_m/best_coco_AP_epoch_*.pth ./

# Download all checkpoints
scp -r azureuser@<vm-ip>:~/equistera-trainer/work_dirs/rtmpose_m ./results/

# Download visualizations
scp -r azureuser@<vm-ip>:~/equistera-trainer/visualizations ./

# With SSH key:
scp -i MLCompute_key.pem azureuser@<vm-ip>:~/equistera-trainer/work_dirs/rtmpose_m/best*.pth ./
```

---

## 💰 Cost Optimization

### 1. Auto-Shutdown

**Via Azure Portal:**
- VM → Auto-shutdown → Enable → Set time (e.g., 2 AM)

**Via command (shutdown after training):**
```bash
# Shutdown in 60 minutes after training completes
python tools/train.py configs/rtmpose_m_ap10k.py && sudo shutdown -h +60
```

### 2. Use Spot Instances
- Up to 90% discount vs pay-as-you-go
- May be preempted (evicted) when Azure needs capacity
- Good for experimentation, not production

### 3. Stop VM When Not Training

```bash
# Stop VM (from local machine with Azure CLI)
az vm deallocate --name <vm-name> --resource-group <rg-name>

# Start when needed
az vm start --name <vm-name> --resource-group <rg-name>

# Check status
az vm list --output table
```

### 4. Estimated Costs

| Phase | Duration | Cost (Pay-as-you-go) |
|-------|----------|----------------------|
| Initial training (800 images) | 10 hours | $5.26 |
| Validation & testing | 2 hours | $1.05 |
| Full training (5k images) | 24 hours | $12.62 |
| **Total** | **36 hours** | **~$19** |

**Storage:**
- OS Disk (128 GB SSD): ~$19.20/month
- Blob Storage (backups): ~$1/50GB/month

**Savings:**
- Spot Instances: Up to 90% off
- 1-year Reserved: 40% off
- 3-year Reserved: 60% off

---

## 🔐 Network Security & Best Practices

### Required Firewall Rules

Add these inbound rules in Azure Portal (VM → Networking):

| Port | Protocol | Source | Purpose |
|------|----------|--------|---------|
| 22 | TCP | Your IP only | SSH |
| 6006 | TCP | Your IP only | TensorBoard |

**⚠️ Security Warning:** Never use "Any" as source - always restrict to your IP address.

### SSH Key Authentication

```bash
# Ensure correct permissions
chmod 600 MLCompute_key.pem

# Connect with key
ssh -i MLCompute_key.pem azureuser@<vm-ip>
```

### Disable Password Authentication (Recommended)

```bash
# On Azure VM
sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no
sudo systemctl restart sshd
```

---

## 🔧 Advanced Configuration

### Persistent Storage with Azure Managed Disk

```bash
# Attach disk in Azure Portal, then on VM:
sudo mkdir -p /mnt/data
sudo mount /dev/sdc1 /mnt/data

# Move data to persistent storage
mv data /mnt/data/
ln -s /mnt/data/data data

mv work_dirs /mnt/data/
ln -s /mnt/data/work_dirs work_dirs
```

### Automated Backups to Azure Blob Storage

```bash
# Install Azure CLI (if not already installed)
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login

# Backup checkpoints every 6 hours
crontab -e
# Add:
0 */6 * * * az storage blob upload-batch --source ~/equistera-trainer/work_dirs --destination backups --account-name <storage-account>
```

### Multi-GPU Training (if using larger VM)

```bash
# For 2+ GPUs
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    tools/train.py configs/rtmpose_m_ap10k.py \
    --launcher pytorch
```

---

## 🐛 Troubleshooting

### Can't Connect to VM

```bash
# Check if VM is running
az vm list --output table

# Check VM status
az vm get-instance-view --name <vm-name> --resource-group <rg-name>

# Start if stopped
az vm start --name <vm-name> --resource-group <rg-name>

# Check Network Security Group rules
az network nsg rule list --nsg-name <nsg-name> --resource-group <rg-name> --output table
```

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# If not found, reinstall drivers
sudo apt-get purge nvidia-*
sudo ubuntu-drivers autoinstall
sudo reboot

# After reboot, verify
nvidia-smi
```

### CUDA Out of Memory

```bash
# Reduce batch size
--cfg-options train_dataloader.batch_size=16

# Or use gradient accumulation (effective batch size stays same)
--cfg-options optim_wrapper.accumulative_counts=2 train_dataloader.batch_size=16

# Monitor GPU memory
watch -n 1 nvidia-smi
```

### Training Stopped/Disconnected

```bash
# Check if training is still running
tmux attach -t training

# If session doesn't exist, check processes
ps aux | grep python

# Check logs
tail -f work_dirs/rtmpose_m/*.log

# Resume from checkpoint
python tools/train.py configs/rtmpose_m_ap10k.py --resume
```

### Slow Data Loading

```bash
# Reduce num_workers if CPU is bottleneck
--cfg-options train_dataloader.num_workers=2

# Increase if CPU is underutilized
--cfg-options train_dataloader.num_workers=8

# Monitor CPU usage
htop
```

### ModuleNotFoundError

```bash
# Ensure PYTHONPATH is set (for V2 config)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify conda environment is activated
conda activate rtmpose

# Reinstall MMPose if needed
cd mmpose
pip install -e .
cd ..
```

### Text Embeddings Missing (V2 config only)

```bash
# Generate embeddings
python tools/generate_text_embeddings.py

# Verify
ls -lh embeddings/
# Should show: horse_global.npy, horse_local.npy
```

---

## ✅ Pre-Flight Checklist

### Before Deployment

- [ ] Azure VM created (NC4as T4 v3, Ubuntu 22.04)
- [ ] SSH key configured with correct permissions (chmod 600)
- [ ] VM IP address known
- [ ] Dataset prepared locally (annotations + images)
- [ ] Network Security Group allows SSH (port 22)

### After Deployment

- [ ] `nvidia-smi` shows T4 GPU
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` returns True
- [ ] Dataset uploaded and verified
- [ ] Network Security Group allows TensorBoard (port 6006)
- [ ] tmux installed (`tmux -V`)
- [ ] Training started in tmux session
- [ ] TensorBoard accessible from browser

### During Training

- [ ] GPU utilization >80% (`nvidia-smi`)
- [ ] Training loss decreasing
- [ ] No OOM errors in logs
- [ ] Checkpoints being saved every 10 epochs
- [ ] Auto-shutdown configured (optional)

---

## 📖 Complete Workflow Example

```bash
# === ON LOCAL MACHINE ===

# 1. Deploy project
./deploy_to_azure.sh 20.123.45.67 azureuser

# 2. Upload dataset (in another terminal)
scp -r data/annotations azureuser@20.123.45.67:~/equistera-trainer/data/
scp -r data/images azureuser@20.123.45.67:~/equistera-trainer/data/

# === ON AZURE VM ===

# 3. SSH into VM
ssh azureuser@20.123.45.67

# 4. Setup environment
cd ~/equistera-trainer
./setup_azure.sh

# If reboot required:
sudo reboot
# Then SSH back and run setup again
cd ~/equistera-trainer
./setup_azure.sh

# 5. Activate environment and verify
conda activate rtmpose
nvidia-smi
python tools/verify_dataset.py

# 6. Start training in tmux
tmux new -s training
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python tools/train.py configs/rtmpose_m_ap10k_v2.py

# 7. Detach from tmux (Ctrl+B, then D)

# 8. Start TensorBoard (new tmux session)
tmux new -s tensorboard
conda activate rtmpose
tensorboard --logdir work_dirs/ --bind_all

# 9. Detach and disconnect
# Ctrl+B, D
# exit

# === MONITOR FROM BROWSER ===

# 10. Open TensorBoard
# http://20.123.45.67:6006

# === DOWNLOAD RESULTS (LOCAL MACHINE) ===

# 11. After training completes
scp azureuser@20.123.45.67:~/equistera-trainer/work_dirs/rtmpose_m_v2/best*.pth ./

# 12. Stop VM to save costs
az vm deallocate --name <vm-name> --resource-group <rg-name>
```

---

## 📞 Quick Reference Commands

```bash
# Deploy
./deploy_to_azure.sh <vm-ip> azureuser

# Setup
./setup_azure.sh

# Activate
conda activate rtmpose

# Train (base)
python tools/train.py configs/rtmpose_m_ap10k.py

# Train (V2)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python tools/train.py configs/rtmpose_m_ap10k_v2.py

# tmux
tmux new -s training        # Create
tmux attach -t training     # Reattach
tmux ls                     # List sessions
Ctrl+B, D                   # Detach

# Monitor
nvidia-smi                  # GPU status
tail -f work_dirs/*/*.log  # Training logs
tensorboard --logdir work_dirs/ --bind_all  # TensorBoard

# Download
scp azureuser@<vm-ip>:~/equistera-trainer/work_dirs/rtmpose_m/best*.pth ./

# Stop VM
az vm deallocate --name <vm-name> --resource-group <rg-name>
```

---

## 🎯 What Gets Deployed

### Included (Small, Fast Upload)
✅ Configuration files (all .py configs)  
✅ Training scripts (tools/*.py)  
✅ Documentation (all .md files)  
✅ Setup scripts (*.sh)  
✅ Requirements.txt  
✅ Makefile  

### Excluded (Created on VM or Uploaded Separately)
❌ `venv/` - Created by setup script  
❌ `mmpose/` - Cloned by setup script  
❌ `work_dirs/` - Generated during training  
❌ `data/images/` - Uploaded separately  
❌ `checkpoints/*.pth` - Downloaded by setup script  

**Result:** Fast deployment (~30 MB, ~5 minutes)

---

## 📚 Related Documentation

- **README.md** - Project overview
- **QUICKSTART.md** - Local training guide
- **TRAINING_GUIDE.md** - Advanced training strategies
- **RTMPOSE_V2_GUIDE.md** - V2 features (text embeddings + diffusion)

---

## 🎉 Summary

Your Azure deployment workflow:

1. **Deploy** → One command from local machine (5 min)
2. **Setup** → Automated environment installation (10 min)
3. **Upload** → Dataset transfer (10-30 min)
4. **Train** → Start in tmux, monitor via TensorBoard (8-10 hours)
5. **Download** → Results back to local machine (2 min)
6. **Stop** → Deallocate VM to save costs

**Total hands-on time:** ~30 minutes  
**Training time:** 8-10 hours (automated)  
**Total cost:** ~$5-6 for 800-image training

---

**Ready to train on Azure! 🚀**

*Last updated: October 13, 2025*
