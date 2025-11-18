# Makefile for Equistera Trainer
# Convenient commands for common tasks

.PHONY: help setup install download-ckpts verify-data train-rtm train-hrnet-ap10k train-hrnet-animal test-rtm test-hrnet export-onnx test-onnx test-onnx-vis visualize monitor tensorboard clean deploy-azure

# Default target
help:
	@echo "Equistera Trainer - Horse Pose Estimation"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup              - Setup environment and install dependencies"
	@echo "  make download-ckpts     - Download pretrained checkpoints"
	@echo "  make verify-data        - Verify dataset annotations"
	@echo ""
	@echo "V2 Enhancements (Anatomical Priors + Diffusion):"
	@echo "  make setup-v2           - Setup V2 components (text embeddings)"
	@echo "  make validate-v2        - Validate V2 modules"
	@echo "  make train-v2           - Train RTMPose V2"
	@echo "  make test-v2 CKPT=<path> - Test V2 (fast mode)"
	@echo "  make test-v2-precise CKPT=<path> - Test V2 (precise mode)"
	@echo ""
	@echo "Azure Deployment:"
	@echo "  make deploy-azure VM=<ip> [KEY=<ssh-key>] - Deploy to Azure VM"
	@echo "  make setup-azure        - Setup Azure VM environment (run on VM)"
	@echo "  make upload-data VM=<ip> [KEY=<ssh-key>] - Upload dataset to Azure VM"
	@echo ""
	@echo "Dataset Preparation:"
	@echo "  make split-dataset      - Split dataset into train/val/test (70/15/15)"
	@echo "  make split-custom TRAIN=0.8 VAL=0.1 TEST=0.1 - Custom split ratios"
	@echo "  make split-symlink      - Split with symlinks (saves space)"
	@echo "  make convert-data       - Convert dataset to COCO format"
	@echo ""
	@echo "Training:"
	@echo "  make train-rtm          - Train RTMPose-M model"
	@echo "  make train-hrnet-ap10k  - Train HRNet-W32 on AP-10K"
	@echo "  make train-hrnet-animal - Train HRNet-W32 on AnimalPose"
	@echo ""
	@echo "ONNX Export and Testing:"
	@echo "  make export-onnx        - Export trained model to ONNX"
	@echo "  make test-onnx          - Test ONNX model on test dataset"
	@echo "  make test-onnx-vis      - Test ONNX with visualizations"
	@echo ""
	@echo "Evaluation:"
	@echo "  make test-rtm           - Test RTMPose-M model"
	@echo "  make test-hrnet         - Test HRNet-W32 model"
	@echo "  make visualize          - Visualize predictions"
	@echo ""
	@echo "Monitoring:"
	@echo "  make monitor            - Analyze training progress"
	@echo "  make tensorboard        - Launch TensorBoard"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean              - Clean temporary files"

# Setup and installation
setup:
	@echo "Setting up environment..."
	chmod +x setup.sh
	./setup.sh

setup-azure:
	@echo "Setting up Azure VM environment..."
	chmod +x setup_azure.sh
	./setup_azure.sh

# Azure deployment
deploy-azure:
	@if [ -z "$(VM)" ]; then \
		echo "Error: Please specify VM IP"; \
		echo "Usage: make deploy-azure VM=20.123.45.67 [KEY=MLCompute_key.pem]"; \
		exit 1; \
	fi
	@echo "Deploying to Azure VM: $(VM)"
	chmod +x deploy_to_azure.sh
	@if [ -n "$(KEY)" ]; then \
		./deploy_to_azure.sh $(VM) azureuser --key $(KEY); \
	else \
		./deploy_to_azure.sh $(VM) azureuser; \
	fi

upload-data:
	@if [ -z "$(VM)" ]; then \
		echo "Error: Please specify VM IP"; \
		echo "Usage: make upload-data VM=20.123.45.67 [KEY=MLCompute_key.pem]"; \
		exit 1; \
	fi
	@echo "Uploading dataset to Azure VM: $(VM)"
	@if [ -n "$(KEY)" ]; then \
		if [ -d "data/annotations" ]; then \
			scp -i $(KEY) -r data/annotations azureuser@$(VM):~/equistera-trainer/data/; \
		fi; \
		if [ -d "data/images" ]; then \
			scp -i $(KEY) -r data/images azureuser@$(VM):~/equistera-trainer/data/; \
		fi; \
	else \
		if [ -d "data/annotations" ]; then \
			scp -r data/annotations azureuser@$(VM):~/equistera-trainer/data/; \
		fi; \
		if [ -d "data/images" ]; then \
			scp -r data/images azureuser@$(VM):~/equistera-trainer/data/; \
		fi; \
	fi
	@echo "✓ Dataset uploaded"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

download-ckpts:
	@echo "Downloading pretrained checkpoints..."
	python tools/download_checkpoints.py

# V2 Setup and Training
setup-v2:
	@echo "Setting up RTMPose V2 enhancements..."
	pip install sentence-transformers>=2.2.0 transformers>=4.30.0
	python tools/setup_v2.py

validate-v2:
	@echo "Validating V2 modules..."
	python tools/validate_v2.py

train-v2:
	@echo "Training RTMPose V2..."
	./train_v2.sh

test-v2:
	@echo "Testing RTMPose V2..."
	@if [ -z "$(CKPT)" ]; then \
		echo "Error: CKPT not specified"; \
		echo "Usage: make test-v2 CKPT=work_dirs/rtmpose_m_ap10k_v2/best_*.pth"; \
		exit 1; \
	fi
	./test_v2.sh $(CKPT)

test-v2-precise:
	@echo "Testing RTMPose V2 (Precise mode)..."
	@if [ -z "$(CKPT)" ]; then \
		echo "Error: CKPT not specified"; \
		echo "Usage: make test-v2-precise CKPT=work_dirs/rtmpose_m_ap10k_v2/best_*.pth"; \
		exit 1; \
	fi
	./test_v2.sh $(CKPT) --cfg-options test_cfg.refine_mode='precise'

# Dataset
verify-data:
	@echo "Verifying dataset..."
	@if [ -f "data/annotations/train.json" ]; then \
		python tools/verify_dataset.py --ann-file data/annotations/train.json; \
	else \
		echo "Error: data/annotations/train.json not found"; \
		echo "Please prepare your dataset first"; \
	fi

convert-data:
	@echo "Converting dataset to COCO format..."
	python tools/convert_dataset.py --input data/raw --output data/annotations

# Split dataset into train/val/test
split-dataset:
	@echo "Splitting dataset (70/15/15)..."
	python tools/split_dataset.py \
		--ann-file data/annotations/horse_all.json \
		--img-dir data/images \
		--out-dir data \
		--train-ratio 0.7 \
		--val-ratio 0.15 \
		--test-ratio 0.15

# Split dataset with custom ratios (e.g., make split-custom TRAIN=0.8 VAL=0.1 TEST=0.1)
split-custom:
	@echo "Splitting dataset (custom ratios)..."
	python tools/split_dataset.py \
		--ann-file data/annotations/horse_all.json \
		--img-dir data/images \
		--out-dir data \
		--train-ratio $(TRAIN) \
		--val-ratio $(VAL) \
		--test-ratio $(TEST)

# Split dataset with symlinks (saves space)
split-symlink:
	@echo "Splitting dataset with symlinks..."
	python tools/split_dataset.py \
		--ann-file data/annotations/horse_all.json \
		--img-dir data/images \
		--out-dir data \
		--symlink

# Training commands
train-rtm:
	@echo "Training RTMPose-M..."
	python tools/train.py configs/rtmpose_m_ap10k.py \
		--work-dir work_dirs/rtmpose_m

train-rtm-amp:
	@echo "Training RTMPose-M with mixed precision..."
	python tools/train.py configs/rtmpose_m_ap10k.py \
		--work-dir work_dirs/rtmpose_m \
		--amp

train-hrnet-ap10k:
	@echo "Training HRNet-W32 on AP-10K..."
	python tools/train.py configs/hrnet_w32_ap10k.py \
		--work-dir work_dirs/hrnet_ap10k

train-hrnet-animal:
	@echo "Training HRNet-W32 on AnimalPose..."
	python tools/train.py configs/hrnet_w32_animalpose.py \
		--work-dir work_dirs/hrnet_animalpose

# Resume training
resume-rtm:
	@echo "Resuming RTMPose-M training..."
	python tools/train.py configs/rtmpose_m_ap10k.py \
		--work-dir work_dirs/rtmpose_m \
		--resume

resume-hrnet:
	@echo "Resuming HRNet-W32 training..."
	python tools/train.py configs/hrnet_w32_ap10k.py \
		--work-dir work_dirs/hrnet_ap10k \
		--resume

# Multi-GPU training
train-rtm-multi:
	@echo "Training RTMPose-M on multiple GPUs..."
	python -m torch.distributed.launch --nproc_per_node=2 \
		tools/train.py configs/rtmpose_m_ap10k.py \
		--launcher pytorch

# Testing
test-rtm:
	@echo "Testing RTMPose-M..."
	@if [ -f "work_dirs/rtmpose_m/best.pth" ]; then \
		python tools/test.py configs/rtmpose_m_ap10k.py \
			work_dirs/rtmpose_m/best.pth; \
	else \
		echo "Error: Checkpoint not found at work_dirs/rtmpose_m/best.pth"; \
	fi

test-hrnet:
	@echo "Testing HRNet-W32..."
	@if [ -f "work_dirs/hrnet_ap10k/best.pth" ]; then \
		python tools/test.py configs/hrnet_w32_ap10k.py \
			work_dirs/hrnet_ap10k/best.pth; \
	else \
		echo "Error: Checkpoint not found at work_dirs/hrnet_ap10k/best.pth"; \
	fi

# ONNX Export and Testing
export-onnx:
	@echo "Exporting model to ONNX..."
	python tools/export_onnx_opset21.py

test-onnx:
	@echo "Testing ONNX model..."
	@if [ -f "work_dirs/rtmpose_m_horse_opset17.onnx" ]; then \
		if [ -f "data/annotations/horse_test.json" ] && [ -d "data/test" ]; then \
			python tools/test_onnx.py \
				--onnx work_dirs/rtmpose_m_horse_opset17.onnx \
				--ann data/annotations/horse_test.json \
				--img-dir data/test; \
		else \
			echo "Error: Test dataset not found"; \
			echo "Expected: data/annotations/horse_test.json and data/test/"; \
		fi; \
	else \
		echo "Error: ONNX model not found at work_dirs/rtmpose_m_horse_opset17.onnx"; \
		echo "Run 'make export-onnx' first"; \
	fi

test-onnx-vis:
	@echo "Testing ONNX model with visualizations..."
	@if [ -f "work_dirs/rtmpose_m_horse_opset17.onnx" ]; then \
		if [ -f "data/annotations/horse_test.json" ] && [ -d "data/test" ]; then \
			python tools/test_onnx.py \
				--onnx work_dirs/rtmpose_m_horse_opset17.onnx \
				--ann data/annotations/horse_test.json \
				--img-dir data/test \
				--show-dir visualizations/onnx_test; \
		else \
			echo "Error: Test dataset not found"; \
			echo "Expected: data/annotations/horse_test.json and data/test/"; \
		fi; \
	else \
		echo "Error: ONNX model not found at work_dirs/rtmpose_m_horse_opset17.onnx"; \
		echo "Run 'make export-onnx' first"; \
	fi


# Visualization
visualize:
	@echo "Visualizing predictions..."
	@if [ -d "data/images/test" ]; then \
		python tools/visualize.py \
			--config configs/rtmpose_m_ap10k.py \
			--checkpoint work_dirs/rtmpose_m/best.pth \
			--img-dir data/images/test \
			--out-dir visualizations; \
	else \
		echo "Error: Test images not found at data/images/test"; \
	fi

visualize-single:
	@echo "Usage: make visualize-single IMG=path/to/image.jpg"
	@if [ -z "$(IMG)" ]; then \
		echo "Error: Please specify IMG variable"; \
		exit 1; \
	fi
	python tools/visualize.py \
		--config configs/rtmpose_m_ap10k.py \
		--checkpoint work_dirs/rtmpose_m/best.pth \
		--img $(IMG) \
		--out-file output.jpg

# Monitoring
monitor:
	@echo "Analyzing training progress..."
	python tools/monitor_training.py --compare \
		work_dirs/rtmpose_m \
		work_dirs/hrnet_ap10k

tensorboard:
	@echo "Launching TensorBoard..."
	@echo "Open http://localhost:6006 in your browser"
	tensorboard --logdir work_dirs/

# Utilities
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "Cleaned!"

clean-checkpoints:
	@echo "Warning: This will delete all training checkpoints!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf work_dirs/*/epoch_*.pth; \
		echo "Checkpoints cleaned!"; \
	fi

# Development
format:
	@echo "Formatting code..."
	black tools/ configs/
	isort tools/ configs/

lint:
	@echo "Linting code..."
	flake8 tools/ configs/
	mypy tools/

# Statistics
stats:
	@echo "Project Statistics:"
	@echo "==================="
	@echo "Python files:"
	@find . -name "*.py" -not -path "./venv/*" -not -path "./mmpose/*" | wc -l
	@echo "Config files:"
	@find configs/ -name "*.py" | wc -l
	@echo "Total lines of code:"
	@find . -name "*.py" -not -path "./venv/*" -not -path "./mmpose/*" -exec wc -l {} + | tail -1
	@if [ -d "data/images" ]; then \
		echo "Training images:"; \
		find data/images/train -type f 2>/dev/null | wc -l || echo "N/A"; \
		echo "Validation images:"; \
		find data/images/val -type f 2>/dev/null | wc -l || echo "N/A"; \
	fi
