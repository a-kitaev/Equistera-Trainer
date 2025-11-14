#!/bin/bash
# Quick deployment script for Azure VM
# Run this on your LOCAL machine to prepare and upload to Azure

set -e

echo "=================================================="
echo "Azure VM Deployment Preparation"
echo "=================================================="
echo ""

# Check if VM IP is provided
if [ -z "$1" ]; then
    echo "Usage: ./deploy_to_azure.sh <vm-ip> [vm-user] [--key <ssh-key-path>]"
    echo ""
    echo "Examples:"
    echo "  ./deploy_to_azure.sh 20.123.45.67 azureuser"
    echo "  ./deploy_to_azure.sh 20.123.45.67 azureuser --key MLCompute_key.pem"
    echo ""
    echo "Using Makefile:"
    echo "  make deploy-azure VM=20.123.45.67"
    echo "  make deploy-azure VM=20.123.45.67 KEY=MLCompute_key.pem"
    exit 1
fi

VM_IP=$1
VM_USER=${2:-azureuser}
SSH_KEY=""

# Parse optional SSH key argument
shift 2 2>/dev/null || shift 1
while [[ $# -gt 0 ]]; do
    case $1 in
        --key)
            SSH_KEY="-i $2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "Target: $VM_USER@$VM_IP"
if [ -n "$SSH_KEY" ]; then
    echo "SSH Key: ${SSH_KEY#-i }"
fi
echo ""

# Create deployment package
echo "Creating deployment package..."
DEPLOY_DIR="deploy_package"
rm -rf $DEPLOY_DIR
mkdir -p $DEPLOY_DIR

# Copy essential files
echo "Copying project files..."
rsync -av --progress \
    --exclude 'venv' \
    --exclude 'mmpose' \
    --exclude 'work_dirs' \
    --exclude 'data/images' \
    --exclude 'data/annotations/*.json' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'checkpoints/*.pth' \
    --exclude 'visualizations' \
    --exclude 'analysis' \
    --exclude '.DS_Store' \
    ./ $DEPLOY_DIR/

echo ""
echo "✓ Deployment package created"
echo ""

# Test SSH connection
echo "Testing SSH connection..."
if ssh $SSH_KEY -o ConnectTimeout=5 $VM_USER@$VM_IP "echo 'Connection successful'"; then
    echo "✓ SSH connection OK"
else
    echo "✗ SSH connection failed!"
    echo ""
    echo "Please ensure:"
    echo "1. VM is running"
    echo "2. SSH key is configured (use --key flag if needed)"
    echo "3. SSH key has correct permissions (chmod 600 MLCompute_key.pem)"
    echo "4. Network Security Group allows SSH (port 22)"
    echo "5. VM IP is correct"
    exit 1
fi

echo ""
read -p "Upload project to Azure VM? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Upload to VM
    echo ""
    echo "Uploading to Azure VM..."
    ssh $SSH_KEY $VM_USER@$VM_IP "mkdir -p ~/equistera-trainer"
    
    rsync -avz --progress -e "ssh $SSH_KEY" $DEPLOY_DIR/ $VM_USER@$VM_IP:~/equistera-trainer/
    
    echo ""
    echo "✓ Upload complete"
    
    # Make scripts executable
    echo ""
    echo "Setting permissions..."
    ssh $SSH_KEY $VM_USER@$VM_IP "cd ~/equistera-trainer && chmod +x *.sh && chmod +x tools/*.py"
    
    echo ""
    echo "=================================================="
    echo "Deployment Complete!"
    echo "=================================================="
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. SSH into your VM:"
    if [ -n "$SSH_KEY" ]; then
        echo "   ssh ${SSH_KEY} $VM_USER@$VM_IP"
    else
        echo "   ssh $VM_USER@$VM_IP"
    fi
    echo ""
    echo "2. Navigate to project:"
    echo "   cd ~/equistera-trainer"
    echo ""
    echo "3. Run setup:"
    echo "   ./setup_azure.sh"
    echo ""
    echo "4. Upload your dataset:"
    echo "   (From local machine)"
    if [ -n "$SSH_KEY" ]; then
        echo "   scp ${SSH_KEY} -r data/annotations $VM_USER@$VM_IP:~/equistera-trainer/data/"
        echo "   scp ${SSH_KEY} -r data/images $VM_USER@$VM_IP:~/equistera-trainer/data/"
    else
        echo "   scp -r data/annotations $VM_USER@$VM_IP:~/equistera-trainer/data/"
        echo "   scp -r data/images $VM_USER@$VM_IP:~/equistera-trainer/data/"
    fi
    echo ""
    echo "5. Start training:"
    echo "   source venv/bin/activate"
    echo "   make train-rtm"
    echo ""
    
    # Clean up
    rm -rf $DEPLOY_DIR
    
    # Offer to SSH
    echo ""
    read -p "SSH into VM now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ssh $SSH_KEY $VM_USER@$VM_IP
    fi
else
    echo "Deployment cancelled"
    rm -rf $DEPLOY_DIR
fi
