#!/bin/bash
#SBATCH --partition=gpu-teaching-9m
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=00:09:00
#SBATCH --job-name=swin_ade20k_seg
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --chdir=/home/pml20/Machine-Learning-Project

# =============================================================================
# ADE20K Semantic Segmentation Training - Swin-T + UperNet
# =============================================================================
# Training Configuration:
# - Swin-T backbone (ImageNet pretrained)
# - UperNet decoder head
# - Fine-tune both encoder and decoder
# - 160 epochs, lr=6e-5, batch_size=16
# - 512x512 input resolution
# =============================================================================

mkdir -p logs

echo "========================================="
echo "ADE20K Segmentation Training"
echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "========================================="
echo ""

# GPU info
echo "GPU Information:"
nvidia-smi
echo ""

# Set PyTorch CUDA memory allocation config for better memory management
# Segmentation with 512x512 requires more memory than classification
export PYTORCH_ALLOC_CONF=max_split_size_mb:512

echo "PYTORCH_ALLOC_CONF: $PYTORCH_ALLOC_CONF"
echo ""

echo "GPU Memory before training:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits
echo ""

# Run segmentation training
# NOTE: ADE20K dataset is loaded from /home/space/datasets/ade20k (shared storage)
#       or downloaded automatically to ~/datasets/ade20k if not available
apptainer run --nv \
    pml.sif \
    python main_segmentation.py

echo ""
echo "========================================="
echo "Training completed at: $(date)"
echo "========================================="
echo ""

echo "GPU Memory after training:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits
echo ""

echo "Job finished"
