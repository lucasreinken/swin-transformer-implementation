#!/bin/bash
#SBATCH --partition=gpu-test
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/validation_%j.out
#SBATCH --error=logs/validation_%j.err
#SBATCH --job-name=ImageNet_Validation
#SBATCH --mem=32G

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Create directories
mkdir -p logs results

# Run validation
apptainer run --nv pml.sif python3 validate_model.py \
    --model swin_tiny_patch4_window7_224 \
    --samples 5000 \
    --batch-size 32 \
    --num-workers 4 \
    --transfer-weights

echo "=========================================="
echo "End time: $(date)"
echo "Results saved to: results/zero_shot_imagenet.log"
