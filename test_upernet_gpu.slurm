#!/bin/bash
#SBATCH --partition=gpu-teaching-7d     # Using 7-day partition (more availability)
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --time=01:00:00                 # 1 hour for testing
#SBATCH --job-name=test_upernet
#SBATCH --output=logs/%j_%x.out         # Same format as your existing jobs
#SBATCH --error=logs/%j_%x.err
#SBATCH --chdir=/home/pml20/Machine-Learning-Project  # Explicit working directory

# Create logs directory
mkdir -p logs

# Print GPU info
echo "========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU information:"
nvidia-smi
echo "========================================="

# Run UperNet test
apptainer run --nv pml.sif python test_upernet.py

# Print completion
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
