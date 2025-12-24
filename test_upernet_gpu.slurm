#!/bin/bash
#SBATCH --partition=gpu-2h              # Short GPU partition for testing
#SBATCH --gpus=1                        # 1 GPU
#SBATCH --time=00:30:00                 # 30 minutes (plenty for test)
#SBATCH --job-name=test_upernet
#SBATCH --output=logs/test_upernet_%j.out
#SBATCH --error=logs/test_upernet_%j.err
#SBATCH --account=pml20

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
