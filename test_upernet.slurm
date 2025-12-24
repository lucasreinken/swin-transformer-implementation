#!/bin/bash
#SBATCH --partition=gpu-teaching-2d
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=test_upernet
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --chdir=/home/pml20/Machine-Learning-Project

mkdir -p logs

echo "========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU information:"
nvidia-smi
echo "========================================="

apptainer run --nv pml.sif python test_upernet.py

echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
