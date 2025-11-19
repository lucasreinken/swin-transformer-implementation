#!/bin/bash
#SBATCH --partition=gpu-test  # or gpu for longer jobs
#SBATCH --gpus=1
#SBATCH --time=00:30:00  # 30 minutes for validation
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=cifar100_validation

mkdir -p logs
apptainer run --nv pml.sif python main.py "$@"
