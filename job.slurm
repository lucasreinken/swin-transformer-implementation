#!/bin/bash
#SBATCH --job-name=cifar100_validation
#SBATCH --partition=gpu-teaching-2h  # Use course partition (2h limit)
#SBATCH --gpus=1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p logs
apptainer run --nv pml.sif python main.py "$@"
