#!/bin/bash
#SBATCH --partition=gpu-test  # or gpu for longer
#SBATCH --gpus=1
#SBATCH --time=00:30:00  # 30 minutes; adjust (e.g., 12:00:00 for 12 hours)
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=ML_Train

mkdir -p logs
apptainer run --nv pml.sif python main.py "$@"

