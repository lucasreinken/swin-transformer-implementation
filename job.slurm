#!/bin/bash
#SBATCH --partition=gpu-teaching-7d # 7 day partition
#SBATCH --gpus=1
#SBATCH --time=168:00:00  # 7 days for training
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=baseline_with_proper_lr
#SBATCH --chdir=/home/pml04/swin_transformer/Machine-Learning-Project

mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
apptainer run --nv --overlay /home/space/datasets-sqfs/imagenet2012.sqfs pml.sif python main.py
