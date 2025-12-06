#!/bin/bash
#SBATCH --partition=gpu-teaching-2d  # 2 days partition
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00  # 2 days for training
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=ML_Train_swin_window_size_4
#SBATCH --chdir=/home/pml20/Machine-Learning-Project

mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
apptainer run --nv --overlay /home/space/datasets-sqfs/imagenet2012.sqfs pml.sif python main.py
