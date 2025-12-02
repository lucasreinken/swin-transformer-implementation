#!/bin/bash
#SBATCH --partition=gpu-teaching-2h  # or gpu for longer
#SBATCH --gpus=1
#SBATCH --time=02:00:00  # 2 hours; adjust (e.g., 12:00:00 for 12 hours)
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=ML_Train
#SBATCH --chdir=/home/pml04/swin_transformer/Machine-Learning-Project

mkdir -p logs
apptainer run --nv --bind /home/space/datasets/imagenet/2012:/mnt/datasets/imagenet/2012 pml.sif python main.py
