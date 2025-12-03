#!/bin/bash
#SBATCH --partition=gpu-teaching-2h  # 2 hour partition
#SBATCH --gpus=1
#SBATCH --time=02:00:00  # 2 hours for training
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=ML_Train
#SBATCH --chdir=/home/pml04/swin_transformer/Machine-Learning-Project

mkdir -p logs
apptainer run --nv --overlay /home/space/datasets-sqfs/imagenet2012.sqfs pml.sif python main.py
