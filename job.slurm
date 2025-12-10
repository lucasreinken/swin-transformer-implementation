#!/bin/bash
#SBATCH --partition=gpu-teaching-5h # 5 hour partition
#SBATCH --gpus=1
#SBATCH --time=05:00:00  # 5 hours for training
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=Ticket2_ablation
#SBATCH --chdir=/home/pml04/swin_transformer/Machine-Learning-Project

mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
apptainer run --nv --overlay /home/space/datasets-sqfs/imagenet2012.sqfs pml.sif python main.py
