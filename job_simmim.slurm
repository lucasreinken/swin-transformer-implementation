#!/bin/bash
#SBATCH --partition=gpu-teaching-5h
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --job-name=simmim         # job name
#SBATCH --output=logs/%j_%x.out   # includes job name → easier to find
#SBATCH --error=logs/%j_%x.err
#SBATCH --chdir=/home/pml17/Machine-Learning-Project
#SBATCH --array=1-1               # Adjust as needed for multiple runs

mkdir -p logs

# === Best-practice memory fixes (keep all of these!) ===
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# === Run your training ===
apptainer run --nv \
    pml.sif \
    python main_simmim.py