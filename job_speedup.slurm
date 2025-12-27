#!/bin/bash
#SBATCH --partition=gpu-teaching-7d    # your long partition
#SBATCH --gpus=1
#SBATCH --exclusive                      # ← NO sharing with other users → fixes node contention
#SBATCH --time=168:00:00                  # 168 hours for 40-epoch training
#SBATCH --job-name=baseline_swin_20         # job name
#SBATCH --output=logs/%j_%x.out          # includes job name → easier to find
#SBATCH --error=logs/%j_%x.err
#SBATCH --chdir=/home/pml17/Machine-Learning-Project
#SBATCH --array=1-1                      # Adjust as needed for multiple runs

mkdir -p logs

# === Best-practice memory fixes (keep all of these!) ===
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0

# Limit OpenMP/MKL threads to avoid exceeding Slurm AssocGrpCpuUsage limits
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# === Run your training ===
apptainer run --nv \
    --overlay /home/space/datasets-sqfs/imagenet2012.sqfs \
    pml.sif \
    python main.py