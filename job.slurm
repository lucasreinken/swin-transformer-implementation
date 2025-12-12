#!/bin/bash
#SBATCH --partition=gpu-teaching-2d    # your long partition
#SBATCH --gpus=1
#SBATCH --exclusive                      # ← NO sharing with other users → fixes node contention
#SBATCH --time=48:00:00                  # 120 hours for 40-epoch training
#SBATCH --job-name=baseline_swin_40         # job name
#SBATCH --output=logs/%j_%x.out          # includes job name → easier to find
#SBATCH --error=logs/%j_%x.err
#SBATCH --chdir=/home/pml04/swin_transformer/Machine-Learning-Project
#SBATCH --array=1-1                      # Adjust as needed for multiple runs

mkdir -p logs

# === Best-practice memory fixes (keep all of these!) ===
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0

# Clear any leftover GPU memory from previous jobs
python - <<'PY'
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
print("GPU memory cleared before job start")
PY

echo "Job started on node: $(hostname)"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"

# === Run your training ===
apptainer run --nv \
    --overlay /home/space/datasets-sqfs/imagenet2012.sqfs \
    pml.sif \
    python main.py \
        --epochs 100 \
        # or whatever you want
        # add more args here if needed

echo "Job finished"