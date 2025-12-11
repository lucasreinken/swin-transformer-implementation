#!/bin/bash
#SBATCH --partition=gpu-teaching-5h      # your long partition
#SBATCH --gpus=1
#SBATCH --exclusive                      # ← NO sharing with other users → fixes node contention
#SBATCH --time=05:00:00                  # 5 hours (adjust if needed)
#SBATCH --job-name=imagenet_100_epochs
#SBATCH --output=logs/%j_%x.out          # includes job name → easier to find
#SBATCH --error=logs/%j_%x.err
#SBATCH --chdir=/home/pml04/swin_transformer/Machine-Learning-Project

# === Exclude known bad / flaky nodes (add more if you discover them) ===
# Example: node04 was giving you OOM with 60 MiB free
#SBATCH --exclude=head058

# === Optional: force specific good nodes (only if you know them) ===
# #SBATCH --nodelist=node01,node02,node03

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