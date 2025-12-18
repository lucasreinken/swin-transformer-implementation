#!/bin/bash
#SBATCH --partition=gpu-teaching-2d    # your long partition
#SBATCH --gpus=1
#SBATCH --exclusive                      # ← NO sharing with other users → fixes node contention
#SBATCH --time=48:00:00                  # 168 hours for 15-epoch training
#SBATCH --job-name=imagenet_model_comparison_15ep        # job name
#SBATCH --output=logs/%j_%x.out          # includes job name → easier to find
#SBATCH --error=logs/%j_%x.err
#SBATCH --chdir=/home/pml04/swin_transformer/Machine-Learning-Project
#SBATCH --array=1-1                      # Adjust as needed for multiple runs

mkdir -p logs

# === Best-practice memory fixes (keep all of these!) ===
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.8
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
export CUDA_VISIBLE_DEVICES=0  # Ensure only GPU 0 is used

# Clear any leftover GPU memory from previous jobs
python - <<'PY'
import torch
import gc
import os

# Force garbage collection
gc.collect()

if torch.cuda.is_available():
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    
    # Empty cache multiple times
    for _ in range(3):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Check memory status
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    
    print(f"GPU Memory Status:")
    print(f"  Total: {total_memory:.2f} GB")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved: {reserved:.2f} GB")
    print(f"  Free: {total_memory - allocated:.2f} GB")
else:
    print("CUDA not available")
PY

echo "Job started on node: $(hostname)"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"

# === Run your training ===
echo "Starting training at $(date)"
echo "GPU Memory before training:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits

apptainer run --nv \
    --overlay /home/space/datasets-sqfs/imagenet2012.sqfs \
    pml.sif \
    python main.py

echo "Training completed at $(date)"
echo "GPU Memory after training:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits

echo "Job finished"