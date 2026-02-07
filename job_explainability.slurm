#!/bin/bash
#SBATCH --partition=gpu-teaching-2h    # Use appropriate partition
#SBATCH --gpus=1
#SBATCH --time=01:00:00                 # 1 hour should be enough for visualization
#SBATCH --job-name=swin_explainability  # Job name
#SBATCH --output=logs/%j_%x.out         # Includes job name
#SBATCH --error=logs/%j_%x.err
#SBATCH --chdir=/home/pml20/swin_transformer/Machine-Learning-Project
#SBATCH --array=1-1                     # Single run (adjust if needed)

mkdir -p logs

# === Early sanity check / debug logging ===
echo "Job started on $(hostname) at $(date)"
nvidia-smi  # This prints GPU info (or errors if none available)

# === GPU memory configuration ===
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.8
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0

# Clear GPU memory from previous jobs
python - <<'PY'
import torch
import gc

gc.collect()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    for _ in range(3):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
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

# === Environment setup ===
echo "========================================="
echo "Swin Transformer Explainability Analysis"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "========================================="

# Activate Apptainer/Singularity container
module load apptainer

# === Python path setup ===
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1  # Force Python to use unbuffered output

# === Run explainability analysis ===
echo "Starting explainability pipeline..."
echo "Configuration: config/explainability_config.py"

apptainer exec --nv \
    --bind /scratch/swin_data:/data \
    --bind ${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR} \
    ${SLURM_SUBMIT_DIR}/pml.sif \
    python ${SLURM_SUBMIT_DIR}/main_explainability.py

EXIT_CODE=$?

# === Cleanup and summary ===
echo "========================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================="

# Final GPU memory check
python - <<'PY'
import torch
if torch.cuda.is_available():
    print("Final GPU Memory:")
    print(f"  Peak allocated: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
    print(f"  Peak reserved: {torch.cuda.max_memory_reserved(0) / 1024**3:.2f} GB")
PY

exit $EXIT_CODE
