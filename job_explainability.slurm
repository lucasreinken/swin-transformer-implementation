#!/bin/bash
#SBATCH --partition=gpu-teaching-2h
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=swin_explainability
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --chdir=/home/pml20/swin_transformer/Machine-Learning-Project
#SBATCH --array=1-1

# ensure logs dir exists
mkdir -p logs

# make failures noisy
trap 'rc=$?; echo "==> SCRIPT EXITED WITH CODE $rc at $(date)"; exit $rc' EXIT

# === Early sanity check / debug logging ===
echo "========================================="
echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST  SLURM_PROCID=$SLURM_PROCID"
echo "USER=$USER  PWD=$(pwd)"
echo "Python from: $(command -v python || echo "python not in PATH")"
echo "Checking GPU availability (nvidia-smi):"
nvidia-smi || echo "nvidia-smi failed or no GPU assigned"
echo "========================================="

# === GPU memory configuration ===
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.8
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0

# Clear GPU memory from previous jobs (best-effort)
python - <<'PY' || echo "Python gpu-clear check failed"
import torch, gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    for _ in range(3):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, Total: {props.total_memory/1024**3:.2f} GB")
else:
    print("CUDA not available")
PY

# show a little of the submit dir for debug
echo "Contents of submit dir: ${SLURM_SUBMIT_DIR:-$PWD}"
ls -lah "${SLURM_SUBMIT_DIR:-$PWD}" | head -n 40

# === Container guard ===
SIF="${SLURM_SUBMIT_DIR:-$PWD}/pml.sif"
if [ ! -f "$SIF" ]; then
  echo "ERROR: container not found at $SIF"
  exit 1
fi
echo "Found container: $SIF (size: $(stat -c%s "$SIF") bytes)"

# load apptainer if needed (safe if module is absent)
module load apptainer || echo "module load apptainer failed (maybe apptainer available already)"

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

# === Run explainability analysis ===
echo "Starting explainability pipeline..."
echo "Configuration: config/explainability_config.py"
set -x
apptainer exec --nv \
    --bind /scratch/swin_data:/data \
    --bind ${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR} \
    "$SIF" \
    python ${SLURM_SUBMIT_DIR}/main_explainability.py
EXIT_CODE=$?
set +x

echo "========================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================="

# Final GPU memory check (best-effort)
python - <<'PY' || true
import torch
if torch.cuda.is_available():
    print("Final GPU peak allocated (GB):", torch.cuda.max_memory_allocated(0)/1024**3)
else:
    print("CUDA not available on final check")
PY

exit $EXIT_CODE
