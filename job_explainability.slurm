#!/bin/bash
#SBATCH --partition=gpu-teaching-2h
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=swin_explainability
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --chdir=/home/pml20/swin_transformer/Machine-Learning-Project
#SBATCH --array=1-1

# Ensure logs dir exists
mkdir -p logs

# Make failures noisy
trap 'rc=$?; echo "==> SCRIPT EXITED WITH CODE $rc at $(date)"; exit $rc' EXIT

echo "========================================="
echo "JOB BOOT"
echo "Host: $(hostname)  Date: $(date)"
echo "User: $USER  PWD: $(pwd)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  NODELIST=$SLURM_JOB_NODELIST  PROCID=$SLURM_PROCID"
echo "========================================="

# Quick GPU check on host (will usually fail inside sbatch if no GPU assigned yet)
echo "Host nvidia-smi (may error if driver not available on head):"
nvidia-smi || echo "nvidia-smi failed or not available on this host"

# Module / container runtime
module load apptainer || echo "module load apptainer failed (apptainer might already be available)"

# Paths
SIF="${SLURM_SUBMIT_DIR:-$PWD}/pml.sif"
OVERLAY="/home/space/datasets-sqfs/imagenet2012.sqfs"

# Show submit dir contents (small)
echo "Submit dir: ${SLURM_SUBMIT_DIR:-$PWD}"
ls -lah "${SLURM_SUBMIT_DIR:-$PWD}" | head -n 40

# Container guard
if [ ! -f "$SIF" ]; then
  echo "ERROR: container not found at $SIF"
  exit 1
fi
echo "Found container: $SIF (size: $(stat -c%s "$SIF") bytes)"

# Overlay guard: prefer overlay if present (this is what worked before)
OVERLAY_ARG=""
if [ -f "$OVERLAY" ]; then
  echo "Found overlay: $OVERLAY"
  OVERLAY_ARG="--overlay $OVERLAY"
else
  echo "Warning: overlay not found at $OVERLAY — continuing WITHOUT overlay"
fi

# === Inside-container quick checks (prints to the log) ===
echo "Container basic checks (inside container):"
apptainer exec --nv "$SIF" bash -c 'echo "container: $(hostname)"; nvidia-smi || echo "container nvidia-smi failed"; python - <<"PY"
import sys
try:
    import torch
    print("torch:", getattr(torch,"__version__","(no torch)"), "cuda:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
except Exception as e:
    print("python inside container error:", e, file=sys.stderr)
PY' || echo "apptainer basic checks returned non-zero (inspect above)"

# Export a couple of envs inside the job (no CUDA_VISIBLE_DEVICES)
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.8
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0

# Run the actual explainability script inside the container using overlay (like your working job)
echo "Starting explainability pipeline (container run) at $(date)"
set -x
apptainer run --nv \
    ${OVERLAY_ARG} \
    --bind ${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR} \
    "$SIF" \
    python ${SLURM_SUBMIT_DIR}/main_explainability.py
EXIT_CODE=$?
set +x

echo "========================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================="

# Best-effort final GPU check (inside container)
apptainer exec --nv "$SIF" bash -c 'python - <<"PY"
import torch,sys
try:
    print("final: torch.cuda.is_available()", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("final: devices", torch.cuda.device_count())
except Exception as e:
    print("final-check error:", e, file=sys.stderr)
PY' || true

exit $EXIT_CODE
