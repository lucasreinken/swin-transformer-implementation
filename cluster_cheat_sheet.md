# Cluster Setup & GPU Training Guide

> **TU Berlin ML Cluster (hydra.ml.tu-berlin.de)**  
> **SLURM + Apptainer**  
> **Goal**: Run your ML code (Swin Transformer, etc.) on GPU


Build Container (CPU node, ~15 min)
srun --partition=gpu-teaching-2h --pty bash -c 'apptainer build --force pml.sif pml.def'


Run GPU Jobs
Interactive Test (Debugging)
srun --partition=gpu-test --gpus=1 --pty bash -c '
apptainer run --nv pml.sif python - <<PY
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
PY
'


Interactive Training
srun --partition=gpu-test --gpus=1 --pty bash -c '
apptainer run --nv pml.sif python train.py --batch-size 64 --epochs 5
'


Batch Job (Long Training)
cat > run_gpu.sh <<'EOS'
#!/bin/bash
#SBATCH --partition=gpu-test      # use gpu for >2h
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=swin_train

mkdir -p logs
apptainer run --nv $HOME/ml_project/pml.sif python train.py "$@"
EOS
chmod +x run_gpu.sh


Submit
sbatch run_gpu.sh --epochs 100 --lr 1e-4


Monitor
squeue -u $USER


View Logs
cat logs/<jobid>.out


One-Liner Quick Test
srun --partition=gpu-test --gpus=1 --pty bash -c '
cd ~/ml_project && git pull origin main && apptainer run --nv pml.sif python train.py --epochs 1
'

