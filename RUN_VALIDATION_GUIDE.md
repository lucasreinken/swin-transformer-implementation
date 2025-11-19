# Running Linear Probing Validation on Cluster

## Overview
This guide walks you through running model validation that compares:
1. **Pretrained model** (imported from timm - swin_tiny_patch4_window7_224)
2. **Custom implementation** (your Swin Transformer with loaded pretrained weights)

Both models will be evaluated on CIFAR-100 to verify your implementation matches the reference.

---

## Step-by-Step Instructions

### 1. Initial Setup (One-time)

After pulling the branch to the cluster, verify the setup:

```bash
# SSH to cluster
ssh pml20@hydra.ml.tu-berlin.de

# Navigate to your project directory
cd ~/Machine-Learning-Project  # or your actual project path

# Check current branch
git branch
# Should show: * validation_trial

# Verify configuration
python3 validate_config.py
# Should show: ✅ ALL CONFIGURATIONS VALID
```

### 2. Build the Container (if not already built)

```bash
# Build on CPU node (takes ~15 minutes)
srun --partition=gpu-teaching-2h --pty bash -c 'apptainer build --force pml.sif pml.def'
```

### 3. Configuration Check

The configuration is already set for validation. Verify:

```bash
grep -A 5 "VALIDATION_CONFIG" config/cifar100_config.py
```

Should show:
```python
VALIDATION_CONFIG = {
    "enable_validation": True,  # ✓ Enabled
    "use_swin_transformer": True,
    "pretrained_model": "swin_tiny_patch4_window7_224",
    "transfer_weights": True,
    "validation_samples": 1000,
}
```

### 4. Test Run (Interactive - Recommended First)

Test the validation interactively to catch any issues:

```bash
srun --partition=gpu-test --gpus=1 --time=00:30:00 --pty bash -c '
    apptainer run --nv pml.sif python main.py
'
```

**What happens:**
- Downloads CIFAR-100 dataset (if not cached)
- Loads pretrained swin_tiny from timm/HuggingFace
- Creates your custom Swin Transformer
- Transfers weights to custom model (173 layers)
- Evaluates both models on 1000 validation samples
- Compares Top-1 and Top-5 accuracy
- Saves results to `runs/run_XX/model_validation_results.json`

**Expected output:**
```
INFO - Loaded pretrained model: swin_tiny_patch4_window7_224 from timm.
INFO - Transferring weights from pretrained to custom model...
INFO - Weight transfer: 173 layers transferred.
INFO - Weight transfer completed: {'transferred': 173, 'missing': 0, 'size_mismatches': 0}
INFO - Using subset of 1000 samples for validation
...
INFO - === MODEL COMPARISON RESULTS ===
INFO - Pretrained Model  - Top-1: XX.XX%, Top-5: XX.XX%
INFO - Custom Model      - Top-1: XX.XX%, Top-5: XX.XX%
INFO - Differences       - Top-1: X.XX%, Top-5: X.XX%
```

### 5. Submit Batch Job (For Unattended Run)

Once the interactive test works, submit as batch job:

```bash
sbatch job.slurm
```

**Monitor the job:**
```bash
# Check job status
squeue -u $USER

# Watch output in real-time (replace JOBID)
tail -f logs/JOBID.out

# Check for errors
tail -f logs/JOBID.err
```

### 6. Review Results

After completion:

```bash
# Find the latest run
ls -lt runs/

# Check validation results
cat runs/run_XX/model_validation_results.json

# View training log
less runs/run_XX/training.log

# Check comparison plot (if generated)
# Download: scp pml20@hydra.ml.tu-berlin.de:~/Machine-Learning-Project/runs/run_XX/model_validation_comparison.png .
```

**Expected results file:**
```json
{
  "custom_model": {
    "top1_accuracy": 65.5,
    "top5_accuracy": 87.2,
    "total_samples": 1000
  },
  "pretrained_model": {
    "top1_accuracy": 65.8,
    "top5_accuracy": 87.5,
    "total_samples": 1000
  },
  "differences": {
    "top1_diff": 0.3,
    "top5_diff": 0.3
  }
}
```

---

## Validation Modes

### Mode 1: Quick Validation (Current Setup)
- **Samples:** 1000 from validation set
- **Time:** ~5-10 minutes
- **Purpose:** Quick verification during development
- **Good for:** Checking if implementation is correct

### Mode 2: Full Validation (Optional)

Edit `config/cifar100_config.py`:
```python
VALIDATION_CONFIG = {
    "enable_validation": True,
    "validation_samples": 10000,  # Full test set
}
```

Then update job time:
```bash
#SBATCH --time=01:00:00  # 1 hour for full validation
```

---

## Troubleshooting

### Issue: "No module named 'timm'"

**Solution:** The container should have timm. Verify:
```bash
srun --partition=gpu-test --gpus=1 --pty bash -c '
    apptainer run --nv pml.sif python -c "import timm; print(timm.__version__)"
'
```

If missing, rebuild container or install in container definition.

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size or validation samples:
```python
DATA_CONFIG = {
    "batch_size": 16,  # Reduce from 32
}
VALIDATION_CONFIG = {
    "validation_samples": 500,  # Reduce from 1000
}
```

### Issue: "Weight transfer failed"

Check the log for specific errors:
```bash
grep -i "transfer\|error\|mismatch" logs/JOBID.err
```

Verify configuration:
```bash
python3 validate_config.py
```

### Issue: "Dataset not found"

The dataset should auto-download. If issues persist:
```bash
# Check dataset path
ls -la /home/space/datasets/cifar-100-python/

# Or use local download
# Edit config/__init__.py:
# DATA_ROOT = "./datasets"
```

---

## After Validation: Run Linear Probing Training

Once validation confirms your implementation is correct, disable validation and run actual training:

### 1. Disable Validation

```python
# config/cifar100_config.py
VALIDATION_CONFIG = {
    "enable_validation": False,  # Disable validation
    "use_swin_transformer": True,
    "pretrained_model": "swin_tiny_patch4_window7_224",
    "transfer_weights": True,
}
```

### 2. Update Job Script for Training

```bash
# job.slurm
#SBATCH --time=04:00:00  # 4 hours for 50 epochs
#SBATCH --job-name=cifar100_linear_probe
```

### 3. Submit Training Job

```bash
sbatch job.slurm
```

**Expected training output:**
- 50 epochs of training (only classification head)
- Validation metrics every epoch
- Test metrics every 5 epochs
- Final test accuracy on full test set

---

## Expected Performance

### Validation Results (Pretrained weights, no training)
- **Top-1 Accuracy:** ~5-15% (random features, untrained head)
- **Top-5 Accuracy:** ~20-35%
- **Difference:** Should be < 1% between models

### After Linear Probing (50 epochs training)
- **Top-1 Accuracy:** ~55-65% (depending on hyperparameters)
- **Top-5 Accuracy:** ~80-90%

---

## Quick Commands Reference

```bash
# Build container
srun --partition=gpu-teaching-2h --pty bash -c 'apptainer build --force pml.sif pml.def'

# Interactive validation test
srun --partition=gpu-test --gpus=1 --time=00:30:00 --pty bash -c 'apptainer run --nv pml.sif python main.py'

# Submit batch job
sbatch job.slurm

# Check job status
squeue -u $USER

# View logs
tail -f logs/JOBID.out

# Validate config
python3 validate_config.py

# Check results
cat runs/run_XX/model_validation_results.json
```

---

## What Gets Validated

✅ **Encoder Implementation**
- Patch embedding layer
- 4 hierarchical stages with Swin Transformer blocks
- Window attention mechanism
- Patch merging layers
- Layer normalization

✅ **Weight Loading**
- All 173 encoder layers transferred successfully
- No dimension mismatches
- No missing keys

✅ **Forward Pass**
- Feature extraction matches pretrained model
- Output dimensions correct
- Numerical stability

✅ **Classification Head**
- Correct input dimension (768 features)
- Correct output dimension (100 classes)
- Proper initialization

---

## Success Criteria

Your implementation is **correct** if:

1. ✅ Weight transfer shows: `transferred: 173, missing: 0, size_mismatches: 0`
2. ✅ Top-1 accuracy difference < 1%
3. ✅ Top-5 accuracy difference < 1%
4. ✅ No runtime errors during forward pass
5. ✅ Validation completes in reasonable time (~10 min for 1000 samples)

---

## Next Steps After Validation

1. ✅ **Verify Results** - Check that accuracies are similar
2. 🔄 **Run Full Training** - Disable validation, train for 50 epochs
3. 📊 **Analyze Results** - Compare training curves, final accuracy
4. 📝 **Report Findings** - Document linear probing performance
5. 🚀 **Fine-tuning** (Optional) - Unfreeze encoder, full fine-tuning

Good luck! 🎉
