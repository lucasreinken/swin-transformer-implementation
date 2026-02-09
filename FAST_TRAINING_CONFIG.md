# Fast Training Configuration (50 Epochs)

**Date**: February 9, 2026
**Reason**: Time constraints for model comparison experiments
**Target**: Quick but meaningful comparison of all Swin variants with corrected position bias

---

## Configuration Changes

### Training Parameters (Updated in `config/imagenet_config.py`)

| Parameter | Original (300 epochs) | New (50 epochs) | Scaling Rationale |
|-----------|----------------------|-----------------|-------------------|
| **Epochs** | 300 | **50** | 6× reduction for time savings |
| **Learning Rate** | 2e-4 | **5e-4** | 2.5× increase for faster convergence |
| **Warmup Epochs** | 30 (10%) | **5** (10%) | Maintain 10% warmup ratio |
| **Weight Decay** | 0.02 | **0.05** | 2.5× increase for stronger regularization (shorter training) |
| **Min LR** | 1e-5 | **1e-5** | Unchanged (cosine decay endpoint) |

### SLURM Job Settings (Updated in `job.slurm`)

| Setting | Original | New | Change |
|---------|----------|-----|--------|
| **Time Limit** | 48:00:00 | **16:00:00** | 16 hours (sufficient for 50 epochs) |
| **Job Name** | imagenet_swin_hybrid_300ep | **imagenet_swin_50ep** | Reflects new epoch count |
| **Partition** | gpu-teaching-2d | **gpu-teaching-2d** | Unchanged (still under 2 days) |

---

## Time Estimates

### Per Model Training Time
- **50 epochs @ ~6 min/epoch** = ~5-6 hours per model
- **With overhead**: ~7-8 hours per model (safe estimate)

### Total Time for All Models
Assuming sequential training:
1. Baseline Swin: ~7 hours
2. Swin V2: ~7 hours
3. Swin Hybrid: ~7 hours
4. Swin Improved: ~7 hours
5. *(Optional) Swin Deformable: ~7 hours*

**Total**: ~28-35 hours for 4-5 models

### Parallel Option
With multiple GPUs or job arrays:
- **4 models in parallel**: ~7-8 hours total
- **Recommended**: Train 2-3 at a time to avoid resource contention

---

## Expected Performance

### With 50 Epochs (vs 300)
- **Typical convergence**: 85-90% of final accuracy
- **300 epochs**: ~41-45% accuracy (with position bias)
- **50 epochs**: ~38-42% accuracy (estimated)

### Performance Ranking (Expected)
1. **Swin V2**: Best overall (all 3 improvements + position bias)
2. **Swin Improved**: Strong (conv stem + inverted FFN + position bias)
3. **Swin Hybrid**: Good (CNN stem + position bias)
4. **Baseline Swin**: Baseline (now with position bias)

**Key Point**: Relative performance ordering should be preserved, even if absolute accuracy is lower than full 300-epoch training.

---

## Training Strategy

### Recommended Sequence
1. **Start with Baseline Swin** (establish corrected baseline)
2. **Run Swin V2** (verify all 3 improvements work)
3. **Run Swin Improved & Hybrid** (your milestone variants)
4. *Optional: Deformable* (if time permits)

### Monitoring
- Check validation accuracy at epochs: 10, 25, 40, 50
- Early indicators by epoch 25 (~halfway)
- Final comparison at epoch 50

---

## Configuration Files Modified

### 1. `config/imagenet_config.py` - TRAINING_CONFIG
```python
TRAINING_CONFIG = {
    "learning_rate": 5e-4,      # Was: 2e-4
    "num_epochs": 50,           # Was: 300
    "warmup_epochs": 5,         # Was: 30
    "weight_decay": 0.05,       # Was: 0.02
    # ... other settings unchanged
}
```

### 2. `job.slurm` - SLURM Headers
```bash
#SBATCH --time=16:00:00                    # Was: 48:00:00
#SBATCH --job-name=imagenet_swin_50ep      # Was: imagenet_swin_hybrid_300ep
```

---

## Model Configurations (All Fixed with Position Bias)

All Swin models now have **`use_relative_bias: True`** (fixed earlier):
- ✅ `swin` - Discrete position bias table
- ✅ `swin_v2` - Log-CPB (continuous position bias)
- ✅ `swin_hybrid` - Discrete position bias table
- ✅ `swin_improved` - Discrete position bias table
- ✅ `swin_deformable` - Discrete position bias table

---

## Running the Experiments

### Submit Job
```bash
# For each model, edit config/imagenet_config.py:
# MODEL_TYPE = "swin"        # Or "swin_v2", "swin_hybrid", "swin_improved"

# Submit job
sbatch job.slurm
```

### Check Progress
```bash
# Monitor job
squeue -u $USER

# Watch training log
tail -f logs/<job_id>_imagenet_swin_50ep.out

# Check GPU usage
srun --jobid=<job_id> nvidia-smi
```

### Compare Results
After all models complete:
```bash
# Extract final accuracies from logs
grep "Epoch 50/50" logs/*.out
```

---

## Important Notes

### Why These Settings?

1. **Higher LR (5e-4 vs 2e-4)**: Faster convergence needed for shorter schedule
2. **Higher Weight Decay (0.05 vs 0.02)**: Prevent overfitting with less epochs
3. **Same Warmup Ratio (10%)**: Maintain training stability
4. **Same Augmentation**: Don't reduce regularization techniques

### Trade-offs

**Pros**:
- ✅ 6× faster experiments
- ✅ Quick model comparison
- ✅ Maintains relative performance ordering
- ✅ Sufficient for architectural comparison

**Cons**:
- ⚠️ Lower absolute accuracy (85-90% of full potential)
- ⚠️ May not fully converge (but good enough for comparison)
- ⚠️ Less reliable for final production numbers

### When to Use Full 300 Epochs

After identifying the best variant with 50 epochs, consider full training:
- Final model selection
- Paper/report results
- Production deployment
- Benchmark comparisons

---

## Success Criteria

### Minimum Goals (50 epochs)
- Baseline Swin: **>38%** accuracy
- Swin V2: **>40%** accuracy (should beat baseline)
- Swin Improved: **>39%** accuracy
- Swin Hybrid: **>38%** accuracy

### Comparison Goals
- Clear ranking between variants
- Swin V2 demonstrates all 3 improvements
- Position bias impact visible (vs old buggy results)

---

## Troubleshooting

### If Training is Still Too Slow
- Reduce `n_train` from 100K to 50K samples
- Increase batch size (if memory allows)
- Use gradient checkpointing for larger models

### If Accuracy is Very Low (<30%)
- Check position bias is enabled (`use_relative_bias=True`)
- Verify data augmentation settings
- Monitor for gradient issues (NaN loss)

### If Running Out of Time
- Priority order: Baseline → V2 → Improved → Hybrid
- Skip deformable if necessary
- Use array jobs for parallel training

---

## Ready to Train!

All configurations are set for fast 50-epoch training. Time savings: **~140 hours saved** (from 168h to 28h for 4 models).

Start with:
```bash
# 1. Set model in config
vim config/imagenet_config.py  # Change MODEL_TYPE

# 2. Submit job
sbatch job.slurm

# 3. Monitor
tail -f logs/*.out
```

Good luck! 🚀
