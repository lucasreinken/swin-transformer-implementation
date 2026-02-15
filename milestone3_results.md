# Milestone 3: Swin Architecture Comparison on ImageNet-1K - Results Summary

## 📊 Table 1: Final Performance Comparison (20 Epochs)

| Model | Test Accuracy | Test Loss | Val Accuracy | Val Loss | Parameters | Δ Params vs Baseline |
|-------|--------------|-----------|--------------|----------|------------|---------------------|
| **Swin Baseline** | **16.82%** | 4.3337 | 16.82% | 4.3337 | 28.29M | - |
| **Swin V2** | **15.32%** | 4.4024 | 15.32% | 4.4024 | 31.51M | +3.22M (+11.4%) |
| **Swin Improved** | **20.00%** | 4.0634 | 20.00% | 4.0634 | 28.52M | +0.23M (+0.8%) |

**Key Findings:**
- ✅ **Swin Improved achieves +3.18% absolute improvement** over baseline with minimal parameter overhead
- ❌ **Swin V2 underperforms baseline by -1.50%** despite 11.4% more parameters
- 🏆 **Best model: Swin Improved** (20.00% accuracy, 28.52M params)

---

## 📈 Table 2: Training Progress by Checkpoint

| Epoch | Swin Baseline | Swin V2 | Swin Improved | Best Model |
|-------|---------------|---------|---------------|------------|
| **5** | 2.44% | 1.97% | 4.72% | Improved (+2.28%) |
| **10** | 7.70% | 5.61% | 11.73% | Improved (+4.03%) |
| **15** | 14.01% | 11.96% | 17.97% | Improved (+3.96%) |
| **20** | 16.82% | 15.32% | 20.00% | Improved (+3.18%) |

**Observations:**
- Swin Improved maintains **consistent lead** throughout training
- Gap widens from epoch 5 (+2.28%) to epoch 10 (+4.03%), then stabilizes
- Swin V2 consistently lags behind both other variants

---

## ⏱️ Table 3: Training Efficiency

| Model | Total Time | Time per Epoch | GPU | Compile Time |
|-------|-----------|----------------|-----|--------------|
| **Swin Baseline** | 7h 8min | ~21.4 min | A100 80GB | 66s (first forward pass) |
| **Swin V2** | 18h 14min | ~54.7 min | A100 80GB | N/A |
| **Swin Improved** | 7h 0min | ~21.0 min | A100 80GB | N/A |

**Notes:**
- Swin Baseline: 19:43 → 02:52 (next day) = 7h 8min
- Swin V2: 12:46 → 07:00 (next day) = 18h 14min ⚠️ (unexpectedly slow)
- Swin Improved: 03:01 → 10:01 = 7h 0min

---

## 🔬 Table 4: Convergence Analysis

| Metric | Swin Baseline | Swin V2 | Swin Improved |
|--------|---------------|---------|---------------|
| **Initial Loss (Epoch 1)** | 6.8848 | 6.9084 | 6.5943 |
| **Final Train Loss** | 4.5997 | 4.6722 | 4.2684 |
| **Total Loss Reduction** | 2.2851 | 2.2362 | 2.3259 |
| **Best Epoch (Val Acc)** | Epoch 20 | Epoch 20 | Epoch 20 |
| **Convergence Quality** | Stable | Stable | **Best** |

---

## 💡 Table 5: Architecture-Specific Details

| Component | Swin Baseline | Swin V2 | Swin Improved |
|-----------|---------------|---------|---------------|
| **Patch Embedding** | Standard 4×4 conv | Standard 4×4 conv | **Overlapping conv stem** (4→3 kernel) |
| **Attention Mechanism** | Shifted Window | **Scaled Cosine + Log-CPB** | Shifted Window |
| **Normalization** | Pre-norm (LN before attn) | **Post-norm** (LN after attn) | Pre-norm |
| **FFN Design** | Standard MLP | Standard MLP | **Inverted Residual** (4× expansion + DWConv) |
| **Position Encoding** | Relative Position Bias | Log-spaced CPB | Relative Position Bias |
| **Key Innovation** | - (baseline) | Temperature-scaled attention | Conv stem + Mobile-FFN |

---

## 📉 Table 6: Loss Progression Comparison

| Epoch | Baseline Train/Val | V2 Train/Val | Improved Train/Val |
|-------|-------------------|--------------|-------------------|
| 1 | 6.88 / 6.77 | 6.91 / 6.79 | 6.59 / 6.36 |
| 5 | 6.18 / 5.91 | 6.28 / 6.06 | 5.87 / 5.47 |
| 10 | 5.53 / 5.14 | 5.75 / 5.37 | 5.16 / 4.72 |
| 15 | 4.91 / 4.55 | 5.05 / 4.67 | 4.57 / 4.23 |
| 20 | 4.60 / 4.33 | 4.67 / 4.40 | 4.27 / 4.06 |

---

## 🎯 Key Takeaways for Report

1. **Swin Improved is the clear winner:**
   - +3.18% absolute accuracy improvement (19% relative improvement)
   - Better convergence (lower final loss: 4.06 vs 4.33)
   - Minimal parameter overhead (+0.8%)
   - Same training time as baseline (~7 hours)

2. **Swin V2 underperformed expectations:**
   - -1.50% worse than baseline
   - 11.4% more parameters with no benefit
   - 2.5× longer training time (18h vs 7h) - potential implementation issue
   - Possible causes: post-norm less suitable for from-scratch training, or implementation bug

3. **Architectural insights:**
   - **Convolutional stem** (Improved) provides better inductive bias than standard patch embedding
   - **Inverted residual FFN** improves feature mixing efficiency
   - **Scaled cosine attention** (V2) may require pretrained weights to be effective

4. **Training configuration:**
   - All models trained with identical settings (LR=5e-4, 20 epochs, warmup=3)
   - Dataset: 100K training samples, 50K validation/test (stratified ImageNet-1K)
   - Hardware: NVIDIA A100 80GB, mixed precision (bf16)

---

## 📝 Raw Data Summary

### Swin Baseline (ms3_swin_baseline_20_epochs_3)
- **Parameters:** 28,288,354 (28.29M)
- **Training Time:** 2026-02-13 19:43:35 → 2026-02-14 02:52:07 (7h 8min)
- **Final Results (Epoch 20):**
  - Train Loss: 4.5997
  - Val Loss: 4.3337
  - Test Loss: 4.3337
  - Test Accuracy: 16.82%

### Swin V2 (ms3_swin_v2_20_epochs_2)
- **Parameters:** 31,507,666 (31.51M)
- **Training Time:** 2026-02-12 12:46:47 → 2026-02-13 07:00:32 (18h 14min)
- **Final Results (Epoch 20):**
  - Train Loss: 4.6722
  - Val Loss: 4.4024
  - Test Loss: 4.4024
  - Test Accuracy: 15.32%

### Swin Improved (ms3_swin_improved_20_epochs_5)
- **Parameters:** 28,518,706 (28.52M)
- **Training Time:** 2026-02-14 03:01:32 → 2026-02-14 10:01:55 (7h 0min)
- **Final Results (Epoch 20):**
  - Train Loss: 4.2684
  - Val Loss: 4.0634
  - Test Loss: 4.0634
  - Test Accuracy: 20.00%
