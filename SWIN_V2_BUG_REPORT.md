# Swin Models Configuration Bug Report & Fix

**Date**: February 9, 2026
**Issue**: ALL Swin models had position bias disabled
**Status**: ✅ FIXED

## Critical Bug Found

### **BUG: Position Bias Disabled for ALL Swin Models** ⚠️ CRITICAL

**Location**: `config/imagenet_config.py`

**Problem**: All Swin variants had `"use_relative_bias": False`, which completely disabled position bias—a fundamental component of Swin Transformers.

**Affected Models**:
- ✅ `swin` (baseline) - **FIXED**
- ✅ `swin_v2` - **FIXED**
- ✅ `swin_hybrid` - **FIXED**
- ✅ `swin_improved` - **FIXED**
- ✅ `swin_deformable` - **FIXED**

---

## Impact Analysis

### What Position Bias Does

Position bias in Swin Transformer encodes **spatial relationships** between patches:
- **Without it**: Attention is purely content-based (no notion of "where" tokens are spatially)
- **With it**: Attention learns that nearby patches are more relevant than distant ones

### Performance Impact

**Expected improvement with position bias enabled**:
- Standard Swin: +2-5% accuracy
- Swin V2 (Log-CPB): +2-5% accuracy + resolution transfer capability

### Why This Explains Poor Performance

Your baseline achieved **39.18% without position bias**. The Swin V2 model was also missing position bias, which explains why it performed worse than expected—it was only using 2 out of 3 improvements:

1. ✅ Post-normalization (active)
2. ✅ Scaled cosine attention (active)  
3. ❌ **Log-CPB position bias (DISABLED)**

---

## Fix Applied

Changed all Swin model configurations from:
```python
"use_relative_bias": False,  # ❌ Wrong
```

To:
```python
"use_relative_bias": True,  # ✅ Correct
```

### Specific Changes

**Lines changed in `config/imagenet_config.py`:**
- Line 33: `swin` baseline
- Line 52: `swin_v2`
- Line 71: `swin_hybrid`
- Line 100: `swin_improved`
- Line 136: `swin_deformable`

---

## Implementation Verification

### ✅ Standard Swin (V1): Uses Discrete Position Bias Table

**Location**: `src/models/swin/window_attention.py` - `WindowAttention` class

When `use_relative_bias=True`:
- Creates learnable parameter table: `(2*W-1)² × num_heads`
- Indexes into table based on relative positions
- Adds bias to attention scores before softmax

**Implementation verified correct** ✓

---

### ✅ Swin V2: Uses Log-CPB (Continuous Position Bias)

**Location**: `src/models/swin/window_attention.py` - `WindowAttentionV2` class

When `use_relative_bias=True` (now enabled):
- Uses `LogCPB` module with 3-layer MLP (2→512→512→heads)
- Log-spacing transformation: `sign(Δ) * log(1 + |Δ|)`
- Generates continuous bias for any window size

**Components verified**:
- ✅ Post-normalization: Correct
- ✅ Scaled cosine attention: Correct (Q/K normalization, temperature scaling)
- ✅ Log-CPB: Correct implementation (but was disabled by config)

**All Swin V2 components now active** ✓

---

## Expected Results After Fix

### Baseline Swin (with position bias)
- **Previous (buggy)**: 39.18% (no position bias)
- **Expected (fixed)**: **41-44%** (+2-5% improvement)

### Swin V2 (with all 3 improvements)
- **Previous (buggy)**: Worse than baseline (only 2/3 improvements)
- **Expected (fixed)**: **42-45%** (+1.5-2.5% over fixed baseline)
  - Post-norm: +0.5-1%
  - Scaled cosine: +0.5-1%  
  - Log-CPB: +0.5-1%

### Other Variants
All Swin variants should see **+2-5% improvement** with position bias enabled.

---

## Why This Bug Existed

The `use_relative_bias=False` setting appears to have been:
1. Originally intended for ablation studies (testing Swin without position bias)
2. Accidentally left disabled in all production configs
3. Never caught because the models still trained (just poorly)

Position bias is **not optional** for proper Swin Transformer performance—it's a core architectural component described in the original paper.

---

## Recommendation

### Immediate Actions
1. ✅ **DONE**: Fixed all config files
2. ⏳ **TODO**: Re-train all Swin models with corrected configs
3. ⏳ **TODO**: Update baseline results in documentation

### Training Priority
1. **Swin baseline** - Establish proper baseline performance
2. **Swin V2** - Verify all 3 improvements work together
3. **Other variants** - Test with corrected position bias

---

## Summary

**Root Cause**: Configuration error disabled position bias across all Swin models

**Fix**: Changed `use_relative_bias` from `False` to `True` in 5 model configs

**Expected Impact**: +2-5% accuracy improvement for all Swin variants

**Code Quality**: All implementations are correct; only config was wrong

**Status**: ✅ Ready for re-training with corrected configurations

