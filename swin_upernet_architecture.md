# Swin-T + UperNet Architecture for ADE20K Semantic Segmentation

## Overview

This document describes the dimensional flow for semantic segmentation using:
- **Encoder**: Swin Transformer Tiny (Swin-T)
- **Decoder**: UperNet (Pyramid Pooling Module + Feature Pyramid Network)
- **Dataset**: ADE20K (150 classes, 512×512 images)

## Architecture Summary 

```
Input [B, 3, 512, 512]
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    SWIN-T ENCODER                           │
├─────────────────────────────────────────────────────────────┤
│  Patch Embed (4×4) ──► [B, 16384, 96]                       │
│       │                                                      │
│  Stage 1: 128×128, 96-dim  ──► C1: [B, 96, 128, 128]        │
│  Stage 2:  64×64, 192-dim  ──► C2: [B, 192, 64, 64]         │
│  Stage 3:  32×32, 384-dim  ──► C3: [B, 384, 32, 32]         │
│  Stage 4:  16×16, 768-dim  ──► C4: [B, 768, 16, 16]         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    UPERNET DECODER                          │
├─────────────────────────────────────────────────────────────┤
│  PPM on C4 ──► [B, 512, 16, 16]                             │
│  FPN Fusion ──► [B, 2048, 512, 512]                         │
│  Bottleneck ──► [B, 512, 512, 512]                          │
│  Classifier ──► [B, 150, 512, 512]                          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Output [B, 150, 512, 512]
```

## Encoder: Swin-T Configuration

| Parameter | Value |
|-----------|-------|
| Variant | Tiny |
| Embed dim | 96 |
| Depths | [2, 2, 6, 2] |
| Num heads | [3, 6, 12, 24] |
| Window size | 7 |
| Patch size | 4 |
| Drop path rate | 0.2 |
| Parameters | ~28M |

### Multi-Scale Feature Extraction

| Stage | Resolution | Tokens | Channels | Padding (for window=7) |
|-------|------------|--------|----------|------------------------|
| 1 | 128×128 | 16,384 | 96 | 128→133 (pad 5) |
| 2 | 64×64 | 4,096 | 192 | 64→70 (pad 6) |
| 3 | 32×32 | 1,024 | 384 | 32→35 (pad 3) |
| 4 | 16×16 | 256 | 768 | 16→21 (pad 5) |

**Note**: Padding is applied inside each transformer block to maintain window_size=7 for proper local attention (MMSegmentation approach).

## Decoder: UperNet Configuration

| Parameter | Value |
|-----------|-------|
| Channels | 512 |
| Pool scales (PPM) | (1, 2, 3, 6) |
| Dropout | 0.1 |
| Num classes | 150 |
| Parameters | ~32M |

### Pyramid Pooling Module (PPM)

Applied to deepest feature C4:
```
C4 [B, 768, 16, 16]
    │
    ├── AdaptiveAvgPool2d(1) → Conv1×1 → [B, 512, 1, 1] → Upsample
    ├── AdaptiveAvgPool2d(2) → Conv1×1 → [B, 512, 2, 2] → Upsample
    ├── AdaptiveAvgPool2d(3) → Conv1×1 → [B, 512, 3, 3] → Upsample
    └── AdaptiveAvgPool2d(6) → Conv1×1 → [B, 512, 6, 6] → Upsample
    │
    ▼
Concat [B, 768+4×512, 16, 16] = [B, 2816, 16, 16]
    │
Bottleneck Conv3×3 → [B, 512, 16, 16]
```

### Feature Pyramid Network (FPN)

```
Lateral Connections (Conv1×1 to 512 channels):
  C1 [B, 96, 128, 128]  → L1 [B, 512, 128, 128]
  C2 [B, 192, 64, 64]   → L2 [B, 512, 64, 64]
  C3 [B, 384, 32, 32]   → L3 [B, 512, 32, 32]
  PPM [B, 512, 16, 16]  → L4 [B, 512, 16, 16]

Top-Down Pathway:
  L4 → Upsample → Add to L3 → L3'
  L3' → Upsample → Add to L2 → L2'
  L2' → Upsample → Add to L1 → L1'

Upsample all to 512×512 and concatenate:
  [B, 4×512, 512, 512] = [B, 2048, 512, 512]
```

### Final Layers

```
FPN Bottleneck: [B, 2048, 512, 512] → [B, 512, 512, 512]
Dropout (0.1)
Classifier (Conv1×1): [B, 512, 512, 512] → [B, 150, 512, 512]
```

## Complete Dimensional Flow

| Component | Input | Output |
|-----------|-------|--------|
| Input Image | - | [B, 3, 512, 512] |
| Patch Embed | [B, 3, 512, 512] | [B, 16384, 96] |
| Stage 1 | [B, 16384, 96] | [B, 16384, 96] |
| Stage 2 | [B, 16384, 96] | [B, 4096, 192] |
| Stage 3 | [B, 4096, 192] | [B, 1024, 384] |
| Stage 4 | [B, 1024, 384] | [B, 256, 768] |
| Reshape C1 | [B, 16384, 96] | [B, 96, 128, 128] |
| Reshape C2 | [B, 4096, 192] | [B, 192, 64, 64] |
| Reshape C3 | [B, 1024, 384] | [B, 384, 32, 32] |
| Reshape C4 | [B, 256, 768] | [B, 768, 16, 16] |
| PPM | [B, 768, 16, 16] | [B, 512, 16, 16] |
| FPN Concat | 4×[B, 512, H, W] | [B, 2048, 512, 512] |
| Fusion | [B, 2048, 512, 512] | [B, 512, 512, 512] |
| Classifier | [B, 512, 512, 512] | [B, 150, 512, 512] |

## Training Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Optimizer | AdamW | Paper |
| Learning rate | 6e-5 | Paper |
| Weight decay | 0.01 | Paper |
| Batch size | 16 | Config |
| Epochs | 160 | Paper (160K iter) |
| Warmup | 1500 iter (~2 epochs) | Paper |
| Pretrained | ImageNet-1K (TIMM) | Paper |

## Key Implementation Details

### Window Attention with Padding

For 512×512 input, resolutions don't divide evenly by window_size=7. We use **padding** (not global attention fallback):

```python
# In SwinTransformerBlock.forward():
pad_r = (self.window_size - W % self.window_size) % self.window_size
pad_b = (self.window_size - H % self.window_size) % self.window_size
x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
# ... apply attention ...
x = x[:, :H, :W, :].contiguous()  # Remove padding
```

This ensures:
- Pretrained weight compatibility (relative_position_bias_table is 13×13)
- True local window attention (not global)
- Matches MMSegmentation official implementation

### Feature Reshaping

Swin outputs `[B, H*W, C]` but UperNet expects `[B, C, H, W]`:

```python
def _reshape_features(features_list):
    for i, features in enumerate(features_list):
        B, N, C = features.shape
        H = W = base_resolution // (2 ** i)  # 128, 64, 32, 16
        features_2d = features.transpose(1, 2).reshape(B, C, H, W)
```

## Parameter Count

| Component | Parameters |
|-----------|------------|
| Swin-T Encoder | 27,517,818 (~28M) |
| UperNet Head | 31,497,366 (~32M) |
| **Total** | **59,015,184 (~60M)** |

## References

- Swin Transformer: [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
- UperNet: [arXiv:1807.10221](https://arxiv.org/abs/1807.10221)
- MMSegmentation: [github.com/open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- Official Swin Segmentation: [github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation)
