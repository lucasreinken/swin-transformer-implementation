import torch
import torch.nn as nn


class ConvDownsample(nn.Module):
    """
    Convolutional Downsampling Layer for Single-Resolution Ablation.

    This replaces PatchMerging in the hierarchical ablation study.
    Instead of concatenating 2x2 neighboring patches and linearly projecting,
    this applies 3x3 convolution with stride 1 to maintain spatial resolution
    while changing the feature processing mechanism.

    Purpose: Alternative to PatchMerging that maintains 56×56 resolution throughout
    Operation: Reshape → 3×3 Conv (stride 1) → Reshape
    Result: Same spatial resolution, different feature transformation

    ┌────────────────────── SINGLE-RESOLUTION ABLATION ──────────────────────┐
    │                                                                       │
    │  Input: [B, H×W, C]  (e.g., [B, 3136, 96] for 56×56)                 │
    │      │                                                                │
    │      ▼ Reshape to spatial                                             │
    │  [B, H, W, C]  (e.g., [B, 56, 56, 96])                               │
    │      │                                                                │
    │      ▼ Transpose for conv                                             │
    │  [B, C, H, W]  (e.g., [B, 96, 56, 56])                               │
    │      │                                                                │
    │      ▼                                                                │
    │  ┌─────────────────────────────────────────┐                          │
    │  │  3×3 Conv, stride 1, padding 1         │                          │
    │  │  Maintains spatial resolution           │                          │
    │  │  Changes feature processing             │                          │
    │  └─────────────────────────────────────────┘                          │
    │      │                                                                │
    │      ▼ Transpose back                                                 │
    │  [B, H, W, C]  (same spatial dims)                                    │
    │      │                                                                │
    │      ▼ Flatten                                                        │
    │  [B, H×W, C]  (same output shape as input)                            │
    │                                                                       │
    └───────────────────────────────────────────────────────────────────────┘

    Why This Ablation Matters:
    1. **No Hierarchical Downsampling**: All stages work at 56×56 resolution
    2. **Different Feature Mixing**: Conv combines features differently than concatenation
    3. **Computational Cost**: Similar cost to PatchMerging but no resolution reduction
    4. **Architecture Comparison**: Tests if hierarchical structure is crucial

    Expected Result (from Swin paper Table 4(e)): ~+2.8% Top-1 accuracy drop
    This shows hierarchical design provides significant performance benefits.
    """

    def __init__(self, dim: int):
        """
        Initialize Convolutional Downsampling layer.

        Args:
            input_resolution: (H, W) spatial dimensions in patches
                            e.g., (56, 56) - maintained throughout network
            dim: Feature dimension per patch
                e.g., 96 for all layers in single-resolution ablation

        Architecture:
            - 3×3 Conv2d with stride 1, padding 1 (maintains resolution)
            - Same input/output dimensions
            - No dimension change (unlike PatchMerging which doubles channels)
        """
        super().__init__()
        self.dim = dim

        # 3×3 convolution with stride 1 to maintain resolution
        # Input: [B, dim, H, W]
        # Output: [B, dim, H, W] (same shape)
        self.conv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,  # Same output channels
            kernel_size=3,
            stride=1,  # No downsampling
            padding=1,  # Maintain spatial dimensions
            bias=False,  # Following Swin paper convention
        )

        # LayerNorm for stabilization (similar to PatchMerging)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        """
        Apply convolutional downsampling while maintaining resolution.

        Args:
            x: Input tensor [B, H*W, C]

        Returns:
            Output tensor [B, H*W, C] (same shape as input)

        Processing:
        1. Reshape flattened patches to spatial layout
        2. Apply 3×3 convolution with stride 1
        3. Normalize and reshape back to sequence format
        """
        B, HW, C = x.shape

        # Reshape to spatial layout for convolution
        # [B, H*W, C] → [B, H, W, C]
        x = x.view(B, H, W, C)

        # Transpose for Conv2d (channels first)
        # [B, H, W, C] → [B, C, H, W]
        x = x.permute(0, 3, 1, 2)

        # Apply 3×3 convolution (maintains H, W dimensions)
        x = self.conv(x)

        # Transpose back to channels last
        # [B, C, H, W] → [B, H, W, C]
        x = x.permute(0, 2, 3, 1)

        # Apply layer norm
        x = self.norm(x)

        # Flatten back to sequence format
        # [B, H, W, C] → [B, H*W, C]
        x = x.view(B, HW, C)

        return x
