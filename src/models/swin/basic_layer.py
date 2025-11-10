import torch
import torch.nn as nn

from .swin_transformer_block import SwinTransformerBlock


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer Layer: A stage in the hierarchical architecture.

    This module represents one stage in Swin Transformer, consisting of:
    1. A sequence of Swin Transformer Blocks (alternating W-MSA and SW-MSA)
    2. Optional Patch Merging for downsampling (except in the last stage)

    ┌─────────────────────── BASIC LAYER STRUCTURE ────────────────────────┐
    │                                                                       │
    │  Input: [B, H×W, C]                                                   │
    │      │                                                                │
    │      ▼                                                                │
    │  ┌─────────────────────────────────────────┐                          │
    │  │  Swin Block 0 (W-MSA, shift_size=0)    │                          │
    │  └─────────────────────────────────────────┘                          │
    │      │                                                                │
    │      ▼                                                                │
    │  ┌─────────────────────────────────────────┐                          │
    │  │  Swin Block 1 (SW-MSA, shift_size=3)   │  ← Shifted windows       │
    │  └─────────────────────────────────────────┘                          │
    │      │                                                                │
    │      ▼                                                                │
    │  ┌─────────────────────────────────────────┐                          │
    │  │  Swin Block 2 (W-MSA, shift_size=0)    │                          │
    │  └─────────────────────────────────────────┘                          │
    │      │                                                                │
    │      ▼                                                                │
    │      ...  (repeat depth times)                                        │
    │      │                                                                │
    │      ▼                                                                │
    │  ┌─────────────────────────────────────────┐                          │
    │  │  Patch Merging (optional)               │  ← Downsample            │
    │  │  [B, H×W, C] → [B, H/2×W/2, 2C]        │                          │
    │  └─────────────────────────────────────────┘                          │
    │      │                                                                │
    │      ▼                                                                │
    │  Output: [B, (H/2)×(W/2), 2C]  (if downsampling)                     │
    │          [B, H×W, C]            (if no downsampling)                 │
    │                                                                       │
    └───────────────────────────────────────────────────────────────────────┘

    Alternating W-MSA and SW-MSA Pattern:
    - **Even blocks (0, 2, 4, ...)**: W-MSA (shift_size = 0)
    - **Odd blocks (1, 3, 5, ...)**: SW-MSA (shift_size = window_size // 2)

    This alternation enables:
    - Local attention within windows (W-MSA)
    - Cross-window connections (SW-MSA)
    - Effective global receptive field with linear complexity

    Typical Swin Architecture:
    - Stage 1: depth=2,  56×56, 96-dim   (no downsampling at start)
    - Stage 2: depth=2,  28×28, 192-dim  (after first merge)
    - Stage 3: depth=6,  14×14, 384-dim  (after second merge)
    - Stage 4: depth=2,  7×7,   768-dim  (after third merge, no merge at end)
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
        downsample: nn.Module = None,
    ):
        """
        Initialize Basic Layer.

        Args:
            dim: Feature dimension
            input_resolution: (H, W) input spatial resolution
            depth: Number of Swin blocks in this layer
            num_heads: Number of attention heads
            window_size: Local window size (default: 7)
            mlp_ratio: Ratio of MLP hidden dim to embedding dim (default: 4.0)
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            drop_path: Stochastic depth rate (can be a list for varying rates)
            downsample: Downsample layer at the end (typically PatchMerging or None)

        Example:
            # Stage 1: 2 blocks at 56×56 resolution, 96 dims, no downsampling
            layer1 = BasicLayer(
                dim=96,
                input_resolution=(56, 56),
                depth=2,
                num_heads=3,
                window_size=7,
                downsample=None
            )

            # Stage 2: 2 blocks at 28×28 resolution, 192 dims, with downsampling
            layer2 = BasicLayer(
                dim=192,
                input_resolution=(28, 28),
                depth=2,
                num_heads=6,
                window_size=7,
                downsample=PatchMerging
            )
        """
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # Build Swin Transformer blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(
                        0 if (i % 2 == 0) else window_size // 2
                    ),  # Alternate W-MSA and SW-MSA
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                )
                for i in range(depth)
            ]
        )

        # Optional downsampling layer (Patch Merging)
        if downsample is not None:
            self.downsample = downsample(input_resolution=input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Basic Layer.

        Args:
            x: Input tensor [B, H*W, C]

        Returns:
            Output tensor [B, H*W, C] or [B, (H/2)*(W/2), 2C] if downsampling

        Processing:
        1. Pass through each Swin block sequentially
        2. Blocks alternate between W-MSA and SW-MSA
        3. Apply optional downsampling at the end
        """
        # Pass through all Swin Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Optional downsampling (e.g., PatchMerging)
        if self.downsample is not None:
            x = self.downsample(x)

        return x
