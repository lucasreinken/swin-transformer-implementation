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
        projection_dropout: float = 0.0,
        drop_path: float = 0.0,
        downsample: nn.Module = None,
        downsample_input_dim: int = None,
        use_shifted_window: bool = True,  # Ablation flag: True for alternating W-MSA/SW-MSA, False for W-MSA only
        use_relative_bias: bool = True,  # Ablation flag: True for learned bias, False for zero bias
        use_absolute_pos_embed: bool = False,  # Ablation flag: True for absolute pos embed (ViT-style)
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
            projection_dropout: Projection dropout rate
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

        # Determine the input resolution for blocks
        # If downsampling exists, blocks work at the downsampled resolution
        if downsample is not None:
            block_input_resolution = [
                input_resolution[0] // 2,
                input_resolution[1] // 2,
            ]
        else:
            block_input_resolution = input_resolution

        # Build Swin Transformer blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=block_input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(
                        0
                        if (i % 2 == 0 or not use_shifted_window)
                        else window_size // 2
                    ),  # Alternate W-MSA and SW-MSA if use_shifted_window=True, else W-MSA only
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    projection_dropout=projection_dropout,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    use_relative_bias=use_relative_bias,  # Pass ablation flag
                    use_absolute_pos_embed=use_absolute_pos_embed,  # Pass ablation flag
                )
                for i in range(depth)
            ]
        )

        # Optional downsampling layer (Patch Merging)
        if downsample is not None:
            downsample_dim = (
                downsample_input_dim if downsample_input_dim is not None else dim
            )
            self.downsample = downsample(
                input_resolution=input_resolution, dim=downsample_dim
            )
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
        1. Apply optional downsampling first (timm-compatible order)
        2. Pass through each Swin block sequentially
        3. Blocks alternate between W-MSA and SW-MSA
        """
        # Optional downsampling first (timm-compatible)
        if self.downsample is not None:
            x = self.downsample(x)

        # Pass through all Swin Transformer blocks
        for block in self.blocks:
            x = block(x)

        return x
