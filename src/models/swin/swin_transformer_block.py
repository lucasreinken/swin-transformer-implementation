import torch
import torch.nn as nn
import logging

from .mlp import MLP
from .window_attention import WindowAttention
from .drop_path import DropPath
from .window_utils import window_partition, window_reverse

logger = logging.getLogger(__name__)


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block: The fundamental building block of Swin Transformer.

    This module implements one complete Swin block that can operate in either:
    - Standard Window Multi-head Self Attention (W-MSA) mode
    - Shifted Window Multi-head Self Attention (SW-MSA) mode

    ┌────────────────────── SWIN BLOCK ARCHITECTURE ──────────────────────┐
    │                                                                      │
    │  Input: [B, H×W, C]                                                  │
    │      │                                                               │
    │      ├─────────────────────────┐                                     │
    │      │                         │  (Residual Connection 1)            │
    │      ▼                         │                                     │
    │  LayerNorm                     │                                     │
    │      │                         │                                     │
    │      ▼                         │                                     │
    │  Window / Shifted-Window       │                                     │
    │  Multi-head Self Attention     │                                     │
    │  (W-MSA or SW-MSA)             │                                     │
    │      │                         │                                     │
    │      ▼                         │                                     │
    │  DropPath (Stochastic Depth)   │                                     │
    │      │                         │                                     │
    │      ▼                         │                                     │
    │  ◄──(+)◄────────────────────────┘                                     │
    │      │                                                               │
    │      ├─────────────────────────┐                                     │
    │      │                         │  (Residual Connection 2)            │
    │      ▼                         │                                     │
    │  LayerNorm                     │                                     │
    │      │                         │                                     │
    │      ▼                         │                                     │
    │  MLP (FC → GELU → FC)          │                                     │
    │      │                         │                                     │
    │      ▼                         │                                     │
    │  DropPath (Stochastic Depth)   │                                     │
    │      │                         │                                     │
    │      ▼                         │                                     │
    │  ◄──(+)◄────────────────────────┘                                     │
    │      │                                                               │
    │      ▼                                                               │
    │  Output: [B, H×W, C]                                                 │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

    Key Design Choices:
    1. **Pre-LayerNorm**: Normalization before attention/MLP (better gradient flow)
    2. **Residual Connections**: Enable deep networks (up to 100+ layers)
    3. **DropPath**: Stochastic depth regularization (drops entire layers randomly)
    4. **MLP Expansion**: Hidden dim = 4× input dim (standard in transformers)

    W-MSA vs SW-MSA:
    - **W-MSA (shift_size=0)**: Attention within fixed non-overlapping windows
    - **SW-MSA (shift_size>0)**: Attention within shifted windows for cross-window connection
    - **Alternating Pattern**: W-MSA → SW-MSA → W-MSA → SW-MSA (enables global modeling)
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        """
        Initialize Swin Transformer Block.

        Args:
            dim: Feature dimension (C)
            input_resolution: (H, W) spatial resolution
            num_heads: Number of attention heads
            window_size: Window size for attention (default: 7×7)
            shift_size: Shift size for SW-MSA (0 for W-MSA, window_size//2 for SW-MSA)
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            projection_dropout: Projection dropout rate
            drop_path: Stochastic depth rate (probability of dropping the layer)

        Notes:
            - For W-MSA blocks: shift_size = 0
            - For SW-MSA blocks: shift_size = window_size // 2
            - Input resolution must be divisible by window_size (after shifting)
        """
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Validate parameters
        if min(self.input_resolution) <= self.window_size:
            # If window size is larger than input resolution, don't partition
            logger.warning(
                f"Window size {self.window_size} >= input resolution {self.input_resolution}. "
                f"Adjusting window_size to {min(self.input_resolution)} and shift_size to 0."
            )
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert (
            0 <= self.shift_size < self.window_size
        ), f"shift_size must be in [0, window_size), got shift_size={self.shift_size}, window_size={self.window_size}"

        # Layer 1: LayerNorm → Window Attention → Residual
        self.norm1 = nn.LayerNorm(dim)

        self.attn = WindowAttention(
            dim=dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            attn_dropout=attention_dropout,
            proj_dropout=projection_dropout
        )

        # Stochastic depth (DropPath) for regularization
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer 2: LayerNorm → MLP → Residual
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, dropout=dropout)

        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Swin Transformer Block.

        Args:
            x: Input tensor [B, H*W, C]

        Returns:
            Output tensor [B, H*W, C]

        Processing Flow:
        1. Save input for residual connection
        2. Apply LayerNorm
        3. Reshape to 2D: [B, H*W, C] → [B, H, W, C]
        4. Apply cyclic shift (if SW-MSA)
        5. Partition into windows
        6. Apply window attention
        7. Merge windows back
        8. Reverse cyclic shift (if SW-MSA)
        9. Reshape to sequence: [B, H, W, C] → [B, H*W, C]
        10. Add residual with DropPath
        11. Apply MLP block with residual connection
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W, f"Input feature has wrong size: {L} vs {H*W}"

        # Save input for residual connection
        shortcut = x

        # ─────────────────────────────────────────────────────────────
        # Block 1: LayerNorm → (Shifted) Window Attention → Residual
        # ─────────────────────────────────────────────────────────────

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift for SW-MSA
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # Partition windows: [B, H, W, C] → [B*num_windows, window_size, window_size, C]
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Apply window attention
        # TODO: Replace with attention mask parameter for SW-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows back: [B*num_windows, window_size*window_size, C] → [B, H, W, C]
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift for SW-MSA
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # Residual connection with stochastic depth
        x = shortcut + self.drop_path1(x)

        # ─────────────────────────────────────────────────────────────
        # Block 2: LayerNorm → MLP → Residual
        # ─────────────────────────────────────────────────────────────

        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x
