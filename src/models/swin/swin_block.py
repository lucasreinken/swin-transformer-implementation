import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for Swin Transformer blocks.
    
    This is the feedforward network applied after attention in each Swin block.
    Uses GELU activation and dropout for regularization.
    
    Architecture:
    Input → Linear(in → hidden) → GELU → Dropout → Linear(hidden → out) → Dropout
    
    Typical expansion: hidden = 4 × in_features (following ViT/Swin convention)
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        dropout: float = 0.0
    ):
        """
        Initialize MLP.
        
        Args:
            in_features: Input feature dimension
            hidden_features: Hidden layer dimension (default: 4 × in_features)
            out_features: Output feature dimension (default: in_features)
            dropout: Dropout rate
        """
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, N, C]
            
        Returns:
            Output tensor [B, N, C]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


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
        
        assert 0 <= self.shift_size < self.window_size, \
            f"shift_size must be in [0, window_size), got shift_size={self.shift_size}, window_size={self.window_size}"
        
        # Layer 1: LayerNorm → Window Attention → Residual
        self.norm1 = nn.LayerNorm(dim)
        
        # TODO: This will be replaced with actual WindowAttention
        # Placeholder for now - replace with: from .window_attention import WindowAttention
        # self.attn = WindowAttention(
        #     dim=dim,
        #     window_size=(self.window_size, self.window_size),
        #     num_heads=num_heads,
        #     dropout=attention_dropout,
        # )
        
        # Temporary placeholder (will be replaced with actual WindowAttention)
        self.attn = nn.Identity()  # Replace this when WindowAttention is ready
        
        # Stochastic depth (DropPath) for regularization
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # Layer 2: LayerNorm → MLP → Residual
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            dropout=dropout
        )
        
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
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Partition windows: [B, H, W, C] → [B*num_windows, window_size, window_size, C]
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Apply window attention
        # TODO: Replace with actual attention computation when WindowAttention is implemented
        # attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = self.attn(x_windows)  # Placeholder
        
        # Merge windows back: [B*num_windows, window_size*window_size, C] → [B, H, W, C]
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift for SW-MSA
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
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
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,  # Alternate W-MSA and SW-MSA
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            )
            for i in range(depth)
        ])
        
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


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample for residual blocks.
    
    Randomly drops entire layers during training to:
    - Reduce overfitting
    - Enable training of very deep networks
    - Act as implicit ensemble of shallower networks
    
    Reference: "Deep Networks with Stochastic Depth" (Huang et al., 2016)
    """
    
    def __init__(self, drop_prob: float = 0.0):
        """
        Initialize DropPath.
        
        Args:
            drop_prob: Probability of dropping the path
        """
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply DropPath.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor (dropped or scaled appropriately)
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        
        # Work with different dimensions (2D, 3D, 4D tensors)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        
        # Scale by keep_prob to maintain expected value
        output = x.div(keep_prob) * random_tensor
        
        return output


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition feature map into non-overlapping windows.
    
    Args:
        x: Input tensor [B, H, W, C]
        window_size: Window size (M)
        
    Returns:
        Windows tensor [B*num_windows, window_size, window_size, C]
        where num_windows = (H/M) * (W/M)
    
    Example:
        Input: [2, 56, 56, 96]  (B=2, H=W=56, C=96)
        Window size: 7
        Output: [128, 7, 7, 96]  (2 * (56/7) * (56/7) = 128 windows)
    """
    B, H, W, C = x.shape
    
    # Reshape to separate windows: [B, H, W, C] → [B, H//M, M, W//M, M, C]
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    
    # Permute to group windows: [B, H//M, M, W//M, M, C] → [B, H//M, W//M, M, M, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    
    # Flatten batch and window dimensions: [B, H//M, W//M, M, M, C] → [B*num_windows, M, M, C]
    windows = x.view(-1, window_size, window_size, C)
    
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse window partition, merging windows back into feature map.
    
    Args:
        windows: Window tensor [B*num_windows, window_size, window_size, C]
        window_size: Window size (M)
        H: Height of feature map
        W: Width of feature map
        
    Returns:
        Feature map [B, H, W, C]
    
    Example:
        Input: [128, 7, 7, 96]  (128 windows)
        H=56, W=56, window_size=7
        Output: [2, 56, 56, 96]  (B=2)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    
    # Reshape to separate windows: [B*num_windows, M, M, C] → [B, H//M, W//M, M, M, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    
    # Permute back to spatial layout: [B, H//M, W//M, M, M, C] → [B, H//M, M, W//M, M, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    
    # Merge windows: [B, H//M, M, W//M, M, C] → [B, H, W, C]
    x = x.view(B, H, W, -1)
    
    return x


def generate_drop_path_rates(drop_path_rate: float, depth: int) -> list:
    """
    Generate linearly increasing drop path rates for stochastic depth.
    
    This creates a schedule where deeper layers have higher drop probabilities,
    which helps training very deep networks by reducing gradient vanishing.
    
    Args:
        drop_path_rate: Maximum drop path rate (for the deepest layer)
        depth: Total number of layers
        
    Returns:
        List of drop path rates, one per layer
        
    Example:
        >>> generate_drop_path_rates(0.2, 4)
        [0.0, 0.0667, 0.1333, 0.2]
    """
    if drop_path_rate <= 0 or depth <= 0:
        return [0.0] * max(1, depth)
    if depth == 1:
        return [drop_path_rate]
    return [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
