import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchMerging(nn.Module):
    """
    Patch Merging Layer for Hierarchical Feature Learning in Swin Transformer.

    This is a KEY COMPONENT that creates the hierarchical structure in Swin,
    similar to pooling in CNNs but preserving rich feature information.

    Purpose: Downsample spatial resolution while increasing feature richness
    Operation: Group 2x2 neighboring patches → Concatenate → Linear projection
    Result: Hierarchical representation (like CNN feature pyramid)

    ┌──────────────────────── HIERARCHICAL DOWNSAMPLING ────────────────────────┐
    │                                                                           │
    │  Input Patches: [B, HxW, C]  (e.g., [B, 3136, 96])                        │
    │                                                                           │
    │  ┌─────────────────────────────────┐  Spatial Layout: 56x56 patches       │
    │  │ p₁  p₂  p₃  p₄  p₅  p₆  p₇  p₈ │  Each patch pᵢ has C=96 features      │
    │  │ p₉  p₁₀ p₁₁ p₁₂ p₁₃ p₁₄ p₁₅ p₁₆│                                       │
    │  │ p₁₇ p₁₈ p₁₉ p₂₀ p₂₁ p₂₂ p₂₃ p₂₄│                                       │
    │  │ p₂₅ p₂₆ p₂₇ p₂₈ p₂₉ p₃₀ p₃₁ p₃₂│                                       │
    │  │ ...  ...  ...  ...  ...  ...  │                                        │
    │  └─────────────────────────────────┘                                      │
    │                    │                                                      │
    │                    ▼ Group 2x2 neighbors                                  │
    │                                                                           │
    │  ┌─────────────────┐               ┌─────────────────┐                    │
    │  │ ┌─────┬─────┐  │               │ ┌─────┬─────┐  │                      │
    │  │ │ p₁  │ p₂  │  │    Merge      │ │ p₃  │ p₄  │  │    Merge             │
    │  │ │ 96  │ 96  │  │    ━━━━━━━━━▶  │ │ 96  │ 96  │  │    ━━━━━━━━━▶       │
    │  │ ├─────┼─────┤  │               │ ├─────┼─────┤  │                      │
    │  │ │ p₉  │ p₁₀ │  │               │ │ p₁₁ │ p₁₂ │  │                      │
    │  │ │ 96  │ 96  │  │               │ │ 96  │ 96  │  │                      │
    │  │ └─────┴─────┘  │               │ └─────┴─────┘  │                      │
    │  └─────────────────┘               └─────────────────┘                    │
    │           │                                 │                             │
    │           ▼ Concatenate [96+96+96+96=384]   ▼                             │
    │                                                                           │
    │  ┌─────────────────┐               ┌─────────────────┐                    │
    │  │   New Patch₁    │               │   New Patch₂    │                    │
    │  │   [384 dims]    │               │   [384 dims]    │                    │
    │  │                 │               │                 │                    │
    │  │ [p₁+p₂+p₉+p₁₀] │               │[p₃+p₄+p₁₁+p₁₂] │                      │
    │  └─────────────────┘               └─────────────────┘                    │
    │           │                                 │                             │
    │           ▼ LayerNorm + Linear(384→192)     ▼                             │
    │                                                                           │
    │  ┌─────────────────┐               ┌─────────────────┐                    │
    │  │   New Patch₁    │               │   New Patch₂    │                    │
    │  │   [192 dims]    │               │   [192 dims]    │                    │
    │  │   ENRICHED      │               │   ENRICHED      │                    │
    │  └─────────────────┘               └─────────────────┘                    │
    │                                                                           │
    │  Result: [B, (H/2)x(W/2), 2C]  (e.g., [B, 784, 192])                      │
    │                                                                           │
    │  Dimensions: 56x56 patches → 28x28 patches                                │
    │  Features:   96 per patch  → 192 per patch                                │
    │  Trade-off:  ½ spatial resolution ↔ 2x feature richness                   │
    │                                                                           │
    └───────────────────────────────────────────────────────────────────────────┘

    Why This Matters:
    1. **Hierarchical Learning**: Like CNN pyramids, different scales capture different features
    2. **Computational Efficiency**: Fewer patches = less attention complexity (O(n²) → O(n²/4))
    3. **Semantic Richness**: Larger receptive fields capture more complex patterns
    4. **Multi-Scale Processing**: Early layers: fine details, later layers: global context

    Swin Architecture Pattern:
    Stage 1: 56x56 patches, 96 dims   → Fine-grained features (textures, edges)
    Stage 2: 28x28 patches, 192 dims  → Medium-scale features (shapes, parts)
    Stage 3: 14x14 patches, 384 dims  → Large-scale features (objects)
    Stage 4: 7x7 patches, 768 dims    → Global features (scene understanding)
    """

    def __init__(self, dim: int):
        """
        Initialize Patch Merging layer.

        Args:
            input_resolution: (H, W) spatial dimensions in patches
                            e.g., (56, 56) for first merging layer
            dim: Input feature dimension per patch
                e.g., 96 for first layer, 192 for second, etc.

        Internal Architecture:
            - LayerNorm(4xdim): Normalize concatenated features
            - Linear(4xdim → 2xdim): Reduce dimension while preserving info
            - No bias: Following Swin paper for better training dynamics

        Mathematical Properties:
            - Input patches:  HxW x dim
            - Output patches: (H/2)x(W/2) x (2xdim)
            - Parameters: 4xdim x 2xdim = 8xdim² weights
            - Memory: 4x reduction in spatial tokens
        """
        super().__init__()
        self.dim = dim

        # Dimension reduction: 4xdim → 2xdim (preserves total information)
        # No bias following Swin paper (helps with training stability)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

        # Normalization before reduction (critical for stable training)
        # Applied to the concatenated 4xdim features
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        """
        Perform hierarchical patch merging.

        Args:
            x: Input tensor [B, HxW, C] where:
               - B: Batch size
               - HxW: Number of patches (must have even H and W)
               - C: Feature dimension per patch

        Returns:
            Merged tensor [B, (H/2)x(W/2), 2xC] with:
            - 4x fewer patches spatially
            - 2x richer features per patch
            - Same total information content

        ⚡ Computational Complexity:
        - Input:  O(HxWxC)
        - Output: O((H/2)x(W/2)x(2C)) = O(HxWxC/2)
        - Next attention: O((HxW/4)²) vs O((HxW)²) → 16x speedup!
        """
        B, L, C = x.shape

        # Validate input dimensions
        assert (
            L == H * W
        ), f"Input sequence length {L} doesn't match resolution {H}x{W}={H*W}"

        # Step 1: Reshape to spatial format for 2x2 grouping
        # [B, HxW, C] → [B, H, W, C]
        # This restores the 2D spatial structure needed for neighbor grouping
        x = x.view(B, H, W, C)

        # padding
        if (H % 2 == 1) or (W % 2 == 1):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        # Step 2: Extract 2x2 neighborhoods using advanced indexing
        # Each variable selects one corner of every 2x2 group
        #
        # Visual example for 4x4 grid:
        # ┌─────┬─────┬─────┬─────┐
        # │ x0  │ x2  │ x0  │ x2  │  x0: top-left     (0::2, 0::2)
        # ├─────┼─────┼─────┼─────┤
        # │ x1  │ x3  │ x1  │ x3  │  x1: bottom-left  (1::2, 0::2)
        # ├─────┼─────┼─────┼─────┤  x2: top-right    (0::2, 1::2)
        # │ x0  │ x2  │ x0  │ x2  │  x3: bottom-right (1::2, 1::2)
        # ├─────┼─────┼─────┼─────┤
        # │ x1  │ x3  │ x1  │ x3  │
        # └─────┴─────┴─────┴─────┘
        x0 = x[:, 0::2, 0::2, :]  # Top-left:     [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left:  [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # Top-right:    [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right: [B, H/2, W/2, C]

        # Step 3: Concatenate the 2x2 neighbors along feature dimension
        # [B, H/2, W/2, C] x 4 → [B, H/2, W/2, 4C]
        # This creates enriched patch representations with 4x more features
        x = torch.cat([x0, x1, x2, x3], dim=-1)

        # Step 4: Reshape back to sequence format for further processing
        # [B, H/2, W/2, 4C] → [B, (H/2)x(W/2), 4C]
        x = x.view(B, -1, 4 * C)

        # Step 5: Normalize the concatenated features (critical for training stability)
        # Each patch now has 4C features from its 2x2 neighborhood
        x = self.norm(x)

        # Step 6: Dimensionality reduction while preserving information
        # [B, (H/2)x(W/2), 4C] → [B, (H/2)x(W/2), 2C]
        # The linear layer learns optimal combination of the 4 patch features
        x = self.reduction(x)

        return x
