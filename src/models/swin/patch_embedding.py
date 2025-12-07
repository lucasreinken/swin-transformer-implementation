import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding for Swin Transformer.

    This module is the FIRST STEP in Swin Transformer that converts raw images
    into a sequence of patch embeddings that can be processed by attention layers.

    Purpose: Transform spatial image data into token-based representation
    Operation: Non-overlapping patch extraction + linear projection + normalization

    ┌─────────────────────────────── VISUAL FLOW ───────────────────────────────┐
    │                                                                           │
    │  Input Image: [B, 3, 224, 224]                                            │
    │  ┌─────────────────────┐                                                  │
    │  │ ░░░░░░░░░░░░░░░░░░░ │  RGB Image (224x224 pixels)                      │
    │  │ ░░░░░░░░░░░░░░░░░░░ │  Each pixel has 3 channels (R,G,B)               │
    │  │ ░░░░░░░░░░░░░░░░░░░ │                                                  │
    │  │ ░░░░░░░░░░░░░░░░░░░ │                                                  │
    │  └─────────────────────┘                                                  │
    │           │                                                               │
    │           ▼ Conv2d(kernel=4x4, stride=4)                                  │
    │                                                                           │
    │  Patch Projection: [B, 96, 56, 56]                                        │
    │  ┌───┬───┬───┬───┬───┐  Each 4x4 pixel region becomes                     │
    │  │ █ │ █ │ █ │ █ │...│  a single 96-dimensional feature vector            │
    │  ├───┼───┼───┼───┼───┤  56x56 = 3136 total patches                        │
    │  │ █ │ █ │ █ │ █ │...│  (224÷4 = 56 patches per dimension)                │
    │  ├───┼───┼───┼───┼───┤                                                    │
    │  │ █ │ █ │ █ │ █ │...│                                                    │
    │  └───┴───┴───┴───┴───┘                                                    │
    │           │                                                               │
    │           ▼ Flatten(2) + Transpose(1,2)                                   │
    │                                                                           │
    │  Sequence Format: [B, 3136, 96]                                           │
    │  ┌─────────────────────────┐                                              │
    │  │ [f₁, f₂, f₃, ..., f₉₆] │ ← Patch 1 features                            │
    │  │ [f₁, f₂, f₃, ..., f₉₆] │ ← Patch 2 features                            │
    │  │ [f₁, f₂, f₃, ..., f₉₆] │ ← Patch 3 features                            │
    │  │         ...             │                                              │
    │  │ [f₁, f₂, f₃, ..., f₉₆] │ ← Patch 3136 features                         │
    │  └─────────────────────────┘                                              │
    │           │                                                               │
    │           ▼ LayerNorm(96)                                                 │
    │                                                                           │
    │  Normalized Embeddings: [B, 3136, 96]                                     │
    │  Ready for Window Attention!                                              │
    │                                                                           │
    └───────────────────────────────────────────────────────────────────────────┘

    Key Insights:
    - Each 4x4 pixel region becomes ONE token in the sequence
    - Spatial locality is preserved through the patching process
    - The embedding dimension (96 for Swin-T) captures rich patch features
    - LayerNorm ensures stable training dynamics
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        embedding_dim: int = 96,
        use_absolute_pos_embed: bool = False,  # Ablation flag: True for absolute pos embed (ViT-style)
    ):
        """
        Initialize the Patch Embedding layer.
        """
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.use_absolute_pos_embed = use_absolute_pos_embed

        # Convolutional projection: efficiently extracts non-overlapping patches
        # This is equivalent to splitting image into patches + linear projection
        # But much more efficient than manual patch extraction
        self.proj = nn.Conv2d(
            in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size
        )

        # Normalization layer: stabilizes training and improves convergence
        # Applied to each patch's feature vector independently
        self.norm = nn.LayerNorm(embedding_dim)

        # Absolute position embedding (ViT-style) - ablation flag
        if self.use_absolute_pos_embed:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, embedding_dim)
            )
            nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)
        else:
            self.absolute_pos_embed = None

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights following Swin Transformer paper guidelines.

        Strategy:
        - Conv weights: Truncated normal (std=0.02) for stable gradients
        - Conv bias: Zero initialization (common practice)
        - LayerNorm: Uses PyTorch defaults (weight=1, bias=0)
        """
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patch embeddings.
        """
        B, C, H, W = x.shape

        # Step 1: Project patches using efficient convolution
        # Input:  [B, 3, 224, 224]
        # Output: [B, 96, 56, 56]
        # Each spatial position (i,j) contains features for patch at (4*i, 4*j)
        x = self.proj(x)

        # Step 2: Reshape to sequence format for attention processing
        # Flatten spatial dimensions: [B, 96, 56, 56] → [B, 96, 3136]
        x = x.flatten(2)

        # Transpose to standard sequence format: [B, 96, 3136] → [B, 3136, 96]
        # Now each row is one patch, each column is one feature dimension
        x = x.transpose(1, 2)

        # Step 3: Normalize each patch's features for stable training
        # Applied independently to each patch's 96-dimensional feature vector
        x = self.norm(x)

        # Step 4: Add absolute position embeddings (ViT-style) if enabled
        if self.use_absolute_pos_embed:
            x = x + self.absolute_pos_embed

        return x
