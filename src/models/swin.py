import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding for Swin Transformer.

    Splits input images into non-overlapping patches using convolution,
    then projects to embedding dimension.

    Input Image: [B, 3, 224, 224]
         │
         ▼ PatchEmbed
    ┌─────────────────────────────────────┐
    │ 1. Conv2d(kernel=4, stride=4)       │
    │    → [B, 96, 56, 56]                │
    │                                     │
    │ 2. Flatten: [B, 96, 56, 56]         │
    │    → [B, 96, 3136]                  │
    │                                     │
    │ 3. Transpose: [B, 96, 3136]         │
    │    → [B, 3136, 96]                  │
    │                                     │
    │ 4. LayerNorm along embed_dim        │
    │    → [B, 3136, 96]                  │
    └─────────────────────────────────────┘
         │
         ▼
    Window Attention Blocks...
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        embedding_dim: int = 96,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        # Convolutional projection: splits image into patches and embeds them
        self.proj = nn.Conv2d(
            in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size
        )

        # Normalization layer
        self.norm = nn.LayerNorm(embedding_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights according to Swin Transformer paper."""
        # Initialize convolutional layer with truncated normal
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to convert images to patch embeddings."""
        B, C, H, W = x.shape

        # Project patches: [B, C, H, W] -> [B, embedding_dim, H/patch_size, W/patch_size]
        x = self.proj(x)

        # Flatten spatial dimnension: [B, embedding_dim, H_p, W_p] -> [B, embedding_dim, num_patches]
        x = x.flatten(2)

        # Transpose to sequence format: [B, embedding_dim, num_patches] -> [B, num_patches, embedding_dim]
        x = x.transpose(1, 2)

        # Apply normalization
        x = self.norm(x)

        return x

    def flops(self) -> float:
        """Calculate FLOPs for the Patch Embedding layer."""
        flops = 0
        flops += (
            self.embedding_dim
            * self.in_channels
            * (self.patch_size**2)
            * self.num_patches
        )
        return flops
