import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """
    Split image into patches and embed them.
    Args:
        img_size (int): Size of the input image (assume square).
        patch_size (int): Size of one patch (assume square).
        in_chans (int): Number of input channels.
        embed_dim (int): Output embedding dimension.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=48):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

def normalize_patches(x):
    """
    Normalize patch embeddings along the embedding dimension.
    Args:
        x: [B, num_patches, embed_dim]
    Returns:
        Normalized tensor of same shape.
    """
    return F.layer_norm(x, x.shape[-1:])
