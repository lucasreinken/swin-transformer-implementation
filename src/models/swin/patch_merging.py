import torch
import torch.nn as nn

class PatchMerging(nn.Module):
    """
    Patch Merging Layer from Swin Transformer.
    Merges 2x2 neighboring patches, concatenates their features, and applies a linear projection.
    Input: [B, H*W, C] (where H and W are even)
    Output: [B, H/2*W/2, 2*2*C]
    """
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution  # (H, W)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        """
        x: [B, H*W, C]
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input feature has wrong size: {L} != {H}*{W}"
        assert H % 2 == 0 and W % 2 == 0, "H and W must be even."

        x = x.view(B, H, W, C)

        # Split into 4 parts and concatenate along last dim
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]
        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]
        return x
