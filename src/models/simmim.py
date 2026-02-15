import torch
import torch.nn as nn

from src.models import SwinTransformerModel

class SwinTransformerForSimMIM(SwinTransformerModel):
    """
    Adapts custom SwinTransformerModel to SimMIM.

    Returns a feature map [B, C, H, W] from the final stage.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        embed_dim = self.config["embedding_dim"]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # patch_embed returns tokens and spatial size
        x, (H, W) = self.patch_embed(x)   # x: [B, L, C], L = H*W

        assert mask is not None, "SimMIM requires a mask"
        B, L, C = x.shape

        # mask expected shape: [B, H, W] with 0/1 (0 keep, 1 mask)
        # flatten to [B, L, 1]
        w = mask.reshape(B, -1).unsqueeze(-1).type_as(x)

        mask_tokens = self.mask_token.expand(B, L, -1).type_as(x)
        x = x * (1.0 - w) + mask_tokens * w

        # run Swin stages
        for layer in self.layers:
            x, H, W = layer(x, H, W)

        # return feature map for decoder
        # x: [B, L, C_last] -> [B, C_last, H, W]
        C_last = x.shape[-1]
        x = x.view(B, H, W, C_last).permute(0, 3, 1, 2).contiguous()
        return x