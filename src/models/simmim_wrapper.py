import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_targets(targets: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Local per-pixel normalization using statistics from a (patch_size x patch_size) window.
    patch_size must be odd.
    """
    assert patch_size % 2 == 1

    targets_count = torch.ones_like(targets)
    targets_square = targets ** 2.0

    targets_mean = F.avg_pool2d(
        targets, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=False
    )
    targets_square_mean = F.avg_pool2d(
        targets_square, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=False
    )
    targets_count = F.avg_pool2d(
        targets_count, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=True
    ) * (patch_size ** 2)

    targets_var = (targets_square_mean - targets_mean ** 2.0) * (targets_count / (targets_count - 1))
    targets_var = torch.clamp(targets_var, min=0.0)

    targets_ = (targets - targets_mean) / (targets_var + 1e-6) ** 0.5

    return targets_


class SimMIM(nn.Module):
    """
    SimMIM (Simple Masked Image Modeling) Wrapper.

    This module implements the framework described in "SimMIM: A Simple Framework for Masked
    Image Modeling". It wraps a vision encoder with a lightweight prediction head (decoder)
    to perform masked image reconstruction.

    The model takes an image and a binary mask as input, encodes the visible patches,
    reconstructs the original image pixels, and computes the loss only on the masked regions.

    Attributes:
        config (dict): Configuration dictionary for loss and normalization.
        encoder (nn.Module): The backbone encoder (e.g., Swin Transformer).
        encoder_stride (int): The spatial downsampling factor of the encoder.
        decoder (nn.Module): A lightweight prediction head (1x1 Conv + PixelShuffle).
        in_chans (int): Number of input image channels.
        patch_size (int): The spatial size of the masking patches.
    """

    def __init__(
        self,
        config: dict,
        encoder: nn.Module,
        encoder_stride: int,
        in_chans: int,
        patch_size: int,
    ):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        if not hasattr(self.encoder, "num_features"):
            raise AttributeError("Encoder must expose .num_features for SimMIM decoder construction.")

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * in_chans,
                kernel_size=1,
            ),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = in_chans
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        mask: [B, H/patch_size, W/patch_size] with 0/1 (0 keep, 1 mask)
        """
        z = self.encoder(x, mask)        # [B, C', H/stride, W/stride]
        x_rec = self.decoder(z)          # [B, C, H, W]

        # Upsample mask from patch-grid to pixel-grid
        mask_pix = (
            mask.repeat_interleave(self.patch_size, 1)
                .repeat_interleave(self.patch_size, 2)
                .unsqueeze(1)
                .contiguous()
        )  # [B, 1, H, W]

        # Optional: normalize target locally
        norm_cfg = self.config.get("norm_target", {})
        if norm_cfg.get("enable", False):
            x = norm_targets(x, int(norm_cfg.get("patch_size", 7)))

        # Loss type (default L1)
        loss_type = self.config.get("loss", {}).get("type", "l1").lower()
        if loss_type == "l1":
            loss_recon = F.l1_loss(x, x_rec, reduction="none")
        elif loss_type in {"l2", "mse"}:
            loss_recon = F.mse_loss(x, x_rec, reduction="none")
        else:
            raise ValueError(f"Unsupported SimMIM loss type: {loss_type}")

        loss = (loss_recon * mask_pix).sum() / (mask_pix.sum() + 1e-5) / self.in_chans
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, "no_weight_decay"):
            return {"encoder." + i for i in self.encoder.no_weight_decay()}
        return set()

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, "no_weight_decay_keywords"):
            return {"encoder." + i for i in self.encoder.no_weight_decay_keywords()}
        return set()
