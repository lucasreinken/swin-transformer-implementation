"""
Swin Transformer V2 Model with Residual Post-Normalization.

This module implements Swin Transformer V2 which uses residual post-normalization
instead of pre-normalization for improved training stability in large/deep models.

Reference: Swin Transformer V2: Scaling Up Capacity and Resolution (Liu et al., 2022)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List, Dict, Any
from torch.utils.checkpoint import checkpoint

try:
    from mmdet.registry import MODELS
except Exception:
    MODELS = None

try:
    from mmengine.model import BaseModule
except Exception:
    BaseModule = None

from .patch_embedding import PatchEmbed
from .basic_layer import BasicLayerV2
from .patch_merging import PatchMerging
from .conv_downsample import ConvDownsample
from .window_utils import generate_drop_path_rates


def _optional_mmdet_register():
    if MODELS is None:

        def deco(cls):
            return cls

        return deco
    return MODELS.register_module()


@_optional_mmdet_register()
class SwinV2TransformerModel((BaseModule if BaseModule is not None else nn.Module)):
    """
    Swin Transformer V2 with Residual Post-Normalization.

    Key difference from V1:
    - Uses BasicLayerV2 which contains SwinV2TransformerBlock
    - Post-norm: x = LayerNorm(x + sublayer(x)) instead of x = x + sublayer(LayerNorm(x))
    - Better training stability for large models
    """

    def __init__(
        self,
        img_size: int | None = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        embedding_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        projection_dropout_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        pretrain_img_size: int | None = None,
        out_indices: tuple | None = None,
        use_shifted_window: bool = True,
        use_relative_bias: bool = True,
        use_absolute_pos_embed: bool = False,
        use_hierarchical_merge: bool = False,
        use_gradient_checkpointing: bool = False,
        init_cfg: dict | None = None,
        **kwargs: Dict[str, Any]
    ):
        if BaseModule is not None:
            super().__init__(init_cfg=init_cfg)
        else:
            super().__init__()
            self.init_cfg = init_cfg

        # Store configuration
        self.config = {
            "img_size": img_size,
            "patch_size": patch_size,
            "in_channels": in_channels,
            "embedding_dim": embedding_dim,
            "depths": depths,
            "num_heads": num_heads,
            "window_size": window_size,
            "mlp_ratio": mlp_ratio,
            "dropout_rate": dropout_rate,
            "attention_dropout_rate": attention_dropout_rate,
            "projection_dropout_rate": projection_dropout_rate,
            "drop_path_rate": drop_path_rate,
            "pretrain_img_size": pretrain_img_size,
            "out_indices": out_indices,
            "use_shifted_window": use_shifted_window,
            "use_relative_bias": use_relative_bias,
            "use_absolute_pos_embed": use_absolute_pos_embed,
            "use_hierarchical_merge": use_hierarchical_merge,
            "use_gradient_checkpointing": use_gradient_checkpointing,
        }

        # Validate configuration
        assert len(depths) == len(
            num_heads
        ), "Depths and num_heads must have the same length"
        if img_size is not None:
            assert (
                img_size % patch_size == 0
            ), "Image size must be divisible by patch size"

        self.num_layers = len(depths)
        self.num_features_list = [embedding_dim << i for i in range(self.num_layers)]
        self.num_features = self.num_features_list[-1]
        self.embedding_dim = embedding_dim
        self.depths = depths
        self.window_size = window_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.out_indices = out_indices if out_indices is not None else []

        # Patch embedding
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            patch_size=patch_size,
        )

        # Absolute position embedding (optional, ViT-style)
        if use_absolute_pos_embed:
            if pretrain_img_size:
                patches_resolution = [
                    pretrain_img_size // patch_size,
                    pretrain_img_size // patch_size,
                ]
            else:
                patches_resolution = [img_size // patch_size, img_size // patch_size]
            num_patches = patches_resolution[0] * patches_resolution[1]
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embedding_dim)
            )
            nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)
        else:
            self.absolute_pos_embed = None

        self.pos_drop = nn.Dropout(p=dropout_rate)

        # Stochastic depth decay rule
        dpr = generate_drop_path_rates(drop_path_rate, sum(depths))

        # Build layers (V2 with post-norm)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = embedding_dim << i_layer

            # Determine downsampling strategy (same logic as V1)
            if use_hierarchical_merge:
                # Single-resolution ablation: ConvDownsample for all stages except first
                if i_layer == 0:
                    downsample = None
                    downsample_input_dim = None
                else:
                    downsample = ConvDownsample
                    downsample_input_dim = embedding_dim
            else:
                # Normal hierarchical: PatchMerging for stages 1, 2, 3 (not stage 0)
                if i_layer == 0:
                    downsample = None
                    downsample_input_dim = None
                else:
                    downsample = PatchMerging
                    downsample_input_dim = int(embedding_dim * (2 ** (i_layer - 1)))

            layer = BasicLayerV2(
                dim=layer_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout_rate,
                attention_dropout=attention_dropout_rate,
                projection_dropout=projection_dropout_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                downsample=downsample,
                downsample_input_dim=downsample_input_dim,
                use_shifted_window=use_shifted_window,
                use_relative_bias=use_relative_bias,
                use_absolute_pos_embed=use_absolute_pos_embed,
            )
            self.layers.append(layer)

        # Layer normalization for each output stage (if using out_indices)
        if len(self.out_indices) > 0:
            self.norm_list = nn.ModuleList()
            for i in range(self.num_layers):
                if i in self.out_indices:
                    layer_dim = self.num_features_list[i]
                    norm_layer = nn.LayerNorm(layer_dim)
                    self.norm_list.append(norm_layer)
                else:
                    self.norm_list.append(nn.Identity())
        else:
            self.norm_list = None

        # Final normalization (for classification head)
        self.norm = nn.LayerNorm(self.num_features)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights following Swin Transformer convention."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features through V2 transformer layers.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Output features [B, L, C] where L = (H/32) * (W/32) for 4 stages
        """
        x, (H, W) = self.patch_embed(x)

        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        outs = []
        for i, layer in enumerate(self.layers):
            if self.use_gradient_checkpointing and self.training:
                x, H, W = checkpoint(layer, x, H, W, use_reentrant=False)
            else:
                x, H, W = layer(x, H, W)

            if i in self.out_indices and self.norm_list is not None:
                norm = self.norm_list[i]
                x_out = norm(x)
                B, L, C = x_out.shape
                x_out = x_out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                outs.append(x_out)

        if len(self.out_indices) > 0:
            return tuple(outs)

        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Features [B, L, C] or tuple of features if out_indices is set
        """
        return self.forward_features(x)
