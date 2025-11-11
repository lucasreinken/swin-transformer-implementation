import torch
import torch.nn as nn
import math
from typing import Optional, List, Dict, Any

from .patch_embedding import PatchEmbed
from .basic_layer import BasicLayer
from .patch_merging import PatchMerging
from .window_utils import generate_drop_path_rates


class SwinTransformerModel(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 1000,
        embedding_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        projection_dropout_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        **kwargs: Dict[str, Any]
    ):
        super().__init__()

        # Store configuration for reference
        self.config = {
            "img_size": img_size,
            "patch_size": patch_size,
            "in_channels": in_channels,
            "num_classes": num_classes,
            "embedding_dim": embedding_dim,
            "depths": depths,
            "num_heads": num_heads,
            "window_size": window_size,
            "mlp_ratio": mlp_ratio,
            "dropout_rate": dropout_rate,
            "attention_dropout_rate": attention_dropout_rate,
            "projection_dropout_rate": projection_dropout_rate,
            "drop_path_rate": drop_path_rate,
        }

        # Validate configuration
        assert len(depths) == len(
            num_heads
        ), "Depths and num_heads must have the same length"
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"

        self.num_layers = len(depths)
        self.num_features = int(embedding_dim * 2 ** (self.num_layers - 1))

        self.patch_embedding = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dim=embedding_dim,
        )

        patches_resolution = self.patch_embedding.patches_resolution
        self.patches_resolution = patches_resolution

        # Generate drop path rates for each layer
        drop_path_rates = generate_drop_path_rates(drop_path_rate, sum(depths))

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            # Calculate dimensions for this stage
            stage_dimension = int(embedding_dim * 2**i)
            state_depth = depths[i]
            stage_num_heads = num_heads[i]

            # Calculate input resolution for stage
            if i == 0:
                input_resolution = (
                    patches_resolution  # First stage is same as patch resolution
                )
            else:
                # Halve resolution
                input_resolution = [
                    patches_resolution[0] // (2**i),
                    patches_resolution[1] // (2**i),
                ]

            # Prepare Stage specific drop path rates
            stage_drop_path_rates = drop_path_rates[
                sum(depths[:i]) : sum(depths[: i + 1])
            ]

            # Create basic layer
            basic_layer = BasicLayer(
                dim=stage_dimension,
                input_resolution=input_resolution,
                depth=state_depth,
                num_heads=stage_num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout_rate,
                attention_dropout=attention_dropout_rate,
                projection_dropout=projection_dropout_rate,
                drop_path=stage_drop_path_rates,
                downsample=None,
            )

            self.layers.append(basic_layer)

            # Add patch merging between stages (not after last stage)
            if i < self.num_layers - 1:
                downsample = PatchMerging(
                    input_resolution=input_resolution, dim=stage_dimension
                )
                self.layers.append(downsample)

        self.norm = norm_layer(self.num_features)  # Feature normalization
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # Average pooling
        self.head = nn.Linear(self.num_features, num_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize model weights according to swin transformer paper."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features through transformer stages"""
        x = self.patch_embedding(x)

        for layer in self.layers:
            x = layer(x)

        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        """Classification head"""

        # Average pooling
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)

        x = self.head(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Complete forward pass through Swin Transformer."""
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration information."""
        return {
            "model_type": "SwinTransformer",
            "config": self.config,
            "num_layers": self.num_layers,
            "num_features": self.num_features,
            "patches_resolution": self.patches_resolution,
            "parameter_count": sum(p.numel() for p in self.parameters()),
        }


def swin_tiny_patch4_window7_224(**kwargs) -> SwinTransformerModel:
    model = SwinTransformerModel(
        img_size=224,
        patch_size=4,
        embedding_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        **kwargs
    )
    return model


def swin_small_patch4_window7_224(**kwargs) -> SwinTransformerModel:
    model = SwinTransformerModel(
        img_size=224,
        patch_size=4,
        embedding_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        **kwargs
    )
    return model


def swin_base_patch4_window7_224(**kwargs) -> SwinTransformerModel:
    model = SwinTransformerModel(
        img_size=224,
        patch_size=4,
        embedding_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        **kwargs
    )
    return model


def swin_large_patch4_window7_224(**kwargs) -> SwinTransformerModel:
    model = SwinTransformerModel(
        img_size=224,
        patch_size=4,
        embedding_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        **kwargs
    )
    return model
