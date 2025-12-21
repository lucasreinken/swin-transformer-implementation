import torch
import torch.nn as nn
import math
from typing import Optional, List, Dict, Any
from torch.utils.checkpoint import checkpoint

from .patch_embedding import PatchEmbed
from .basic_layer import BasicLayer
from .patch_merging import PatchMerging
from .conv_downsample import ConvDownsample
from .window_utils import generate_drop_path_rates


class SwinTransformerModel(nn.Module):
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
        pretrain_img_size: int = 224,
        out_indices: tuple | None = None, # which Swin stages should output features (0, 1, 2, 3)
        use_shifted_window: bool = True,  # Ablation flag: True for SW-MSA, False for W-MSA only
        use_relative_bias: bool = True,  # Ablation flag: True for learned bias, False for zero bias
        use_absolute_pos_embed: bool = False,  # Ablation flag: True for absolute pos embed (ViT-style), False for relative bias
        use_hierarchical_merge: bool = False,  # Ablation flag: False for hierarchical PatchMerging, True for single-resolution conv
        use_gradient_checkpointing: bool = False,  # Enable gradient checkpointing to save memory
        **kwargs: Dict[str, Any]
    ):
        super().__init__()

        # Store configuration for reference
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
            assert img_size % patch_size == 0, "Image size must be divisible by patch size"

        self.num_layers = len(depths)

        self.num_features_list = [
            embedding_dim << i for i in range(self.num_layers)
        ]

        self.num_features = (
            embedding_dim
            if use_hierarchical_merge
            else self.num_features_list[-1]
        )

        self.out_indices = out_indices

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            pretrain_img_size=pretrain_img_size,
            use_absolute_pos_embed=use_absolute_pos_embed,
        )

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Generate drop path rates for each layer
        drop_path_rates = generate_drop_path_rates(drop_path_rate, sum(depths))

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            # For hierarchical ablation: all stages use same dimension and resolution
            if use_hierarchical_merge:
                stage_dimension = embedding_dim  # All stages: 96 dims
                stage_num_heads = num_heads[0]  # All stages: same num heads as stage 1
            else:
                # Normal hierarchical Swin: dimensions double each stage
                stage_dimension = int(embedding_dim * (2**i))  # 96, 192, 384, 768
                stage_num_heads = num_heads[i]

            state_depth = depths[i]

            # Prepare Stage specific drop path rates
            stage_drop_path_rates = drop_path_rates[
                sum(depths[:i]) : sum(depths[: i + 1])
            ]

            # Create basic layer with appropriate downsampling
            if use_hierarchical_merge:
                # Single-resolution ablation: use ConvDownsample for all stages except first
                if i == 0:  # First stage - no downsampling
                    downsample = None
                    downsample_input_dim = None
                else:  # stages 1, 2, 3 - use ConvDownsample (maintains resolution)
                    downsample = ConvDownsample
                    downsample_input_dim = embedding_dim  # Always 96 for all stages
            else:
                # Normal hierarchical: use PatchMerging for stages 1, 2, 3
                if i == 0:  # First stage - no downsampling
                    downsample = None
                    downsample_input_dim = None
                else:  # stages 1, 2, 3 - with PatchMerging downsampling
                    downsample = PatchMerging
                    downsample_input_dim = int(embedding_dim * (2 ** (i - 1)))

            basic_layer = BasicLayer(
                dim=stage_dimension,
                depth=state_depth,
                num_heads=stage_num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout_rate,
                attention_dropout=attention_dropout_rate,
                projection_dropout=projection_dropout_rate,
                drop_path=stage_drop_path_rates,
                downsample=downsample,
                downsample_input_dim=downsample_input_dim,
                use_shifted_window=use_shifted_window,  # Pass ablation flag
                use_relative_bias=use_relative_bias,  # Pass ablation flag
                use_absolute_pos_embed=use_absolute_pos_embed,  # Pass ablation flag
            )

            self.layers.append(basic_layer)

            if out_indices is not None:
                # create a separate LayerNorm for each selected output stage
                for i in out_indices:
                    layer = nn.LayerNorm(self.num_features_list[i])
                    layer_name = f'norm{i}'
                    self.add_module(layer_name, layer)

        self.use_gradient_checkpointing = use_gradient_checkpointing

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
        x, (H, W) = self.patch_embed(x)
        if self.out_indices is not None:
            outs = []
            for i, layer in enumerate(self.layers):
                x, H, W = layer(x, H, W)
                if i in self.out_indices:
                    x_out = getattr(self, f"norm{i}")(x)
                    out = x_out.view(-1, H, W, self.num_features_list[i]).permute(0, 3, 1, 2).contiguous()
                    outs.append(out)
            x = tuple(outs)
        else:
            for layer in self.layers:
                x, H, W = layer(x, H, W)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Complete forward pass through Swin Transformer."""
        x = self.forward_features(x)
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
