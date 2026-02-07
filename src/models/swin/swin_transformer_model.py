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
from .basic_layer import BasicLayer
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
class SwinTransformerModel((BaseModule if BaseModule is not None else nn.Module)):

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
        pretrain_img_size: int | None = None, # image size the model was pretrained on
        out_indices: tuple | None = None, # which Swin stages should output features (0, 1, 2, 3)
        use_shifted_window: bool = True,  # Ablation flag: True for SW-MSA, False for W-MSA only
        use_relative_bias: bool = True,  # Ablation flag: True for learned bias, False for zero bias
        use_absolute_pos_embed: bool = False,  # Ablation flag: True for absolute pos embed (ViT-style), False for relative bias
        use_hierarchical_merge: bool = False,  # Ablation flag: False for hierarchical PatchMerging, True for single-resolution conv
        use_gradient_checkpointing: bool = False,  # Enable gradient checkpointing to save memory
        return_attention_maps: bool = False,  # Explainability flag: True to capture attention weights for visualization
        init_cfg: dict | None = None,   # MMEngine weight initialization config (optional)
        **kwargs: Dict[str, Any]
    ):
        if BaseModule is not None:
            super().__init__(init_cfg=init_cfg)
        else:
            super().__init__()
            # In plain PyTorch mode, init_cfg is not used.
            self.init_cfg = init_cfg

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
            "pretrain_img_size": pretrain_img_size,
            "out_indices": out_indices,
            "use_shifted_window": use_shifted_window,
            "use_relative_bias": use_relative_bias,
            "use_absolute_pos_embed": use_absolute_pos_embed,
            "use_hierarchical_merge": use_hierarchical_merge,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "return_attention_maps": return_attention_maps,
        }
        
        # Attention map storage (only used when return_attention_maps=True)
        self.return_attention_maps = return_attention_maps
        self.attention_maps = []

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
                return_attention=return_attention_maps,  # Pass explainability flag
            )

            self.layers.append(basic_layer)

            if out_indices is not None:
                # create a separate LayerNorm for each selected output stage
                for i in out_indices:
                    layer = nn.LayerNorm(self.num_features_list[i])
                    layer_name = f'norm{i}'
                    self.add_module(layer_name, layer)

        self.use_gradient_checkpointing = use_gradient_checkpointing

    def init_weights(self):
        """Initialize weights.

        If init_cfg is set and MMEngine is available, MMEngine will handle it.
        Otherwise do Swin default init.
        """
        if BaseModule is not None and getattr(self, "init_cfg", None) is not None:
            super().init_weights()
            return

        # Default Swin initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features through transformer stages"""
        # Clear attention maps from previous forward pass
        if self.return_attention_maps:
            self.attention_maps = []
        
        x, (H, W) = self.patch_embed(x)
        if self.out_indices is not None:
            outs = []
            for i, layer in enumerate(self.layers):
                x, H, W = layer(x, H, W)
                
                # Collect attention maps if enabled
                if self.return_attention_maps:
                    self._collect_attention_from_layer(layer, stage_idx=i)
                
                if i in self.out_indices:
                    x_out = getattr(self, f"norm{i}")(x)
                    out = x_out.view(-1, H, W, self.num_features_list[i]).permute(0, 3, 1, 2).contiguous()
                    outs.append(out)
            x = tuple(outs)
        else:
            for i, layer in enumerate(self.layers):
                x, H, W = layer(x, H, W)
                
                # Collect attention maps if enabled
                if self.return_attention_maps:
                    self._collect_attention_from_layer(layer, stage_idx=i)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Complete forward pass through Swin Transformer."""
        x = self.forward_features(x)
        return x
    
    def _collect_attention_from_layer(self, layer, stage_idx: int):
        """
        Collect attention maps from all blocks in a layer.
        
        Args:
            layer: BasicLayer instance
            stage_idx: Stage index (0-3 for Swin-Tiny)
        """
        for block_idx, block in enumerate(layer.blocks):
            if hasattr(block, '_block_metadata') and block._block_metadata is not None:
                metadata = block._block_metadata.copy()
                metadata['stage'] = stage_idx
                metadata['block'] = block_idx
                metadata['dim'] = block.dim
                metadata['num_heads'] = block.num_heads
                self.attention_maps.append(metadata)
    
    def get_attention_maps(self):
        """
        Get collected attention maps from the last forward pass.
        
        Returns:
            List of dictionaries containing attention weights and metadata:
            - 'stage': stage index
            - 'block': block index within stage
            - 'is_shifted': whether this is SW-MSA (True) or W-MSA (False)
            - 'attention': attention weights [num_windows*B, num_heads, N, N]
            - 'resolution': (H, W) original feature map resolution
            - 'padded_resolution': (Hp, Wp) padded resolution
            - 'window_size': window size
            - 'shift_size': shift size
            - 'num_windows': (nH, nW) number of windows
            - 'dim': feature dimension
            - 'num_heads': number of attention heads
        """
        if not self.return_attention_maps:
            raise ValueError(
                "return_attention_maps=False. Create model with return_attention_maps=True "
                "to collect attention weights."
            )
        return self.attention_maps

    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration information."""
        return {
            "model_type": "SwinTransformer",
            "config": self.config,
            "num_layers": self.num_layers,
            "num_features_list": self.num_features_list,
            "num_features": self.num_features,
            "parameter_count": sum(p.numel() for p in self.parameters()),
        }
