"""
Segmentation model wrapper for combining encoder + segmentation head.

Similar to ModelWrapper for classification, but designed for segmentation tasks.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class SegmentationModelWrapper(nn.Module):
    """
    Wrapper for segmentation models combining an encoder and segmentation head.
    
    Architecture:
        Input Image → Encoder (multi-scale features) → Segmentation Head → Output
    
    Args:
        encoder: Feature extractor (e.g., SwinTransformerModel)
        seg_head: Segmentation head (e.g., UperNetHead)
        freeze_encoder: If True, freeze encoder weights during training
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        seg_head: nn.Module,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.seg_head = seg_head
        self.freeze_encoder = freeze_encoder
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder and segmentation head.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        # Extract multi-scale features from encoder
        # For Swin-T: returns list of [B, H*W, C] tensors
        features_list = self.encoder(x, return_multi_scale=True)
        
        # Convert features from [B, H*W, C] to [B, C, H, W] format
        # This is needed because Swin outputs flattened spatial dimensions
        features_spatial = self._reshape_features(features_list)
        
        # Pass through segmentation head
        output = self.seg_head(features_spatial)
        
        return output
    
    def _reshape_features(self, features_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Reshape features from [B, H*W, C] to [B, C, H, W].
        
        For Swin-T with 512×512 input:
            Stage 1: [B, 128*128, 96] → [B, 96, 128, 128]
            Stage 2: [B, 64*64, 192] → [B, 192, 64, 64]
            Stage 3: [B, 32*32, 384] → [B, 384, 32, 32]
            Stage 4: [B, 16*16, 768] → [B, 768, 16, 16]
        
        Args:
            features_list: List of features with shape [B, H*W, C]
        
        Returns:
            List of features with shape [B, C, H, W]
        """
        reshaped_features = []
        
        # Get patch resolution from encoder
        # For 512×512 input with patch_size=4: 128×128 patches initially
        base_resolution = self.encoder.patches_resolution[0]  # 128 for 512×512
        
        for i, features in enumerate(features_list):
            B, N, C = features.shape  # [B, H*W, C]
            
            # Calculate spatial dimensions for this stage
            # Stage i reduces resolution by 2^i from base
            H = W = base_resolution // (2 ** i)
            
            # Reshape to spatial format
            features_2d = features.transpose(1, 2).reshape(B, C, H, W)
            reshaped_features.append(features_2d)
        
        return reshaped_features
    
    def train(self, mode: bool = True):
        """Override train mode to keep encoder frozen if requested."""
        super().train(mode)
        
        if self.freeze_encoder:
            # Keep encoder in eval mode even when wrapper is in train mode
            self.encoder.eval()
        
        return self
    
    def get_num_params(self) -> dict:
        """Get parameter counts for encoder and head separately."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        head_params = sum(p.numel() for p in self.seg_head.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "encoder": encoder_params,
            "head": head_params,
            "total": encoder_params + head_params,
            "trainable": trainable_params,
        }
