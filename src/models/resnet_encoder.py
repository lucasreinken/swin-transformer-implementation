"""
ResNet Feature Extractor for semantic segmentation.

Wraps torchvision ResNet to extract multi-scale features compatible with UperNet.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet101_Weights, ResNet50_Weights
from typing import List, Optional


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet backbone that extracts multi-scale features for segmentation.
    
    Extracts features from layer1, layer2, layer3, layer4 to provide
    hierarchical features similar to Swin Transformer stages.
    
    Feature channels for ResNet-101/50:
        - layer1: 256 channels, stride 4 (H/4, W/4)
        - layer2: 512 channels, stride 8 (H/8, W/8)
        - layer3: 1024 channels, stride 16 (H/16, W/16)
        - layer4: 2048 channels, stride 32 (H/32, W/32)
    
    Args:
        variant: 'resnet50' or 'resnet101'
        pretrained: If True, load ImageNet pretrained weights
        replace_stride_with_dilation: Replace stride with dilation in later layers
                                      for denser feature maps (not used by default)
    """
    
    # Feature channels for each layer
    LAYER_CHANNELS = {
        'resnet50': [256, 512, 1024, 2048],
        'resnet101': [256, 512, 1024, 2048],
    }
    
    def __init__(
        self,
        variant: str = 'resnet101',
        pretrained: bool = True,
        replace_stride_with_dilation: Optional[List[bool]] = None,
    ):
        super().__init__()
        self.variant = variant
        
        # Load base ResNet model
        if variant == 'resnet101':
            if pretrained:
                self.resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            else:
                self.resnet = models.resnet101(weights=None)
        elif variant == 'resnet50':
            if pretrained:
                self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.resnet = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported ResNet variant: {variant}. Use 'resnet50' or 'resnet101'")
        
        # Store channel info for UperNet
        self.out_channels = self.LAYER_CHANNELS[variant]
        self.num_features = self.out_channels[-1]  # For compatibility
        
        # For compatibility with SegmentationModelWrapper
        # ResNet with 512x512 input has these feature map sizes:
        # layer1: 128x128, layer2: 64x64, layer3: 32x32, layer4: 16x16
        # This matches Swin-T stages for 512x512 input
        self._img_size = 512  # Default, can be updated
        
    @property
    def patches_resolution(self):
        """
        Return base resolution for compatibility with SegmentationModelWrapper.
        For ResNet with stride-4 first layer, this is img_size // 4.
        """
        return [self._img_size // 4, self._img_size // 4]
    
    def set_img_size(self, img_size: int):
        """Set input image size for resolution calculations."""
        self._img_size = img_size
        
    def forward(
        self, 
        x: torch.Tensor, 
        return_multi_scale: bool = False
    ) -> List[torch.Tensor]:
        """
        Extract multi-scale features from ResNet.
        
        Args:
            x: Input images [B, 3, H, W]
            return_multi_scale: If True, return features from all stages.
                               If False, return only final features (for classification).
        
        Returns:
            If return_multi_scale=True:
                List of features [layer1_out, layer2_out, layer3_out, layer4_out]
                Each with shape [B, C, H, W]
            If return_multi_scale=False:
                Final features [B, 2048, H/32, W/32]
        """
        # Stem: conv1 + bn1 + relu + maxpool
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        # Extract features from each layer
        c1 = self.resnet.layer1(x)    # [B, 256, H/4, W/4]
        c2 = self.resnet.layer2(c1)   # [B, 512, H/8, W/8]
        c3 = self.resnet.layer3(c2)   # [B, 1024, H/16, W/16]
        c4 = self.resnet.layer4(c3)   # [B, 2048, H/32, W/32]
        
        if return_multi_scale:
            return [c1, c2, c3, c4]
        else:
            return c4
    
    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class ResNetSegmentationWrapper(nn.Module):
    """
    Wrapper for ResNet encoder + segmentation head.
    
    Similar to SegmentationModelWrapper but optimized for ResNet.
    ResNet outputs [B, C, H, W] directly, so no reshaping needed.
    
    Args:
        encoder: ResNetFeatureExtractor
        seg_head: Segmentation head (e.g., UperNetHead)
        freeze_encoder: If True, freeze encoder weights during training
    """
    
    def __init__(
        self,
        encoder: ResNetFeatureExtractor,
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
        # Extract multi-scale features from ResNet
        # Returns list of [B, C, H, W] tensors (already spatial format)
        features_list = self.encoder(x, return_multi_scale=True)
        
        # Pass through segmentation head (no reshaping needed for ResNet)
        output = self.seg_head(features_list)
        
        return output
    
    def train(self, mode: bool = True):
        """Override train mode to keep encoder frozen if requested."""
        super().train(mode)
        
        if self.freeze_encoder:
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
