"""
UperNet segmentation head for semantic segmentation.

Based on the paper:
"Unified Perceptual Parsing for Scene Understanding" (https://arxiv.org/abs/1807.10221)

UperNet combines:
1. Pyramid Pooling Module (PPM) for multi-scale context
2. Feature Pyramid Network (FPN) for multi-scale feature fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM) for capturing multi-scale context.
    
    Applies adaptive average pooling at multiple scales, then upsamples
    and concatenates the features.
    
    Args:
        in_channels: Number of input channels
        pool_scales: Tuple of pooling output sizes (e.g., (1, 2, 3, 6))
        channels: Number of channels for each pooled feature
    """
    
    def __init__(
        self,
        in_channels: int,
        pool_scales: Tuple[int, ...] = (1, 2, 3, 6),
        channels: int = 512,
    ):
        super().__init__()
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.channels = channels
        
        # Create pooling + conv layers for each scale
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=scale),
                nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
            for scale in pool_scales
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: Input features [B, C, H, W]
            
        Returns:
            List of upsampled pooled features at original resolution
        """
        ppm_outs = []
        input_size = x.shape[2:]  # (H, W)
        
        for stage in self.stages:
            # Pool at specific scale
            pooled = stage(x)  # [B, channels, scale, scale]
            
            # Upsample back to input size
            upsampled = F.interpolate(
                pooled,
                size=input_size,
                mode='bilinear',
                align_corners=False
            )
            ppm_outs.append(upsampled)
        
        return ppm_outs


class UperNetHead(nn.Module):
    """
    UperNet segmentation head.
    
    Combines Pyramid Pooling Module (PPM) and Feature Pyramid Network (FPN)
    to perform multi-scale feature fusion for semantic segmentation.
    
    Architecture:
    1. PPM on the last (deepest) feature map
    2. Lateral connections from all feature maps
    3. Top-down FPN pathway
    4. Final segmentation prediction
    
    Args:
        in_channels: List of input channels for each feature level
                     e.g., [96, 192, 384, 768] for Swin-T
        num_classes: Number of segmentation classes
        channels: Number of channels in FPN (default: 512)
        pool_scales: Scales for PPM (default: (1, 2, 3, 6))
        dropout: Dropout rate before final classifier (default: 0.1)
    """
    
    def __init__(
        self,
        in_channels: List[int],
        num_classes: int,
        channels: int = 512,
        pool_scales: Tuple[int, ...] = (1, 2, 3, 6),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = channels
        
        # =================================================================
        # Pyramid Pooling Module on deepest feature
        # =================================================================
        self.ppm = PyramidPoolingModule(
            in_channels=in_channels[-1],  # 768 for Swin-T
            pool_scales=pool_scales,
            channels=channels,
        )
        
        # Bottleneck after PPM: fuse original features + pooled features
        # Input: in_channels[-1] + len(pool_scales) * channels
        # Output: channels
        ppm_output_channels = in_channels[-1] + len(pool_scales) * channels
        self.ppm_bottleneck = nn.Sequential(
            nn.Conv2d(ppm_output_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # =================================================================
        # FPN lateral connections (1x1 convs to match channels)
        # =================================================================
        self.lateral_convs = nn.ModuleList()
        for i, in_ch in enumerate(in_channels[:-1]):  # Skip last (used in PPM)
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # =================================================================
        # FPN output convs (3x3 convs after upsampling)
        # =================================================================
        self.fpn_convs = nn.ModuleList()
        for _ in range(len(in_channels) - 1):  # One per lateral connection
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # =================================================================
        # Final fusion bottleneck
        # =================================================================
        # Concatenate all FPN outputs (including PPM)
        # Total: len(in_channels) * channels
        fpn_output_channels = len(in_channels) * channels
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(fpn_output_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # =================================================================
        # Segmentation head
        # =================================================================
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through UperNet head.
        
        Args:
            features: List of multi-scale features from encoder
                      [C1, C2, C3, C4] where Ci has shape [B, in_channels[i], Hi, Wi]
                      For Swin-T with 512x512 input:
                        C1: [B, 96, 128, 128]
                        C2: [B, 192, 64, 64]
                        C3: [B, 384, 32, 32]
                        C4: [B, 768, 16, 16]
        
        Returns:
            Segmentation logits [B, num_classes, H, W] at original resolution
        """
        # Fusion happens at first feature level resolution (H/4, W/4)
        # This avoids memory issues with large tensors at full resolution
        # Final upsampling to input resolution happens after bottleneck fusion
        first_feature_size = features[0].shape[2:]  # (H/4, W/4) - fusion resolution
        input_size = (first_feature_size[0] * 4, first_feature_size[1] * 4)  # (H, W) - final output
        
        # =================================================================
        # Step 1: Apply PPM to deepest feature (C4)
        # =================================================================
        ppm_feature = features[-1]  # [B, 768, 16, 16]
        ppm_outs = self.ppm(ppm_feature)  # List of [B, 512, 16, 16]
        
        # Concatenate original feature + all PPM outputs
        ppm_concat = torch.cat([ppm_feature] + ppm_outs, dim=1)  # [B, 768+4*512, 16, 16]
        ppm_output = self.ppm_bottleneck(ppm_concat)  # [B, 512, 16, 16]
        
        # =================================================================
        # Step 2: Build FPN lateral connections
        # =================================================================
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(features[i]))  # [B, 512, Hi, Wi]
        
        # Add PPM output as the last lateral
        laterals.append(ppm_output)  # [B, 512, 16, 16]
        
        # =================================================================
        # Step 3: Top-down FPN pathway
        # =================================================================
        # Start from deepest level, add upsampled features to shallower levels
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample deeper feature to match shallower feature size
            prev_shape = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=False
            )
            # Add to shallower lateral
            laterals[i - 1] = laterals[i - 1] + upsampled
        
        # =================================================================
        # Step 4: Apply FPN convs
        # =================================================================
        fpn_outs = []
        for i in range(len(self.fpn_convs)):
            fpn_outs.append(self.fpn_convs[i](laterals[i]))
        
        # Add PPM output (already processed)
        fpn_outs.append(laterals[-1])
        
        # =================================================================
        # Step 5: Upsample all FPN outputs to fusion size (H/4 x W/4) and concatenate
        # =================================================================
        # Important: We fuse at first_feature_size to avoid memory issues
        # Upsampling 512 channels to 512x512 would create tensors exceeding INT_MAX
        upsampled_fpn_outs = []
        for fpn_out in fpn_outs:
            upsampled = F.interpolate(
                fpn_out,
                size=first_feature_size,  # Fuse at H/4 x W/4
                mode='bilinear',
                align_corners=False
            )
            upsampled_fpn_outs.append(upsampled)
        
        # Concatenate all upsampled FPN features
        fpn_concat = torch.cat(upsampled_fpn_outs, dim=1)  # [B, 4*512, H/4, W/4]
        
        # Final fusion - reduces channels from 2048 to 512
        fused = self.fpn_bottleneck(fpn_concat)  # [B, 512, H/4, W/4]
        
        # =================================================================
        # Step 6: Segmentation prediction
        # =================================================================
        fused = self.dropout(fused)
        output = self.classifier(fused)  # [B, num_classes, H/4, W/4]
        
        # Upsample segmentation output to input resolution
        # Now we only upsample num_classes channels (150 for ADE20K), not 512
        output = F.interpolate(
            output,
            size=input_size,  # Upsample to full resolution (H, W)
            mode='bilinear',
            align_corners=False
        )
        
        return output
    
    def init_weights(self):
        """Initialize weights following official implementation."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
