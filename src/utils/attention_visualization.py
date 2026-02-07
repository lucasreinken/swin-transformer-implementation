"""
Attention visualization utilities for Swin Transformer.

This module provides tools to visualize and analyze attention patterns from
Swin Transformer, including W-MSA and SW-MSA comparisons, multi-stage analysis,
and attention rollout visualizations.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict
import logging

from .window_mapping import (
    attention_to_spatial_map,
    get_attention_for_all_queries,
    compute_attention_distance,
    create_window_grid_mask,
)

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """
    Main class for visualizing Swin Transformer attention patterns.
    
    This class provides a high-level interface for extracting and visualizing
    attention weights from a Swin Transformer model with return_attention_maps=True.
    
    Example:
        >>> from src.models.swin.swin_transformer_model import SwinTransformerModel
        >>> model = SwinTransformerModel(return_attention_maps=True, ...)
        >>> visualizer = AttentionVisualizer(model, device='cuda')
        >>> image = torch.randn(1, 3, 224, 224)
        >>> visualizer.extract_attention_maps(image)
        >>> visualizer.visualize_stage(0, save_path='stage0.png')
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda',
        colormap: str = 'jet',
        overlay_alpha: float = 0.6
    ):
        """
        Initialize the AttentionVisualizer.
        
        Args:
            model: Swin Transformer model with return_attention_maps=True
            device: Device to run on ('cuda' or 'cpu')
            colormap: Matplotlib colormap for heatmaps
            overlay_alpha: Transparency for attention overlay (0-1)
        """
        self.model = model.to(device)
        self.device = device
        self.colormap = colormap
        self.overlay_alpha = overlay_alpha
        
        # Verify model has attention capture enabled
        if not getattr(model, 'return_attention_maps', False):
            raise ValueError(
                "Model must have return_attention_maps=True. "
                "Create model with this flag enabled."
            )
        
        self.attention_maps = []
        self.last_image = None
        
    def extract_attention_maps(
        self,
        image: torch.Tensor,
        target_stages: Optional[List[int]] = None,
        target_blocks: Optional[Dict[str, List[int]]] = None
    ) -> List[Dict]:
        """
        Run forward pass and extract attention maps.
        
        Args:
            image: Input tensor [B, 3, H, W] (typically B=1 for visualization)
            target_stages: List of stage indices to extract (None = all)
            target_blocks: Dict mapping stage names to block indices
                Example: {'stage_0': [0, 1], 'stage_2': [0, 2, 4]}
        
        Returns:
            List of attention map dictionaries
        """
        self.model.eval()
        image = image.to(self.device)
        self.last_image = image
        
        with torch.no_grad():
            _ = self.model(image)
            all_attention_maps = self.model.get_attention_maps()
        
        # Filter by target stages/blocks if specified
        if target_stages is not None:
            all_attention_maps = [
                m for m in all_attention_maps if m['stage'] in target_stages
            ]
        
        if target_blocks is not None:
            filtered = []
            for m in all_attention_maps:
                stage_key = f"stage_{m['stage']}"
                if stage_key in target_blocks:
                    if m['block'] in target_blocks[stage_key]:
                        filtered.append(m)
            all_attention_maps = filtered
        
        self.attention_maps = all_attention_maps
        logger.info(f"Extracted {len(self.attention_maps)} attention maps")
        
        return self.attention_maps
    
    def create_attention_heatmap(
        self,
        image: Union[torch.Tensor, Image.Image, np.ndarray],
        attn_map: torch.Tensor,
        overlay: bool = True,
        show_colorbar: bool = True,
        title: Optional[str] = None,
        window_boundaries: bool = False,
        window_size: Optional[int] = None,
        shift_size: int = 0
    ) -> Image.Image:
        """
        Create attention heatmap visualization.
        
        Args:
            image: Original image (tensor, PIL, or numpy)
            attn_map: Attention map [H, W]
            overlay: Whether to overlay on original image
            show_colorbar: Whether to show colorbar
            title: Optional title for the plot
            window_boundaries: Whether to show window boundaries
            window_size: Window size (required if window_boundaries=True)
            shift_size: Shift size for SW-MSA boundaries
        
        Returns:
            PIL Image of the visualization
        """
        # Convert image to numpy
        if isinstance(image, torch.Tensor):
            if image.ndim == 4:
                image = image[0]  # Remove batch dim
            image_np = image.cpu().permute(1, 2, 0).numpy()
            # Denormalize if needed (assuming ImageNet normalization)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = std * image_np + mean
            image_np = np.clip(image_np, 0, 1)
        elif isinstance(image, Image.Image):
            image_np = np.array(image) / 255.0
        else:
            image_np = image
        
        # Convert attention to numpy
        attn_np = attn_map.cpu().numpy()
        
        # Normalize attention map
        attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
        
        # Resize attention to match image if needed
        if attn_np.shape != image_np.shape[:2]:
            attn_np = np.array(
                Image.fromarray((attn_np * 255).astype(np.uint8)).resize(
                    (image_np.shape[1], image_np.shape[0]), Image.BILINEAR
                )
            ) / 255.0
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        if overlay:
            # Show original image
            ax.imshow(image_np)
            # Overlay attention heatmap
            im = ax.imshow(attn_np, cmap=self.colormap, alpha=self.overlay_alpha)
        else:
            # Show only heatmap
            im = ax.imshow(attn_np, cmap=self.colormap)
        
        # Add window boundaries if requested
        if window_boundaries and window_size is not None:
            H, W = attn_np.shape
            mask = create_window_grid_mask(H, W, window_size, shift_size, linewidth=2)
            ax.imshow(mask, cmap='gray', alpha=0.3)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.axis('off')
        
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Attention Weight', fontsize=10)
        
        # Convert to PIL Image
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        image_pil = Image.frombytes(
            'RGBA', (w, h), fig.canvas.buffer_rgba()
        ).convert('RGB')
        
        plt.close(fig)
        
        return image_pil
    
    def compare_w_msa_sw_msa(
        self,
        query_position: Tuple[int, int],
        stage_idx: int,
        save_path: Optional[str] = None
    ) -> Image.Image:
        """
        Create side-by-side comparison of W-MSA vs SW-MSA attention.
        
        Args:
            query_position: (h, w) query coordinates
            stage_idx: Which stage to visualize
            save_path: Optional path to save the comparison
        
        Returns:
            PIL Image of the comparison
        """
        # Find W-MSA and SW-MSA blocks in the target stage
        stage_maps = [m for m in self.attention_maps if m['stage'] == stage_idx]
        
        wmsa_block = None
        swmsa_block = None
        
        for m in stage_maps:
            if not m['is_shifted'] and wmsa_block is None:
                wmsa_block = m
            elif m['is_shifted'] and swmsa_block is None:
                swmsa_block = m
            
            if wmsa_block is not None and swmsa_block is not None:
                break
        
        if wmsa_block is None or swmsa_block is None:
            raise ValueError(f"Could not find both W-MSA and SW-MSA blocks in stage {stage_idx}")
        
        # Scale query position from image coordinates to feature-map coordinates
        feat_H, feat_W = wmsa_block['resolution']
        img_H, img_W = 224, 224  # Swin-Tiny input resolution
        scaled_query = (
            min(query_position[0] * feat_H // img_H, feat_H - 1),
            min(query_position[1] * feat_W // img_W, feat_W - 1),
        )
        
        # Get attention maps (feature-map resolution)
        wmsa_attn = attention_to_spatial_map(
            wmsa_block['attention'],
            scaled_query,
            wmsa_block,
            aggregate_heads='mean'
        )
        
        swmsa_attn = attention_to_spatial_map(
            swmsa_block['attention'],
            scaled_query,
            swmsa_block,
            aggregate_heads='mean'
        )
        
        # Create comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Prepare image
        if self.last_image is not None:
            img = self.last_image[0].cpu().permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
        else:
            img = np.zeros((224, 224, 3))
        
        # Original image with query point
        axes[0].imshow(img)
        axes[0].plot(query_position[1], query_position[0], 'r*', markersize=20)
        axes[0].set_title('Original Image\n(Red star = Query position)', fontsize=12)
        axes[0].axis('off')
        
        # Helper: normalize and resize attention map to image resolution
        def _prep_attn(attn_tensor):
            a = attn_tensor.cpu().numpy()
            a = (a - a.min()) / (a.max() - a.min() + 1e-8)
            a = np.array(
                Image.fromarray((a * 255).astype(np.uint8)).resize(
                    (img.shape[1], img.shape[0]), Image.BILINEAR
                )
            ) / 255.0
            return a
        
        # W-MSA attention
        wmsa_np = _prep_attn(wmsa_attn)
        axes[1].imshow(img)
        im1 = axes[1].imshow(wmsa_np, cmap=self.colormap, alpha=self.overlay_alpha)
        axes[1].set_title(
            f'W-MSA (Stage {stage_idx}, Block {wmsa_block["block"]})\n'
            f'Window-based attention ({feat_H}×{feat_W})',
            fontsize=12
        )
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # SW-MSA attention
        swmsa_np = _prep_attn(swmsa_attn)
        axes[2].imshow(img)
        im2 = axes[2].imshow(swmsa_np, cmap=self.colormap, alpha=self.overlay_alpha)
        axes[2].set_title(
            f'SW-MSA (Stage {stage_idx}, Block {swmsa_block["block"]})\n'
            f'Shifted window attention ({feat_H}×{feat_W})',
            fontsize=12
        )
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Convert to PIL
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        comparison_pil = Image.frombytes(
            'RGBA', (w, h), fig.canvas.buffer_rgba()
        ).convert('RGB')
        
        if save_path:
            comparison_pil.save(save_path, dpi=(300, 300))
            logger.info(f"Saved W-MSA vs SW-MSA comparison to {save_path}")
        
        plt.close(fig)
        
        return comparison_pil
    
    def visualize_stage_evolution(
        self,
        query_position: Tuple[int, int],
        save_path: Optional[str] = None
    ) -> Image.Image:
        """
        Visualize how attention evolves across all stages.
        
        Args:
            query_position: (h, w) query coordinates (in stage 0 resolution)
            save_path: Optional path to save visualization
        
        Returns:
            PIL Image showing attention across all stages
        """
        stages = sorted(set(m['stage'] for m in self.attention_maps))
        
        fig, axes = plt.subplots(1, len(stages) + 1, figsize=(5 * (len(stages) + 1), 5))
        
        # Original image
        if self.last_image is not None:
            img = self.last_image[0].cpu().permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
        else:
            img = np.zeros((224, 224, 3))
        
        axes[0].imshow(img)
        axes[0].plot(query_position[1], query_position[0], 'r*', markersize=15)
        axes[0].set_title('Original\n(Query: red star)', fontsize=10)
        axes[0].axis('off')
        
        # Each stage
        for idx, stage_idx in enumerate(stages):
            stage_maps = [m for m in self.attention_maps if m['stage'] == stage_idx]
            if not stage_maps:
                continue
            
            # Use first block of each stage
            block = stage_maps[0]
            
            # Scale query position to this stage's resolution
            H, W = block['resolution']
            stage_query = (
                query_position[0] * H // 224,
                query_position[1] * W // 224
            )
            stage_query = (
                min(stage_query[0], H - 1),
                min(stage_query[1], W - 1)
            )
            
            attn_map = attention_to_spatial_map(
                block['attention'],
                stage_query,
                block,
                aggregate_heads='mean'
            )
            
            # Upsample to original resolution for visualization
            attn_np = attn_map.cpu().numpy()
            attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
            attn_upsampled = np.array(
                Image.fromarray((attn_np * 255).astype(np.uint8)).resize(
                    (224, 224), Image.BILINEAR
                )
            ) / 255.0
            
            axes[idx + 1].imshow(img)
            im = axes[idx + 1].imshow(attn_upsampled, cmap=self.colormap, alpha=self.overlay_alpha)
            axes[idx + 1].set_title(
                f'Stage {stage_idx}\n{H}×{W} resolution',
                fontsize=10
            )
            axes[idx + 1].axis('off')
            plt.colorbar(im, ax=axes[idx + 1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Convert to PIL
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        evolution_pil = Image.frombytes(
            'RGBA', (w, h), fig.canvas.buffer_rgba()
        ).convert('RGB')
        
        if save_path:
            evolution_pil.save(save_path, dpi=(300, 300))
            logger.info(f"Saved stage evolution to {save_path}")
        
        plt.close(fig)
        
        return evolution_pil
    
    def compute_attention_statistics(self) -> Dict[str, List[float]]:
        """
        Compute statistical metrics for all captured attention maps.
        
        Returns:
            Dictionary with metrics per layer:
            - 'entropy': Attention distribution entropy
            - 'sparsity': Percentage of near-zero attention weights
            - 'avg_distance': Average spatial distance of attended tokens
            - 'max_attention': Maximum attention value
        """
        stats = {
            'stage': [],
            'block': [],
            'is_shifted': [],
            'entropy': [],
            'sparsity': [],
            'avg_distance': [],
            'max_attention': [],
        }
        
        for attn_map in self.attention_maps:
            attn = attn_map['attention']  # [nW*B, nH, N, N]
            
            # Aggregate heads
            attn_agg = attn.mean(dim=1)  # [nW*B, N, N]
            
            # Compute entropy
            eps = 1e-8
            entropy = -(attn_agg * torch.log(attn_agg + eps)).sum(dim=-1).mean().item()
            
            # Compute sparsity (percentage below threshold)
            threshold = 0.01
            sparsity = (attn_agg < threshold).float().mean().item() * 100
            
            # Compute average distance
            avg_dist = compute_attention_distance(attn, attn_map, aggregate_heads='mean')
            
            # Max attention
            max_attn = attn_agg.max().item()
            
            stats['stage'].append(attn_map['stage'])
            stats['block'].append(attn_map['block'])
            stats['is_shifted'].append(attn_map['is_shifted'])
            stats['entropy'].append(entropy)
            stats['sparsity'].append(sparsity)
            stats['avg_distance'].append(avg_dist)
            stats['max_attention'].append(max_attn)
        
        return stats
    
    def save_all_attention_maps(
        self,
        output_dir: Union[str, Path],
        query_positions: Optional[List[Tuple[int, int]]] = None,
        prefix: str = ''
    ):
        """
        Save all attention maps to directory.
        
        Args:
            output_dir: Directory to save visualizations
            query_positions: List of query positions to visualize
                If None, uses center and corners
            prefix: Prefix for filenames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default query positions in image coordinates (224x224)
        if query_positions is None:
            query_positions = [
                (112, 112),  # Center
                (28, 28),    # Top-left
                (28, 196),   # Top-right
                (196, 28),   # Bottom-left
                (196, 196),  # Bottom-right
            ]
        
        logger.info(f"Saving attention visualizations to {output_dir}")
        
        for query_idx, query_pos in enumerate(query_positions):
            for attn_map in self.attention_maps:
                # Scale query position from image coords to feature-map coords
                feat_H, feat_W = attn_map['resolution']
                scaled_query = (
                    min(query_pos[0] * feat_H // 224, feat_H - 1),
                    min(query_pos[1] * feat_W // 224, feat_W - 1),
                )
                
                # Generate attention map
                spatial_attn = attention_to_spatial_map(
                    attn_map['attention'],
                    scaled_query,
                    attn_map,
                    aggregate_heads='mean'
                )
                
                # Create visualization
                title = (
                    f"Stage {attn_map['stage']}, Block {attn_map['block']} "
                    f"({'SW-MSA' if attn_map['is_shifted'] else 'W-MSA'})\n"
                    f"Query: ({query_pos[0]}, {query_pos[1]})"
                )
                
                vis = self.create_attention_heatmap(
                    self.last_image,
                    spatial_attn,
                    overlay=True,
                    title=title,
                    window_boundaries=True,
                    window_size=attn_map['window_size'],
                    shift_size=attn_map['shift_size'] if attn_map['is_shifted'] else 0
                )
                
                # Save
                filename = (
                    f"{prefix}query{query_idx}_"
                    f"stage{attn_map['stage']}_"
                    f"block{attn_map['block']}_"
                    f"{'swmsa' if attn_map['is_shifted'] else 'wmsa'}.png"
                )
                vis.save(output_dir / filename)
        
        logger.info(f"Saved {len(self.attention_maps) * len(query_positions)} visualizations")
