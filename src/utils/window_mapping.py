"""
Window-to-spatial mapping utilities for Swin Transformer attention visualization.

This module provides functions to convert window-based attention weights back to
spatial image coordinates, handling both W-MSA and SW-MSA (shifted window) cases.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def window_partition_inverse(
    windows: torch.Tensor,
    window_size: int,
    H: int,
    W: int
) -> torch.Tensor:
    """
    Reverse window partitioning to reconstruct spatial feature map.
    
    This is the inverse operation of window_partition() used in Swin Transformer.
    Converts window-based representation back to full spatial resolution.
    
    Args:
        windows: Windowed tensor [num_windows*B, window_size, window_size, C]
        window_size: Size of each window
        H: Height of the feature map
        W: Width of the feature map
    
    Returns:
        x: Spatial tensor [B, H, W, C]
    
    Example:
        >>> windows = torch.randn(64, 7, 7, 96)  # 64 windows, 7x7 each
        >>> spatial = window_partition_inverse(windows, 7, 56, 56)
        >>> spatial.shape
        torch.Size([1, 56, 56, 96])
    """
    nH = H // window_size
    nW = W // window_size
    B = windows.shape[0] // (nH * nW)
    
    x = windows.view(B, nH, nW, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    
    return x


def shifted_window_partition_inverse(
    windows: torch.Tensor,
    window_size: int,
    H: int,
    W: int,
    shift_size: int
) -> torch.Tensor:
    """
    Reverse shifted window partitioning (undo cyclic shift).
    
    For SW-MSA blocks, features are cyclically shifted before window partitioning.
    This function reverses both the partitioning and the cyclic shift.
    
    Args:
        windows: Windowed tensor [num_windows*B, window_size, window_size, C]
        window_size: Size of each window
        H: Height of the feature map
        W: Width of the feature map
        shift_size: Amount of cyclic shift (typically window_size // 2)
    
    Returns:
        x: Spatial tensor [B, H, W, C] with shift reversed
    
    Example:
        >>> windows = torch.randn(64, 7, 7, 96)
        >>> spatial = shifted_window_partition_inverse(windows, 7, 56, 56, 3)
        >>> spatial.shape
        torch.Size([1, 56, 56, 96])
    """
    # First reverse the window partitioning
    x = window_partition_inverse(windows, window_size, H, W)
    
    # Then reverse the cyclic shift
    if shift_size > 0:
        x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))
    
    return x


def attention_to_spatial_map(
    attn_weights: torch.Tensor,
    query_position: Tuple[int, int],
    window_metadata: dict,
    aggregate_heads: str = 'mean'
) -> torch.Tensor:
    """
    Convert window-based attention weights to spatial attention map.
    
    This is the core function for attention visualization. It takes attention weights
    from window-based self-attention and maps them back to image coordinates, showing
    which parts of the image a specific query position attends to.
    
    Args:
        attn_weights: Attention tensor [num_windows*B, num_heads, N, N]
            where N = window_size^2
        query_position: (h, w) image coordinates of the query token
        window_metadata: Dictionary containing:
            - 'resolution': (H, W) original resolution
            - 'padded_resolution': (Hp, Wp) after padding
            - 'window_size': window size
            - 'shift_size': shift amount (0 for W-MSA, >0 for SW-MSA)
            - 'is_shifted': bool indicating SW-MSA
            - 'num_windows': (nH, nW) number of windows
        aggregate_heads: How to combine multi-head attention
            - 'mean': Average across heads
            - 'max': Maximum across heads
            - 'min': Minimum across heads
            - int: Use specific head index
    
    Returns:
        spatial_map: [H, W] attention map in original image space
        
    Example:
        >>> attn = model.get_attention_maps()[0]['attention']  # [nW*B, nH, N, N]
        >>> metadata = model.get_attention_maps()[0]
        >>> heatmap = attention_to_spatial_map(attn, (28, 28), metadata)
        >>> heatmap.shape
        torch.Size([56, 56])
    """
    H, W = window_metadata['resolution']
    Hp, Wp = window_metadata['padded_resolution']
    window_size = window_metadata['window_size']
    shift_size = window_metadata['shift_size']
    is_shifted = window_metadata['is_shifted']
    
    nH = Hp // window_size
    nW = Wp // window_size
    N = window_size * window_size
    
    # Aggregate attention heads
    if isinstance(aggregate_heads, int):
        # Use specific head
        attn = attn_weights[:, aggregate_heads, :, :]  # [nW*B, N, N]
    elif aggregate_heads == 'mean':
        attn = attn_weights.mean(dim=1)  # [nW*B, N, N]
    elif aggregate_heads == 'max':
        attn = attn_weights.max(dim=1)[0]  # [nW*B, N, N]
    elif aggregate_heads == 'min':
        attn = attn_weights.min(dim=1)[0]  # [nW*B, N, N]
    else:
        raise ValueError(f"Unknown aggregation method: {aggregate_heads}")
    
    # Map query position to window coordinates
    query_h, query_w = query_position
    
    # Handle shifted windows - need to apply shift to query position
    if is_shifted and shift_size > 0:
        # Apply the same cyclic shift that was used during forward pass
        query_h_shifted = (query_h - shift_size) % Hp
        query_w_shifted = (query_w - shift_size) % Wp
    else:
        query_h_shifted = query_h
        query_w_shifted = query_w
    
    # Determine which window contains the query
    window_h_idx = query_h_shifted // window_size
    window_w_idx = query_w_shifted // window_size
    window_idx = window_h_idx * nW + window_w_idx
    
    # Position within the window
    in_window_h = query_h_shifted % window_size
    in_window_w = query_w_shifted % window_size
    query_token_idx = in_window_h * window_size + in_window_w
    
    # Extract attention for this query token
    # attn shape: [nW*B, N, N] -> we want [nW*B, N] for the query
    query_attn = attn[:, query_token_idx, :]  # [nW*B, N]
    
    # Reshape to window grid: [B, nH, nW, N]
    B = attn.shape[0] // (nH * nW)
    query_attn = query_attn.view(B, nH, nW, N)
    
    # Reshape each window's attention to 2D: [B, nH, nW, window_size, window_size]
    query_attn = query_attn.view(B, nH, nW, window_size, window_size)
    
    # Reconstruct spatial map: [B, Hp, Wp]
    spatial_map = torch.zeros(B, Hp, Wp, device=attn_weights.device)
    
    for h_idx in range(nH):
        for w_idx in range(nW):
            h_start = h_idx * window_size
            h_end = h_start + window_size
            w_start = w_idx * window_size
            w_end = w_start + window_size
            
            spatial_map[:, h_start:h_end, w_start:w_end] = query_attn[:, h_idx, w_idx, :, :]
    
    # Reverse cyclic shift if this was SW-MSA
    if is_shifted and shift_size > 0:
        spatial_map = torch.roll(spatial_map, shifts=(shift_size, shift_size), dims=(1, 2))
    
    # Crop to original resolution (remove padding)
    spatial_map = spatial_map[:, :H, :W]
    
    # Return single image (assume batch size 1)
    spatial_map = spatial_map[0]  # [H, W]
    
    return spatial_map


def get_attention_for_all_queries(
    attn_weights: torch.Tensor,
    window_metadata: dict,
    aggregate_heads: str = 'mean',
    stride: int = 1
) -> torch.Tensor:
    """
    Generate attention maps for all query positions (or strided subset).
    
    This creates a 4D tensor where each spatial position has its own attention map.
    Useful for visualizing attention patterns across the entire image.
    
    Args:
        attn_weights: [num_windows*B, num_heads, N, N]
        window_metadata: Metadata dict from model
        aggregate_heads: Head aggregation method
        stride: Spatial stride for query positions (1 = all, 2 = every other, etc.)
    
    Returns:
        all_attention: [H//stride, W//stride, H, W] attention maps
            all_attention[i, j] is the attention map when query is at (i*stride, j*stride)
    
    Example:
        >>> attn = model.get_attention_maps()[0]['attention']
        >>> metadata = model.get_attention_maps()[0]
        >>> all_attn = get_attention_for_all_queries(attn, metadata, stride=4)
        >>> all_attn.shape
        torch.Size([14, 14, 56, 56])  # 14x14 query positions -> 56x56 attention each
    """
    H, W = window_metadata['resolution']
    
    query_positions = []
    for h in range(0, H, stride):
        for w in range(0, W, stride):
            query_positions.append((h, w))
    
    attention_maps = []
    for query_pos in query_positions:
        attn_map = attention_to_spatial_map(
            attn_weights, query_pos, window_metadata, aggregate_heads
        )
        attention_maps.append(attn_map)
    
    # Reshape to [H//stride, W//stride, H, W]
    attention_maps = torch.stack(attention_maps)  # [num_queries, H, W]
    num_h = H // stride
    num_w = W // stride
    attention_maps = attention_maps.view(num_h, num_w, H, W)
    
    return attention_maps


def compute_attention_distance(
    attn_weights: torch.Tensor,
    window_metadata: dict,
    aggregate_heads: str = 'mean'
) -> float:
    """
    Compute average attention distance (how far tokens attend on average).
    
    This metric measures whether attention is primarily local (small distances)
    or global (large distances). Useful for comparing W-MSA vs SW-MSA.
    
    Args:
        attn_weights: [num_windows*B, num_heads, N, N]
        window_metadata: Metadata dict
        aggregate_heads: Head aggregation method
    
    Returns:
        avg_distance: Average Euclidean distance of attended tokens
    
    Example:
        >>> # W-MSA typically has smaller average distance (local attention)
        >>> wmsa_dist = compute_attention_distance(wmsa_attn, wmsa_metadata)
        >>> # SW-MSA may have larger distance (enables cross-window attention)
        >>> swmsa_dist = compute_attention_distance(swmsa_attn, swmsa_metadata)
    """
    H, W = window_metadata['resolution']
    window_size = window_metadata['window_size']
    
    # Create coordinate grid
    coords_h = torch.arange(H, device=attn_weights.device)
    coords_w = torch.arange(W, device=attn_weights.device)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # [2, H, W]
    coords = coords.float()
    
    # Sample query positions (every 4th to save computation)
    sample_stride = 4
    total_distance = 0.0
    num_samples = 0
    
    for h in range(0, H, sample_stride):
        for w in range(0, W, sample_stride):
            # Get attention map for this query
            attn_map = attention_to_spatial_map(
                attn_weights, (h, w), window_metadata, aggregate_heads
            )  # [H, W]
            
            # Compute weighted distance
            query_coords = torch.tensor([h, w], dtype=torch.float32, device=attn_weights.device)
            
            # Compute distances from query to all positions
            distances = torch.sqrt(
                (coords[0] - query_coords[0])**2 + 
                (coords[1] - query_coords[1])**2
            )  # [H, W]
            
            # Weighted average distance
            avg_dist = (attn_map * distances).sum().item()
            total_distance += avg_dist
            num_samples += 1
    
    return total_distance / num_samples if num_samples > 0 else 0.0


def create_window_grid_mask(
    H: int,
    W: int,
    window_size: int,
    shift_size: int = 0,
    linewidth: int = 1
) -> np.ndarray:
    """
    Create a binary mask showing window boundaries.
    
    Useful for overlaying window boundaries on attention visualizations
    to show how the image is partitioned.
    
    Args:
        H: Image height
        W: Image width
        window_size: Window size
        shift_size: Shift amount (0 for W-MSA, >0 for SW-MSA)
        linewidth: Width of boundary lines in pixels
    
    Returns:
        mask: [H, W] binary mask (1 at boundaries, 0 elsewhere)
    
    Example:
        >>> mask = create_window_grid_mask(56, 56, 7, shift_size=0)
        >>> # Overlay on visualization to show W-MSA windows
    """
    mask = np.zeros((H, W), dtype=np.float32)
    
    # Apply shift if SW-MSA
    if shift_size > 0:
        offset_h = shift_size
        offset_w = shift_size
    else:
        offset_h = 0
        offset_w = 0
    
    # Draw vertical lines
    for w in range(offset_w, W, window_size):
        w_start = max(0, w - linewidth // 2)
        w_end = min(W, w + linewidth // 2 + 1)
        mask[:, w_start:w_end] = 1.0
    
    # Draw horizontal lines
    for h in range(offset_h, H, window_size):
        h_start = max(0, h - linewidth // 2)
        h_end = min(H, h + linewidth // 2 + 1)
        mask[h_start:h_end, :] = 1.0
    
    return mask
