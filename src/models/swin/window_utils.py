import torch


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition feature map into non-overlapping windows.

    Args:
        x: Input tensor [B, H, W, C]
        window_size: Window size (M)

    Returns:
        Windows tensor [B*num_windows, window_size, window_size, C]
        where num_windows = (H/M) * (W/M)

    Example:
        Input: [2, 56, 56, 96]  (B=2, H=W=56, C=96)
        Window size: 7
        Output: [128, 7, 7, 96]  (2 * (56/7) * (56/7) = 128 windows)
    """
    B, H, W, C = x.shape

    # Reshape to separate windows: [B, H, W, C] → [B, H//M, M, W//M, M, C]
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)

    # Permute to group windows: [B, H//M, M, W//M, M, C] → [B, H//M, W//M, M, M, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    # Flatten batch and window dimensions: [B, H//M, W//M, M, M, C] → [B*num_windows, M, M, C]
    windows = x.view(-1, window_size, window_size, C)

    return windows


def window_reverse(
    windows: torch.Tensor, window_size: int, H: int, W: int
) -> torch.Tensor:
    """
    Reverse window partition, merging windows back into feature map.

    Args:
        windows: Window tensor [B*num_windows, window_size, window_size, C]
        window_size: Window size (M)
        H: Height of feature map
        W: Width of feature map

    Returns:
        Feature map [B, H, W, C]

    Example:
        Input: [128, 7, 7, 96]  (128 windows)
        H=56, W=56, window_size=7
        Output: [2, 56, 56, 96]  (B=2)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))

    # Reshape to separate windows: [B*num_windows, M, M, C] → [B, H//M, W//M, M, M, C]
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )

    # Permute back to spatial layout: [B, H//M, W//M, M, M, C] → [B, H//M, M, W//M, M, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    # Merge windows: [B, H//M, M, W//M, M, C] → [B, H, W, C]
    x = x.view(B, H, W, -1)

    return x


def generate_drop_path_rates(drop_path_rate: float, depth: int) -> list:
    """
    Generate linearly increasing drop path rates for stochastic depth.

    This creates a schedule where deeper layers have higher drop probabilities,
    which helps training very deep networks by reducing gradient vanishing.

    Args:
        drop_path_rate: Maximum drop path rate (for the deepest layer)
        depth: Total number of layers

    Returns:
        List of drop path rates, one per layer

    Example:
        >>> generate_drop_path_rates(0.2, 4)
        [0.0, 0.0667, 0.1333, 0.2]
    """
    if drop_path_rate <= 0 or depth <= 0:
        return [0.0] * max(1, depth)
    if depth == 1:
        return [drop_path_rate]
    return [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
