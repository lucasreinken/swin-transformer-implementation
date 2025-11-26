"""
Base configuration shared across all datasets.
Contains settings that are identical regardless of dataset choice.
"""

# Swin Transformer architecture presets (identical for all datasets)
SWIN_PRESETS = {
    "tiny": {"embed_dim": 96, "depths": [2, 2, 6, 2], "num_heads": [3, 6, 12, 24]},
    "small": {"embed_dim": 96, "depths": [2, 2, 18, 2], "num_heads": [3, 6, 12, 24]},
    "base": {"embed_dim": 128, "depths": [2, 2, 18, 2], "num_heads": [4, 8, 16, 32]},
    "large": {"embed_dim": 192, "depths": [2, 2, 18, 2], "num_heads": [6, 12, 24, 48]},
}

# Visualization configuration (identical for all datasets)
VIZ_CONFIG = {
    "figsize": (10, 10),
    "output_file": "visualization.png",
}

# Seed configuration for reproducibility (identical for all datasets)
SEED_CONFIG = {
    "seed": 42,
    "deterministic": False,
}


def get_pretrained_swin_name(swin_config: dict) -> str:
    """
    Generate TIMM model name based on SWIN_CONFIG.

    Args:
        swin_config: Dictionary containing variant, patch_size, window_size, img_size

    Returns:
        TIMM model name string like "swin_tiny_patch4_window7_224"
    """
    variant = swin_config["variant"]
    patch_size = swin_config["patch_size"]
    window_size = swin_config["window_size"]
    img_size = swin_config["img_size"]
    return f"swin_{variant}_patch{patch_size}_window{window_size}_{img_size}"


def apply_swin_preset(swin_config: dict, swin_presets: dict) -> None:
    """
    Apply preset values to SWIN_CONFIG for None fields.
    Modifies swin_config in place.

    Args:
        swin_config: The SWIN_CONFIG dictionary to update
        swin_presets: The SWIN_PRESETS dictionary with variant configurations
    """
    variant = swin_config["variant"]
    preset = swin_presets[variant]

    if swin_config.get("embed_dim") is None:
        swin_config["embed_dim"] = preset["embed_dim"]

    if swin_config.get("depths") is None:
        swin_config["depths"] = preset["depths"]

    if swin_config.get("num_heads") is None:
        swin_config["num_heads"] = preset["num_heads"]
