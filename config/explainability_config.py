"""
Configuration file for Swin Transformer attention visualization and explainability.

This configuration enables the capture and visualization of attention weights from
W-MSA (Window Multi-head Self-Attention) and SW-MSA (Shifted Window MSA) blocks
to understand how the model attends to different spatial regions.
"""

from .base_config import SWIN_PRESETS, apply_swin_preset

# =============================================================================
# Model Configuration
# =============================================================================

# Model type selection for explainability (use pretrained Swin-Tiny)
MODEL_TYPE = "swin"  # Only "swin" is supported for attention visualization

# Swin Transformer configuration with attention capture enabled
SWIN_CONFIG = {
    "type": "swin",
    "variant": "tiny",  # Use Swin-Tiny for faster processing
    "patch_size": 4,
    "embed_dim": None,  # Auto-set from preset
    "depths": None,  # Auto-set from preset
    "num_heads": None,  # Auto-set from preset
    "window_size": 7,
    "mlp_ratio": 4.0,
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "projection_dropout": 0.0,
    "drop_path_rate": 0.0,
    "use_shifted_window": True,  # Keep default Swin behavior
    "use_relative_bias": True,   # Keep default Swin behavior
    "use_absolute_pos_embed": False,
    "use_hierarchical_merge": False,
    "use_gradient_checkpointing": False,
    "return_attention_maps": True,  # CRITICAL: Enable attention capture
}

# Apply preset values
apply_swin_preset(SWIN_CONFIG, SWIN_PRESETS)

# Pretrained weights configuration
PRETRAINED_CONFIG = {
    "use_pretrained": True,
    "pretrained_model": "swin_tiny_patch4_window7_224",  # TIMM model name
    "freeze_model": True,  # Don't update weights during visualization
}

# =============================================================================
# Visualization Configuration
# =============================================================================

VIZ_CONFIG = {
    # Sample settings
    "num_samples": 5,  # Number of images to visualize
    "sampling_strategy": "first",  # Options: 'first', 'random', 'diverse'
    
    # Which layers to visualize
    "target_stages": None,  # None = all stages [0, 1, 2, 3]
    "comparison_stages": [0, 1, 2],  # Stages for W-MSA vs SW-MSA comparison
    
    # Query token selection (which tokens to visualize attention FROM)
    "query_strategy": "corners_center",  # Options: 'center', 'grid', 'corners_center'
    "grid_points": 3,  # For 'grid' strategy
    
    # Visualization style
    "colormap": "jet",  # Matplotlib colormap: 'jet', 'hot', 'viridis', 'turbo'
    "overlay_alpha": 0.6,  # Transparency of attention heatmap overlay (0-1)
    
    # Output control
    "save_all_maps": False,  # Save all individual attention maps (can be large)
}

# =============================================================================
# Data Configuration
# =============================================================================

DATA_CONFIG = {
    "dataset": "imagenet",  # Dataset to sample images from
    "data_path": "/data/imagenet",  # Path inside container (from overlay)
    "split": "val",  # Use validation set
    "num_samples": 5,  # Number of images to visualize
    "img_size": 224,  # Input image size
    "batch_size": 1,  # Process one image at a time for visualization
    "shuffle": False,  # Don't shuffle for reproducibility
    "num_workers": 4,  # DataLoader workers
    "pin_memory": True,  # Pin memory for GPU transfer
    "augmentation_strength": "none",  # No augmentation for visualization
}

# =============================================================================
# Output Configuration
# =============================================================================

OUTPUT_CONFIG = {
    "base_dir": "./visualization_outputs",
    "experiment_name": "swin_tiny_attention_analysis",
    "create_subdirs": True,  # Create subdirectories per image
    
    # What to save
    "save_individruns",
    "experiment_name": "explainabilityarisons
    "save_rollout": True,  # Save attention rollout visualizations
    "save_statistics": True,  # Save attention statistics (CSV)
    "save_raw_attention": False,  # Save raw attention tensors (.pt files)
    
    # File formats
    "image_format": "png",  # Options: 'png', 'jpg', 'pdf', 'svg'

    
    # Distance-based analysis
    "analyze_attention_distance": True,  # Average distance of attended tokens
    "distance_metric": "euclidean",  # Options: 'euclidean', 'manhattan'
    
    # Comparative analysis
    "compare_w_msa_sw_msa_stats": True,  # Statistical comparison
    "compare_across_stages_stats": True,  # Track evolution across stages
    "compare_across_heads_stats": True,  # Head specialization analysis
    
    # Thresholds
    "attention_threshold": 0.1,  # Minimum attention value to consider
    "top_k_tokens": 10,  # Number of top attended tokens to track
}

# =============================================================================
# Seed Configuration (for reproducibility)
# =============================================================================

SEED_CONFIG = {
    "seed": 42,
    "deterministic": True,
}"compute_statistics": True,  # Compute attention statistics
        "viz": VIZ_CONFIG,
        "data": DATA_CONFIG,
        "output": OUTPUT_CONFIG,
        "analysis": ANALYSIS_CONFIG,
        "seed": SEED_CONFIG,
    }

# =============================================================================
# Validation
# =============================================================================

def validate_config():
    """Validate configuration parameters."""
    errors = []
    
    # Validate model config
  Build complete MODEL_CONFIG for pipeline
MODEL_CONFIG = {
    **SWIN_CONFIG,
    "pretrained_weights": PRETRAINED_CONFIG["pretrained_model"] if PRETRAINED_CONFIG["use_pretrained"] else None,
    "num_classes": 1000,  # ImageNet classes
}