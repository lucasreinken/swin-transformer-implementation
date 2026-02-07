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
    # Which layers to visualize
    "target_stages": [0, 1, 2, 3],  # All 4 stages (None = all)
    "target_blocks": None,  # Specific blocks per stage (None = all blocks)
    # Example: {"stage_0": [0, 1], "stage_2": [0, 2, 4]} to select specific blocks
    
    # Attention head selection
    "head_aggregation": "mean",  # Options: 'mean', 'max', 'min', or int (specific head index)
    "visualize_individual_heads": False,  # Save separate visualizations per head
    "heads_to_visualize": None,  # Specific heads (None = all, or list like [0, 2, 5])
    
    # Query token selection (which tokens to visualize attention FROM)
    "query_strategy": "grid",  # Options: 'grid', 'center', 'corners', 'salient', 'all'
    "num_query_points": 9,  # For 'grid' strategy (3x3 grid)
    "query_positions": None,  # Manual positions: [(h1, w1), (h2, w2), ...]
    
    # Comparison settings
    "compare_w_msa_sw_msa": True,  # Create side-by-side W-MSA vs SW-MSA comparisons
    "compare_across_stages": True,  # Show how attention evolves across stages
    "show_window_boundaries": True,  # Draw window boundaries on visualizations
    
    # Visualization style
    "colormap": "jet",  # Matplotlib colormap: 'jet', 'hot', 'viridis', 'turbo'
    "overlay_alpha": 0.6,  # Transparency of attention heatmap overlay (0-1)
    "normalize_attention": True,  # Normalize attention to [0, 1] for visualization
    
    # Advanced visualization
    "show_rollout": True,  # Compute attention rollout across layers
    "create_animation": False,  # Create video animation (requires ffmpeg)
    "animation_fps": 2,  # Frames per second for animation
}

# =============================================================================
# Data Configuration
# =============================================================================

DATA_CONFIG = {
    "dataset": "ImageNet",  # Dataset to sample images from
    "root": "./datasets",
    "split": "val",  # Use validation set
    "num_samples": 50,  # Number of images to visualize
    "img_size": 224,  # Input image size
    "batch_size": 1,  # Process one image at a time for visualization
    
    # Image selection strategy
    "selection_strategy": "random",  # Options: 'random', 'first', 'specific'
    "specific_indices": None,  # For 'specific' strategy: [0, 100, 500, ...]
    "class_filter": None,  # Filter by class indices: [1, 5, 10] or None for all
}

# =============================================================================
# Output Configuration
# =============================================================================

OUTPUT_CONFIG = {
    "base_dir": "./visualization_outputs",
    "experiment_name": "swin_tiny_attention_analysis",
    "create_subdirs": True,  # Create subdirectories per image
    
    # What to save
    "save_individual_maps": True,  # Save per-layer attention maps
    "save_comparison_grids": True,  # Save grid comparisons
    "save_rollout": True,  # Save attention rollout visualizations
    "save_statistics": True,  # Save attention statistics (CSV)
    "save_raw_attention": False,  # Save raw attention tensors (.pt files)
    
    # File formats
    "image_format": "png",  # Options: 'png', 'jpg', 'pdf', 'svg'
    "dpi": 300,  # Resolution for saved images
    "figure_size": (12, 8),  # Default figure size (width, height) in inches
    
    # Naming convention
    "include_metadata_in_filename": True,  # Add stage/block/head info to filenames
}

# =============================================================================
# Analysis Configuration
# =============================================================================

ANALYSIS_CONFIG = {
    # Statistical analysis of attention patterns
    "compute_entropy": True,  # Measure attention distribution entropy
    "compute_sparsity": True,  # Measure attention sparsity
    "compute_locality": True,  # Measure local vs global attention
    
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
}

# =============================================================================
# Helper Functions
# =============================================================================

def get_model_config():
    """Get complete model configuration for visualization."""
    return {
        "model_type": MODEL_TYPE,
        "swin_config": SWIN_CONFIG,
        "pretrained": PRETRAINED_CONFIG,
    }

def get_visualization_config():
    """Get complete visualization configuration."""
    return {
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
    if not SWIN_CONFIG.get("return_attention_maps"):
        errors.append("return_attention_maps must be True for explainability")
    
    # Validate visualization config
    valid_strategies = ['grid', 'center', 'corners', 'salient', 'all']
    if VIZ_CONFIG["query_strategy"] not in valid_strategies:
        errors.append(f"query_strategy must be one of {valid_strategies}")
    
    # Validate data config
    if DATA_CONFIG["batch_size"] != 1:
        errors.append("batch_size must be 1 for attention visualization")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    return True

# Validate on import
validate_config()
