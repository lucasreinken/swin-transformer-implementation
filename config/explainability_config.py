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
    "num_samples": 30,  # 10 classes × 3 samples each
    "sampling_strategy": "diverse_classes",  # Options: 'first', 'random', 'diverse_classes'
    "num_classes": 10,        # Number of distinct classes to sample
    "samples_per_class": 3,   # Samples per class
    
    # Which layers to visualize
    "target_stages": None,  # None = all stages [0, 1, 2, 3]
    "evolution_stages": [0, 1, 2, 3],  # All stages; Stage 3 shows register artifact (not a bug)
    "comparison_stages": [0, 1, 2],  # Stages for W-MSA vs SW-MSA comparison
    
    # Query token selection (which tokens to visualize attention FROM)
    "query_strategy": "center",  # Options: 'center', 'grid', 'corners_center'
    "grid_points": 3,  # For 'grid' strategy
    
    # Visualization style
    "colormap": "jet",  # Matplotlib colormap: 'jet', 'hot', 'viridis', 'turbo'
    "overlay_alpha": 0.6,  # Transparency of attention heatmap overlay (0-1)
    
    # Additional visualizations
    "save_all_maps": False,   # Save all individual attention maps (can be large)
    "per_head_viz": True,     # Visualize individual attention heads
    "per_head_stages": [1, 2],  # Stages for per-head viz (0 has too few heads to interpret)
    "combined_wmsa_swmsa_overlay": True,  # Overlay both W-MSA & SW-MSA windows on one image
    "attention_summary_plot": True,  # Bar-chart summary of entropy/sparsity per stage
    
    # Grad-CAM settings (class-discriminative saliency via gradient-weighted activations)
    "gradcam_enabled": True,          # Generate Grad-CAM visualizations
    "gradcam_stages": [0, 1, 2, 3],  # All stages; Stage 3 shows register artifact
    "gradcam_comparison_stage": 2,    # Stage used for attention-vs-GradCAM side-by-side
}

# =============================================================================
# Data Configuration
# =============================================================================

DATA_CONFIG = {
    "dataset": "ImageNet",  # Must be exact case to match dataloader.py
    "data_path": "/",  # Overlay mounts ImageNet at root: /train_set, /val_set
    "split": "val",  # Use validation set
    "num_samples": 30,  # 10 classes × 3 samples each
    "img_size": 224,  # Input image size
    "batch_size": 1,  # Process one image at a time for visualization
    "shuffle": False,  # Don't shuffle for reproducibility
    "num_workers": 2,  # DataLoader workers (cluster recommends ≤2)
    "pin_memory": True,  # Pin memory for GPU transfer
}

# =============================================================================
# Output Configuration
# =============================================================================

OUTPUT_CONFIG = {
    "base_dir": "runs",
    "experiment_name": "explainability",
}

# =============================================================================
# Analysis Configuration
# =============================================================================

ANALYSIS_CONFIG = {
    "compute_statistics": True,  # Compute attention statistics
}

# =============================================================================
# Seed Configuration (for reproducibility)
# =============================================================================

SEED_CONFIG = {
    "seed": 42,
    "deterministic": True,
}

# =============================================================================
# Build complete MODEL_CONFIG for pipeline
# =============================================================================

MODEL_CONFIG = {
    **SWIN_CONFIG,
    "pretrained_weights": PRETRAINED_CONFIG["pretrained_model"] if PRETRAINED_CONFIG["use_pretrained"] else None,
    "num_classes": 1000,  # ImageNet classes
}