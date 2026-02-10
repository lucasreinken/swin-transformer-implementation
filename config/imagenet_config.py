"""
Configuration file for ImageNet-1K dataset.
"""

from .base_config import (
    SWIN_PRESETS,
    VIZ_CONFIG,
    SEED_CONFIG,
    apply_swin_preset,
    TrainingMode,
    get_training_mode_settings,
)

# Model type selection for comparison experiments
MODEL_TYPE = "swin"  # Options: "swin", "swin_v2", "swin_hybrid", "swin_improved", "swin_deformable", "vit", "resnet"

# Model configurations for all types
MODEL_CONFIGS = {
    "swin": {
        "type": "swin",
        "variant": "tiny",
        "patch_size": 4,
        "embed_dim": None,  # Auto-set from preset
        "depths": None,  # Auto-set from preset
        "num_heads": None,  # Auto-set from preset
        "window_size": 7,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "projection_dropout": 0.0,
        "drop_path_rate": 0.08,
        "use_shifted_window": True,
        "use_relative_bias": True,  # FIXED: Enable relative position bias (essential for Swin)
        "use_absolute_pos_embed": False,
        "use_hierarchical_merge": False,
        "use_gradient_checkpointing": False,  # Enable for memory efficiency during long training
    },
    "swin_v2": {
        "type": "swin_v2",
        "variant": "tiny",
        "patch_size": 4,
        "embed_dim": None,  # Auto-set from preset
        "depths": None,  # Auto-set from preset
        "num_heads": None,  # Auto-set from preset
        "window_size": 7,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "projection_dropout": 0.0,
        "drop_path_rate": 0.08,
        "use_shifted_window": True,
        "use_relative_bias": True,  # FIXED: Enable Log-CPB position bias (critical for V2)
        "use_absolute_pos_embed": False,
        "use_hierarchical_merge": False,
        "use_gradient_checkpointing": False,  # Enable for memory efficiency
    },
    "swin_hybrid": {
        "type": "swin_hybrid",
        "variant": "tiny",
        "patch_size": 4,
        "embed_dim": None,  # Auto-set from preset
        "depths": None,  # Auto-set from preset
        "num_heads": None,  # Auto-set from preset
        "window_size": 7,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "projection_dropout": 0.0,
        "drop_path_rate": 0.08,
        "use_shifted_window": True,
        "use_relative_bias": True,  # FIXED: Enable relative position bias (essential for Swin)
        "use_absolute_pos_embed": False,
        "use_hierarchical_merge": False,
        "use_gradient_checkpointing": False,  # Enable for memory efficiency
        # CNN stem configuration for hybrid
        "use_cnn_stem": True,  # Enable CNN-Swin early fusion
        "cnn_stem_config": {
            "channels": [32, 64],  # Intermediate channels before final projection
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "activation": "gelu",  # GELU activation
            "use_batch_norm": True,
        },
    },
    "swin_improved": {
        "type": "swin_improved",
        "variant": "tiny",
        "patch_size": 4,
        "embed_dim": None,  # Auto-set from preset
        "depths": None,  # Auto-set from preset
        "num_heads": None,  # Auto-set from preset
        "window_size": 7,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "projection_dropout": 0.0,
        "drop_path_rate": 0.08,
        "use_shifted_window": True,
        "use_relative_bias": True,  # FIXED: Enable relative position bias (essential for Swin)
        "use_absolute_pos_embed": False,
        "use_hierarchical_merge": False,
        "use_gradient_checkpointing": True,  # CRITICAL: Enable to prevent OOM
        # Convolutional stem (replaces vanilla patch embedding)
        "use_conv_stem": True,
        "conv_stem_config": {
            "channels": [48, 96],  # Overlapping conv stem channels
            "kernel_sizes": [4, 3],  # First conv kernel size 4, second 3
            "strides": [4, 2],  # Downsampling strides
            "paddings": [0, 1],  # Corresponding paddings
            "activation": "gelu",
            "use_batch_norm": True,
        },
        # Inverted residual FFN configuration
        "use_inverted_ffn": True,
        "ffn_config": {
            "expand_ratio": 4,  # Expansion ratio for inverted residual
            "use_depthwise_conv": True,  # Enable depthwise convolution
            "activation": "gelu",  # Activation function
        },
    },
    "swin_deformable": {
        "type": "swin_deformable",
        "variant": "tiny",
        "patch_size": 4,
        "embed_dim": None,  # Auto-set from preset
        "depths": None,  # Auto-set from preset
        "num_heads": None,  # Auto-set from preset
        "window_size": 7,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "projection_dropout": 0.0,
        "drop_path_rate": 0.08,
        "use_shifted_window": True,
        "use_relative_bias": True,  # FIXED: Enable relative position bias (essential for Swin)
        "use_absolute_pos_embed": False,
        "use_hierarchical_merge": False,
        "use_gradient_checkpointing": True,  # Enable for memory efficiency
        # Deformable hybrid attention configuration
        "use_deformable_attn": True,
        "deformable_config": {
            "num_points": 4,  # Number of sampling points per query (4-8 typical)
        },
    },
    "vit": {
        "type": "vit",
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 448,  # Increased for ~30M parameters (very close to Swin)
        "depth": 12,
        "num_heads": 7,  # Increased for better capacity
        "mlp_ratio": 4.0,
        "num_classes": 1000,
        "use_gradient_checkpointing": False,  # Enable for memory efficiency
    },
    "resnet": {
        "type": "resnet",
        "layers": [3, 4, 6, 3],  # ResNet-50
        "num_classes": 1000,
        "use_gradient_checkpointing": False,  # Enable for memory efficiency
    },
}

# Selected model configuration
MODEL_CONFIG = MODEL_CONFIGS[MODEL_TYPE]

# Data configuration
DATA_CONFIG = {
    "dataset": "ImageNet",
    "use_batch_for_val": False,
    "val_batch": 5,
    "batch_size": 128,  # Match baseline: 128 batch size
    "num_workers": 2,  # Reduced from 8 to match system recommendation and avoid slowness
    "root": "./datasets",
    "img_size": 224,
    # Subset configuration for faster training
    "n_train": 100000,  # Increased training samples for better generalization
    "n_test": 50000,  # Number of validation/test samples (None for full dataset) - using full validation set
    "stratified": True,  # Maintain class distribution in train/test split for balanced sampling
}

# Swin Transformer configuration (legacy - kept for compatibility)
SWIN_CONFIG = MODEL_CONFIG if MODEL_TYPE == "swin" else {}

# Apply preset values for None fields (only for Swin variants)
for model_type in [
    "swin",
    "swin_v2",
    "swin_hybrid",
    "swin_improved",
    "swin_deformable",
]:
    if model_type in MODEL_CONFIGS:
        apply_swin_preset(MODEL_CONFIGS[model_type], SWIN_PRESETS)

# =============================================================================
# Downstream Task Configuration
# =============================================================================
# Training mode: "linear_probe" or "from_scratch"
_TRAINING_MODE = TrainingMode.FROM_SCRATCH
_mode_settings = get_training_mode_settings(_TRAINING_MODE)

DOWNSTREAM_CONFIG = {
    "mode": _TRAINING_MODE,
    "head_type": "linear_classification",
    "num_classes": 1000,
    "hidden_dim": None,
    # Auto-set based on mode
    "freeze_encoder": _mode_settings["freeze_encoder"],
    "use_pretrained": _mode_settings["use_pretrained"],
}

# Training configuration
TRAINING_CONFIG = {
    "seed": 42,  # Random seed for reproducibility
    "deterministic": False,  # Set to True for fully reproducible (but slower) training
    "learning_rate": 1e-3,  # Increased for faster convergence (20 epochs - aggressive)
    "num_epochs": 20,  # Reduced to 20 for very fast comparison
    "warmup_epochs": 2,  # Scaled down from 30 (10% warmup ratio)
    "warmup_start_factor": 0.01,  # Start from very low LR
    "weight_decay": 0.05,  # Increased for shorter training (stronger regularization)
    "min_lr": 1e-4,  # CRITICAL: Higher min_lr for short 20-epoch training (was 1e-5, too aggressive decay)
    "lr_scheduler_type": "cosine",  # Pure cosine annealing as in Swin paper
    "mixed_precision": True,
    "compile": True,  # Re-enabled to match baseline configuration
    # Early stopping configuration
    "early_stopping": {
        "enabled": False,  # Disabled for ablation studies to ensure consistent training duration
        "patience": 5,
        "min_delta": 0.01,
        "mode": "min",  # 'min' for loss, 'max' for accuracy
    },
}

# Augmentation configuration
AUGMENTATION_CONFIG = {
    "use_augmentation": True,
    "rand_augment_m": 9,
    "rand_augment_n": 2,
    "mixup_alpha": 0.8,
    "random_erase_prob": 0.25,
    "mean": [0.485, 0.456, 0.406],  # ImageNet normalization
    "std": [0.229, 0.224, 0.225],
}

# Model Validation Configuration
VALIDATION_CONFIG = {
    "enable_validation": False,
    "pretrained_model": "swin_tiny_patch4_window7_224",
    "transfer_weights": True,
    "validation_samples": 50000,  # Full ImageNet validation
}
