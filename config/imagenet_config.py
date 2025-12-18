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
MODEL_TYPE = "resnet"  # Options: "swin", "vit", "resnet"

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
        "use_relative_bias": True,
        "use_absolute_pos_embed": False,
        "use_hierarchical_merge": False,
        "use_gradient_checkpointing": True,
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
        "use_gradient_checkpointing": True,  # Enable for memory efficiency
    },
    "resnet": {
        "type": "resnet",
        "layers": [3, 4, 6, 3],  # ResNet-50
        "num_classes": 1000,
        "use_gradient_checkpointing": True,  # Enable for memory efficiency
    },
}

# Selected model configuration
MODEL_CONFIG = MODEL_CONFIGS[MODEL_TYPE]

# Data configuration
DATA_CONFIG = {
    "dataset": "ImageNet",
    "use_batch_for_val": False,
    "val_batch": 5,
    "batch_size": 128,  # Increased for better gradient estimates on ImageNet
    "num_workers": 0,  # Set to 0 to avoid worker process issues
    "root": "./datasets",
    "img_size": 224,
    # Subset configuration for faster training
    "n_train": 100000,  # Increased training samples for better generalization
    "n_test": 50000,  # Number of validation/test samples (None for full dataset) - using full validation set
}

# Swin Transformer configuration (legacy - kept for compatibility)
SWIN_CONFIG = MODEL_CONFIG if MODEL_TYPE == "swin" else {}

# Apply preset values for None fields (only for Swin)
if MODEL_TYPE == "swin":
    apply_swin_preset(SWIN_CONFIG, SWIN_PRESETS)

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
    "learning_rate": 1.5e-3,  # Increased for 15 epochs: ~3x higher than 50-epoch LR for faster convergence
    "num_epochs": 15,  # Reduced for testing stability
    "warmup_epochs": 1,  # ~7% of 15 epochs for stability
    "warmup_start_factor": 0.01,  # Start from very low LR
    "weight_decay": 0.02,  # Reduced for training schedule (less regularization needed)
    "min_lr": 1e-4,  # Adjusted for shorter training
    "lr_scheduler_type": "cosine",  # Pure cosine annealing as in Swin paper (no hybrid approaches)
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
