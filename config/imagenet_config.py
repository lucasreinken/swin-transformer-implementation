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

# Data configuration
DATA_CONFIG = {
    "dataset": "ImageNet",
    "use_batch_for_val": False,
    "val_batch": 5,
    "batch_size": 128,  # Reduced from 128 to fit in GPU memory
    "num_workers": 0,  # Set to 0 to avoid worker process issues
    "root": "./datasets",
    "img_size": 224,
    # Subset configuration for faster training
    "n_train": 50000,  # Number of training samples (None for full dataset) - good balance of speed vs representativeness
    "n_test": 5000,  # Number of validation/test samples (None for full dataset)
}

# Swin Transformer configuration
SWIN_CONFIG = {
    "img_size": 224,
    "variant": "tiny",  # Choose: "tiny", "small", "base", "large"
    "patch_size": 4,
    "embed_dim": None,  # Auto-set from preset
    "depths": None,  # Auto-set from preset
    "num_heads": None,  # Auto-set from preset
    "window_size": 7,
    "mlp_ratio": 4.0,
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "projection_dropout": 0.0,
    "drop_path_rate": 0.1,
    "use_shifted_window": True,  # Enable shifted windows (normal model)
    "use_relative_bias": True,  # Ablation flag for experiment #2: True for learned bias, False for zero bias
}

# Apply preset values for None fields
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
    "learning_rate": 5e-4,  # Higher LR for from-scratch training
    "num_epochs": 15,  # Temporarily reduced for testing memory usage
    "warmup_epochs": 3,  # Longer warmup for stability
    "warmup_start_factor": 0.01,  # Start from very low LR
    "weight_decay": 0.05,  # Higher weight decay for regularization
    "min_lr": 1e-6,  # Minimum LR for cosine annealing
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
