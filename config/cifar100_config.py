"""
Configuration file for CIFAR-100 dataset.
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
    "dataset": "CIFAR100",
    "use_batch_for_val": True,
    "val_batch": 5,
    "batch_size": 32,
    "num_workers": 2,
    "root": "./datasets",
    "img_size": 224,  # Resized to 224 for ImageNet-pretrained weights compatibility
    # Subset configuration (optional, set to None to use full dataset)
    "n_train": None,  # Number of training samples
    "n_test": None,  # Number of validation/test samples
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
}

# Apply preset values for None fields
apply_swin_preset(SWIN_CONFIG, SWIN_PRESETS)

# =============================================================================
# Downstream Task Configuration
# =============================================================================
# Training mode: "linear_probe" or "from_scratch"
_TRAINING_MODE = TrainingMode.LINEAR_PROBE
_mode_settings = get_training_mode_settings(_TRAINING_MODE)

DOWNSTREAM_CONFIG = {
    "mode": _TRAINING_MODE,
    "head_type": "linear_classification",
    "num_classes": 100,
    "hidden_dim": None,
    # Auto-set based on mode
    "freeze_encoder": _mode_settings["freeze_encoder"],
    "use_pretrained": _mode_settings["use_pretrained"],
}

# Training configuration
TRAINING_CONFIG = {
    "learning_rate": 0.001,
    "num_epochs": 1,
    "warmup_epochs": 0,
    "warmup_start_factor": 0.1,  # LR multiplier at start of warmup
    "weight_decay": 1e-4,
    "checkpoint_frequency": 10,
    "resume_from_checkpoint": None,
}

# Augmentation configuration
AUGMENTATION_CONFIG = {
    "use_augmentation": False,
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
    "validation_samples": 1000,
}
