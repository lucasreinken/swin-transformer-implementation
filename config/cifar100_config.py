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
    "batch_size": 128,  # Reduced batch size for 224x224 images
    "num_workers": 2,
    "root": "./datasets",
    "img_size": 224,  # Resized to 224 for ImageNet-pretrained weights compatibility
    # Subset configuration (optional, set to None to use full dataset)
    "n_train": None,  # Number of training samples (50,000 for full CIFAR-100)
    "n_test": None,  # Number of validation/test samples (10,000 for full CIFAR-100)
}

# Swin Transformer configuration
SWIN_CONFIG = {
    "img_size": 224,  # Resized to 224 for ImageNet-pretrained weights compatibility
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
_TRAINING_MODE = TrainingMode.FROM_SCRATCH
_mode_settings = get_training_mode_settings(_TRAINING_MODE)

DOWNSTREAM_CONFIG = {
    "mode": _TRAINING_MODE,
    "head_type": "linear_classification",
    "num_classes": 100,  # CIFAR-100 has 100 classes
    "hidden_dim": None,
    # Auto-set based on mode
    "freeze_encoder": _mode_settings["freeze_encoder"],
    "use_pretrained": _mode_settings["use_pretrained"],
}

# Training configuration
TRAINING_CONFIG = {
    "learning_rate": 1.25e-4,  # Scaled for batch_size=128: 5e-4 * (128/512) = 1.25e-4
    "num_epochs": 100,  # Full training for CIFAR-100 convergence
    "warmup_epochs": 7,  # Scaled from Swin paper: 20 epochs warmup for 300 epochs = 6.7%, so ~7 epochs for 100 epochs
    "warmup_start_factor": 0.01,  # Start from very low LR
    "weight_decay": 0.05,  # Higher weight decay for regularization
    "min_lr": 1e-6,  # Lower min LR for longer training (allow more decay)
    "lr_scheduler_type": "cosine",  # Pure cosine annealing
    # Early stopping configuration
    "early_stopping": {
        "enabled": False,  # Keep consistent training duration
        "patience": 10,
        "min_delta": 0.005,
        "mode": "max",  # 'max' for accuracy
    },
}

# Augmentation configuration
AUGMENTATION_CONFIG = {
    "use_augmentation": True,
    "rand_augment_m": 9,
    "rand_augment_n": 2,
    "mixup_alpha": 0.8,
    "random_erase_prob": 0.25,
    "mean": [0.5071, 0.4867, 0.4408],  # CIFAR-100 normalization
    "std": [0.2675, 0.2565, 0.2761],
}

# Model Validation Configuration
VALIDATION_CONFIG = {
    "enable_validation": False,
    "pretrained_model": "swin_tiny_patch4_window7_224",
    "transfer_weights": True,
    "validation_samples": 1000,
}
