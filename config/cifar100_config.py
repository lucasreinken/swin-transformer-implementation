"""
Configuration file for CIFAR-100 dataset.
"""

from .base_config import (
    SWIN_PRESETS,
    VIZ_CONFIG,
    SEED_CONFIG,
    apply_swin_preset,
)

# Data configuration
DATA_CONFIG = {
    "dataset": "CIFAR100",
    "use_batch_for_val": True,
    "val_batch": 5,
    "batch_size": 32,
    "num_workers": 4,
    "root": "./datasets",
    "img_size": 224,  # Resized to 224 for ImageNet-pretrained weights compatibility
}

# Swin Transformer configuration
SWIN_CONFIG = {
    "img_size": 224,
    "variant": "tiny",  # Choose: "tiny", "small", "base", "large"
    "pretrained_weights": True,
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

# Downstream task configuration
DOWNSTREAM_CONFIG = {
    "mode": "linear_probe",
    "head_type": "linear_classification",
    "num_classes": 100,
    "hidden_dim": None,
}

# Auto-set freeze based on mode
DOWNSTREAM_CONFIG["freeze_encoder"] = DOWNSTREAM_CONFIG["mode"] == "linear_probe"

# Training configuration
TRAINING_CONFIG = {
    "learning_rate": 0.001,
    "num_epochs": 50,
    "warmup_epochs": 2,
    "warmup_start_factor": 0.1,  # LR multiplier at start of warmup
    "weight_decay": 1e-4,
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
