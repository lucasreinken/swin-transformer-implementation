"""
Configuration file for SCIN masked image modeling (SimMIM) pretraining with Swin + SimMIM.
"""

from .base_config import (
    SWIN_PRESETS,
    SEED_CONFIG,
    apply_swin_preset,
)

# =============================================================================
# Data configuration
# =============================================================================
DATA_CONFIG = {
    "dataset": "SCIN",
    "use_batch_for_val": False,
    "val_batch": None,
    "batch_size": 128,
    "num_workers": 8,
    "root": "./datasets",
    "img_size": 224,     # Use 224 to match ImageNet pretrained Swin window=7 weights
    # Case-level split
    "val_frac": 0.10,
    "test_frac": 0.00,
    # Optional subset
    "n_train": None,
    "n_test": None,
    "stratified": True,
}

# =============================================================================
# Swin Transformer configuration (encoder for SimMIM)
# =============================================================================
SWIN_CONFIG = {
    "img_size": 224,
    "variant": "tiny",   # "tiny", "small", "base", "large"
    "patch_size": 4,
    "embed_dim": None,
    "depths": None,
    "num_heads": None,
    "window_size": 7,
    "mlp_ratio": 4.0,
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "projection_dropout": 0.0,
    "drop_path_rate": 0.05,
    "use_gradient_checkpointing": False,  # enable if memory is tight
}

apply_swin_preset(SWIN_CONFIG, SWIN_PRESETS)

# =============================================================================
# SimMIM configuration
# =============================================================================
SIMMIM_CONFIG = {
    "type": "simmim",
    # Masking
    "mask_ratio": 0.9,           # higher ratio performed better
    "mask_type": "random",        # random patch masking
    # Target normalization
    "norm_target": {
        "enable": True,
        "patch_size": 7,          # must be odd
    },
    # Loss
    "loss": {
        "type": "l1",
    },
}

# =============================================================================
# Training configuration
# =============================================================================
TRAINING_CONFIG = {
    # best performing hyperparameters based on basic grid search
    "learning_rate": 2e-4,
    "num_epochs": 100,
    "warmup_epochs": 10,
    "warmup_start_factor": 0.1,
    "weight_decay": 0.05,
    "mixed_precision": True,
    "compile": False,
    "checkpoint_frequency": 1,
    # Regime selection
    # False = Pure SSL (random init)
    # True  = ImageNet + SSL (load ImageNet pretrained Swin encoder first)
    "load_imagenet_pretrained": True,
    # Early stopping (optional, requires val_loader)
    "early_stopping": {
        "enabled": False,
        "patience": 20,
        "min_delta": 1e-4,
        "mode": "min",
    },
}

# =============================================================================
# Augmentation configuration (image-only)
# =============================================================================
AUGMENTATION_CONFIG = {
    "use_augmentation": True,
    # Keep augmentation moderate
    "random_resized_crop": True,
    "crop_scale": (0.6, 1.0),
    "horizontal_flip": True,
    "color_jitter": 0.2,
    # Normalization (ImageNet)
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

# =============================================================================
# Optional validation config (sanity checks)
# =============================================================================
VALIDATION_CONFIG = {
    "enable_validation": False,
    "validation_samples": 100,
}
