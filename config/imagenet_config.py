"""
Configuration file for ImageNet-1K dataset.
"""

# Data configuration
DATA_CONFIG = {
    "dataset": "ImageNet",
    "use_batch_for_val": False,
    "val_batch": 5,
    "batch_size": 128,
    "num_workers": 8,
    "root": "./datasets",
    "img_size": 224,
}

# Model configuration
MODEL_CONFIG = {
    "input_dim": 3 * 224 * 224,
    "hidden_dims": [1024, 512, 256],
    "num_classes": 1000,
    "dropout_rate": 0.3,
    "use_batch_norm": True,
}

SWIN_PRESETS = {
    "tiny": {"embed_dim": 96, "depths": [2, 2, 6, 2], "num_heads": [3, 6, 12, 24]},
    "small": {"embed_dim": 96, "depths": [2, 2, 18, 2], "num_heads": [3, 6, 12, 24]},
    "base": {"embed_dim": 128, "depths": [2, 2, 18, 2], "num_heads": [4, 8, 16, 32]},
    "large": {"embed_dim": 192, "depths": [2, 2, 18, 2], "num_heads": [6, 12, 24, 48]},
}

SWIN_CONFIG = {
    "img_size": 224,  # Changed from 32 to 224
    "variant": "base",
    "pretrained_weights": True,
    "patch_size": 4,
    "embed_dim": None,
    "depths": None,
    "num_heads": None,
    "window_size": 7,
    "mlp_ratio": 4.0,
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "projection_dropout": 0.0,
    "drop_path_rate": 0.1,
}

variant = SWIN_CONFIG["variant"]
preset = SWIN_PRESETS[variant]

# auto-set if None in SWIN_CONFIG
if SWIN_CONFIG.get("embed_dim") is None:
    SWIN_CONFIG["embed_dim"] = preset["embed_dim"]

if SWIN_CONFIG.get("depths") is None:
    SWIN_CONFIG["depths"] = preset["depths"]

if SWIN_CONFIG.get("num_heads") is None:
    SWIN_CONFIG["num_heads"] = preset["num_heads"]

DOWNSTREAM_CONFIG = {
    "mode": "linear_probe",
    "head_type": "linear_classification",
    "num_classes": 1000,  # Changed from 10 for ImageNet
    "hidden_dim": None,
}

# auto-set
DOWNSTREAM_CONFIG["freeze_encoder"] = DOWNSTREAM_CONFIG["mode"] == "linear_probe"


# Generate pretrained model name based on Swin variant
def get_pretrained_swin_name():
    """Generate TIMM model name based on SWIN_CONFIG variant."""
    variant = SWIN_CONFIG["variant"]
    patch_size = SWIN_CONFIG["patch_size"]
    window_size = SWIN_CONFIG["window_size"]
    img_size = SWIN_CONFIG["img_size"]
    return f"swin_{variant}_patch{patch_size}_window{window_size}_{img_size}"


# Training configuration
TRAINING_CONFIG = {
    "learning_rate": 0.0001,
    "num_epochs": 90,
    "warmup_epochs": 2,  # Number of warmup epochs for learning rate scheduler
    "weight_decay": 1e-4,
}

# Visualization configuration
VIZ_CONFIG = {
    "figsize": (10, 10),
    "output_file": "visualization.png",
}

# Seed configuration for reproducibility
SEED_CONFIG = {
    "seed": 42,
    "deterministic": False,
}

AUGMENTATION_CONFIG = {
    "use_augmentation": True,
    "rand_augment_m": 9,
    "rand_augment_n": 2,
    "mixup_alpha": 0.8,
    "random_erase_prob": 0.25,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

# Scheduler configuration
SCHEDULER_CONFIG = {
    "use_scheduler": True,
    "optimizer": "AdamW",
    "lr": 0.0001,
    "weight_decay": 0.05,
    "warmup_epochs": 20,
    "total_epochs": 90,
}

# Model Validation Configuration
VALIDATION_CONFIG = {
    "enable_validation": False,
    "use_swin_transformer": True,
    "pretrained_model": "swin_tiny_patch4_window7_224",
    "transfer_weights": True,
    "validation_samples": 50000,  # Full ImageNet validation
}
