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

SWIN_CONFIG = {
    "img_size": 224,
    "patch_size": 4,
    "embed_dim": 96,  # Can be changed to 128 for Base, 192 for Large
    "depths": [2, 2, 6, 2],  # Can be changed to [2,2,18,2] for Base/Large
    "num_heads": [
        3,
        6,
        12,
        24,
    ],  # Can be changed to [4,8,16,32] for Base, [6,12,24,48] for Large
    "window_size": 7,
    "mlp_ratio": 4.0,
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "projection_dropout": 0.0,
    "drop_path_rate": 0.1,
    "num_classes": 1000,
}

# Training configuration
TRAINING_CONFIG = {
    "learning_rate": 0.0001,
    "num_epochs": 90,
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
    "enable_validation": True,
    "pretrained_model": "swin_tiny_patch4_window7_224",
    "transfer_weights": True,
    "validation_samples": 50000,  # Full ImageNet validation
}
