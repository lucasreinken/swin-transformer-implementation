"""
Configuration file for CIFAR-10 dataset.
"""

# Data configuration
DATA_CONFIG = {
    "dataset": "CIFAR10",
    "use_batch_for_val": True,
    "val_batch": 5,
    "batch_size": 32,
    "num_workers": 4,
    "root": "./datasets",
    "img_size": 32,
}

# Model configuration
MODEL_CONFIG = {
    "input_dim": 3 * 32 * 32,
    "hidden_dims": [512, 256, 128],
    "num_classes": 10,
    "dropout_rate": 0.3,
    "use_batch_norm": True,
}

SWIN_CONFIG = {
    "img_size": 32,
    "patch_size": 4,
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": 7,
    "mlp_ratio": 4.0,
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "projection_dropout": 0.0,
    "drop_path_rate": 0.1,
    "num_classes": 10,
}

# Training configuration
TRAINING_CONFIG = {
    "learning_rate": 0.001,
    "num_epochs": 20,
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
    "use_augmentation": False,
    "rand_augment_m": 9,
    "rand_augment_n": 2,
    "mixup_alpha": 0.8,
    "random_erase_prob": 0.25,
    "mean": [0.485, 0.456, 0.406],  # ImageNet mean (works for CIFAR too)
    "std": [0.229, 0.224, 0.225],  # ImageNet std (works for CIFAR too)
}

# Scheduler configuration
SCHEDULER_CONFIG = {
    "use_scheduler": False,
    "optimizer": "AdamW",
    "lr": 0.001,
    "weight_decay": 1e-4,
    "warmup_epochs": 2,
    "total_epochs": 20,
}

# Model Validation Configuration
VALIDATION_CONFIG = {
    "enable_validation": True,
    "pretrained_model": "swin_tiny_patch4_window7_224",
    "transfer_weights": True,
    "validation_samples": 1000,
}
