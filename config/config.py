"""
Configuration file for the machine learning project.
"""

# Data configuration
DATA_CONFIG = {
    "dataset": "ImageNet",  # Changed from CIFAR10
    "use_batch_for_val": False,  # Changed for ImageNet
    "val_batch": 5,
    "batch_size": 128,  # Changed from 32 for ImageNet
    "num_workers": 4,
    "root": "./datasets",
    "img_size": 224,  # Changed from 32 for ImageNet
}

# Model configuration
MODEL_CONFIG = {
    "input_dim": 3 * 224 * 224,  # Changed for ImageNet
    "hidden_dims": [1024, 512, 256],  # Changed for ImageNet
    "num_classes": 1000,  # Changed from 10 for ImageNet
    "dropout_rate": 0.3,
    "use_batch_norm": True,
}

SWIN_CONFIG = {
    "img_size": 224,  # or 32 for CIFAR-10
    "patch_size": 4,
    "embed_dim": 128,  # Changed from 96 for Base model
    "depths": [2, 2, 18, 2],  # Changed from [2,2,6,2] for Base model
    "num_heads": [4, 8, 16, 32],  # Changed from [3,6,12,24] for Base model
    "window_size": 7,
    "mlp_ratio": 4.0,
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "projection_dropout": 0.0,
    "drop_path_rate": 0.1,
}

DOWNSTREAM_CONFIG = {
    "mode": "linear_probe",
    "head_type": "linear_classification",
    "num_classes": 1000,  # Changed from 10 for ImageNet
    "hidden_dim": None,
}

# auto-set
DOWNSTREAM_CONFIG["freeze_encoder"] = (DOWNSTREAM_CONFIG["mode"] == "linear_probe")

# Training configuration
TRAINING_CONFIG = {
    "learning_rate": 0.0001,  # Changed from 0.001 for ImageNet
    "num_epochs": 90,  # Changed from 20 for ImageNet
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

# Augmentation configuration
AUGMENTATION_CONFIG = {
    "use_augmentation": True,  # Changed to True for ImageNet
    "rand_augment_m": 9,  # Magnitude for RandAugment
    "rand_augment_n": 2,  # Number of operations per image
    "mixup_alpha": 0.8,  # Alpha for Mixup
    "random_erase_prob": 0.25,  # Probability for RandomErasing
    "mean": [0.485, 0.456, 0.406],  # ImageNet mean (works for both)
    "std": [0.229, 0.224, 0.225],  # ImageNet std
}

# Scheduler configuration
SCHEDULER_CONFIG = {
    "use_scheduler": True,  # Enable for ImageNet training on cluster
    "optimizer": "AdamW",
    "lr": 0.0001,  # Changed for ImageNet
    "weight_decay": 0.05,  # Changed for ImageNet
    "warmup_epochs": 20,  # Changed for ImageNet
    "total_epochs": 90,  # Changed for ImageNet
}

# Model Validation Configuration
VALIDATION_CONFIG = {
    "enable_validation": True,  # Enable for Swin validation
    "pretrained_model": "swin_tiny_patch4_window7_224",
    "transfer_weights": True,  # Transfer weights before comparison
    "validation_samples": 1000,  # Limit samples for faster validation
}
