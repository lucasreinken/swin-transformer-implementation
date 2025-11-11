"""
Configuration module for the ML pipeline.
"""

import os

# Dataset selection - change this directly to switch between datasets
# Options: "cifar10", "cifar100", "imagenet"
# DATASET = os.getenv("DATASET", "cifar10").lower()
DATASET = "cifar10"  # Uncomment and change this line


def _load_config():
    """Load the appropriate config based on DATASET environment variable."""
    global AUGMENTATION_CONFIG, DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG
    global VIZ_CONFIG, SEED_CONFIG, SCHEDULER_CONFIG, VALIDATION_CONFIG, SWIN_CONFIG

    if DATASET == "cifar10":
        from .cifar10_config import (
            AUGMENTATION_CONFIG,
            DATA_CONFIG,
            MODEL_CONFIG,
            TRAINING_CONFIG,
            VIZ_CONFIG,
            SEED_CONFIG,
            SCHEDULER_CONFIG,
            VALIDATION_CONFIG,
            SWIN_CONFIG,
        )
    elif DATASET == "cifar100":
        from .cifar100_config import (
            AUGMENTATION_CONFIG,
            DATA_CONFIG,
            MODEL_CONFIG,
            TRAINING_CONFIG,
            VIZ_CONFIG,
            SEED_CONFIG,
            SCHEDULER_CONFIG,
            VALIDATION_CONFIG,
            SWIN_CONFIG,
        )
    elif DATASET == "imagenet":
        from .imagenet_config import (
            AUGMENTATION_CONFIG,
            DATA_CONFIG,
            MODEL_CONFIG,
            TRAINING_CONFIG,
            VIZ_CONFIG,
            SEED_CONFIG,
            SCHEDULER_CONFIG,
            VALIDATION_CONFIG,
            SWIN_CONFIG,
        )
    else:
        raise ValueError(
            f"Unknown dataset: {DATASET}. Choose from: cifar10, cifar100, imagenet"
        )


# Load the config
_load_config()

__all__ = [
    "AUGMENTATION_CONFIG",
    "DATA_CONFIG",
    "MODEL_CONFIG",
    "TRAINING_CONFIG",
    "VIZ_CONFIG",
    "SEED_CONFIG",
    "SCHEDULER_CONFIG",
    "VALIDATION_CONFIG",
    "SWIN_CONFIG",
]
