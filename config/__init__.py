"""
Configuration module for the ML pipeline.
"""

# Dataset selection - choose one dataset
# DATASET = "cifar10"
DATASET = "cifar100"
# DATASET = "imagenet"

# Data root configuration - choose one based on environment
# # For local development:
# DATA_ROOT = "./datasets"
# For cluster:
DATA_ROOT = "/home/space/datasets"


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

    # Override data root based on environment
    DATA_CONFIG["root"] = DATA_ROOT


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
