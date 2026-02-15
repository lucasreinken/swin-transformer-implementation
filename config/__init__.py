"""
Configuration module for the ML pipeline.
"""

from .base_config import (
    SWIN_PRESETS,
    VIZ_CONFIG,
    SEED_CONFIG,
    TrainingMode,
    get_pretrained_swin_name as _get_pretrained_swin_name,
)

# Dataset selection - choose one dataset
# DATASET = "cifar10"
# DATASET = "cifar100"
# DATASET = "ade20k"
# DATASET = "scin"
DATASET = "imagenet"

# Data root configuration - choose one based on environment
# For local development:
# DATA_ROOT = "./datasets"
# For cluster:
DATA_ROOT = "/home/space/datasets"


def _load_config():
    """Load the appropriate config based on DATASET environment variable."""
    global AUGMENTATION_CONFIG, DATA_CONFIG, DOWNSTREAM_CONFIG, TRAINING_CONFIG
    global VALIDATION_CONFIG, SWIN_CONFIG
    global MODEL_TYPE, MODEL_CONFIGS, MODEL_CONFIGS

    if DATASET == "cifar10":
        from .cifar10_config import (
            AUGMENTATION_CONFIG,
            DATA_CONFIG,
            DOWNSTREAM_CONFIG,
            TRAINING_CONFIG,
            VALIDATION_CONFIG,
            SWIN_CONFIG,
        )
        from .imagenet_config import (
            MODEL_TYPE,
            MODEL_CONFIGS
        )
    elif DATASET == "cifar100":
        from .cifar100_config import (
            AUGMENTATION_CONFIG,
            DATA_CONFIG,
            DOWNSTREAM_CONFIG,
            TRAINING_CONFIG,
            VALIDATION_CONFIG,
            SWIN_CONFIG,
        )
        from .imagenet_config import (
            MODEL_TYPE,
            MODEL_CONFIGS
        )
    elif DATASET == "imagenet":
        from .imagenet_config import (
            AUGMENTATION_CONFIG,
            DATA_CONFIG,
            DOWNSTREAM_CONFIG,
            TRAINING_CONFIG,
            VALIDATION_CONFIG,
            SWIN_CONFIG,
        )
        from .imagenet_config import (
            MODEL_TYPE,
            MODEL_CONFIGS
        )
    elif DATASET == "ade20k":
        from .ade20k_config import (
            AUGMENTATION_CONFIG,
            DATA_CONFIG,
            DOWNSTREAM_CONFIG,
            TRAINING_CONFIG,
            VALIDATION_CONFIG,
            SWIN_CONFIG,
        )
        from .imagenet_config import (
            MODEL_TYPE,
            MODEL_CONFIGS
        )
    elif DATASET == "scin":
        from .simmim_config import (
            AUGMENTATION_CONFIG,
            DATA_CONFIG,
            DOWNSTREAM_CONFIG,
            TRAINING_CONFIG,
            VALIDATION_CONFIG,
            SWIN_CONFIG,
            MODEL_TYPE
        )
        from .imagenet_config import (
            MODEL_CONFIGS
        )
    else:
        raise ValueError(
            f"Unknown dataset: {DATASET}. Choose from: cifar10, cifar100, imagenet, ade20k"
        )

    # Override data root based on environment
    if DATASET == "imagenet":
        DATA_CONFIG["root"] = "/"
    elif DATASET == "ade20k":
        DATA_CONFIG["root"] = DATA_ROOT
    else:
        DATA_CONFIG["root"] = DATA_ROOT


# Load the config
_load_config()


def get_pretrained_swin_name():
    """Generate TIMM model name based on current SWIN_CONFIG."""
    return _get_pretrained_swin_name(SWIN_CONFIG)


__all__ = [
    "AUGMENTATION_CONFIG",
    "DATA_CONFIG",
    "SWIN_PRESETS",
    "DOWNSTREAM_CONFIG",
    "TRAINING_CONFIG",
    "VIZ_CONFIG",
    "SEED_CONFIG",
    "VALIDATION_CONFIG",
    "SWIN_CONFIG",
    "TrainingMode",
    "MODEL_TYPE",
    "MODEL_CONFIGS",
    "get_pretrained_swin_name",
]
