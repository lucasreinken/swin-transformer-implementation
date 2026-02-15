"""
Configuration file for SD-198 dataset (Skin Disease 198 classes).

Designed for:
- Fine-Tuning on SimMIM pretrained backbone
"""

from .base_config import (
    SWIN_PRESETS,
    VIZ_CONFIG,
    SEED_CONFIG,
    apply_swin_preset,
    TrainingMode,
    get_training_mode_settings,
)

# =============================================================================
# Model type selection
# =============================================================================
MODEL_TYPE = "swin"  # Options: "swin", "vit", "resnet"

MODEL_CONFIGS = {
    "swin": {
        "type": "swin",
        "variant": "tiny",
        "patch_size": 4,
        "embed_dim": None,  # auto-set by preset
        "depths": None,
        "num_heads": None,
        "window_size": 7,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "projection_dropout": 0.0,
        "drop_path_rate": 0.2,
        "use_shifted_window": True,
        "use_relative_bias": True,
        "use_absolute_pos_embed": False,
        "use_hierarchical_merge": False,
        "use_gradient_checkpointing": False,
    },
    "vit": {
        "type": "vit",
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "num_classes": 198,
        "use_gradient_checkpointing": False,
    },
    "resnet": {
        "type": "resnet",
        "layers": [3, 4, 6, 3],  # ResNet-50
        "num_classes": 198,
        "use_gradient_checkpointing": False,
    },
}

MODEL_CONFIG = MODEL_CONFIGS[MODEL_TYPE]
SWIN_CONFIG = None

# Apply preset values (Swin only)
if MODEL_TYPE == "swin":
    apply_swin_preset(MODEL_CONFIG, SWIN_PRESETS)
    SWIN_CONFIG = {
        **{k: v for k, v in MODEL_CONFIG.items() if k != "type"},
        "img_size": 224,
    }

# =============================================================================
# Data configuration
# =============================================================================
DATA_CONFIG = {
    "dataset": "SD198",
    "use_batch_for_val": False,
    "val_batch": None,
    "batch_size": 128,
    "num_workers": 8,
    "root": "./datasets",
    "img_size": 224,
    "val_frac": 0,
    "test_frac": 0.50,
    "n_train": None,
    "n_test": None,
    "stratified": True,
}

# =============================================================================
# Downstream task configuration
# =============================================================================
# Choose:
#   TrainingMode.LINEAR_PROBE
#   TrainingMode.FROM_SCRATCH
_TRAINING_MODE = TrainingMode.FINE_TUNE
_mode_settings = get_training_mode_settings(_TRAINING_MODE)

DOWNSTREAM_CONFIG = {
    "mode": _TRAINING_MODE,
    "head_type": "linear_classification",
    "num_classes": 198,
    "hidden_dim": None,
    "use_reference_model": False,   # True only when you want the timm-vs-custom sanity check

    # SimMIM checkpoint path (None: timm pretrained weights; "random": random initialized weights)
    "pretrained_path": "/home/pml17/Machine-Learning-Project/runs/run_438/best_model.pth",
    # "/home/pml17/Machine-Learning-Project/runs/run_452/best_model.pth"

    # Auto-set by mode
    "freeze_encoder": _mode_settings["freeze_encoder"],
    "use_pretrained": _mode_settings["use_pretrained"],
}

# =============================================================================
# Training configuration
# =============================================================================
TRAINING_CONFIG = {
    "learning_rate": 5e-4,  # Reduced from 0.0015 for fine-tuning
    "num_epochs": 50,
    "warmup_epochs": 5,     # Increase slightly to stabilize early training
    "warmup_start_factor": 1e-6, # Start very small
    "weight_decay": 0.1,        # Increased from 0.05
    "layer_decay": 0.65,        # Critical for preserving pre-trained features
    
    "min_lr": 1e-6,
    "lr_scheduler_type": "cosine",
    "mixed_precision": True,
    "compile": False,
    "early_stopping": {
        "enabled": False,
        "patience": 10,
        "min_delta": 1e-4,
        "mode": "min"       # or 'max' if monitoring accuracy
    },
}

# =============================================================================
# Augmentation configuration
# =============================================================================
AUGMENTATION_CONFIG = {
    "use_augmentation": True,
    "random_resized_crop": True,
    "crop_scale": (0.25, 1.0),
    "vertical_flip": True,
    "horizontal_flip": True,
    "rotation": 0,
    
    "color_jitter": 0.2,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
# =============================================================================
# Validation configuration
# =============================================================================
VALIDATION_CONFIG = {
    "enable_validation": True,
    "validation_samples": None,
}
