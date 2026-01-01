"""
Configuration file for ADE20K semantic segmentation dataset.
"""

from .base_config import (
    SWIN_PRESETS,
    VIZ_CONFIG,
    SEED_CONFIG,
    apply_swin_preset,
    TrainingMode,
    get_training_mode_settings,
)

# Data configuration
DATA_CONFIG = {
    "dataset": "ADE20K",
    "use_batch_for_val": False,
    "val_batch": 5,
    "batch_size": 4,  # Reduced for 12GB GPU (RTX 4070); use 16 for 24GB+ GPUs
    "num_workers": 4,  # Reduced to avoid DataLoader warnings
    "root": "./datasets",  # Will check shared storage first, then download if needed
    "img_size": 512,  # ADE20K resolution (512 is standard despite window_size mismatch)
    # Subset configuration (optional, set to None to use full dataset)
    "n_train": None,  # 20,210 training images total
    "n_test": None,   # 2,000 validation images total
    "stratified": False,  # Not applicable for segmentation
}

# Swin Transformer configuration
SWIN_CONFIG = {
    "img_size": 512,
    "variant": "tiny",  # Choose: "tiny", "small", "base", "large"
    "patch_size": 4,
    "embed_dim": None,  # Auto-set from preset
    "depths": None,  # Auto-set from preset
    "num_heads": None,  # Auto-set from preset
    "window_size": 7,  # Keep at 7 to match ImageNet pretrained weights
    "mlp_ratio": 4.0,
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "projection_dropout": 0.0,
    "drop_path_rate": 0.2,  # Higher for segmentation following paper
    "use_gradient_checkpointing": True,  # Enable for memory efficiency on 12GB GPUs
}

# Apply preset values for None fields
apply_swin_preset(SWIN_CONFIG, SWIN_PRESETS)

# =============================================================================
# ResNet Configuration (for ResNet-101/50 + UperNet experiments)
# =============================================================================
RESNET_CONFIG = {
    "type": "resnet",
    "variant": "resnet101",  # Choose: "resnet50" or "resnet101"
    "pretrained": True,  # Load ImageNet pretrained weights from torchvision
    "img_size": 512,  # ADE20K resolution
    "layers": [3, 4, 23, 3],  # ResNet-101 layers (use [3, 4, 6, 3] for ResNet-50)
    "use_gradient_checkpointing": False,  # Enable for memory efficiency if needed
}

# =============================================================================
# Downstream Task Configuration
# =============================================================================
# Training mode: "linear_probe" or "from_scratch"
_TRAINING_MODE = TrainingMode.FROM_SCRATCH
_mode_settings = get_training_mode_settings(_TRAINING_MODE)

DOWNSTREAM_CONFIG = {
    "mode": _TRAINING_MODE,
    "head_type": "upernet",  # Segmentation head (UperNet)
    "num_classes": 150,  # ADE20K has 150 semantic categories
    "hidden_dim": None,
    # Fine-tuning setup: Load ImageNet pretrained weights, train both encoder + head
    "freeze_encoder": False,  # Fine-tune encoder (not frozen)
    "use_pretrained": True,   # Load ImageNet pretrained weights from TIMM
}

# Training configuration (following paper settings)
TRAINING_CONFIG = {
    "learning_rate": 6e-5,  # AdamW learning rate from paper
    "num_epochs": 160,  # 160K iterations / ~1K iterations per epoch
    "warmup_epochs": 2,  # 1500 iterations warmup (~1.5 epochs)
    "warmup_start_factor": 0.1,
    "weight_decay": 0.01,  # Following paper
    "mixed_precision": True,  # Enable for memory efficiency
    "compile": False,  # Disable torch.compile for compatibility
}

# Augmentation configuration (following paper)
AUGMENTATION_CONFIG = {
    "use_augmentation": True,
    "rand_augment_m": 9,
    "rand_augment_n": 2,
    "mixup_alpha": 0.0,  # Not used for segmentation
    "random_erase_prob": 0.0,  # Not used for segmentation
    # ADE20K-specific augmentations
    "random_flip": True,
    "random_scale": [0.5, 2.0],  # Random re-scaling range
    "random_crop": True,
    "photometric_distortion": True,
    "mean": [0.485, 0.456, 0.406],  # ImageNet normalization
    "std": [0.229, 0.224, 0.225],
}

# Model Validation Configuration
VALIDATION_CONFIG = {
    "enable_validation": False,
    "pretrained_model": "swin_tiny_patch4_window7_224",  # ImageNet pretrained (224×224, window=7)
    "transfer_weights": True,  # Load ImageNet weights, window size will auto-adjust for 512×512
    "validation_samples": 100,
}
