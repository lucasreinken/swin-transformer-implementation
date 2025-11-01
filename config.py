"""
Configuration file for the machine learning project.
"""

# Data configuration
DATA_CONFIG = {
    "dataset": "CIFAR10",
    "use_batch_for_val": True,
    "val_batch": 5,
    "batch_size": 32,
    "num_workers": 4,
    "root": "./datasets",
    "img_size": 224,
}

# Model configuration
MODEL_CONFIG = {
    "input_dim": 3 * 32 * 32,
    "hidden_dims": [512, 256, 128],
    "num_classes": 10,
    "dropout_rate": 0.3,
    "use_batch_norm": True,
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
