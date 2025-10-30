"""
Configuration file for the machine learning project.
"""

# Data configuration
DATA_CONFIG = {
    "dataset": "CIFAR10",
    "n_train": 40000,
    "n_test": 10000,
    "batch_size": 32,
    "num_workers": 4,  # number of data loading workers
    "root": "./datasets",
    "img_size": 224,  # target image size for imageNet
}

# Model configuration
MODEL_CONFIG = {
    "input_dim": 3 * 32 * 32,
    "hidden_dims": [512, 256, 128],  # Reduced depth to prevent vanishing gradients
    "num_classes": 10,
    "dropout_rate": 0.3,  # Slightly higher dropout
    "use_batch_norm": True,  # Enable batch normalization
}

# Training configuration
TRAINING_CONFIG = {
    "learning_rate": 0.001,  # Much lower learning rate
    "num_epochs": 20,  # More epochs
    "weight_decay": 1e-4,  # L2 regularization
}

# Visualization configuration
VIZ_CONFIG = {
    "figsize": (10, 10),
    "output_file": "visualization.png",
}
