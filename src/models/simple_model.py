import torch
import torch.nn as nn
from typing import List


class SimpleModel(nn.Module):
    """
    Improved dynamic feedforward neural network for CIFAR-10 classification.

    Args:
        input_dim: Input dimension (3*32*32 for CIFAR-10)
        hidden_dims: List of hidden layer dimensions (e.g., [512, 256, 128])
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization (default: 0.2)
        use_batch_norm: Whether to use batch normalization (default: True)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super(SimpleModel, self).__init__()

        # Build the network layers dynamically
        layers = []
        prev_dim = input_dim

        # Add hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Add batch normalization for better training stability
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Use LeakyReLU to prevent dead neurons
            layers.append(nn.LeakyReLU(0.1))

            # Add dropout for regularization (but not too much early on)
            if i > 0:  # Skip dropout on first layer
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        # Add final output layer (no activation, CrossEntropyLoss handles it)
        layers.append(nn.Linear(prev_dim, num_classes))

        # Create the sequential network
        self.network = nn.Sequential(*layers)

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input (batch_size, channels, height, width) -> (batch_size, input_dim)
        x = x.view(x.size(0), -1)
        return self.network(x)
