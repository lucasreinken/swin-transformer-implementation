import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for Swin Transformer blocks.

    This is the feedforward network applied after attention in each Swin block.
    Uses GELU activation and dropout for regularization.

    Architecture:
    Input → Linear(in → hidden) → GELU → Dropout → Linear(hidden → out) → Dropout

    Typical expansion: hidden = 4 × in_features (following ViT/Swin convention)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        dropout: float = 0.0,
    ):
        """
        Initialize MLP.

        Args:
            in_features: Input feature dimension
            hidden_features: Hidden layer dimension (default: 4 × in_features)
            out_features: Output feature dimension (default: in_features)
            dropout: Dropout rate
        """
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, N, C]

        Returns:
            Output tensor [B, N, C]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
