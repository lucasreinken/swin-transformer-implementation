import torch
import torch.nn as nn


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample for residual blocks.

    Randomly drops entire layers during training to:
    - Reduce overfitting
    - Enable training of very deep networks
    - Act as implicit ensemble of shallower networks

    Reference: "Deep Networks with Stochastic Depth" (Huang et al., 2016)
    """

    def __init__(self, drop_prob: float = 0.0):
        """
        Initialize DropPath.

        Args:
            drop_prob: Probability of dropping the path
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply DropPath.

        Args:
            x: Input tensor

        Returns:
            Output tensor (dropped or scaled appropriately)
        """
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob

        # Work with different dimensions (2D, 3D, 4D tensors)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize

        # Scale by keep_prob to maintain expected value
        output = x.div(keep_prob) * random_tensor

        return output
