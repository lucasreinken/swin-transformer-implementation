"""
Model checkpointing and persistence functions.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Tuple, Optional
import os


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    loss: float,
    filepath: str = "checkpoints/checkpoint.pth",
) -> None:
    """ "Save full training checkpoint."""
    os.makedirs(os.path.dirname(f"checkpoints/{filepath}"), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    filepath: str = "checkpoints/model_checkpoint.pth",
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Optional[Optimizer], int, float]:
    """Load full training checkpoint."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", 0.0)

    print(f"Checkpoint loaded: {filepath} (epoch {epoch}, loss {loss:.4f})")


def save_model_weights(model: nn.Module, filepath: str = "model_weights.pth") -> None:
    """Save model weights for inference."""
    os.makedirs(os.path.dirname(f"checkpoints/{filepath}"), exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{filepath}")
    print(f"Model weights saved: {filepath}")


def load_model_weights(
    model: nn.Module,
    filepath: str = "model_weights.pth",
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load model weights for inference."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Weights file not found: {filepath}")

    state_dict = torch.load(filepath, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model weights loaded: {filepath}")
    return model
