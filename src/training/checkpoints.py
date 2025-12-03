"""
Model checkpointing and persistence functions.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Tuple, Optional
import os
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    loss: float,
    filepath: str = "checkpoints/checkpoint_epoch_{epoch}.pth",
    metadata: Optional[dict] = None,
) -> None:
    """Save full training checkpoint."""
    # Format filename with epoch if placeholder used
    if "{epoch}" in filepath:
        filepath = filepath.format(epoch=epoch)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    if metadata:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, filepath)
    logger.info(f"✅ Checkpoint saved: {filepath}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    filepath: str = "checkpoints/checkpoint_epoch_10.pth",
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Optional[Optimizer], int, float, Optional[dict]]:
    """Load full training checkpoint."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", 0.0)
    metadata = checkpoint.get("metadata", None)

    logger.info(f"✅ Checkpoint loaded: {filepath} (epoch {epoch}, loss {loss:.4f})")
    return model, optimizer, epoch, loss, metadata


def save_model_weights(
    model: nn.Module,
    filepath: str = "trained_models/model_weights.pth",
    metadata: Optional[dict] = None,
) -> None:
    """Save model weights for inference."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    state_dict = model.state_dict()
    if metadata:
        checkpoint = {
            "model_state_dict": state_dict,
            "metadata": metadata,
        }
        torch.save(checkpoint, filepath)
    else:
        torch.save(state_dict, filepath)

    logger.info(f"✅ Model weights saved: {filepath}")


def load_model_weights(
    model: nn.Module,
    filepath: str = "trained_models/model_weights.pth",
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load model weights for inference."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Weights file not found: {filepath}")

    state_dict = torch.load(filepath, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(f"✅ Model weights loaded: {filepath}")
    return model
