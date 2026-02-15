"""
Model checkpointing and persistence functions.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Tuple, Optional
import os
import logging

from config import DOWNSTREAM_CONFIG, TrainingMode

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
    encoder_only: bool = False,
) -> nn.Module:
    """
    Load model weights robustly.
    Handles 'encoder.', 'backbone.', or no-prefix keys for SimMIM compatibility.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Weights file not found: {filepath}")

    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device or 'cpu', weights_only=False)
    
    # Handle wrapping (e.g. {'model': ...} or {'state_dict': ...})
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if encoder_only:
        if not hasattr(model, "encoder"):
            raise AttributeError("Model has no 'encoder' attribute")

        new_state_dict = {}
        
        # 1. Check if keys generally have prefixes
        has_encoder_prefix = any(k.startswith("encoder.") for k in state_dict.keys())
        has_backbone_prefix = any(k.startswith("backbone.") for k in state_dict.keys())
        
        for k, v in state_dict.items():
            # Skip decoder/head keys entirely
            if "decoder" in k or "head" in k or "mask_token" in k:
                continue

            # Standardize keys to match timm (remove prefixes)
            new_k = k
            if has_encoder_prefix and k.startswith("encoder."):
                new_k = k.replace("encoder.", "", 1)
            elif has_backbone_prefix and k.startswith("backbone."):
                new_k = k.replace("backbone.", "", 1)
            
            # Skip relative_position_index (causes shape mismatch errors in Swin)
            if "relative_position_index" in new_k:
                continue
                
            new_state_dict[new_k] = v

        if not new_state_dict:
            # Try loading directly if keys match loosely
            logger.warning("No prefixed keys found. Attempting to load state_dict directly into encoder.")
            new_state_dict = {k: v for k, v in state_dict.items() if "decoder" not in k}

        # Load into the encoder submodule
        missing, unexpected = model.encoder.load_state_dict(
            new_state_dict,
            strict=False,
        )

        logger.info(
            f"Encoder weights loaded from {filepath} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    else:
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Full model weights loaded from {filepath}")
    
    mode = DOWNSTREAM_CONFIG.get("mode", None)
    if not mode or mode is not TrainingMode.LINEAR_PROBE:
        model.eval()
    
    return model
