"""
SimMIM-specific training utilities (SimMIM style).

- Model contract: loss = model(images, mask)
- Mask contract: mask is [B, Gh, Gw] with {0,1}, 1 = masked.

Assumes:
- train_loader yields images, or (images, _) where labels are ignored
- val_loader optional, used for monitoring reconstruction loss
"""

import logging
from typing import Optional, Dict, List, Tuple, Callable, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .early_stopping import EarlyStopping
from .checkpoints import save_checkpoint

logger = logging.getLogger(__name__)


def _unpack_images(batch: Any) -> torch.Tensor:
    """
    Accepts batches shaped like:
    - images
    - (images, labels)
    - dict with key 'image' or 'images'
    """
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, (list, tuple)) and len(batch) >= 1:
        return batch[0]
    if isinstance(batch, dict):
        if "images" in batch:
            return batch["images"]
        if "image" in batch:
            return batch["image"]
    raise ValueError("Unsupported batch format. Expected tensor, tuple/list, or dict with image(s).")


def _infer_patch_grid(images: torch.Tensor, patch_size: int) -> Tuple[int, int]:
    """
    Infer Gh, Gw for mask from images and patch_size.
    images: [B, C, H, W]
    """
    _, _, H, W = images.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"Image size ({H},{W}) must be divisible by patch_size={patch_size}.")
    return H // patch_size, W // patch_size


def train_one_epoch_simmim(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mask_generator: Callable[[int, int, int, torch.device], torch.Tensor],
    patch_size: int,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> float:
    """
    Train SimMIM model for one epoch.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        images = _unpack_images(batch).to(device, non_blocking=True)

        B = images.shape[0]
        Gh, Gw = _infer_patch_grid(images, patch_size=patch_size)
        mask = mask_generator(B, Gh, Gw, device=device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=bool(amp_dtype),
        ):
            # Model returns scalar loss
            loss = model(images, mask)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item())
        num_batches += 1

    return running_loss / max(1, num_batches)


@torch.no_grad()
def evaluate_simmim(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    mask_generator: Callable[[int, int, int, torch.device], torch.Tensor],
    patch_size: int,
    amp_dtype: Optional[torch.dtype] = None,
    max_batches: Optional[int] = None,
) -> float:
    """
    Evaluate SimMIM model (reconstruction loss) on a validation set.

    Returns:
        Average loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = _unpack_images(batch).to(device, non_blocking=True)

        B = images.shape[0]
        Gh, Gw = _infer_patch_grid(images, patch_size=patch_size)
        mask = mask_generator(B, Gh, Gw, device=device)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=bool(amp_dtype),
        ):
            loss = model(images, mask)

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(1, num_batches)


def run_simmim_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    mask_generator: Callable[[int, int, int, torch.device], torch.Tensor],
    patch_size: int,
    amp_dtype: Optional[torch.dtype] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    start_epoch: int = 0,
    run_dir=None,
    checkpoint_frequency: int = 10,
    early_stopping_config: Optional[Dict] = None,
) -> Tuple[Dict[str, List], List[float]]:
    """
    Run the SimMIM training loop.

    Metrics tracked:
    - train_loss
    - val_loss (if val_loader provided)
    """
    metrics_history = {
        "train_loss": [],
        "val_loss": [],
    }
    lr_history: List[float] = []

    early_stopper = None
    if early_stopping_config and early_stopping_config.get("enabled", False) and val_loader is not None:
        early_stopper = EarlyStopping(
            patience=early_stopping_config.get("patience", 10),
            min_delta=early_stopping_config.get("min_delta", 1e-4),
            mode=early_stopping_config.get("mode", "min"),
        )
        logger.info(
            f"Early stopping enabled: patience={early_stopping_config.get('patience', 10)}, "
            f"mode={early_stopping_config.get('mode', 'min')}"
        )

    best_val_loss = float("inf")

    logger.info("Starting SIMMIM training...")
    logger.info(f"Training for {num_epochs} epochs, {len(train_loader)} batches/epoch")

    for epoch in range(start_epoch, num_epochs):
        train_loss = train_one_epoch_simmim(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            mask_generator=mask_generator,
            patch_size=patch_size,
            scaler=scaler,
            amp_dtype=amp_dtype,
        )

        if val_loader is not None:
            val_loss = evaluate_simmim(
                model=model,
                dataloader=val_loader,
                device=device,
                mask_generator=mask_generator,
                patch_size=patch_size,
                amp_dtype=amp_dtype,
            )
        else:
            val_loss = float("nan")

        metrics_history["train_loss"].append(train_loss)
        metrics_history["val_loss"].append(val_loss)

        # LR update
        current_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(current_lr)

        if scheduler:
            scheduler.step()
        else:
            lr_history.append(optimizer.param_groups[0]["lr"])

        # Logging
        if val_loader is not None:
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}"
            )
        else:
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss: {train_loss:.6f}"
            )

        # Best model (by val loss)
        if val_loader is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"  → New best val loss: {best_val_loss:.6f}")

            if run_dir:
                best_path = run_dir / "best_model.pth"
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    loss=val_loss,
                    filepath=str(best_path),
                )
                logger.info(f"  → Best model saved to {best_path}")

        # Early stopping
        if early_stopper:
            if early_stopper(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        # Periodic checkpoint
        if run_dir and (epoch + 1) % checkpoint_frequency == 0:
            checkpoint_path = run_dir / "last_model.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=(val_loss if val_loader is not None else train_loss),
                filepath=str(checkpoint_path),
            )
            logger.info(f"Checkpoint saved (latest): {checkpoint_path}")

    if val_loader is not None:
        logger.info(f"Training complete! Best val loss: {best_val_loss:.6f}")
    else:
        logger.info("Training complete!")

    return metrics_history, lr_history, best_val_loss
