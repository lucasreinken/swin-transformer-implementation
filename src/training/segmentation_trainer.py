"""
Segmentation-specific training utilities.

Separate from classification trainer to avoid affecting existing pipelines.
Follows the same patterns but uses mIoU instead of accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple
import numpy as np

from .metrics import calculate_segmentation_metrics
from .early_stopping import EarlyStopping
from .checkpoints import save_checkpoint

import logging

logger = logging.getLogger(__name__)


def train_one_epoch_segmentation(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> float:
    """
    Train segmentation model for one epoch.
    
    Args:
        model: Segmentation model
        train_loader: Training data loader (yields images, masks)
        criterion: Loss function (CrossEntropyLoss with ignore_index)
        optimizer: Optimizer
        device: Device to train on
        scaler: GradScaler for mixed precision (optional)
        amp_dtype: Mixed precision dtype (optional)
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=bool(amp_dtype),
        ):
            # Forward pass: [B, num_classes, H, W]
            outputs = model(images)
            
            # Loss computation
            # outputs: [B, num_classes, H, W]
            # masks: [B, H, W] with class indices
            loss = criterion(outputs, masks)
        
        # Backward pass with mixed precision handling
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        num_batches += 1
        
        # Log progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            logger.info(
                f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                f"Loss = {loss.item():.4f}"
            )
    
    return running_loss / num_batches


def evaluate_segmentation(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 150,
    ignore_index: int = 255,
    amp_dtype: Optional[torch.dtype] = None,
) -> Tuple[float, float, Dict]:
    """
    Evaluate segmentation model.
    
    Args:
        model: Segmentation model
        dataloader: Validation/test data loader
        criterion: Loss function
        device: Device to evaluate on
        num_classes: Number of semantic classes (150 for ADE20K)
        ignore_index: Label to ignore (255 for ADE20K)
        amp_dtype: Mixed precision dtype (optional)
    
    Returns:
        Tuple of (loss, mean_iou, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Accumulate intersection and union for mIoU
    total_intersection = np.zeros(num_classes)
    total_union = np.zeros(num_classes)
    total_correct = 0
    total_pixels = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(amp_dtype),
            ):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions
            preds = outputs.argmax(dim=1)  # [B, H, W]
            
            # Move to CPU for metric computation
            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            # Accumulate metrics for each sample in batch
            for pred, mask in zip(preds_np, masks_np):
                valid = mask != ignore_index
                
                if valid.sum() == 0:
                    continue
                
                pred_valid = pred[valid]
                mask_valid = mask[valid]
                
                # Pixel accuracy
                total_correct += (pred_valid == mask_valid).sum()
                total_pixels += valid.sum()
                
                # Per-class IoU accumulation
                for cls in range(num_classes):
                    pred_cls = pred_valid == cls
                    mask_cls = mask_valid == cls
                    total_intersection[cls] += np.logical_and(pred_cls, mask_cls).sum()
                    total_union[cls] += np.logical_or(pred_cls, mask_cls).sum()
    
    # Compute final metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # IoU per class
    with np.errstate(divide='ignore', invalid='ignore'):
        iou_per_class = total_intersection / total_union
        iou_per_class = np.nan_to_num(iou_per_class, nan=0.0)
    
    # Class present mask
    class_present = total_union > 0
    
    # Mean IoU (only over present classes)
    if class_present.sum() > 0:
        mean_iou = iou_per_class[class_present].mean() * 100
    else:
        mean_iou = 0.0
    
    # Pixel accuracy
    if total_pixels > 0:
        pixel_accuracy = 100.0 * total_correct / total_pixels
    else:
        pixel_accuracy = 0.0
    
    metrics = {
        "mean_iou": mean_iou,
        "pixel_accuracy": pixel_accuracy,
        "iou_per_class": iou_per_class * 100,
        "class_present": class_present,
        "num_classes_present": int(class_present.sum()),
    }
    
    return avg_loss, mean_iou, metrics


def run_segmentation_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    num_classes: int = 150,
    ignore_index: int = 255,
    amp_dtype: Optional[torch.dtype] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    start_epoch: int = 0,
    run_dir=None,
    checkpoint_frequency: int = 10,
    early_stopping_config: Optional[Dict] = None,
) -> Tuple[Dict[str, List], List[float]]:
    """
    Run the segmentation training loop.
    
    Args:
        model: Segmentation model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Total number of epochs
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on
        num_classes: Number of semantic classes
        ignore_index: Label to ignore in loss/metrics
        amp_dtype: Mixed precision dtype
        scaler: GradScaler for mixed precision
        start_epoch: Starting epoch (for resume)
        run_dir: Directory for checkpoints
        checkpoint_frequency: Save checkpoint every N epochs
        early_stopping_config: Early stopping configuration dict
    
    Returns:
        Tuple of (metrics_history, lr_history)
    """
    # Initialize metrics history
    metrics_history = {
        "train_loss": [],
        "val_loss": [],
        "val_miou": [],
        "val_pixel_acc": [],
    }
    lr_history = []
    
    # Early stopping setup
    early_stopper = None
    if early_stopping_config and early_stopping_config.get("enabled", False):
        early_stopper = EarlyStopping(
            patience=early_stopping_config.get("patience", 10),
            min_delta=early_stopping_config.get("min_delta", 0.1),
            mode=early_stopping_config.get("mode", "max"),  # Maximize mIoU
        )
        logger.info(
            f"Early stopping enabled: patience={early_stopping_config.get('patience', 10)}, "
            f"mode={early_stopping_config.get('mode', 'max')}"
        )
    
    best_miou = 0.0
    
    logger.info("Starting segmentation training...")
    logger.info(f"Training for {num_epochs} epochs, {len(train_loader)} batches/epoch")
    
    for epoch in range(start_epoch, num_epochs):
        # Training
        train_loss = train_one_epoch_segmentation(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp_dtype=amp_dtype,
        )
        
        # Validation
        val_loss, val_miou, val_metrics = evaluate_segmentation(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            ignore_index=ignore_index,
            amp_dtype=amp_dtype,
        )
        
        # Store metrics
        metrics_history["train_loss"].append(train_loss)
        metrics_history["val_loss"].append(val_loss)
        metrics_history["val_miou"].append(val_miou)
        metrics_history["val_pixel_acc"].append(val_metrics["pixel_accuracy"])
        
        # Update learning rate
        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            lr_history.append(current_lr)
        else:
            lr_history.append(optimizer.param_groups[0]["lr"])
        
        # Logging
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val mIoU: {val_miou:.2f}%, "
            f"Val PixelAcc: {val_metrics['pixel_accuracy']:.2f}%"
        )
        
        # Track best model
        if val_miou > best_miou:
            best_miou = val_miou
            logger.info(f"  → New best mIoU: {best_miou:.2f}%")
            
            # Save best model
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
        
        # Early stopping check (maximize mIoU)
        if early_stopper:
            if early_stopper(val_miou):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                break
        
        # Periodic checkpoint
        if run_dir and (epoch + 1) % checkpoint_frequency == 0:
            checkpoint_path = run_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=val_loss,
                filepath=str(checkpoint_path),
            )
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    logger.info(f"Training complete! Best mIoU: {best_miou:.2f}%")
    
    return metrics_history, lr_history
