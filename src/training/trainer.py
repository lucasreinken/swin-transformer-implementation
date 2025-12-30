import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Union, Dict
from .metrics import calculate_classification_metrics
from config import AUGMENTATION_CONFIG, TRAINING_CONFIG
import numpy as np

from src.training.early_stopping import EarlyStopping
from src.training.checkpoints import save_checkpoint, save_model_weights

import logging

logger = logging.getLogger(__name__)


class Mixup:
    """Mixup augmentation for label smoothing."""

    def __init__(self, alpha: float = 0.8):
        self.alpha = alpha

    def __call__(self, x, y):
        if self.alpha > 0 and AUGMENTATION_CONFIG["use_augmentation"]:
            lam = np.random.beta(self.alpha, self.alpha)
            index = torch.randperm(x.size(0))
            x_mix = lam * x + (1 - lam) * x[index]
            y_a, y_b = y, y[index]
            return x_mix, y_a, y_b, lam
        return x, y, y, 1.0  # No mixup if disabled


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixup: Optional[Mixup] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: The neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on

    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if mixup:
            inputs, labels_a, labels_b, lam = mixup(inputs, labels)

        optimizer.zero_grad()
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=bool(amp_dtype),
        ):
            outputs = model(inputs)

            if mixup:
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(
                    outputs, labels_b
                )
            else:
                loss = criterion(outputs, labels)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 10,
    detailed_metrics: bool = True,
    amp_dtype: Optional[torch.dtype] = None
) -> Union[Tuple[float, float], Tuple[float, float, Dict]]:
    """
    Evaluate the model on data.

    Args:
        model: The neural network model
        dataloader: Data loader
        criterion: Loss function
        device: Device to run on
        num_classes: Number of classes
        detailed_metrics: Whether to compute detailed metrics

    Returns:
        If detailed_metrics=False: (loss, accuracy)
        If detailed_metrics=True: (loss, accuracy, metrics_dict)
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(amp_dtype),
            ):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(dataloader)
    accuracy = 100.0 * correct / total

    if detailed_metrics:
        metrics = calculate_classification_metrics(
            model, dataloader, device, num_classes=num_classes, amp_dtype=amp_dtype
        )
        return test_loss, accuracy, metrics
    else:
        return test_loss, accuracy


def run_training_loop(
    model,
    train_generator,
    val_generator,
    test_generator,
    num_epochs,
    learning_rate,
    criterion,
    optimizer,
    scheduler,
    metrics_history,
    lr_history,
    mixup,
    device,
    amp_dtype,
    scaler,
    start_epoch=0,
    run_dir=None,
    checkpoint_frequency=10,
):
    """Run the main training loop."""
    # Early stopping setup - configurable for different experiments
    early_stopping_config = TRAINING_CONFIG.get("early_stopping", {})
    if early_stopping_config.get("enabled", False):
        early_stopper = EarlyStopping(
            patience=early_stopping_config.get("patience", 5),
            min_delta=early_stopping_config.get("min_delta", 0.01),
            mode=early_stopping_config.get("mode", "min"),
        )
    else:
        early_stopper = None

    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, num_epochs):
        train_loss = train_one_epoch(
            model,
            train_generator,
            criterion,
            optimizer,
            device,
            mixup=mixup,
            scaler=scaler,
            amp_dtype=amp_dtype
        )

        # Validate every epoch
        val_loss, val_accuracy, val_metrics = evaluate_model(
            model,
            val_generator,
            criterion,
            device,
            amp_dtype=amp_dtype
        )

        # Store metrics
        metrics_history["train_loss"].append(train_loss)
        metrics_history["val_loss"].append(val_loss)
        metrics_history["val_accuracy"].append(val_accuracy)
        if "f1_score" in val_metrics:
            metrics_history["val_f1"].append(val_metrics["f1_score"])
        if "precision" in val_metrics:
            metrics_history["val_precision"].append(val_metrics["precision"])
        if "recall" in val_metrics:
            metrics_history["val_recall"].append(val_metrics["recall"])
        if "f1_per_class" in val_metrics:
            metrics_history["val_f1_per_class"].append(val_metrics["f1_per_class"])

        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            lr_history.append(current_lr)
            logger.info(f"Epoch {epoch+1}/{num_epochs}: LR: {current_lr:.6f}")
        else:
            lr_history.append(learning_rate)

        # Early stopping check
        if early_stopper and early_stopper(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch+1}.")
            break

        # Test periodically (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            test_loss, test_accuracy, test_metrics = evaluate_model(
                model, test_generator, criterion, device, amp_dtype=amp_dtype
            )

            metrics_history["test_loss"].append(test_loss)
            metrics_history["test_accuracy"].append(test_accuracy)

            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%"
            )
        else:
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
            )

        if run_dir and (epoch + 1) % checkpoint_frequency == 0:
            checkpoint_path = run_dir / f"checkpoint_epoch_{epoch+1}.pth"
            logger.info(f"Saving checkpoint for epoch {epoch+1}...")
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                train_loss,
                str(checkpoint_path),
            )
