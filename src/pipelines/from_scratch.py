"""
From-scratch training pipeline: Train custom Swin model without pretrained weights.

No reference model, no comparison - just train our model from random initialization.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.imagenet_config import MODEL_TYPE, MODEL_CONFIGS
from config import (
    DATA_CONFIG,
    DOWNSTREAM_CONFIG,
    TRAINING_CONFIG,
)

from src.models import (
    SwinTransformerModel,
    ModelWrapper,
    LinearClassificationHead,
)

from src.training import run_training_loop
from src.training.checkpoints import load_checkpoint
from src.utils.experiment import ExperimentTracker
from src.utils.load_weights import transfer_weights

# Import shared utilities
from src.pipelines.utils import (
    setup_training_components,
    generate_reports,
    save_final_model,
)

logger = logging.getLogger(__name__)


def create_model_from_scratch(device: torch.device) -> nn.Module:
    """
    Create a model with random initialization based on MODEL_TYPE.

    Args:
        device: Device to place the model on

    Returns:
        ModelWrapper containing the model ready for training
    """
    from src.models.model_factory import create_model

    logger.info(f"Initializing {MODEL_TYPE.upper()} model from scratch...")

    # Get model config
    model_config = MODEL_CONFIGS[MODEL_TYPE].copy()
    model_config["type"] = MODEL_TYPE

    # Log model configuration
    logger.info(f"Model architecture: {MODEL_TYPE.upper()}")
    logger.info(f"Model config: {model_config}")

    # Create model using factory
    model = create_model(model_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with random initialization")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Improve CUDA tensor memory access with mixed precision
    if TRAINING_CONFIG.get("mixed_precision", False) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    model = model.to(device)

    # GPU model compilation for performance
    if TRAINING_CONFIG.get("compile", False) and device.type == "cuda":
        logger.info("Compiling model with torch.compile()")
        model = torch.compile(model)

    return model

def setup_mixed_precision(device) -> tuple[torch.amp.GradScaler | None, torch.dtype | None]:
    """
    Configure mixed-precision settings for the current training device.

    Args:
        device: Target device on which the model will run.

    Returns:
        amp_dtype: selected mixed-precision dtype (bf16/float16) or None for fp32.
        scaler: GradScaler when float16 is used, otherwise None.
    """
    use_mp = TRAINING_CONFIG.get("mixed_precision", False)

    if not use_mp:
        logger.info("Mixed precision disabled → training in float32 precision")
        return None, None

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            logger.info("Mixed precision enabled → CUDA bf16 selected (hardware supported)")
            return torch.bfloat16, None

        logger.info("Mixed precision enabled → CUDA float16 selected (bf16 unsupported)")
        return torch.float16, torch.amp.GradScaler(device.type)

    if device.type == "cpu":
        logger.info("Mixed precision enabled → CPU bf16 selected")
        return torch.bfloat16, None

    logger.info("Mixed precision requested, but device unsupported → falling back to float32")
    return None, None


def _train_single_model(
    model: nn.Module,
    train_generator: DataLoader,
    val_generator: DataLoader,
    test_generator: DataLoader,
    total_epochs: int,
    warmup_epochs: int,
    learning_rate: float,
    device: torch.device,
    amp_dtype: torch.dtype,
    scaler: torch.amp.GradScaler,
    start_epoch: int = 0,
    run_dir: Path = None,
    checkpoint_frequency: int = 10,
) -> Tuple[nn.Module, List[float], Dict[str, List[float]]]:
    """Run training loop for one model and return training artifacts."""
    criterion, optimizer, scheduler = setup_training_components(
        model, total_epochs, warmup_epochs, learning_rate
    )

    metrics_history = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "val_accuracy": [],
        "test_accuracy": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1_per_class": [],
    }

    lr_history = []
    mixup = None

    run_training_loop(
        model,
        train_generator,
        val_generator,
        test_generator,
        total_epochs,
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
        start_epoch,
        run_dir,
        checkpoint_frequency,
    )

    return criterion, lr_history, metrics_history


def _finalize_training(
    model,
    variant,
    test_generator,
    criterion,
    lr_history,
    metrics_history,
    device,
    amp_dtype,
    run_dir,
    tracker,
):
    """Generate reports, log results, and save model."""
    final_test_metrics = generate_reports(
        model,
        variant,
        test_generator,
        criterion,
        lr_history,
        metrics_history,
        device,
        amp_dtype,
        run_dir,
        None,
    )

    tracker.log_results(
        variant,
        final_metrics=final_test_metrics,
        training_history=metrics_history,
    )

    tracker.finalize(variant)
    save_final_model(model, variant, run_dir, config=DOWNSTREAM_CONFIG)

    return final_test_metrics


def run_from_scratch(
    train_generator: DataLoader,
    val_generator: DataLoader,
    test_generator: DataLoader,
    total_epochs: int,
    warmup_epochs: int,
    learning_rate: float,
    device: torch.device,
    run_dir: Path,
) -> None:
    """
    Run from-scratch training: train custom Swin model without pretrained weights.

    No reference model, no comparison - just train our model from random initialization.

    Args:
        train_generator: Training data loader
        val_generator: Validation data loader
        test_generator: Test data loader
        total_epochs: Number of training epochs
        warmup_epochs: Number of warmup epochs
        learning_rate: Learning rate
        device: Device to train on
        run_dir: Directory to save results
    """
    logger.info(f"Training {MODEL_TYPE.upper()} from scratch")

    # Create model with random initialization
    model = create_model_from_scratch(device)

    start_epoch = 0

    amp_dtype, scaler = setup_mixed_precision(device)

    logger.info("Starting from-scratch training...")
    criterion, lr_history, metrics_history = _train_single_model(
        model,
        train_generator,
        val_generator,
        test_generator,
        total_epochs,
        warmup_epochs,
        learning_rate,
        device,
        amp_dtype,
        scaler,
        start_epoch,
        run_dir,
        TRAINING_CONFIG.get("checkpoint_frequency", 10),
    )
    save_final_model(model, "from_scratch", run_dir, config=DOWNSTREAM_CONFIG)
    logger.info("Training completed!")
