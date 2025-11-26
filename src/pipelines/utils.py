"""
Shared training utilities used by multiple pipelines.

Contains setup functions, reporting, and validation that are common across
linear probing and from-scratch training.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from config import (
    DOWNSTREAM_CONFIG,
    TRAINING_CONFIG,
    SWIN_PRESETS,
)

from src.training import evaluate_model
from src.training.checkpoints import save_model_weights
from src.training.metrics import (
    plot_confusion_matrix,
    plot_lr_schedule,
    plot_training_curves,
    plot_model_validation_comparison,
)

from src.utils.visualization import CIFAR100_CLASSES

logger = logging.getLogger(__name__)


# =============================================================================
# Validation Functions
# =============================================================================


def validate_pretrained_model_name(model_name: str) -> None:
    """Validate that the pretrained model name is supported."""
    supported_architectures = ["swin", "resnet", "vit", "deit"]

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Model name must be a non-empty string")

    model_lower = model_name.lower()
    if not any(arch in model_lower for arch in supported_architectures):
        logger.warning(
            f"Model '{model_name}' may not be supported. Known architectures: {supported_architectures}"
        )

    # Special validation for Swin models
    if "swin" in model_lower:
        found_size = None
        for size in SWIN_PRESETS.keys():
            if size in model_lower:
                found_size = size
                break

        if found_size is None:
            logger.warning(
                f"Could not detect Swin model size in '{model_name}'. Available: {list(SWIN_PRESETS.keys())}"
            )

    logger.debug(f"Pretrained model name validated: {model_name}")


# =============================================================================
# Training Setup Functions
# =============================================================================


def setup_training_components(
    model: nn.Module, total_epochs: int, warmup_epochs: int, learning_rate: float
):
    """Setup optimizer, scheduler, and loss criterion.

    For linear probing (freeze_encoder=True): only train the classification head.
    For from-scratch (freeze_encoder=False): train the full model.
    """
    criterion = nn.CrossEntropyLoss()

    # Select parameters to train based on mode
    if DOWNSTREAM_CONFIG["freeze_encoder"]:
        # Linear probing: only train the head
        params_to_train = model.pred_head.parameters()
        logger.info("Optimizer: training head parameters only (encoder frozen)")
    else:
        # From-scratch: train full model
        params_to_train = model.parameters()
        logger.info("Optimizer: training all model parameters")

    optimizer = torch.optim.AdamW(
        params_to_train,
        lr=learning_rate,
        weight_decay=TRAINING_CONFIG["weight_decay"],
    )

    warmup_start_factor = TRAINING_CONFIG.get("warmup_start_factor", 0.1)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                total_iters=warmup_epochs,
            ),
            CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
            ),
        ],
        milestones=[warmup_epochs],
    )

    return criterion, optimizer, scheduler


# =============================================================================
# Reporting Functions
# =============================================================================


def generate_reports(
    model,
    variant,
    test_generator,
    criterion,
    lr_history,
    metrics_history,
    device,
    run_dir,
    validation_results=None,
):
    """Generate training reports and visualizations."""
    logger.info("Generating training curves...")
    plot_training_curves(
        metrics_history, save_path=str(run_dir / f"training_curves_{variant}")
    )

    logger.info("Generating confusion matrix on test set...")
    _, _, final_test_metrics = evaluate_model(
        model,
        test_generator,
        criterion,
        device,
        num_classes=DOWNSTREAM_CONFIG["num_classes"],
        detailed_metrics=True,
    )
    plot_confusion_matrix(
        final_test_metrics["confusion_matrix"],
        CIFAR100_CLASSES,
        save_path=str(run_dir / f"confusion_matrix_{variant}.png"),
    )

    # Final evaluation on test set
    logger.info(f"Performing final evaluation of {variant} model on test set...")
    final_test_loss, final_test_accuracy, final_test_metrics = evaluate_model(
        model, test_generator, criterion, device, detailed_metrics=True
    )
    logger.info(
        f"Final Test Results of {variant} model: Loss: {final_test_loss:.4f}, Accuracy: {final_test_accuracy:.2f}%"
    )

    if lr_history:
        logger.info(f"Generating LR schedule plot of {variant} model...")
        plot_lr_schedule(
            lr_history, save_path=str(run_dir / f"lr_schedule_{variant}.png")
        )

    if validation_results:
        logger.info(
            f"Generating model validation comparison plot of {variant} model..."
        )
        plot_model_validation_comparison(
            validation_results,
            save_path=str(run_dir / f"model_validation_comparison_{variant}.png"),
        )

    logger.info(f"\nFinal Test Results:")
    logger.info(f"Loss: {final_test_loss:.4f}")
    logger.info(f"Accuracy: {final_test_accuracy:.2f}%")
    logger.info(f"Precision: {final_test_metrics['precision']:.2f}%")
    logger.info(f"Recall: {final_test_metrics['recall']:.2f}%")
    logger.info(f"F1 Score: {final_test_metrics['f1_score']:.2f}%")

    return final_test_metrics


def save_final_model(model, variant):
    """Save the final trained model weights."""
    save_model_weights(
        model, f"trained_models/CIFAR100_final_model_{variant}_weights.pth"
    )
