"""
Shared training utilities used by multiple pipelines.

Contains setup functions, reporting, and validation that are common across
linear probing and from-scratch training.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
    MultiStepLR,
    StepLR,
    ExponentialLR,
)

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

from src.utils.visualization import CIFAR100_CLASSES, IMAGENET_CLASSES

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
    min_lr = TRAINING_CONFIG.get("min_lr", 0.0)
    scheduler_type = TRAINING_CONFIG.get("lr_scheduler_type", "cosine")

    # Create warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        total_iters=warmup_epochs,
    )

    # Create main scheduler based on type
    if scheduler_type == "cosine":
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=min_lr,
        )
    elif scheduler_type == "multi_step":
        milestones = TRAINING_CONFIG.get("lr_decay_milestones", [30, 60, 90])
        gamma = TRAINING_CONFIG.get("lr_decay_gamma", 0.1)
        main_scheduler = MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
        )
    elif scheduler_type == "step":
        step_size = TRAINING_CONFIG.get("lr_step_size", 30)
        gamma = TRAINING_CONFIG.get("lr_decay_gamma", 0.1)
        main_scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    elif scheduler_type == "exponential":
        gamma = TRAINING_CONFIG.get("lr_decay_gamma", 0.95)
        main_scheduler = ExponentialLR(
            optimizer,
            gamma=gamma,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )

    logger.info(f"LR Scheduler: {scheduler_type} with {warmup_epochs} warmup epochs")

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
    amp_dtype,
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
        amp_dtype=amp_dtype
    )
    plot_confusion_matrix(
        final_test_metrics["confusion_matrix"],
        (
            IMAGENET_CLASSES
            if DOWNSTREAM_CONFIG["num_classes"] == 1000
            else CIFAR100_CLASSES
        ),
        save_path=str(run_dir / f"confusion_matrix_{variant}.png"),
    )

    # Final evaluation on test set
    logger.info(f"Performing final evaluation of {variant} model on test set...")
    final_test_loss, final_test_accuracy, final_test_metrics = evaluate_model(
        model, test_generator, criterion, device, detailed_metrics=True, amp_dtype=amp_dtype
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


def save_final_model(model, variant, run_dir=None, config=None):
    """Save the final trained model weights."""
    if run_dir:
        weights_path = run_dir / f"final_model_{variant}_weights.pth"
        metadata_path = run_dir / f"final_model_{variant}_metadata.json"
    else:
        weights_path = f"trained_models/final_model_{variant}_weights.pth"
        metadata_path = f"trained_models/final_model_{variant}_metadata.json"

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "variant": variant,
        "config": config or {},
    }

    save_model_weights(model, str(weights_path), metadata=metadata)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Final model saved: {weights_path}")
    logger.info(f"Final model metadata saved: {metadata_path}")


# =============================================================================
# Segmentation-Specific Functions
# =============================================================================


def setup_segmentation_training_components(
    model: nn.Module,
    total_epochs: int,
    warmup_epochs: int,
    learning_rate: float,
    weight_decay: float = 0.01,
    freeze_encoder: bool = False,
    ignore_index: int = 255,
):
    """
    Setup optimizer, scheduler, and loss criterion for segmentation.
    
    Similar to setup_training_components but:
    - Uses CrossEntropyLoss with ignore_index for segmentation
    - Supports encoder freezing for fine-tuning scenarios
    
    Args:
        model: Segmentation model (SegmentationModelWrapper)
        total_epochs: Total number of training epochs
        warmup_epochs: Number of warmup epochs
        learning_rate: Initial learning rate
        weight_decay: Weight decay for AdamW
        freeze_encoder: If True, only train segmentation head
        ignore_index: Label index to ignore in loss (255 for ADE20K)
    
    Returns:
        Tuple of (criterion, optimizer, scheduler)
    """
    # Segmentation loss with ignore_index for unlabeled pixels
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    # Select parameters to train
    if freeze_encoder:
        # Fine-tuning: only train the segmentation head
        params_to_train = model.seg_head.parameters()
        logger.info("Optimizer: training segmentation head only (encoder frozen)")
    else:
        # Full training: train encoder + head
        params_to_train = model.parameters()
        logger.info("Optimizer: training all model parameters (encoder + head)")
    
    optimizer = torch.optim.AdamW(
        params_to_train,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Warmup + cosine annealing (standard for segmentation)
    warmup_start_factor = 0.1
    min_lr = learning_rate * 0.01  # 1% of initial LR
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        total_iters=warmup_epochs,
    )
    
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=min_lr,
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )
    
    logger.info(f"LR Scheduler: cosine with {warmup_epochs} warmup epochs")
    logger.info(f"Learning rate: {learning_rate}, min_lr: {min_lr}")
    
    return criterion, optimizer, scheduler


def generate_segmentation_reports(
    model,
    variant: str,
    val_loader,
    criterion,
    lr_history: List[float],
    metrics_history: Dict[str, List],
    device,
    amp_dtype,
    run_dir,
    num_classes: int = 150,
    ignore_index: int = 255,
):
    """
    Generate segmentation training reports and visualizations.
    
    Args:
        model: Trained segmentation model
        variant: Model variant name (e.g., "swin_upernet")
        val_loader: Validation data loader
        criterion: Loss function
        lr_history: Learning rate history
        metrics_history: Training metrics history
        device: Device
        amp_dtype: Mixed precision dtype
        run_dir: Directory to save reports
        num_classes: Number of segmentation classes
        ignore_index: Label to ignore
    
    Returns:
        Final validation metrics dict
    """
    from src.training import (
        evaluate_segmentation,
        plot_segmentation_training_curves,
        plot_iou_per_class,
        plot_lr_schedule,
    )
    
    logger.info("Generating segmentation training curves...")
    plot_segmentation_training_curves(
        metrics_history,
        save_path=str(run_dir / f"segmentation_curves_{variant}"),
    )
    
    logger.info("Performing final evaluation on validation set...")
    final_loss, final_miou, final_metrics = evaluate_segmentation(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        ignore_index=ignore_index,
        amp_dtype=amp_dtype,
    )
    
    logger.info("Generating per-class IoU plot...")
    plot_iou_per_class(
        iou_per_class=final_metrics["iou_per_class"],
        save_path=str(run_dir / f"iou_per_class_{variant}.png"),
    )
    
    if lr_history:
        logger.info("Generating LR schedule plot...")
        plot_lr_schedule(
            lr_history,
            save_path=str(run_dir / f"lr_schedule_{variant}.png"),
        )
    
    # Log final results
    logger.info(f"\n{'='*50}")
    logger.info(f"Final Segmentation Results ({variant})")
    logger.info(f"{'='*50}")
    logger.info(f"Loss: {final_loss:.4f}")
    logger.info(f"mIoU: {final_miou:.2f}%")
    logger.info(f"Pixel Accuracy: {final_metrics['pixel_accuracy']:.2f}%")
    logger.info(f"Classes Present: {final_metrics['num_classes_present']}/{num_classes}")
    
    # Save metrics to JSON
    metrics_json = {
        "loss": float(final_loss),
        "mean_iou": float(final_miou),
        "pixel_accuracy": float(final_metrics["pixel_accuracy"]),
        "num_classes_present": int(final_metrics["num_classes_present"]),
        "iou_per_class": final_metrics["iou_per_class"].tolist(),
    }
    
    metrics_path = run_dir / f"final_metrics_{variant}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    logger.info(f"Final metrics saved to {metrics_path}")
    
    return final_metrics
