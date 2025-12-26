"""
Segmentation pipeline: Train Swin-T + UperNet on ADE20K.

Fine-tunes ImageNet pretrained Swin-T encoder with UperNet segmentation head.
Completely separate from classification pipelines.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.model_factory import create_segmentation_model
from src.training import run_segmentation_training_loop
from src.training.checkpoints import load_checkpoint, save_model_weights
from src.utils.experiment import ExperimentTracker

from src.pipelines.utils import (
    setup_segmentation_training_components,
    generate_segmentation_reports,
    save_final_model,
)

logger = logging.getLogger(__name__)


def setup_mixed_precision(device: torch.device) -> Tuple[Optional[torch.dtype], Optional[torch.amp.GradScaler]]:
    """
    Configure mixed-precision settings for segmentation training.
    
    Args:
        device: Target device
    
    Returns:
        Tuple of (amp_dtype, scaler)
    """
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            logger.info("Mixed precision: CUDA bf16 selected (hardware supported)")
            return torch.bfloat16, None
        else:
            logger.info("Mixed precision: CUDA float16 selected")
            return torch.float16, torch.amp.GradScaler(device.type)
    elif device.type == "cpu":
        logger.info("Mixed precision: CPU bf16 selected")
        return torch.bfloat16, None
    else:
        logger.info("Mixed precision: disabled (unsupported device)")
        return None, None


def create_segmentation_model_for_training(
    swin_config: Dict,
    downstream_config: Dict,
    device: torch.device,
    load_pretrained: bool = True,
) -> nn.Module:
    """
    Create segmentation model and move to device.
    
    Args:
        swin_config: Swin Transformer configuration
        downstream_config: Downstream task configuration
        device: Device to place model on
        load_pretrained: Whether to load ImageNet pretrained weights
    
    Returns:
        SegmentationModelWrapper on device
    """
    logger.info("Creating Swin-T + UperNet segmentation model...")
    
    model = create_segmentation_model(
        swin_config=swin_config,
        downstream_config=downstream_config,
        load_pretrained=load_pretrained,
    )
    
    # Log parameter counts
    param_counts = model.get_num_params()
    logger.info(f"Model parameters:")
    logger.info(f"  Encoder: {param_counts['encoder']:,}")
    logger.info(f"  Head: {param_counts['head']:,}")
    logger.info(f"  Total: {param_counts['total']:,}")
    logger.info(f"  Trainable: {param_counts['trainable']:,}")
    
    model = model.to(device)
    
    return model


def run_segmentation_pipeline(
    train_loader: DataLoader,
    val_loader: DataLoader,
    swin_config: Dict,
    downstream_config: Dict,
    training_config: Dict,
    device: torch.device,
    run_dir: Path,
    resume_checkpoint: Optional[str] = None,
) -> Dict:
    """
    Run the full segmentation training pipeline.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        swin_config: Swin Transformer configuration (SWIN_CONFIG)
        downstream_config: Downstream configuration (DOWNSTREAM_CONFIG)
        training_config: Training configuration (TRAINING_CONFIG)
        device: Device to train on
        run_dir: Directory to save results
        resume_checkpoint: Path to checkpoint to resume from (optional)
    
    Returns:
        Final metrics dictionary
    """
    variant = "swin_upernet"
    
    logger.info("=" * 60)
    logger.info("Segmentation Pipeline: Swin-T + UperNet on ADE20K")
    logger.info("=" * 60)
    
    # Create model
    model = create_segmentation_model_for_training(
        swin_config=swin_config,
        downstream_config=downstream_config,
        device=device,
        load_pretrained=downstream_config.get("use_pretrained", True),
    )
    
    # Setup mixed precision
    use_mp = training_config.get("mixed_precision", True)
    if use_mp:
        amp_dtype, scaler = setup_mixed_precision(device)
    else:
        logger.info("Mixed precision: disabled")
        amp_dtype, scaler = None, None
    
    # Training parameters
    num_epochs = training_config.get("num_epochs", 160)
    warmup_epochs = training_config.get("warmup_epochs", 2)
    learning_rate = training_config.get("learning_rate", 6e-5)
    weight_decay = training_config.get("weight_decay", 0.01)
    num_classes = downstream_config.get("num_classes", 150)
    freeze_encoder = downstream_config.get("freeze_encoder", False)
    checkpoint_frequency = training_config.get("checkpoint_frequency", 10)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        model, _, start_epoch, _, _ = load_checkpoint(
            model, None, resume_checkpoint, device
        )
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Setup training components
    criterion, optimizer, scheduler = setup_segmentation_training_components(
        model=model,
        total_epochs=num_epochs,
        warmup_epochs=warmup_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        freeze_encoder=freeze_encoder,
        ignore_index=255,  # ADE20K unlabeled
    )
    
    # Early stopping config
    early_stopping_config = training_config.get("early_stopping", {})
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(run_dir)
    
    logger.info(f"Training configuration:")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Warmup epochs: {warmup_epochs}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Weight decay: {weight_decay}")
    logger.info(f"  Freeze encoder: {freeze_encoder}")
    logger.info(f"  Num classes: {num_classes}")
    logger.info(f"  Mixed precision: {amp_dtype}")
    
    # Run training
    logger.info("Starting training...")
    metrics_history, lr_history = run_segmentation_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=num_classes,
        ignore_index=255,
        amp_dtype=amp_dtype,
        scaler=scaler,
        start_epoch=start_epoch,
        run_dir=run_dir,
        checkpoint_frequency=checkpoint_frequency,
        early_stopping_config=early_stopping_config,
    )
    
    logger.info("Training complete!")
    
    # Generate reports
    logger.info("Generating reports...")
    final_metrics = generate_segmentation_reports(
        model=model,
        variant=variant,
        val_loader=val_loader,
        criterion=criterion,
        lr_history=lr_history,
        metrics_history=metrics_history,
        device=device,
        amp_dtype=amp_dtype,
        run_dir=run_dir,
        num_classes=num_classes,
        ignore_index=255,
    )
    
    # Save final model
    save_final_model(
        model=model,
        variant=variant,
        run_dir=run_dir,
        config={
            "swin_config": swin_config,
            "downstream_config": downstream_config,
            "training_config": training_config,
        },
    )
    
    # Log to tracker
    tracker.log_results(
        variant,
        final_metrics={
            "mean_iou": final_metrics["mean_iou"],
            "pixel_accuracy": final_metrics["pixel_accuracy"],
        },
        training_history=metrics_history,
    )
    tracker.finalize(variant)
    
    logger.info("=" * 60)
    logger.info("SEGMENTATION PIPELINE COMPLETE")
    logger.info(f"Final mIoU: {final_metrics['mean_iou']:.2f}%")
    logger.info(f"Final Pixel Acc: {final_metrics['pixel_accuracy']:.2f}%")
    logger.info("=" * 60)
    
    return final_metrics
