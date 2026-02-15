"""
Masked Image Modeling pipeline: Implements the SSL pretraining stage.

- Supports 2 regimes:
  1) Pure SimMIM (random init Swin, pretrain with SimMIM)
  2) ImageNet + SimMIM (load ImageNet weights into Swin encoder, then SimMIM pretrain)
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.model_factory import create_simmim_model
from src.training import run_simmim_training_loop
from src.training.checkpoints import load_checkpoint
from src.utils.experiment import ExperimentTracker

from src.pipelines.utils import (
    setup_simmim_training_components,
    generate_simmim_reports,
    save_final_model,
)

logger = logging.getLogger(__name__)


def setup_mixed_precision(device: torch.device) -> Tuple[Optional[torch.dtype], Optional[torch.amp.GradScaler]]:
    """
    Configure mixed-precision settings for SimMIM training.
    """
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            logger.info("Mixed precision: CUDA bf16 selected (hardware supported)")
            return torch.bfloat16, None
        logger.info("Mixed precision: CUDA float16 selected")
        return torch.float16, torch.amp.GradScaler(device.type)
    if device.type == "cpu":
        logger.info("Mixed precision: CPU bf16 selected")
        return torch.bfloat16, None
    logger.info("Mixed precision: disabled (unsupported device)")
    return None, None


def create_simmim_model_for_training(
    swin_config: Dict,
    simmim_config: Dict,
    device: torch.device,
    load_pretrained: bool = False,
) -> nn.Module:
    """
    Create SimMIM model and move to device.
    """
    logger.info("Creating Swin + SimMIM model...")

    model = create_simmim_model(
        swin_config=swin_config,
        simmim_config=simmim_config,
        load_pretrained=load_pretrained,
    )

    # Log parameter counts
    if hasattr(model, "get_num_params"):
        param_counts = model.get_num_params()
        logger.info("Model parameters:")
        logger.info(f"  Encoder: {param_counts.get('encoder', 0):,}")
        logger.info(f"  Head/Decoder: {param_counts.get('head', param_counts.get('decoder', 0)):,}")
        logger.info(f"  Total: {param_counts.get('total', 0):,}")
        logger.info(f"  Trainable: {param_counts.get('trainable', 0):,}")
    else:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: total={total:,}, trainable={trainable:,}")

    model = model.to(device)
    return model


def run_simmim_pipeline(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    swin_config: Dict,
    simmim_config: Dict,
    training_config: Dict,
    device: torch.device,
    run_dir: Path,
    resume_checkpoint: Optional[str] = None,
) -> Dict:
    """
    Run the full SimMIM pretraining pipeline.

    Args:
        train_loader: returns images (and optionally metadata). Labels are ignored.
        val_loader: optional SSL-val loader to track reconstruction loss.
        swin_config: Swin config dict
        simmim_config: SimMIM config dict
        training_config: training dict
        device: torch device
        run_dir: run directory
        resume_checkpoint: optional resume ckpt path

    Returns:
        Final metrics dictionary
    """
    variant = "swin_simmim"
    logger.info("=" * 60)
    logger.info("SimMIM Pipeline: Swin + SimMIM")
    logger.info("=" * 60)

    # Create model
    model = create_simmim_model_for_training(
        swin_config=swin_config,
        simmim_config=simmim_config,
        device=device,
        load_pretrained=training_config.get("load_imagenet_pretrained", False),
    )

    # GPU model compilation for performance
    if training_config.get("compile", False) and device.type == "cuda":
        logger.info("Compiling model with torch.compile()")
        model = torch.compile(model)

    # Mixed precision
    use_mp = training_config.get("mixed_precision", True)
    if use_mp:
        amp_dtype, scaler = setup_mixed_precision(device)
    else:
        logger.info("Mixed precision: disabled")
        amp_dtype, scaler = None, None

    # Training parameters
    num_epochs = training_config.get("num_epochs", 100)
    warmup_epochs = training_config.get("warmup_epochs", 10)
    learning_rate = training_config.get("learning_rate", 2e-4)
    weight_decay = training_config.get("weight_decay", 0.05)
    checkpoint_frequency = training_config.get("checkpoint_frequency", 10)

    # Resume
    start_epoch = 0
    if resume_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        model, optimizer, start_epoch, scheduler, _ = load_checkpoint(
            model=model,
            optimizer=None,
            checkpoint_path=resume_checkpoint,
            device=device,
        )
        logger.info(f"Resumed from epoch {start_epoch}")
    else:
        optimizer = None
        scheduler = None

    # Setup optimizer/scheduler
    criterion, optimizer, scheduler, mask_generator = setup_simmim_training_components(
        model=model,
        total_epochs=num_epochs,
        warmup_epochs=warmup_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        training_config=training_config,
        simmim_config=simmim_config,
    )

    # Early stopping config (optional)
    early_stopping_config = training_config.get("early_stopping", {})

    # Tracker
    tracker = ExperimentTracker(run_dir)

    logger.info("Training configuration:")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Warmup epochs: {warmup_epochs}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Weight decay: {weight_decay}")
    logger.info(f"  Mixed precision: {amp_dtype}")

    logger.info("SimMIM configuration:")
    logger.info(f"  Type: {simmim_config.get('type')}")
    logger.info(f"  Mask type: {simmim_config.get('mask_type')}")
    logger.info(f"  Mask ratio: {simmim_config.get('mask_ratio')}")

    norm_cfg = simmim_config.get("norm_target", {})
    logger.info(
        f"  Norm target: enable={norm_cfg.get('enable')}, "
        f"patch_size={norm_cfg.get('patch_size')}"
    )

    loss_cfg = simmim_config.get("loss", {})
    logger.info(f"  Loss type: {loss_cfg.get('type')}")

    # Train
    logger.info("Starting SimMIM pretraining...")
    metrics_history, lr_history, best_val_loss = run_simmim_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        amp_dtype=amp_dtype,
        scaler=scaler,
        start_epoch=start_epoch,
        run_dir=run_dir,
        checkpoint_frequency=checkpoint_frequency,
        early_stopping_config=early_stopping_config,
        mask_generator=mask_generator,  # expected: callable that returns [B, H/p, W/p]
        patch_size=swin_config["patch_size"],
    )

    logger.info("SIMMIM pretraining complete!")

    best_path = run_dir / "best_model.pth"

    if val_loader is not None:
        if best_path.exists():
            logger.info(f"Loading best model from {best_path}")
            checkpoint = torch.load(best_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Best model successfully loaded for final evaluation.")
        else:
            logger.warning("Best model checkpoint not found. Using last epoch weights.")

    # Reports (loss curves, final losses, etc.)
    logger.info("Generating reports...")
    final_metrics = generate_simmim_reports(
        model=model,
        variant=variant,
        val_loader=val_loader,
        lr_history=lr_history,
        metrics_history=metrics_history,
        device=device,
        amp_dtype=amp_dtype,
        run_dir=run_dir,
    )

    # Save final model
    save_final_model(
        model=model,
        variant=variant,
        run_dir=run_dir,
        config={
            "swin_config": swin_config,
            "simmim_config": simmim_config,
            "training_config": training_config,
        },
    )

    # Log to tracker
    tracker.log_results(
        variant,
        final_metrics=final_metrics,
        training_history=metrics_history,
    )
    tracker.finalize(variant)

    logger.info("=" * 60)
    logger.info("SimMIM PIPELINE COMPLETE")
    if "train_loss" in final_metrics and metrics_history["train_loss"]:
        logger.info(f"Final train loss: {final_metrics['train_loss']:.6f}")
    if "val_loss" in final_metrics and metrics_history["train_loss"]:
        logger.info(f"Final val loss: {final_metrics['val_loss']:.6f}")
    if best_val_loss is not None and best_val_loss != float("inf"):
        logger.info(f"Best val loss: {best_val_loss:.6f}")
    logger.info("=" * 60)

    return final_metrics
