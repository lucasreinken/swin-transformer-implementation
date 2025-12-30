"""
Main entry point for ADE20K semantic segmentation training.

This is separate from the classification main.py to avoid any interference.
Uses the segmentation pipeline with Swin-T + UperNet.
"""

import logging

import torch

from src.data import load_data
from src.data.segmentation_transforms import ADE20KTransform
from src.utils.seeds import set_all_seeds, get_worker_init_fn
from src.utils.experiment import setup_run_directory, setup_logging

from src.pipelines import run_segmentation_pipeline

# Import ADE20K-specific config
from config.ade20k_config import (
    DATA_CONFIG,
    SWIN_CONFIG,
    DOWNSTREAM_CONFIG,
    TRAINING_CONFIG,
    SEED_CONFIG,
)

logger = logging.getLogger(__name__)


def setup_device() -> torch.device:
    """Setup and return the appropriate device for training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"GPU memory: {gpu_mem:.1f}GB")
    else:
        logger.info("Using CPU (no GPU available)")

    return device


def main():
    """Main segmentation training pipeline."""
    try:
        logger.info("=" * 60)
        logger.info("ADE20K Semantic Segmentation Training")
        logger.info("Model: Swin-T + UperNet")
        logger.info("=" * 60)

        # Setup run directory and logging
        run_dir = setup_run_directory()
        setup_logging(run_dir)
        logger.info(f"Experiment directory: {run_dir}")

        # Set random seeds for reproducibility
        seed = SEED_CONFIG.get("seed", 42)
        logger.info(f"Setting random seeds (seed: {seed})...")
        set_all_seeds(seed=seed, deterministic=SEED_CONFIG.get("deterministic", False))

        # Setup device
        device = setup_device()

        # Enable CuDNN benchmarking
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            logger.info("CuDNN benchmark mode enabled")

        # Log configuration
        logger.info(f"Data config: {DATA_CONFIG}")
        logger.info(f"Swin config: {SWIN_CONFIG}")
        logger.info(f"Downstream config: {DOWNSTREAM_CONFIG}")
        logger.info(f"Training config: {TRAINING_CONFIG}")

        # Setup transforms (synchronized for image + mask)
        img_size = DATA_CONFIG.get("img_size", 512)
        train_transform = ADE20KTransform(img_size=img_size, is_training=True)
        val_transform = ADE20KTransform(img_size=img_size, is_training=False)

        # Load ADE20K data
        logger.info("Loading ADE20K dataset...")
        train_loader, val_loader, test_loader = load_data(
            dataset="ADE20K",
            transformation=train_transform,
            val_transformation=val_transform,
            n_train=DATA_CONFIG.get("n_train"),
            n_test=DATA_CONFIG.get("n_test"),
            stratified=False,  # Not used for segmentation
            use_batch_for_val=False,
            val_batch=None,
            batch_size=DATA_CONFIG.get("batch_size", 16),
            num_workers=DATA_CONFIG.get("num_workers", 8),
            root=DATA_CONFIG.get("root", "./datasets"),
            img_size=img_size,
            worker_init_fn=get_worker_init_fn(seed),
        )

        logger.info(
            f"Dataset loaded: "
            f"train={len(train_loader.dataset)} ({len(train_loader)} batches), "
            f"val={len(val_loader.dataset)} ({len(val_loader)} batches)"
        )

        # Run segmentation pipeline
        final_metrics = run_segmentation_pipeline(
            train_loader=train_loader,
            val_loader=val_loader,
            swin_config=SWIN_CONFIG,
            downstream_config=DOWNSTREAM_CONFIG,
            training_config=TRAINING_CONFIG,
            device=device,
            run_dir=run_dir,
            resume_checkpoint=TRAINING_CONFIG.get("resume_from_checkpoint"),
        )

        logger.info("Segmentation training completed successfully!")
        logger.info(f"Final mIoU: {final_metrics['mean_iou']:.2f}%")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.exception("Full traceback:")
        raise RuntimeError(f"Segmentation training failed: {e}") from e


if __name__ == "__main__":
    main()
