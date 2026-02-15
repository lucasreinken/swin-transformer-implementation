"""
Main entry point for masked image modeling (SimMIM) pretraining.

Separate from classification and segmentation mains to avoid interference.
Uses a SimMIM pipeline (e.g., Swin + SimMIM).

Expected outputs:
- SSL-pretrained checkpoint for later fine-tuning on a downstream task.
"""

import logging
import torch

import json
from pathlib import Path

from src.data import load_data
from src.data.transforms import get_default_transforms
from src.utils.seeds import set_all_seeds, get_worker_init_fn
from src.utils.experiment import setup_run_directory, setup_logging

from src.pipelines import run_simmim_pipeline

# Import SimMIM-specific config
from config.simmim_config import (
    DATA_CONFIG,
    SWIN_CONFIG,
    SIMMIM_CONFIG,
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
    try:
        # Dataset selection from config
        dataset_name = DATA_CONFIG["dataset"]
        
        # Set random seeds for reproducibility
        seed = SEED_CONFIG.get("seed", 42)
        
        # Setup device
        device = setup_device()

        # Setup run directory
        run_dir = setup_run_directory()
        setup_logging(run_dir)
        
        # Set seeds
        set_all_seeds(seed=seed, deterministic=SEED_CONFIG.get("deterministic", False))
        
        # Enable CuDNN benchmarking
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        
        # Setup transforms (image-only)
        img_size = DATA_CONFIG.get("img_size", 224)
        train_transform = get_default_transforms(dataset_name, img_size=img_size, is_training=True)
        val_transform = get_default_transforms(dataset_name, img_size=img_size, is_training=False)
        
        # Load data
        logger.info(f"Loading dataset: {dataset_name}...")
        
        train_loader, val_loader, test_loader = load_data(
            dataset=dataset_name,
            transformation=train_transform,
            val_transformation=val_transform,
            n_train=DATA_CONFIG.get("n_train"),
            n_test=DATA_CONFIG.get("n_test"),
            stratified=DATA_CONFIG.get("stratified", False),
            use_batch_for_val=DATA_CONFIG.get("use_batch_for_val", False),
            val_batch=DATA_CONFIG.get("val_batch"),
            batch_size=DATA_CONFIG.get("batch_size", 128),
            num_workers=DATA_CONFIG.get("num_workers", 8),
            root=DATA_CONFIG.get("root", "./datasets"),
            img_size=img_size,
            worker_init_fn=get_worker_init_fn(seed),
        )
        
        logger.info(f"Dataset loaded: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")

        logger.info(f"Experiment directory: {run_dir}")
        
        # Log configuration
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Data config: {DATA_CONFIG}")
        logger.info(f"Swin config: {SWIN_CONFIG}")
        logger.info(f"SimMIM config: {SIMMIM_CONFIG}")
        logger.info(f"Training config: {TRAINING_CONFIG}")
        
        # Run SimMIM pipeline
        final_metrics = run_simmim_pipeline(
            train_loader=train_loader,
            val_loader=val_loader,
            swin_config=SWIN_CONFIG,
            simmim_config=SIMMIM_CONFIG,
            training_config=TRAINING_CONFIG,
            device=device,
            run_dir=run_dir,
            resume_checkpoint=None,
        )

        logger.info("SIMMIM training completed successfully!")
        logger.info(f"Best val loss: {final_metrics.get('best_val_loss', 'N/A')}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.exception("Full traceback:")
        raise RuntimeError(f"SimMIM training search failed: {e}") from e


if __name__ == "__main__":
    main()