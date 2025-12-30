"""
Main orchestration file for the machine learning pipeline.

This file serves as the entry point and orchestrator for different training modes:
- LINEAR_PROBE: Compare reference (TIMM) vs custom model with pretrained weights
- FROM_SCRATCH: Train custom model from random initialization

The actual training logic is in src/pipelines/
"""

import logging

import torch

from src.data import load_data
from src.data.transforms import get_default_transforms
from src.utils.seeds import set_all_seeds, get_worker_init_fn
from src.utils.experiment import setup_run_directory, setup_logging

from config.imagenet_config import (
    MODEL_TYPE,
    MODEL_CONFIGS,
    TRAINING_CONFIG,
    SEED_CONFIG,
)
from config import (
    DATA_CONFIG,
    SWIN_CONFIG,
    SWIN_PRESETS,
    DOWNSTREAM_CONFIG,
    TrainingMode,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Validation
# =============================================================================


def validate_configuration() -> None:
    """Validate all configuration parameters for consistency and correctness."""
    logger.info("Validating configuration parameters...")

    # Validate SWIN_PRESETS
    for size, preset in SWIN_PRESETS.items():
        required_keys = {"embed_dim", "depths", "num_heads"}
        if not all(key in preset for key in required_keys):
            raise ValueError(
                f"SWIN preset '{size}' missing required keys: {required_keys - preset.keys()}"
            )

        if len(preset["depths"]) != len(preset["num_heads"]):
            raise ValueError(
                f"SWIN preset '{size}' has mismatched depths and num_heads lengths"
            )

        if not all(isinstance(d, int) and d > 0 for d in preset["depths"]):
            raise ValueError(
                f"SWIN preset '{size}' has invalid depths (must be positive integers)"
            )

        if not all(isinstance(h, int) and h > 0 for h in preset["num_heads"]):
            raise ValueError(
                f"SWIN preset '{size}' has invalid num_heads (must be positive integers)"
            )

    # Validate SEED_CONFIG
    if not isinstance(SEED_CONFIG.get("seed"), int) or SEED_CONFIG["seed"] < 0:
        raise ValueError("SEED_CONFIG['seed'] must be a non-negative integer")

    if not isinstance(SEED_CONFIG.get("deterministic"), bool):
        raise ValueError("SEED_CONFIG['deterministic'] must be a boolean")

    logger.info("Configuration validation passed!")


def validate_training_parameters(
    total_epochs: int, warmup_epochs: int, learning_rate: float
) -> None:
    """Validate training parameters."""
    if not isinstance(total_epochs, int) or total_epochs <= 0:
        raise ValueError(f"total_epochs must be a positive integer, got {total_epochs}")

    if not isinstance(warmup_epochs, int) or warmup_epochs < 0:
        raise ValueError(
            f"warmup_epochs must be a non-negative integer, got {warmup_epochs}"
        )

    if warmup_epochs >= total_epochs:
        raise ValueError(
            f"warmup_epochs ({warmup_epochs}) must be less than total_epochs ({total_epochs})"
        )

    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError(
            f"learning_rate must be a positive number, got {learning_rate}"
        )

    logger.debug(
        f"Training parameters validated: epochs={total_epochs}, warmup={warmup_epochs}, lr={learning_rate}"
    )


# =============================================================================
# Setup Functions
# =============================================================================


def setup_device() -> torch.device:
    """Setup and return the appropriate device for training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Clear GPU memory at startup to prevent fragmentation issues
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(
            f"GPU memory cleared at startup. Available: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f}GB"
        )

    logger.info(f"Using device: {device}")
    return device


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main training pipeline."""
    try:
        mode = DOWNSTREAM_CONFIG["mode"]
        logger.info(f"Starting {mode} experiment...")

        # Validate configuration before proceeding
        validate_configuration()

        # Setup run directory and logging first
        run_dir = setup_run_directory()
        setup_logging(run_dir)
        logger.info(f"Experiment directory: {run_dir}")

        # Set random seeds for reproducibility
        logger.info(
            f"Setting random seeds for reproducibility (seed: {SEED_CONFIG['seed']})..."
        )
        set_all_seeds(
            seed=SEED_CONFIG["seed"], deterministic=SEED_CONFIG["deterministic"]
        )

        # Setup device
        device = setup_device()

        # Enable CuDNN benchmarking for faster convolution algorithms
        if bool(DATA_CONFIG.get("img_size", False)) and device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Get training parameters from config
        total_epochs = TRAINING_CONFIG.get("num_epochs", 50)
        warmup_epochs = TRAINING_CONFIG.get("warmup_epochs", 2)
        learning_rate = TRAINING_CONFIG.get("learning_rate", 0.001)

        validate_training_parameters(total_epochs, warmup_epochs, learning_rate)
        logger.info(
            f"Training configuration: epochs={total_epochs}, warmup={warmup_epochs}, lr={learning_rate}"
        )

        # Log model configuration for ablation tracking
        if MODEL_TYPE == "swin":
            logger.info(
                f"SWIN configuration: variant={MODEL_CONFIGS['swin']['variant']}, use_shifted_window={MODEL_CONFIGS['swin']['use_shifted_window']}, use_relative_bias={MODEL_CONFIGS['swin']['use_relative_bias']}, use_absolute_pos_embed={MODEL_CONFIGS['swin']['use_absolute_pos_embed']}, use_hierarchical_merge={MODEL_CONFIGS['swin']['use_hierarchical_merge']}, use_gradient_checkpointing={MODEL_CONFIGS['swin'].get('use_gradient_checkpointing', False)}"
            )
            logger.info(
                f"SWIN details: embed_dim={MODEL_CONFIGS['swin']['embed_dim']}, depths={MODEL_CONFIGS['swin']['depths']}, num_heads={MODEL_CONFIGS['swin']['num_heads']}, window_size={MODEL_CONFIGS['swin']['window_size']}"
            )
        else:
            logger.info(
                f"{MODEL_TYPE.upper()} configuration: {MODEL_CONFIGS[MODEL_TYPE]}"
            )

        # Load dataset
        logger.info("Loading dataset...")
        # Configure subset sizes for faster training/testing
        # Use config values if available, otherwise None for full dataset
        n_train = DATA_CONFIG.get("n_train")
        n_test = DATA_CONFIG.get("n_test")
        # Preserves class distribution when limiting datasets
        stratified = DATA_CONFIG.get("stratified", False)
        # Set transforms
        train_transformation = get_default_transforms(
            DATA_CONFIG["dataset"], DATA_CONFIG["img_size"], is_training=True
        )
        val_transformation = get_default_transforms(
            DATA_CONFIG["dataset"], DATA_CONFIG["img_size"], is_training=False
        )
        train_generator, val_generator, test_generator = load_data(
            dataset=DATA_CONFIG["dataset"],
            transformation=train_transformation,
            val_transformation=val_transformation,
            n_train=n_train,
            n_test=n_test,
            stratified=stratified,
            use_batch_for_val=DATA_CONFIG.get("use_batch_for_val", True),
            val_batch=DATA_CONFIG.get("val_batch", 5),
            batch_size=DATA_CONFIG["batch_size"],
            num_workers=DATA_CONFIG["num_workers"],
            root=DATA_CONFIG["root"],
            img_size=DATA_CONFIG["img_size"],
            worker_init_fn=get_worker_init_fn(SEED_CONFIG["seed"]),
        )
        logger.info(
            f"Dataset loaded: train={len(train_generator.dataset)} samples ({len(train_generator)} batches), "
            f"val={len(val_generator.dataset)} samples ({len(val_generator)} batches), "
            f"test={len(test_generator.dataset)} samples ({len(test_generator)} batches)"
        )

        # Run mode-specific training pipeline
        if mode == TrainingMode.LINEAR_PROBE:
            from src.pipelines import run_linear_probing

            run_linear_probing(
                train_generator,
                val_generator,
                test_generator,
                total_epochs,
                warmup_epochs,
                learning_rate,
                device,
                run_dir,
            )
        elif mode == TrainingMode.FROM_SCRATCH:
            from src.pipelines import run_from_scratch

            run_from_scratch(
                train_generator,
                val_generator,
                test_generator,
                total_epochs,
                warmup_epochs,
                learning_rate,
                device,
                run_dir,
            )
        else:
            raise ValueError(f"Unknown training mode: {mode}")

        logger.info(f"{mode} experiment completed successfully!")

    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        logger.exception("Full traceback:")
        raise RuntimeError(f"Experiment failed: {e}") from e


if __name__ == "__main__":
    main()
