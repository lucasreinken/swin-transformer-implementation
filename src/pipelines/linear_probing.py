"""
Linear probing pipeline: Compare reference (TIMM) vs custom Swin model.

Both models use pretrained weights with frozen encoder, training only the head.
This validates that our custom Swin implementation matches the official one.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from timm import create_model

from config import (
    DATA_CONFIG,
    DOWNSTREAM_CONFIG,
    SWIN_PRESETS,
    TRAINING_CONFIG,
    get_pretrained_swin_name,
    SWIN_CONFIG,
)

from src.models import (
    SwinTransformerModel,
    ModelWrapper,
    LinearClassificationHead,
)

from src.training import run_training_loop
from src.utils.experiment import ExperimentTracker
from src.utils.load_weights import transfer_weights

# Import shared utilities
from src.pipelines.utils import (
    validate_pretrained_model_name,
    setup_training_components,
    generate_reports,
    save_final_model,
)

logger = logging.getLogger(__name__)


def create_reference_model(pretrained_model: str, device: torch.device) -> nn.Module:
    """
    Create the TIMM reference Swin model wrapped for linear probing.

    Args:
        pretrained_model: Name of the pretrained model to load
        device: Device to place the model on

    Returns:
        ModelWrapper containing the reference model

    Raises:
        RuntimeError: If model creation or setup fails
    """
    try:
        logger.info("Initializing reference model...")
        logger.info(f"Loading TIMM model: {pretrained_model} with pretrained weights")

        encoder = create_model(pretrained_model, pretrained=True)

        if encoder is None:
            raise RuntimeError(f"Failed to load model: {pretrained_model}")

        logger.info(f"Successfully loaded {pretrained_model} from TIMM")

        if pretrained_model.lower().startswith("swin"):
            encoder.prune_intermediate_layers(
                indices=None,  # keep all transformer blocks
                prune_norm=True,  # remove final LN
                prune_head=True,  # remove classifier
            )
        else:
            encoder.prune_intermediate_layers(
                indices=None,  # keep all transformer blocks
                prune_head=True,  # remove classifier
            )

        # Verify encoder has expected attributes
        if not hasattr(encoder, "num_features"):
            raise RuntimeError("Encoder missing 'num_features' attribute")

        logger.info(f"Encoder loaded: {encoder.num_features} output features")

        pred_head = LinearClassificationHead(
            num_features=encoder.num_features,
            num_classes=DOWNSTREAM_CONFIG["num_classes"],
        )

        logger.info(
            f"Created classification head: {encoder.num_features} -> {DOWNSTREAM_CONFIG['num_classes']} classes"
        )

        model = ModelWrapper(
            encoder=encoder,
            pred_head=pred_head,
            freeze=DOWNSTREAM_CONFIG["freeze_encoder"],
        )

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mode_desc = "head only" if DOWNSTREAM_CONFIG["freeze_encoder"] else "full model"
        logger.info(
            f"Reference model created. Trainable params ({mode_desc}): {trainable_params:,}"
        )

        return model.to(device)

    except Exception as e:
        logger.error(f"Failed to create reference model {pretrained_model}: {e}")
        raise RuntimeError(f"Model creation failed: {e}") from e


def create_custom_model(
    reference_model: nn.Module, model_size: str, device: torch.device
) -> nn.Module:
    """Initialize and setup the custom Swin model, then transfer weights."""
    try:
        # Input validation
        if reference_model is None:
            raise ValueError("Reference model cannot be None")
        if model_size not in SWIN_PRESETS:
            raise ValueError(
                f"Invalid model size '{model_size}'. Available: {list(SWIN_PRESETS.keys())}"
            )
        if device is None:
            raise ValueError("Device cannot be None")

        # Validate reference model has required encoder attribute
        if not hasattr(reference_model, "encoder"):
            raise AttributeError(
                f"Reference model missing 'encoder' attribute. Available: {dir(reference_model)}"
            )

        logger.info(f"Initializing custom Swin model (size: {model_size})...")

        # Get preset configuration
        preset = SWIN_PRESETS[model_size]

        logger.info(
            f"Custom model will match architecture: embed_dim={preset['embed_dim']}, "
            f"depths={preset['depths']}, num_heads={preset['num_heads']}"
        )

        cfg = {
            "img_size": 224,
            "patch_size": 4,
            "window_size": 7,
            "mlp_ratio": 4.0,
            "drop_path_rate": 0.1,
        }

        cfg["embed_dim"] = preset["embed_dim"]
        cfg["depths"] = preset["depths"]
        cfg["num_heads"] = preset["num_heads"]

        logger.debug(f"Creating SwinTransformerModel with config: {cfg}")
        encoder = SwinTransformerModel(
            img_size=cfg["img_size"],
            patch_size=cfg["patch_size"],
            embedding_dim=cfg["embed_dim"],
            depths=cfg["depths"],
            num_heads=cfg["num_heads"],
            window_size=cfg["window_size"],
            mlp_ratio=cfg["mlp_ratio"],
            drop_path_rate=cfg["drop_path_rate"],
        )

        # Validate encoder has num_features attribute
        if not hasattr(encoder, "num_features"):
            raise AttributeError(
                f"Custom encoder missing 'num_features' attribute. Available: {dir(encoder)}"
            )

        logger.debug(
            f"Creating classification head for {encoder.num_features} features -> "
            f"{DOWNSTREAM_CONFIG['num_classes']} classes"
        )
        pred_head = LinearClassificationHead(
            num_features=encoder.num_features,
            num_classes=DOWNSTREAM_CONFIG["num_classes"],
        )

        logger.debug("Wrapping encoder and head in ModelWrapper")
        model = ModelWrapper(
            encoder=encoder,
            pred_head=pred_head,
            freeze=DOWNSTREAM_CONFIG["freeze_encoder"],
        )

        logger.info("Transferring weights from reference to custom encoder...")
        logger.info(
            f"Weight transfer: {reference_model.encoder.__class__.__name__} -> "
            f"{encoder.__class__.__name__}"
        )
        transfer_stats = transfer_weights(
            model,
            encoder_only=True,
            pretrained_model=reference_model.encoder,
            device=device,
        )
        logger.info(f"Weight transfer completed: {transfer_stats}")

        # Verify successful weight transfer
        if transfer_stats["transferred"] == 0:
            logger.warning("No weights were transferred! Check model compatibility.")
        elif transfer_stats["missing"] > 0:
            logger.warning(f"{transfer_stats['missing']} layers could not be matched")
        else:
            logger.info("✓ All expected weights transferred successfully")

        # Verify model state
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable_params == 0:
            logger.warning("No trainable parameters found in model")

        mode_desc = "head only" if DOWNSTREAM_CONFIG["freeze_encoder"] else "full model"
        logger.info(
            f"Custom model created. Trainable params ({mode_desc}): {trainable_params:,}"
        )
        return model.to(device)

    except Exception as e:
        logger.error(f"Failed to create custom model (size: {model_size}): {e}")
        raise RuntimeError(f"Custom model creation failed: {e}") from e


def _train_single_model(
    model: nn.Module,
    train_generator: DataLoader,
    val_generator: DataLoader,
    test_generator: DataLoader,
    total_epochs: int,
    warmup_epochs: int,
    learning_rate: float,
    device: torch.device,
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
    )

    return criterion, lr_history, metrics_history


def _finalize_validation(
    model,
    variant,
    test_generator,
    criterion,
    lr_history,
    metrics_history,
    device,
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
        run_dir,
        None,
    )

    tracker.log_results(
        variant,
        final_metrics=final_test_metrics,
        training_history=metrics_history,
    )

    tracker.finalize(variant)
    save_final_model(model, variant)

    return final_test_metrics


def run_linear_probing(
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
    Run linear probing experiment: compare reference (TIMM) vs custom model.

    Both models use pretrained weights with frozen encoder, training only the head.
    This validates that our custom Swin implementation matches the official one.

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
    pretrained_model = get_pretrained_swin_name()
    validate_pretrained_model_name(pretrained_model)

    logger.info(f"Using pretrained model: {pretrained_model}")
    logger.info(f"Model architecture: Swin-{SWIN_CONFIG['variant'].title()}")

    # Create reference model (TIMM pretrained)
    reference_model = create_reference_model(pretrained_model, device)

    # Create custom model and transfer weights from reference
    custom_model = None
    if pretrained_model.lower().startswith("swin"):
        logger.info("Detected Swin Transformer architecture")
        model_size = None
        for p in pretrained_model.lower().split("_"):
            if p in SWIN_PRESETS:
                model_size = p
                break

        if model_size is None:
            raise ValueError(
                f"Could not detect model size from '{pretrained_model}'. "
                f"Available sizes: {list(SWIN_PRESETS.keys())}"
            )

        logger.info(f"Detected model size: {model_size}")
        custom_model = create_custom_model(reference_model, model_size, device)

    # Train reference model
    reference_tracker = ExperimentTracker(run_dir)
    logger.info("Starting reference model training...")
    reference_criterion, reference_lr_history, reference_metrics_history = (
        _train_single_model(
            reference_model,
            train_generator,
            val_generator,
            test_generator,
            total_epochs,
            warmup_epochs,
            learning_rate,
            device,
        )
    )
    logger.info("Reference model training completed!")

    final_reference_metrics = _finalize_validation(
        reference_model,
        "reference",
        test_generator,
        reference_criterion,
        reference_lr_history,
        reference_metrics_history,
        device,
        run_dir,
        reference_tracker,
    )

    # Train custom model and compare
    if custom_model is not None:
        custom_tracker = ExperimentTracker(run_dir)
        logger.info("Starting custom model training...")
        custom_criterion, custom_lr_history, custom_metrics_history = (
            _train_single_model(
                custom_model,
                train_generator,
                val_generator,
                test_generator,
                total_epochs,
                warmup_epochs,
                learning_rate,
                device,
            )
        )
        logger.info("Custom model training completed!")

        final_custom_metrics = _finalize_validation(
            custom_model,
            "custom",
            test_generator,
            custom_criterion,
            custom_lr_history,
            custom_metrics_history,
            device,
            run_dir,
            custom_tracker,
        )

        # Compare results
        diff = abs(
            final_reference_metrics["accuracy"] - final_custom_metrics["accuracy"]
        )
        dataset = DATA_CONFIG.get("dataset", "dataset")
        logger.info(
            f"=== MODEL COMPARISON RESULTS (linear_probe on {dataset.upper()}) ==="
        )
        logger.info(f"Reference (TIMM): {final_reference_metrics['accuracy']:.2f}%")
        logger.info(f"Custom          : {final_custom_metrics['accuracy']:.2f}%")
        logger.info(f"Difference      : {diff:.2f}%")
    else:
        logger.info("Only reference model trained (non-Swin architecture)")
