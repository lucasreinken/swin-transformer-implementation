"""
Main orchestration file for the machine learning pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from config import SWIN_CONFIG
from src.data import load_data

from src.models import (
    SwinTransformerModel,
    ModelWrapper,
    LinearClassificationHead,
)

from src.training import evaluate_model, run_training_loop
from src.training.checkpoints import save_model_weights
from src.training.metrics import (
    plot_confusion_matrix,
    plot_lr_schedule,
    plot_training_curves,
    plot_model_validation_comparison,
)
from src.training.trainer import Mixup

from src.utils.visualization import show_batch
from src.utils.seeds import set_random_seeds, get_worker_init_fn
from src.utils.experiment import setup_run_directory, setup_logging, ExperimentTracker
from src.utils.model_validation import ModelValidator
from src.utils.load_weights import transfer_weights

from config import (
    AUGMENTATION_CONFIG,
    DATA_CONFIG,
    SWIN_PRESETS,
    DOWNSTREAM_CONFIG,
    TRAINING_CONFIG,
    VIZ_CONFIG,
    SEED_CONFIG,
    VALIDATION_CONFIG,
)

from src.utils.visualization import CIFAR100_CLASSES

# Setup logging
import logging

logger = logging.getLogger(__name__)


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


try:
    from timm import create_model

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm library not found. Cannot transfer weights.")


def get_swin_name():
    variant = SWIN_CONFIG["variant"]  # "tiny", "small", "base", "large"
    patch_size = SWIN_CONFIG.get("patch_size", 4)  # default 4
    window = SWIN_CONFIG.get("window_size", 7)  # default 7
    img = SWIN_CONFIG.get("img_size", 224)  # default 224

    return f"swin_{variant}_patch{patch_size}_window{window}_{img}"


def setup_device() -> torch.device:
    """Setup and return the appropriate device for training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def setup_data(
    device: torch.device, run_dir: Path
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load and setup data loaders and visualization."""
    logger.info("Loading data...")

    # For validation, use ImageNet size; otherwise use config size
    img_size = (
        224
        if VALIDATION_CONFIG.get("enable_validation", False)
        else DATA_CONFIG["img_size"]
    )

    train_generator, val_generator, test_generator = load_data(
        dataset=DATA_CONFIG["dataset"],
        n_train=DATA_CONFIG.get("n_train"),
        n_test=DATA_CONFIG.get("n_test"),
        use_batch_for_val=DATA_CONFIG.get("use_batch_for_val", False),
        val_batch=DATA_CONFIG.get("val_batch", 5),
        batch_size=DATA_CONFIG["batch_size"],
        num_workers=DATA_CONFIG["num_workers"],
        root=DATA_CONFIG["root"],
        img_size=img_size,
        worker_init_fn=get_worker_init_fn(SEED_CONFIG["seed"]),
    )

    # Visualize first batch
    logger.info("Visualizing first batch...")
    show_batch(
        dataloader=train_generator,
        dataset=DATA_CONFIG["dataset"],
        n_images=16,
        outfile=str(run_dir / VIZ_CONFIG["output_file"]),
        figsize=VIZ_CONFIG["figsize"],
        # show_patch_overlay = True, # Enable patch overlays
        # patch_size = 4 # Enable patch overlays for Swin Transformer debugging
    )

    return train_generator, val_generator, test_generator


def setup_model(device):
    """Initialize and setup the model."""
    logger.info("Initializing model...")

    if VALIDATION_CONFIG.get("enable_validation", False):
        # For validation, use ImageNet-compatible Swin-Tiny
        from src.models import swin_tiny_patch4_window7_224

        model = swin_tiny_patch4_window7_224(num_classes=1000)  # ImageNet classes
        logger.info(
            "Created Swin-Tiny model for validation against pretrained weights."
        )
    else:
        # For regular training, use Swin Transformer config
        from src.models import (
            SwinTransformerModel,
            ModelWrapper,
            LinearClassificationHead,
        )

        encoder = SwinTransformerModel(
            img_size=SWIN_CONFIG["img_size"],
            patch_size=SWIN_CONFIG["patch_size"],
            embedding_dim=SWIN_CONFIG["embed_dim"],
            depths=SWIN_CONFIG["depths"],
            num_heads=SWIN_CONFIG["num_heads"],
            window_size=SWIN_CONFIG["window_size"],
            mlp_ratio=SWIN_CONFIG["mlp_ratio"],
            dropout=SWIN_CONFIG["dropout"],
            attention_dropout=SWIN_CONFIG["attention_dropout"],
            projection_dropout=SWIN_CONFIG["projection_dropout"],
            drop_path_rate=SWIN_CONFIG["drop_path_rate"],
        )

        if DOWNSTREAM_CONFIG["head_type"] == "linear_classification":
            pred_head = LinearClassificationHead(
                num_features=encoder.num_features,
                num_classes=DOWNSTREAM_CONFIG["num_classes"],
            )
        else:
            raise AssertionError(f"Unknown head type: {DOWNSTREAM_CONFIG['head_type']}")

        model = ModelWrapper(
            encoder=encoder,
            pred_head=pred_head,
            freeze=DOWNSTREAM_CONFIG["freeze_encoder"],
        )
        logger.info("Created SwinTransformerModel training.")

        if SWIN_CONFIG.get("pretrained_weights", False):
            model_name = get_swin_name()
            logger.info("Transferring weights from pretrained to custom model...")
            transfer_stats = transfer_weights(
                model, encoder_only=True, model_name=model_name, device=device
            )
            logger.info(f"Weight transfer completed: {transfer_stats}")
    logger.info(
        f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    return model


def setup_training_components(
    model: nn.Module, total_epochs: int, warmup_epochs: int, learning_rate
):
    """Setup optimizer, scheduler, and loss criterion for linear probing."""
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.pred_head.parameters(),
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


def initialize_metrics_tracking():
    """Initialize metrics history dictionaries."""
    metrics_history = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],  # Note: only populated every 5 epochs
        "val_accuracy": [],
        "test_accuracy": [],  # Note: only populated every 5 epochs
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1_per_class": [],
    }

    lr_history = []

    mixup = (
        Mixup(alpha=AUGMENTATION_CONFIG["mixup_alpha"])
        if AUGMENTATION_CONFIG.get("mixup_alpha")
        else None
    )

    return metrics_history, lr_history, mixup


def generate_reports(
    model,
    variant,
    test_generator,
    criterion,
    lr_history,
    metrics_history,
    device,
    run_dir,
    validation_results,
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


def validate_model_if_enabled(model, val_generator, run_dir, device):
    """Validate model implementation"""
    if not VALIDATION_CONFIG.get("enable_validation", False):
        return None

    validator = ModelValidator(device=device)
    return validator.validate_model_implementation(
        custom_model=model,
        val_dataloader=val_generator,
        run_dir=run_dir,
        validation_config=VALIDATION_CONFIG,
    )


def create_reference_model(pretrained_model: str, device: torch.device) -> nn.Module:
    """
    Create the HuggingFace reference Swin model wrapped for linear probing.

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
            raise RuntimeError(f"Encoder missing 'num_features' attribute")

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
            freeze=True,  # freeze encoder, train head only
        )

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Reference model created. Trainable params (head only): {trainable_params:,}"
        )

        return model.to(device)

    except Exception as e:
        logger.error(f"Failed to create reference model {pretrained_model}: {e}")
        raise RuntimeError(f"Model creation failed: {e}") from e
    return model.to(device)


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

        # Get preset configuration before using it in logging
        preset = SWIN_PRESETS[model_size]

        logger.info(
            f"Custom model will match architecture: embed_dim={preset['embed_dim']}, depths={preset['depths']}, num_heads={preset['num_heads']}"
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
            f"Creating classification head for {encoder.num_features} features -> {DOWNSTREAM_CONFIG['num_classes']} classes"
        )
        pred_head = LinearClassificationHead(
            num_features=encoder.num_features,
            num_classes=DOWNSTREAM_CONFIG["num_classes"],
        )

        logger.debug("Wrapping encoder and head in ModelWrapper")
        model = ModelWrapper(
            encoder=encoder,
            pred_head=pred_head,
            freeze=True,  # linear probe: freeze encoder
        )

        logger.info("Transferring weights from reference to custom encoder...")
        logger.info(
            f"Weight transfer: {reference_model.encoder.__class__.__name__} -> {encoder.__class__.__name__}"
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

        logger.info(
            f"Custom model created. Trainable params (head only): {trainable_params:,}"
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
    """Run training loop for one model and return best validation accuracy."""
    criterion, optimizer, scheduler = setup_training_components(
        model, total_epochs, warmup_epochs, learning_rate
    )

    metrics_history = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],  # only populated every 5 epochs
        "val_accuracy": [],
        "test_accuracy": [],  # only populated every 5 epochs
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


def main():
    """Main training pipeline."""
    try:
        logger.info("Starting linear probing experiment...")

        # Validate configuration before proceeding
        validate_configuration()

        # Setup run directory and logging first
        run_dir = setup_run_directory()
        setup_logging(run_dir)
        logger.info(f"Experiment directory: {run_dir}")

        # Initialize experiment tracker
        tracker = ExperimentTracker(run_dir)

        # Set random seeds first to ensure reproducibility
        logger.info(
            f"Setting random seeds for reproducibility (seed: {SEED_CONFIG['seed']})..."
        )
        set_random_seeds(
            seed=SEED_CONFIG["seed"], deterministic=SEED_CONFIG["deterministic"]
        )

        # Setup components
        logger.info("Setting up device...")
        device = setup_device()
        logger.info(f"Using device: {device}")

        # Get training parameters from config (with fallbacks for missing keys)
        total_epochs = TRAINING_CONFIG.get("num_epochs", 50)
        warmup_epochs = TRAINING_CONFIG.get(
            "warmup_epochs", 2
        )  # Add this to config if needed
        learning_rate = TRAINING_CONFIG.get("learning_rate", 0.001)

        # Validate training parameters
        validate_training_parameters(total_epochs, warmup_epochs, learning_rate)
        logger.info(
            f"Training configuration: epochs={total_epochs}, warmup={warmup_epochs}, lr={learning_rate}"
        )

        # Get pretrained model from config (automatically matches current dataset)
        from config import get_pretrained_swin_name, SWIN_CONFIG

        pretrained_model = get_pretrained_swin_name()
        # Alternative: uncomment for ResNet50
        # pretrained_model = "resnet50"

        validate_pretrained_model_name(pretrained_model)
        logger.info(f"Using pretrained model: {pretrained_model}")
        logger.info(f"Model architecture: Swin-{SWIN_CONFIG['variant'].title()}")
        logger.info(
            f"Model config: embed_dim={SWIN_CONFIG['embed_dim']}, depths={SWIN_CONFIG['depths']}, num_heads={SWIN_CONFIG['num_heads']}"
        )
        logger.info(
            f"Image size: {SWIN_CONFIG['img_size']}x{SWIN_CONFIG['img_size']}, patch_size: {SWIN_CONFIG['patch_size']}x{SWIN_CONFIG['patch_size']}"
        )

        logger.info("Loading dataset...")
        train_generator, val_generator, test_generator = load_data(
            dataset="CIFAR100",
            use_batch_for_val=True,
            val_batch=5,
            batch_size=32,
            num_workers=4,
            root="./datasets",
            img_size=224,
            worker_init_fn=get_worker_init_fn(42),
        )
        logger.info(
            f"Dataset loaded: train={len(train_generator)}, val={len(val_generator)}, test={len(test_generator)} batches"
        )

        # Create models with error handling
        reference_model = create_reference_model(pretrained_model, device)

        custom_model = None
        if pretrained_model.lower().startswith("swin"):
            logger.info("Detected Swin Transformer architecture")
            logger.info("Extracting model size from TIMM model name...")
            model_size = None

            for p in pretrained_model.lower().split("_"):
                if p in SWIN_PRESETS:
                    model_size = p
                    break

            if model_size is None:
                raise ValueError(
                    f"Could not detect model size from '{pretrained_model}'. Available sizes: {list(SWIN_PRESETS.keys())}"
                )

            logger.info(f"Detected model size: {model_size}")
            logger.info(f"Will create custom model with: {SWIN_PRESETS[model_size]}")
            custom_model = create_custom_model(reference_model, model_size, device)
        else:
            logger.info(f"Non-Swin architecture detected: {pretrained_model}")
            logger.info(
                "Skipping custom model creation (only reference model will be trained)"
            )

        reference_tracker = ExperimentTracker(run_dir)

        # 3) Train & collect best validation accuracies
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

        if pretrained_model.lower().startswith("swin") and custom_model is not None:

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

            diff = abs(
                final_reference_metrics["accuracy"] - final_custom_metrics["accuracy"]
            )

            # 4) Log + save
            logger.info(
                "=== MODEL COMPARISON RESULTS (Linear Probing on CIFAR-100) ==="
            )
            logger.info(f"Reference (HF): {final_reference_metrics['accuracy']:.2f}%")
            logger.info(f"Custom        : {final_custom_metrics['accuracy']:.2f}%")
            logger.info(f"Difference    : {diff:.2f}% (custom - reference)")
        else:
            logger.info(
                "Only reference model training completed (no custom model for non-Swin architectures)"
            )

        logger.info("Linear probing experiment completed successfully!")

    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        logger.exception("Full traceback:")
        raise RuntimeError(f"Linear probing experiment failed: {e}") from e


if __name__ == "__main__":
    main()
