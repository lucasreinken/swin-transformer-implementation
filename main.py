"""
Main orchestration file for the machine learning pipeline.
"""
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from config import SWIN_CONFIG
from src.data import load_data

from src.models import SwinTransformerModel, ModelWrapper, LinearClassificationHead, SimpleModel

from src.training import evaluate_model, run_training_loop
from src.training.checkpoints import save_model_weights
from src.training.metrics import (
    plot_confusion_matrix,
    plot_lr_schedule,
    plot_training_curves,
    plot_model_validation_comparison,
)
from src.training.trainer import Mixup

from src.utils.visualization import CIFAR10_CLASSES, show_batch
from src.utils.seeds import set_random_seeds, get_worker_init_fn
from src.utils.experiment import setup_run_directory, setup_logging, ExperimentTracker
from src.utils.model_validation import ModelValidator
from src.utils.load_weights import transfer_weights

from config import (
    AUGMENTATION_CONFIG,
    DATA_CONFIG,
    SWIN_PRESETS,
    MODEL_CONFIG,
    DOWNSTREAM_CONFIG,
    TRAINING_CONFIG,
    VIZ_CONFIG,
    SEED_CONFIG,
    SCHEDULER_CONFIG,
    VALIDATION_CONFIG,
)

from src.utils.visualization import CIFAR100_CLASSES

# Setup logging
import logging

logger = logging.getLogger(__name__)

try:
    from timm import create_model
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning(
        "timm library not found. Cannot transfer weights."
    )


def get_swin_name():
    variant = SWIN_CONFIG["variant"]            # "tiny", "small", "base", "large"
    patch_size = SWIN_CONFIG.get("patch_size", 4)      # default 4
    window = SWIN_CONFIG.get("window_size", 7)  # default 7
    img = SWIN_CONFIG.get("img_size", 224)      # default 224

    return f"swin_{variant}_patch{patch_size}_window{window}_{img}"


def setup_device():
    """Setup and return the appropriate device for training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def setup_data(device, run_dir):
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
    elif VALIDATION_CONFIG.get("use_swin_transformer", False):
        # For regular training, use CIFAR-10 Swin config
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
            transfer_stats = transfer_weights(model, encoder_only=True, model_name=model_name, device=device)
            logger.info(f"Weight transfer completed: {transfer_stats}")
    else:
        input_dim = 3 * DATA_CONFIG["img_size"] * DATA_CONFIG["img_size"]
        model = SimpleModel(
            input_dim=input_dim,
            hidden_dims=MODEL_CONFIG["hidden_dims"],
            num_classes=MODEL_CONFIG["num_classes"],
            dropout_rate=MODEL_CONFIG["dropout_rate"],
            use_batch_norm=MODEL_CONFIG["use_batch_norm"],
        ).to(device)

        # Print model architecture
        logger.info(
            f"Model architecture: Input({MODEL_CONFIG['input_dim']}) -> "
            f"Hidden{MODEL_CONFIG['hidden_dims']} -> Output({MODEL_CONFIG['num_classes']})"
        )
    logger.info(
        f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    return model


def setup_training_components(model: nn.Module, total_epochs: int, warmup_epochs: int, learning_rate):
    """Setup optimizer, scheduler, and loss criterion for linear probing."""
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.pred_head.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(
                optimizer,
                start_factor=0.1,
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
    plot_training_curves(metrics_history, save_path=str(run_dir / f"training_curves_{variant}"))

    logger.info("Generating confusion matrix on test set...")
    _, _, final_test_metrics = evaluate_model(
        model,
        test_generator,
        criterion,
        device,
        num_classes=100,
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
        plot_lr_schedule(lr_history, save_path=str(run_dir / f"lr_schedule_{variant}.png"))

    if validation_results:
        logger.info(f"Generating model validation comparison plot of {variant} model...")
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

def create_reference_model(pretrained_model:str, device) -> nn.Module:
    """Create the HuggingFace reference Swin model wrapped for linear probing."""
    logger.info("Initializing reference model...")

    encoder = create_model(pretrained_model, pretrained=True)

    if pretrained_model.lower().startswith("swin"):
        encoder.prune_intermediate_layers(
            indices=None,     # keep all transformer blocks
            prune_norm=True,  # remove final LN
            prune_head=True   # remove classifier
        )
    else:
        encoder.prune_intermediate_layers(
            indices=None,     # keep all transformer blocks
            prune_head=True   # remove classifier
        )

    pred_head = LinearClassificationHead(
        num_features=encoder.num_features,
        num_classes=100,
    )

    model = ModelWrapper(
        encoder=encoder,
        pred_head=pred_head,
        freeze=True,  # freeze encoder, train head only
    )

    logger.info(
        "Reference model created. Trainable params (head only): "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    return model.to(device)


def create_custom_model(reference_model, model_size, device) -> nn.Module:
    """Initialize and setup the custom Swin model, then transfer weights."""
    logger.info("Initializing custom Swin model...")

    cfg = {
        "img_size": 224,
        "patch_size": 4,
        "window_size": 7,
        "mlp_ratio": 4.0,
        "drop_path_rate": 0.1,
    }

    preset = SWIN_PRESETS[model_size]

    cfg["embed_dim"] = preset["embed_dim"]
    cfg["depths"] = preset["depths"]
    cfg["num_heads"] = preset["num_heads"]

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

    pred_head = LinearClassificationHead(
        num_features=encoder.num_features,
        num_classes=100,
    )

    model = ModelWrapper(
        encoder=encoder,
        pred_head=pred_head,
        freeze=True,  # linear probe: freeze encoder
    )

    logger.info("Transferring weights from reference to custom encoder...")
    transfer_stats = transfer_weights(
        model, encoder_only=True, pretrained_model=reference_model.encoder, device=device
    )
    logger.info(f"Weight transfer completed: {transfer_stats}")

    logger.info(
        "Custom model created. Trainable params (head only): "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    return model.to(device)

def _train_single_model(
    model: nn.Module,
    train_generator: DataLoader,
    val_generator: DataLoader,
    test_generator: DataLoader,
    total_epochs: int,
    warmup_epochs,
    learning_rate: float,
    device
):
    """Run training loop for one model and return best validation accuracy."""
    criterion, optimizer, scheduler = setup_training_components(model, total_epochs, warmup_epochs, learning_rate)

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
    # Setup run directory and logging first
    run_dir = setup_run_directory()
    setup_logging(run_dir)

    # Initialize experiment tracker
    tracker = ExperimentTracker(run_dir)

    # Set random seeds first to ensure reproducibility
    set_random_seeds(
        seed=SEED_CONFIG["seed"], deterministic=SEED_CONFIG["deterministic"]
    )

    # Setup components
    device = setup_device()

    total_epochs = 50
    warmup_epochs = 2
    learning_rate = 0.001

    # pretrained_model = "swin_tiny_patch4_window7_224"
    pretrained_model = "resnet50"

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

    reference_model = create_reference_model(pretrained_model, device)
    if pretrained_model.lower().startswith("swin"):
        model_size = None

        for p in pretrained_model.lower().split("_"):
            if p in SWIN_PRESETS:
                model_size = p
                break

        custom_model = create_custom_model(reference_model, model_size, device)

    reference_tracker = ExperimentTracker(run_dir)

    # 3) Train & collect best validation accuracies
    logger.info("Training reference model (HF)...")
    reference_criterion, reference_lr_history, reference_metrics_history = _train_single_model(
        reference_model, train_generator, val_generator, test_generator, total_epochs, warmup_epochs, learning_rate, device
    )

    logger.info("Finished training reference model (HF)!")

    final_reference_metrics = _finalize_validation(
        reference_model,
        "reference",
        test_generator,
        reference_criterion,
        reference_lr_history,
        reference_metrics_history,
        device,
        run_dir,
        reference_tracker
    )

    if pretrained_model.lower().startswith("swin"):

        custom_tracker = ExperimentTracker(run_dir)

        logger.info("Training custom model...")
        custom_criterion, custom_lr_history, custom_metrics_history = _train_single_model(
            custom_model, train_generator, val_generator, test_generator, total_epochs, warmup_epochs, learning_rate, device
        )

        logger.info("Finished training custom model (HF)!")


        final_custom_metrics = _finalize_validation(
            custom_model,
            "custom",
            test_generator,
            custom_criterion,
            custom_lr_history,
            custom_metrics_history,
            device,
            run_dir,
            custom_tracker
        )

        diff = abs(final_reference_metrics["accuracy"] - final_custom_metrics["accuracy"])

        # 4) Log + save
        logger.info("=== MODEL COMPARISON RESULTS (Linear Probing on CIFAR-100) ===")
        logger.info(f"Reference (HF): {final_reference_metrics['accuracy']:.2f}%")
        logger.info(f"Custom        : {final_custom_metrics['accuracy']:.2f}%")
        logger.info(f"Difference    : {diff:.2f}% (custom - reference)")



if __name__ == "__main__":
    main()
