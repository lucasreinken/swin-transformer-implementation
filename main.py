"""
Main orchestration file for the machine learning pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from config import SWIN_CONFIG
from src.data import load_data

from src.models import SimpleModel

from src.training import train_one_epoch, evaluate_model
from src.training.early_stopping import EarlyStopping
from src.training.checkpoints import save_checkpoint, save_model_weights
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

from config import (
    AUGMENTATION_CONFIG,
    DATA_CONFIG,
    MODEL_CONFIG,
    TRAINING_CONFIG,
    VIZ_CONFIG,
    SEED_CONFIG,
    SCHEDULER_CONFIG,
    VALIDATION_CONFIG,
)

# Additional imports for dataset plotting
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import os

# Setup logging
import logging

logger = logging.getLogger(__name__)


def setup_device():
    """Setup and return the appropriate device for training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def plot_cifar100_class_distribution(run_dir):
    """Generate CIFAR-100 class distribution plot and save to run directory."""
    try:
        logger.info("Generating CIFAR-100 class distribution plot...")

        # Use same data root as training
        dataset = datasets.CIFAR100(root=DATA_CONFIG["root"], train=True, download=False)

        # Count samples per fine class
        train_counts = np.bincount(dataset.targets, minlength=100)

        # Count per superclass (5 fine classes each)
        superclass_counts = [sum(train_counts[i * 5 : (i + 1) * 5]) for i in range(20)]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Fine-grained classes
        ax1.bar(range(100), train_counts, color="skyblue", edgecolor="navy", alpha=0.8)
        ax1.set_title("CIFAR-100: 100 Fine-Grained Classes", fontsize=14, pad=20)
        ax1.set_xlabel("Class Index")
        ax1.set_ylabel("Training Samples")
        ax1.axhline(500, color="red", linestyle="--", linewidth=1.5, label="Expected: 500")
        ax1.legend()
        ax1.grid(True, axis="y", alpha=0.3)

        # Superclasses
        ax2.bar(
            range(20), superclass_counts, color="lightcoral", edgecolor="darkred", alpha=0.8
        )
        ax2.set_title("20 Superclasses (5 classes each)", fontsize=14, pad=20)
        ax2.set_xlabel("Superclass Index")
        ax2.set_ylabel("Training Samples")
        ax2.set_xticks(range(20))
        ax2.set_xticklabels([f"{i}" for i in range(20)])
        ax2.axhline(
            2500, color="red", linestyle="--", linewidth=1.5, label="Expected: 2500"
        )
        ax2.legend()
        ax2.grid(True, axis="y", alpha=0.3)

        plt.suptitle("CIFAR-100 Dataset Distribution", fontsize=16, y=0.98)
        plt.tight_layout()

        # Save to run directory
        output_path = run_dir / "cifar100_class_distribution.pdf"
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()

        logger.info(f"Saved CIFAR-100 distribution plot to {output_path}")

    except Exception as e:
        logger.warning(f"Could not generate CIFAR-100 distribution plot: {e}")


def plot_upsampling_comparison(run_dir):
    """Generate upsampling comparison plot and save to run directory."""
    try:
        logger.info("Generating upsampling comparison plot...")

        # Use same data root as training
        dataset = datasets.CIFAR100(root=DATA_CONFIG["root"], train=True, download=False)
        img, label = dataset[42]  # Choose a visually clear image

        # Original
        img_np = np.array(img)

        # Resized
        resize = transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)
        img_resized = resize(img)
        img_resized_np = np.array(img_resized)

        # Plot side-by-side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        ax1.imshow(img_np)
        ax1.set_title("Original 32×32", fontsize=14)
        ax1.axis("off")

        ax2.imshow(img_resized_np)
        ax2.set_title("Resized 224×224 (bicubic)", fontsize=14)
        ax2.axis("off")

        plt.tight_layout()

        # Save to run directory
        combined_path = run_dir / "upsampling_comparison.pdf"
        plt.savefig(combined_path, bbox_inches="tight", dpi=300)
        plt.close()

        logger.info(f"Saved upsampling comparison plot to {combined_path}")

    except Exception as e:
        logger.warning(f"Could not generate upsampling comparison plot: {e}")


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
        from src.models.swin import swin_tiny_patch4_window7_224

        model = swin_tiny_patch4_window7_224(num_classes=1000)  # ImageNet classes
        logger.info(
            "Created Swin-Tiny model for validation against pretrained weights."
        )
    elif VALIDATION_CONFIG.get("use_swin_transformer", False):
        # For regular training, use CIFAR-10 Swin config
        from src.models.swin.swin_transformer_model import SwinTransformerModel

        model = SwinTransformerModel(
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
            num_classes=SWIN_CONFIG["num_classes"],
        )
        logger.info("Created SwinTransformerModel training.")
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


def setup_training_components(model):
    """Setup optimizer, scheduler, and loss criterion."""
    criterion = nn.CrossEntropyLoss()

    if SCHEDULER_CONFIG["use_scheduler"]:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=SCHEDULER_CONFIG["lr"],
            weight_decay=SCHEDULER_CONFIG["weight_decay"],
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer,
                    start_factor=0.1,
                    total_iters=SCHEDULER_CONFIG["warmup_epochs"],
                ),
                CosineAnnealingLR(
                    optimizer,
                    T_max=SCHEDULER_CONFIG["total_epochs"]
                    - SCHEDULER_CONFIG["warmup_epochs"],
                ),
            ],
            milestones=[SCHEDULER_CONFIG["warmup_epochs"]],
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
        )
        scheduler = None

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


def run_training_loop(
    model,
    train_generator,
    val_generator,
    test_generator,
    criterion,
    optimizer,
    scheduler,
    metrics_history,
    lr_history,
    mixup,
    device,
):
    """Run the main training loop."""
    # Early stopping setup
    early_stopper = EarlyStopping(patience=5, min_delta=0.01, mode="min")

    # Training loop
    logger.info("Starting training...")
    for epoch in range(TRAINING_CONFIG["num_epochs"]):
        train_loss = train_one_epoch(
            model, train_generator, criterion, optimizer, device, mixup=mixup
        )

        # Validate every epoch
        val_loss, val_accuracy, val_metrics = evaluate_model(
            model,
            val_generator,
            criterion,
            device,
        )

        # Store metrics
        metrics_history["train_loss"].append(train_loss)
        metrics_history["val_loss"].append(val_loss)
        metrics_history["val_accuracy"].append(val_accuracy)
        if "f1_score" in val_metrics:
            metrics_history["val_f1"].append(val_metrics["f1_score"])
        if "precision" in val_metrics:
            metrics_history["val_precision"].append(val_metrics["precision"])
        if "recall" in val_metrics:
            metrics_history["val_recall"].append(val_metrics["recall"])
        if "f1_per_class" in val_metrics:
            metrics_history["val_f1_per_class"].append(val_metrics["f1_per_class"])

        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            lr_history.append(current_lr)
            logger.info(
                f"Epoch {epoch+1}/{TRAINING_CONFIG['num_epochs']}: LR: {current_lr:.6f}"
            )
        else:
            lr_history.append(TRAINING_CONFIG["learning_rate"])

        # Early stopping check
        if early_stopper(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch+1}.")
            break

        # Test periodically (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            test_loss, test_accuracy, test_metrics = evaluate_model(
                model, test_generator, criterion, device
            )

            metrics_history["test_loss"].append(test_loss)
            metrics_history["test_accuracy"].append(test_accuracy)

            logger.info(
                f"Epoch {epoch+1}/{TRAINING_CONFIG['num_epochs']}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%"
            )
        else:
            logger.info(
                f"Epoch {epoch+1}/{TRAINING_CONFIG['num_epochs']}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
            )

        if (epoch + 1) % 10 == 0:
            logger.info(f"Saving checkpoint for epoch {epoch+1}...")
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                train_loss,
                f"checkpoints/checkpoint_epoch_{epoch+1}.pth",
            )

    logger.info("Training completed!")


def generate_reports(
    model,
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
    plot_training_curves(metrics_history, save_path=str(run_dir / "training_curves"))

    logger.info("Generating confusion matrix on test set...")
    _, _, final_test_metrics = evaluate_model(
        model,
        test_generator,
        criterion,
        device,
        num_classes=MODEL_CONFIG["num_classes"],
        detailed_metrics=True,
    )
    plot_confusion_matrix(
        final_test_metrics["confusion_matrix"],
        CIFAR10_CLASSES,
        save_path=str(run_dir / "confusion_matrix.png"),
    )

    # Final evaluation on test set
    logger.info("Performing final evaluation on test set...")
    final_test_loss, final_test_accuracy, final_test_metrics = evaluate_model(
        model, test_generator, criterion, device, detailed_metrics=True
    )
    logger.info(
        f"Final Test Results: Loss: {final_test_loss:.4f}, Accuracy: {final_test_accuracy:.2f}%"
    )

    if lr_history:
        logger.info("Generating LR schedule plot...")
        plot_lr_schedule(lr_history, save_path=str(run_dir / "lr_schedule.png"))

    if validation_results:
        logger.info("Generating model validation comparison plot...")
        plot_model_validation_comparison(
            validation_results,
            save_path=str(run_dir / "model_validation_comparison.png"),
        )

    # Generate dataset analysis plots
    if DATA_CONFIG["dataset"] in ["CIFAR10", "CIFAR100"]:
        if DATA_CONFIG["dataset"] == "CIFAR100":
            plot_cifar100_class_distribution(run_dir)
        plot_upsampling_comparison(run_dir)

    logger.info(f"\nFinal Test Results:")
    logger.info(f"Loss: {final_test_loss:.4f}")
    logger.info(f"Accuracy: {final_test_accuracy:.2f}%")
    logger.info(f"Precision: {final_test_metrics['precision']:.2f}%")
    logger.info(f"Recall: {final_test_metrics['recall']:.2f}%")
    logger.info(f"F1 Score: {final_test_metrics['f1_score']:.2f}%")

    return final_test_metrics


def save_final_model(model):
    """Save the final trained model weights."""
    save_model_weights(
        model, f"trained_models/{DATA_CONFIG['dataset']}_final_model_weights.pth"
    )


def validate_model_if_enabled(model, val_generator, run_dir):
    """Validate model implementation"""
    if not VALIDATION_CONFIG.get("enable_validation", False):
        return None

    validator = ModelValidator(device=next(model.parameters()).device.type)
    return validator.validate_model_implementation(
        custom_model=model,
        val_dataloader=val_generator,
        run_dir=run_dir,
        validation_config=VALIDATION_CONFIG,
    )


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
    train_generator, val_generator, test_generator = setup_data(device, run_dir)
    model = setup_model(device)

    # Log all configurations
    tracker.log_config(
        DATA_CONFIG=DATA_CONFIG,
        MODEL_CONFIG=MODEL_CONFIG,
        TRAINING_CONFIG=TRAINING_CONFIG,
        AUGMENTATION_CONFIG=AUGMENTATION_CONFIG,
        SCHEDULER_CONFIG=SCHEDULER_CONFIG,
        VALIDATION_CONFIG=VALIDATION_CONFIG,
        SEED_CONFIG=SEED_CONFIG,
        VIZ_CONFIG=VIZ_CONFIG,
    )

    # Validate model implementation
    validation_results = validate_model_if_enabled(model, val_generator, run_dir)

    criterion, optimizer, scheduler = setup_training_components(model)
    metrics_history, lr_history, mixup = initialize_metrics_tracking()

    # Run training
    run_training_loop(
        model,
        train_generator,
        val_generator,
        test_generator,
        criterion,
        optimizer,
        scheduler,
        metrics_history,
        lr_history,
        mixup,
        device,
    )

    # Generate reports and save model
    final_test_metrics = generate_reports(
        model,
        test_generator,
        criterion,
        lr_history,
        metrics_history,
        device,
        run_dir,
        validation_results,
    )

    tracker.log_results(
        final_metrics=final_test_metrics,
        training_history=metrics_history,
        validation_results=validation_results,
    )

    # Finalize experiment
    tracker.finalize()

    save_final_model(model)


if __name__ == "__main__":
    main()
