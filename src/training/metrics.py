"""
Enhanced Metrics Module for
"""

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


def calculate_classification_metrics(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    # Overall accuracy
    accuracy = np.mean(all_preds == all_labels)

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = (
        precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1_score": f1 * 100,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "true_labels": all_labels,
    }
"""
Enhanced Metrics Module for
"""

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

def save_tensor_image_as_pdf(
    img_tensor: torch.Tensor,
    outfile: str,
    imagenet_denormalize: bool = True,
):
    """
    Save a single PyTorch tensor image as a PDF.

    Args:
        img_tensor: Tensor [C, H, W] or [H, W, C]
        outfile: Path like 'images/example.pdf'
        imagenet_denormalize: Apply mean/std inverse if image is normalized
    """
    img = img_tensor

    # Move to CPU and convert to numpy
    if isinstance(img, torch.Tensor):
        img = img.cpu()

    # Ensure format [H, W, C]
    if img.dim() == 3 and img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)  # [C,H,W] -> [H,W,C]

    img = img.numpy()

    # Detect and undo ImageNet normalization
    if imagenet_denormalize and (img.min() < 0 or img.max() > 1):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)

    # If still not [0,1], normalize from [0,255]
    if img.max() > 1.0:
        img = img.astype(np.float32) / 255.0

    # Save as PDF
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(outfile, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved image to '{outfile}'")

def calculate_classification_metrics(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # --- filtering by label + correctness for saving images ---
            # batch_size = 1, so we can use .item()
            y_true = labels.item()
            y_pred = preds.item()

            target_labels = [9, 11, 39, 55, 68, 74]

            if y_true in target_labels:
                # correct predictions
                if y_pred == y_true:
                    outfile = f"images/correct_label{y_true}_{y_pred}.pdf"
                    save_tensor_image_as_pdf(inputs[0], str(outfile))

                # incorrect predictions
                if y_pred != y_true:
                    outfile = f"images/incorrect_label{y_true}_{y_pred}.pdf"
                    save_tensor_image_as_pdf(inputs[0], str(outfile))

            if y_pred in target_labels:
                # incorrect predictions
                if y_pred != y_true:
                    outfile = f"images/incorrect_pred{y_true}_{y_pred}.pdf"
                    save_tensor_image_as_pdf(inputs[0], str(outfile))

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    # Overall accuracy
    accuracy = np.mean(all_preds == all_labels)

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = (
        precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": float(accuracy * 100),
        "precision": float(precision * 100),
        "recall": float(recall * 100),
        "f1_score": float(f1 * 100),
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "confusion_matrix": cm.tolist(),
        "predictions": all_preds.tolist(),
        "true_labels": all_labels.tolist(),
    }


def plot_confusion_matrix(
    cm: np.ndarray, class_names: List[str], save_path: str = None
):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_loss_curves(
    epochs: range,
    metrics_history: Dict[str, List[float]],
    base_path: str,
    figsize: Tuple[int, int],
):
    """Plot training and validation loss curves."""
    plt.figure(figsize=figsize)
    plt.plot(
        epochs, metrics_history["train_loss"], "b-", label="Train Loss", linewidth=2
    )
    plt.plot(epochs, metrics_history["val_loss"], "r-", label="Val Loss", linewidth=2)
    plt.title("Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_path}_loss.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_accuracy_curves(
    epochs: range,
    metrics_history: Dict[str, List[float]],
    base_path: str,
    figsize: Tuple[int, int],
):
    """Plot validation accuracy curves."""
    plt.figure(figsize=figsize)
    plt.plot(
        epochs, metrics_history["val_accuracy"], "r-", label="Val Accuracy", linewidth=2
    )
    plt.title("Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_path}_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_f1_curves(
    epochs: range,
    metrics_history: Dict[str, List[float]],
    base_path: str,
    figsize: Tuple[int, int],
):
    """Plot validation F1 score curves."""
    if "val_f1" in metrics_history:
        plt.figure(figsize=figsize)
        plt.plot(epochs, metrics_history["val_f1"], "r-", label="Val F1", linewidth=2)
        plt.title("Validation F1 Score")
        plt.xlabel("Epochs")
        plt.ylabel("F1 Score (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{base_path}_f1.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_precision_recall_curves(
    epochs: range,
    metrics_history: Dict[str, List[float]],
    base_path: str,
    figsize: Tuple[int, int],
):
    """Plot validation precision vs recall curves."""
    if "val_precision" in metrics_history and "val_recall" in metrics_history:
        plt.figure(figsize=figsize)
        plt.plot(
            epochs,
            metrics_history["val_precision"],
            "r-",
            label="Val Precision",
            linewidth=2,
        )
        plt.plot(
            epochs, metrics_history["val_recall"], "g-", label="Val Recall", linewidth=2
        )
        plt.title("Validation Precision vs Recall")
        plt.xlabel("Epochs")
        plt.ylabel("Percentage (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{base_path}_precision_recall.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_per_class_f1_curves(
    epochs: range,
    metrics_history: Dict[str, List[float]],
    base_path: str,
    figsize: Tuple[int, int],
):
    """Plot per-class F1 score curves."""
    if (
        "val_f1_per_class" in metrics_history
        and len(metrics_history["val_f1_per_class"]) > 0
    ):
        plt.figure(figsize=(figsize[0] * 1.5, figsize[1]))  # Wider for multiple lines
        f1_per_class = np.array(metrics_history["val_f1_per_class"])
        class_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        for i, class_name in enumerate(class_names):
            plt.plot(epochs, f1_per_class[:, i], label=class_name, linewidth=2)
        plt.title("Validation F1 Score per Class")
        plt.xlabel("Epochs")
        plt.ylabel("F1 Score (%)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{base_path}_per_class_f1.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_training_curves(
    metrics_history: Dict[str, List[float]],
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot training curves, saving each as a separate image.

    Args:
        metrics_history: Dict with keys like 'train_loss', 'val_accuracy', etc.
        save_path: Base path for saving (will append metric names)
        figsize: Figure size for each plot
    """
    epochs = range(1, len(metrics_history["train_loss"]) + 1)
    base_path = save_path or "training_curves"

    # Plot each metric type
    plot_loss_curves(epochs, metrics_history, base_path, figsize)
    plot_accuracy_curves(epochs, metrics_history, base_path, figsize)
    plot_f1_curves(epochs, metrics_history, base_path, figsize)
    plot_precision_recall_curves(epochs, metrics_history, base_path, figsize)
    plot_per_class_f1_curves(epochs, metrics_history, base_path, figsize)

    logger.info(
        f"Training curves saved as separate images with base name '{base_path}'"
    )


def plot_lr_schedule(
    lr_history: List[float],
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """Plot the learning rate schedule over epochs."""
    epochs = range(1, len(lr_history) + 1)

    plt.figure(figsize=figsize)
    plt.plot(epochs, lr_history, "b-", label="Learning Rate", linewidth=2)
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"LR schedule plot saved to '{save_path}'")
    else:
        plt.show()
    plt.close()


def plot_model_validation_comparison(
    validation_results: Dict,
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 6),
):
    """
    Plot comparison between custom model and pretrained model performance.
    """
    if not validation_results:
        logger.warning("No validation results to plot")
        return

    custom_results = validation_results.get("custom_model", {})
    pretrained_results = validation_results.get("pretrained_model", {})
    differences = validation_results.get("differences", {})

    # Prepare data
    models = ["Custom Model", "Pretrained Model"]
    top1_scores = [
        custom_results.get("top1_accuracy", 0),
        pretrained_results.get("top1_accuracy", 0),
    ]
    top5_scores = [
        custom_results.get("top5_accuracy", 0),
        pretrained_results.get("top5_accuracy", 0),
    ]

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Top-1 Accuracy
    bars1 = ax1.bar(models, top1_scores, color=["skyblue", "lightcoral"], alpha=0.8)
    ax1.set_title("Top-1 Accuracy Comparison", fonsize=14, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, score in zip(bars1, top1_scores):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{score:.1f}%",
            ha="center",
            fontweight="bold",
        )

    # Add difference annonation
    top1_diff = differences.get("top1_diff", 0)
    top5_diff = differences.get("top5_diff", 0)

    fig.suptitle(
        f"Model Validation Results:\n"
        f"Top-1 delta: {top1_diff:+.2f}% | Top-5 delta: {top5_diff:+.2f}%,",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Model validation comparison plot saved to '{save_path}")
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray, class_names: List[str], save_path: str = None
):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_loss_curves(
    epochs: range,
    metrics_history: Dict[str, List[float]],
    base_path: str,
    figsize: Tuple[int, int],
):
    """Plot training and validation loss curves."""
    plt.figure(figsize=figsize)
    plt.plot(
        epochs, metrics_history["train_loss"], "b-", label="Train Loss", linewidth=2
    )
    plt.plot(epochs, metrics_history["val_loss"], "r-", label="Val Loss", linewidth=2)
    plt.title("Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_path}_loss.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_accuracy_curves(
    epochs: range,
    metrics_history: Dict[str, List[float]],
    base_path: str,
    figsize: Tuple[int, int],
):
    """Plot validation accuracy curves."""
    plt.figure(figsize=figsize)
    plt.plot(
        epochs, metrics_history["val_accuracy"], "r-", label="Val Accuracy", linewidth=2
    )
    plt.title("Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_path}_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_f1_curves(
    epochs: range,
    metrics_history: Dict[str, List[float]],
    base_path: str,
    figsize: Tuple[int, int],
):
    """Plot validation F1 score curves."""
    if "val_f1" in metrics_history:
        plt.figure(figsize=figsize)
        plt.plot(epochs, metrics_history["val_f1"], "r-", label="Val F1", linewidth=2)
        plt.title("Validation F1 Score")
        plt.xlabel("Epochs")
        plt.ylabel("F1 Score (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{base_path}_f1.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_precision_recall_curves(
    epochs: range,
    metrics_history: Dict[str, List[float]],
    base_path: str,
    figsize: Tuple[int, int],
):
    """Plot validation precision vs recall curves."""
    if "val_precision" in metrics_history and "val_recall" in metrics_history:
        plt.figure(figsize=figsize)
        plt.plot(
            epochs,
            metrics_history["val_precision"],
            "r-",
            label="Val Precision",
            linewidth=2,
        )
        plt.plot(
            epochs, metrics_history["val_recall"], "g-", label="Val Recall", linewidth=2
        )
        plt.title("Validation Precision vs Recall")
        plt.xlabel("Epochs")
        plt.ylabel("Percentage (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{base_path}_precision_recall.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_per_class_f1_curves(
    epochs: range,
    metrics_history: Dict[str, List[float]],
    base_path: str,
    figsize: Tuple[int, int],
):
    """Plot per-class F1 score curves."""
    if (
        "val_f1_per_class" in metrics_history
        and len(metrics_history["val_f1_per_class"]) > 0
    ):
        plt.figure(figsize=(figsize[0] * 1.5, figsize[1]))  # Wider for multiple lines
        f1_per_class = np.array(metrics_history["val_f1_per_class"])
        class_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        for i, class_name in enumerate(class_names):
            plt.plot(epochs, f1_per_class[:, i], label=class_name, linewidth=2)
        plt.title("Validation F1 Score per Class")
        plt.xlabel("Epochs")
        plt.ylabel("F1 Score (%)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{base_path}_per_class_f1.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_training_curves(
    metrics_history: Dict[str, List[float]],
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot training curves, saving each as a separate image.

    Args:
        metrics_history: Dict with keys like 'train_loss', 'val_accuracy', etc.
        save_path: Base path for saving (will append metric names)
        figsize: Figure size for each plot
    """
    epochs = range(1, len(metrics_history["train_loss"]) + 1)
    base_path = save_path or "training_curves"

    # Plot each metric type
    plot_loss_curves(epochs, metrics_history, base_path, figsize)
    plot_accuracy_curves(epochs, metrics_history, base_path, figsize)
    plot_f1_curves(epochs, metrics_history, base_path, figsize)
    plot_precision_recall_curves(epochs, metrics_history, base_path, figsize)
    plot_per_class_f1_curves(epochs, metrics_history, base_path, figsize)

    logger.info(
        f"Training curves saved as separate images with base name '{base_path}'"
    )


def plot_lr_schedule(
    lr_history: List[float],
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """Plot the learning rate schedule over epochs."""
    epochs = range(1, len(lr_history) + 1)

    plt.figure(figsize=figsize)
    plt.plot(epochs, lr_history, "b-", label="Learning Rate", linewidth=2)
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"LR schedule plot saved to '{save_path}'")
    else:
        plt.show()
    plt.close()


def plot_model_validation_comparison(
    validation_results: Dict,
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 6),
):
    """
    Plot comparison between custom model and pretrained model performance.
    """
    if not validation_results:
        logger.warning("No validation results to plot")
        return

    custom_results = validation_results.get("custom_model", {})
    pretrained_results = validation_results.get("pretrained_model", {})
    differences = validation_results.get("differences", {})

    # Prepare data
    models = ["Custom Model", "Pretrained Model"]
    top1_scores = [
        custom_results.get("top1_accuracy", 0),
        pretrained_results.get("top1_accuracy", 0),
    ]
    top5_scores = [
        custom_results.get("top5_accuracy", 0),
        pretrained_results.get("top5_accuracy", 0),
    ]

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Top-1 Accuracy
    bars1 = ax1.bar(models, top1_scores, color=["skyblue", "lightcoral"], alpha=0.8)
    ax1.set_title("Top-1 Accuracy Comparison", fonsize=14, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, score in zip(bars1, top1_scores):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{score:.1f}%",
            ha="center",
            fontweight="bold",
        )

    # Add difference annonation
    top1_diff = differences.get("top1_diff", 0)
    top5_diff = differences.get("top5_diff", 0)

    fig.suptitle(
        f"Model Validation Results:\n"
        f"Top-1 delta: {top1_diff:+.2f}% | Top-5 delta: {top5_diff:+.2f}%,",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Model validation comparison plot saved to '{save_path}")
    else:
        plt.show()
    plt.close()
