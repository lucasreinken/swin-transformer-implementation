"""
Enhanced Metrics Module for
"""

import torch
from typing import Optional
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
    amp_dtype: Optional[torch.dtype] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(amp_dtype),
            ):
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


# =============================================================================
# Segmentation Metrics (mIoU, per-class IoU)
# =============================================================================


def compute_iou_per_class(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Intersection over Union (IoU) for each class.
    
    IoU = TP / (TP + FP + FN) for each class
    
    Args:
        pred: Predicted segmentation map [H, W] or flattened
        target: Ground truth segmentation map [H, W] or flattened
        num_classes: Number of semantic classes
        ignore_index: Label to ignore in computation (default: 255 for ADE20K)
    
    Returns:
        Tuple of (iou_per_class, valid_mask):
            - iou_per_class: IoU for each class [num_classes]
            - valid_mask: Boolean mask indicating which classes are present [num_classes]
    """
    pred = pred.flatten()
    target = target.flatten()
    
    # Create mask for valid pixels (not ignore_index)
    valid_mask = target != ignore_index
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    iou_per_class = np.zeros(num_classes)
    class_present = np.zeros(num_classes, dtype=bool)
    
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        
        if union > 0:
            iou_per_class[cls] = intersection / union
            class_present[cls] = True
        else:
            # Class not present in either pred or target
            iou_per_class[cls] = 0.0
            class_present[cls] = False
    
    return iou_per_class, class_present


def compute_mean_iou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> Tuple[float, np.ndarray]:
    """
    Compute mean Intersection over Union (mIoU).
    
    mIoU = (1/K) * sum(IoU_k) where K is number of classes present
    
    Args:
        pred: Predicted segmentation map [H, W] or flattened
        target: Ground truth segmentation map [H, W] or flattened
        num_classes: Number of semantic classes
        ignore_index: Label to ignore in computation (default: 255 for ADE20K)
    
    Returns:
        Tuple of (mean_iou, iou_per_class):
            - mean_iou: Mean IoU across all present classes
            - iou_per_class: IoU for each class [num_classes]
    """
    iou_per_class, class_present = compute_iou_per_class(
        pred, target, num_classes, ignore_index
    )
    
    # Only average over classes that are present
    if class_present.sum() > 0:
        mean_iou = iou_per_class[class_present].mean()
    else:
        mean_iou = 0.0
    
    return mean_iou, iou_per_class


def compute_pixel_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    ignore_index: int = 255,
) -> float:
    """
    Compute pixel-wise accuracy.
    
    Args:
        pred: Predicted segmentation map [H, W] or flattened
        target: Ground truth segmentation map [H, W] or flattened
        ignore_index: Label to ignore in computation (default: 255)
    
    Returns:
        Pixel accuracy as percentage (0-100)
    """
    pred = pred.flatten()
    target = target.flatten()
    
    # Mask for valid pixels
    valid_mask = target != ignore_index
    
    if valid_mask.sum() == 0:
        return 0.0
    
    correct = (pred[valid_mask] == target[valid_mask]).sum()
    total = valid_mask.sum()
    
    return 100.0 * correct / total


def calculate_segmentation_metrics(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 150,
    ignore_index: int = 255,
    amp_dtype: Optional[torch.dtype] = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive segmentation metrics over a dataset.
    
    Similar to calculate_classification_metrics but for segmentation tasks.
    Computes mIoU, per-class IoU, and pixel accuracy.
    
    Args:
        model: Segmentation model
        data_loader: DataLoader yielding (images, masks)
        device: Device to run on
        num_classes: Number of semantic classes (150 for ADE20K)
        ignore_index: Label to ignore (255 for ADE20K unlabeled pixels)
        amp_dtype: Mixed precision dtype (optional)
    
    Returns:
        Dict containing:
            - mean_iou: Mean IoU across all classes (%)
            - pixel_accuracy: Pixel-wise accuracy (%)
            - iou_per_class: IoU for each class
            - class_present: Boolean mask for present classes
    """
    model.eval()
    
    # Accumulate intersection and union for each class across batches
    total_intersection = np.zeros(num_classes)
    total_union = np.zeros(num_classes)
    total_correct = 0
    total_pixels = 0
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(amp_dtype),
            ):
                outputs = model(images)  # [B, num_classes, H, W]
            
            # Get predictions
            preds = outputs.argmax(dim=1)  # [B, H, W]
            
            # Move to CPU for metric computation
            preds = preds.cpu().numpy()
            masks = masks.cpu().numpy()
            
            # Accumulate metrics for each sample in batch
            for pred, mask in zip(preds, masks):
                # Valid pixels mask
                valid = mask != ignore_index
                
                if valid.sum() == 0:
                    continue
                
                pred_valid = pred[valid]
                mask_valid = mask[valid]
                
                # Pixel accuracy accumulation
                total_correct += (pred_valid == mask_valid).sum()
                total_pixels += valid.sum()
                
                # Per-class intersection and union accumulation
                for cls in range(num_classes):
                    pred_cls = pred_valid == cls
                    mask_cls = mask_valid == cls
                    
                    total_intersection[cls] += np.logical_and(pred_cls, mask_cls).sum()
                    total_union[cls] += np.logical_or(pred_cls, mask_cls).sum()
    
    # Compute final metrics
    # IoU per class (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou_per_class = total_intersection / total_union
        iou_per_class = np.nan_to_num(iou_per_class, nan=0.0)
    
    # Class present mask (classes with non-zero union)
    class_present = total_union > 0
    
    # Mean IoU (only over present classes)
    if class_present.sum() > 0:
        mean_iou = iou_per_class[class_present].mean() * 100
    else:
        mean_iou = 0.0
    
    # Pixel accuracy
    if total_pixels > 0:
        pixel_accuracy = 100.0 * total_correct / total_pixels
    else:
        pixel_accuracy = 0.0
    
    logger.info(f"Segmentation metrics: mIoU={mean_iou:.2f}%, PixelAcc={pixel_accuracy:.2f}%")
    
    return {
        "mean_iou": mean_iou,
        "pixel_accuracy": pixel_accuracy,
        "iou_per_class": iou_per_class * 100,  # Convert to percentage
        "class_present": class_present,
        "num_classes_present": class_present.sum(),
    }


def plot_iou_per_class(
    iou_per_class: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (14, 6),
    top_k: int = 20,
):
    """
    Plot per-class IoU as a bar chart.
    
    Args:
        iou_per_class: IoU values for each class [num_classes]
        class_names: Optional list of class names
        save_path: Path to save the figure
        figsize: Figure size
        top_k: Number of top/bottom classes to show (if too many classes)
    """
    num_classes = len(iou_per_class)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # If too many classes, show top and bottom k
    if num_classes > 2 * top_k:
        # Sort by IoU
        sorted_indices = np.argsort(iou_per_class)[::-1]  # Descending
        top_indices = sorted_indices[:top_k]
        bottom_indices = sorted_indices[-top_k:]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Top k classes
        top_ious = iou_per_class[top_indices]
        top_names = [class_names[i] for i in top_indices]
        bars1 = ax1.barh(range(top_k), top_ious, color='steelblue', alpha=0.8)
        ax1.set_yticks(range(top_k))
        ax1.set_yticklabels(top_names, fontsize=8)
        ax1.set_xlabel('IoU (%)')
        ax1.set_title(f'Top {top_k} Classes by IoU')
        ax1.set_xlim(0, 100)
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Bottom k classes
        bottom_ious = iou_per_class[bottom_indices]
        bottom_names = [class_names[i] for i in bottom_indices]
        bars2 = ax2.barh(range(top_k), bottom_ious, color='coral', alpha=0.8)
        ax2.set_yticks(range(top_k))
        ax2.set_yticklabels(bottom_names, fontsize=8)
        ax2.set_xlabel('IoU (%)')
        ax2.set_title(f'Bottom {top_k} Classes by IoU')
        ax2.set_xlim(0, 100)
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
    else:
        # Show all classes
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.bar(range(num_classes), iou_per_class, color='steelblue', alpha=0.8)
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('IoU (%)')
        ax.set_xlabel('Class')
        ax.set_title('Per-Class IoU')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Per-class IoU plot saved to '{save_path}'")
    else:
        plt.show()
    plt.close()


def plot_segmentation_training_curves(
    metrics_history: Dict[str, List[float]],
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 4),
):
    """
    Plot segmentation training curves (loss, mIoU, pixel accuracy).
    
    Args:
        metrics_history: Dict with keys like 'train_loss', 'val_miou', 'val_pixel_acc'
        save_path: Base path for saving (will append metric names)
        figsize: Figure size for each plot
    """
    epochs = range(1, len(metrics_history.get("train_loss", [])) + 1)
    
    if len(epochs) == 0:
        logger.warning("No training history to plot")
        return
    
    base_path = save_path or "segmentation_curves"
    
    # Plot 1: Loss curves
    if "train_loss" in metrics_history:
        plt.figure(figsize=figsize)
        plt.plot(epochs, metrics_history["train_loss"], 'b-', label='Train Loss', linewidth=2)
        if "val_loss" in metrics_history:
            val_epochs = range(1, len(metrics_history["val_loss"]) + 1)
            plt.plot(val_epochs, metrics_history["val_loss"], 'r-', label='Val Loss', linewidth=2)
        plt.title('Loss Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{base_path}_loss.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 2: mIoU curve
    if "val_miou" in metrics_history:
        plt.figure(figsize=figsize)
        val_epochs = range(1, len(metrics_history["val_miou"]) + 1)
        plt.plot(val_epochs, metrics_history["val_miou"], 'g-', label='Val mIoU', linewidth=2)
        plt.title('Validation mIoU')
        plt.xlabel('Epochs')
        plt.ylabel('mIoU (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{base_path}_miou.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Pixel accuracy curve
    if "val_pixel_acc" in metrics_history:
        plt.figure(figsize=figsize)
        val_epochs = range(1, len(metrics_history["val_pixel_acc"]) + 1)
        plt.plot(val_epochs, metrics_history["val_pixel_acc"], 'm-', label='Val Pixel Acc', linewidth=2)
        plt.title('Validation Pixel Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{base_path}_pixel_acc.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Segmentation training curves saved with base name '{base_path}'")


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
        try:
            # Try to create numpy array, but handle inhomogeneous shapes
            f1_per_class_data = metrics_history["val_f1_per_class"]

            # Check if all arrays have the same shape
            shapes = [
                np.array(arr).shape
                for arr in f1_per_class_data
                if len(np.array(arr).shape) > 0
            ]
            if len(set(shapes)) > 1:
                logger.warning(
                    f"Inhomogeneous shapes in per-class F1 data: {shapes}. Skipping per-class plotting."
                )
                return

            f1_per_class = np.array(f1_per_class_data)
            num_classes = f1_per_class.shape[1]

            # Skip if too many classes
            if num_classes > 50:
                logger.info(
                    f"Too many classes ({num_classes}) for per-class plotting, skipping"
                )
                return

            plt.figure(
                figsize=(figsize[0] * 1.5, figsize[1])
            )  # Wider for multiple lines

            # Generate class names
            if num_classes == 10:
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
            else:
                class_names = [f"Class {i}" for i in range(num_classes)]

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

        except (ValueError, IndexError) as e:
            logger.warning(
                f"Could not plot per-class F1 curves due to data format issue: {e}. Skipping."
            )
            return


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
