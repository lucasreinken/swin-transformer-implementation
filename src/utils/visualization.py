import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Tuple
from torch.utils.data import DataLoader
import math
from pathlib import Path


CIFAR_CLASSES = [
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


def show(
    x: Union[np.ndarray, torch.Tensor],
    y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    outfile: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> None:
    """
    Visualize a batch of images in a grid.
    """
    if isinstance(x, torch.Tensor):
        x = x.numpy()

    batch_size = x.shape[0]

    # Create a grid
    grid_size = int(np.ceil(np.sqrt(batch_size)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()

    for i in range(batch_size):
        # Convert CHW to HWC and ensure values are in [0, 1]
        img = x[i].transpose(1, 2, 0)
        axes[i].imshow(img)

        if y is not None:
            axes[i].set_title(f"Label: {y[i].item()}")
        else:
            axes[i].set_title(f"Img {i}")
        axes[i].axis("off")

    for j in range(batch_size, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()


def denormalize_image(
    img: np.ndarray, mean: List[float], std: List[float]
) -> np.ndarray:
    """
    Denormalize an image tensor back to [0,1] range for visualization.

    Args:
        img: Normalized image
        mean: Mean values
        std: Standard deviation values

    Returns:
        Denormalized image
    """
    img = img * np.array(std) + np.array(mean)
    return np.clip(img, 0, 1)


def show_batch(
    loader: DataLoader,
    class_names: List[str],
    n_images: int = 16,
    outfile: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 12),
    dataset: str = "CIFAR10",
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> None:
    """
    Display a grid of images from a DataLoader with their labels.

    Args:
        loader: DataLoader containing the images and labels
        class_names: List of class names
        n_images: Number of images to display
        outfile: Path to save the figure
        figsize: Figure size
        dataset: Dataset name
        mean: Mean values for denormalization
        std: Standard deviation values for denormalization
    """
    if class_names is None:
        if dataset == "CIFAR10":
            class_names = CIFAR_CLASSES
        else:
            raise ValueError("Class names must be provided for unknown datasets.")

    # Set default normalization values based on dataset
    if mean is None or std is None:
        if dataset == "CIFAR10":
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

    images, labels = next(iter(loader))
    images = images[:n_images]
    labels = labels[:n_images]

    # Calculate grid
    grid_size = int(math.ceil(math.sqrt(len(images))))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)

    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(len(images)):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = denormalize_image(img, mean, std)
        axes[i].imshow(img)

        label_idx = labels[i].item()
        class_name = (
            class_names[label_idx]
            if label_idx < len(class_names)
            else f"Class {label_idx}"
        )
        axes[i].set_title(f"{class_name}", fontsize=8)
        axes[i].axis("off")

    # Hide empty subplots
    for j in range(len(images), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()


def show_raw_batch(
    dataset: str = "CIFAR10",
    n_images: int = 16,
    outfile: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 12),
) -> None:
    if dataset != "CIFAR10":
        raise ValueError("show_raw_batch currently only supports CIFAR10 dataset.")

    # Load raw data without transformations
    data_dir = Path("./datasets/cifar-10-batches-py")
    if not data_dir.exists():
        raise FileNotFoundError(
            "CIFAR-10 dataset not found in the specified directory."
        )

    # Load first batch for visualization
    with open(data_dir / "data_batch_1", "rb") as f:
        batch = pickle.load(f, encoding="bytes")
        images = batch[b"data"]
        labels = batch[b"labels"]

    # Take first n_images
    images = images[:n_images]
    labels = labels[:n_images]

    # Grid
    grid_size = int(math.ceil(math.sqrt(n_images)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)

    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(n_images):
        img_flat = images[i]
        img = img_flat.reshape(3, 32, 32).transpose(1, 2, 0)

        img = img.astype(np.float32) / 255.0  # Scale to [0, 1]

        axes[i].imshow(img)
        axes[i].set_title(f"Label: {CIFAR_CLASSES[labels[i]]}", fontsize=8)
        axes[i].axis("off")

    # Hide empty subplots
    for j in range(n_images, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        print(f"Saved raw batch visualization to {outfile}")
    else:
        plt.show()
