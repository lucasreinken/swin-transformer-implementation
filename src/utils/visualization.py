import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import math
import torch
from torch.utils.data import DataLoader

import logging

logger = logging.getLogger(__name__)

# CIFAR-10 class names
CIFAR10_CLASSES = [
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

# CIFAR-100 class names (coarse labels for simplicity)
CIFAR100_CLASSES = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]


def get_class_names(dataset: str) -> list:
    """
    Get class names for different datasets.

    Args:
        dataset: Dataset name ("CIFAR10", "CIFAR100", or "ImageNet")

    Returns:
        List of class names or indices for ImageNet
    """
    if dataset == "CIFAR10":
        return CIFAR10_CLASSES
    elif dataset == "CIFAR100":
        return CIFAR100_CLASSES
    elif dataset == "ImageNet":
        # For ImageNet, return class indices since there are 1000 classes
        return [f"class_{i}" for i in range(1000)]
    else:
        return [f"class_{i}" for i in range(10)]  # fallback


def show_batch(
    dataloader: DataLoader,
    dataset: str = "CIFAR10",
    n_images: int = 16,
    outfile: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 12),
    show_patch_overlay: bool = False,
    patch_size: int = 4,
    patch_color: str = "red",
    patch_alpha: float = 0.3,
    patch_linewidth: float = 1.0,
) -> None:
    """
    Enhanced visualization function for dataset batches with optional patch overlays.

    This function visualizes images from a PyTorch DataLoader, showing the actual
    transformed/preprocessed images that the model sees during training.

    Features:
    - Multi-dataset support (CIFAR-10, CIFAR-100, ImageNet)
    - Optional patch overlays for Swin Transformer debugging
    - Grid layout with class labels
    - High-quality image rendering

    Args:
        dataloader: PyTorch DataLoader to visualize
        dataset: Dataset name for class names ("CIFAR10", "CIFAR100", "ImageNet")
        n_images: Number of images to display in grid
        outfile: Path to save the visualization. If None, displays interactively.
        figsize: Figure size as (width, height) in inches
        show_patch_overlay: Whether to draw patch boundary grid lines
        patch_size: Size of patches for overlay grid (e.g., 4 for 4x4 patches)
        patch_color: Color of patch overlay lines
        patch_alpha: Transparency of patch overlay (0.0 = transparent, 1.0 = opaque)
        patch_linewidth: Width of patch overlay lines
    """

    # Get class names for the dataset
    class_names = get_class_names(dataset)

    # Get first batch from dataloader
    logger.info(f"Visualizing from DataLoader with {len(dataloader)} batches")

    data_iter = iter(dataloader)
    batch_images, batch_labels = next(data_iter)

    # Take first n_images from the batch
    n_available = min(n_images, len(batch_images))
    images = batch_images[:n_available]
    labels = batch_labels[:n_available].tolist()

    # Convert tensors to numpy arrays for matplotlib
    if isinstance(images, torch.Tensor):
        # Handle different tensor formats
        if images.dim() == 4:  # [B, C, H, W] - standard PyTorch format
            images = images.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, C]
        elif images.dim() == 3:  # [C, H, W] - single image
            images = images.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            images = images[None, ...]  # Add batch dimension
            labels = [labels]  # Make it a list

    logger.info(f"Loaded {len(images)} images from dataloader")

    # Calculate grid dimensions
    grid_size = int(math.ceil(math.sqrt(len(images))))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)

    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(len(images)):
        img = images[i]

        # Handle different image formats
        if img.max() > 1.0 and img.min() >= 0.0:
            # Images are in [0, 255] range, convert to [0, 1]
            img = img.astype(np.float32) / 255.0
        elif img.min() < -0.5 or img.max() > 2.0:
            # Images are normalized with ImageNet stats, denormalize to [0, 1]
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean  # Denormalize
            img = np.clip(img, 0.0, 1.0)  # Ensure valid range
        # If images are already in [0, 1] range, leave them as-is

        axes[i].imshow(img)

        # Add class label
        class_name = (
            class_names[labels[i]]
            if labels[i] < len(class_names)
            else f"class_{labels[i]}"
        )
        axes[i].set_title(f"{class_name}", fontsize=8)
        axes[i].axis("off")

        # Add patch overlay if requested
        if show_patch_overlay:
            _add_patch_overlay(
                axes[i],
                img.shape[:2],
                patch_size,
                patch_color,
                patch_alpha,
                patch_linewidth,
            )

    # Hide empty subplots
    for j in range(len(images), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        logger.info(f"Saved enhanced dataset visualization to {outfile}")
    else:
        plt.show()


def _add_patch_overlay(ax, img_shape, patch_size, color, alpha, linewidth):
    """
    Add patch boundary overlay to an image axis.

    Args:
        ax: Matplotlib axis to draw on
        img_shape: (height, width) of the image
        patch_size: Size of each patch (e.g., 4 for 4x4 patches)
        color: Color of the grid lines
        alpha: Transparency of the grid lines
        linewidth: Width of the grid lines
    """
    h, w = img_shape

    # Calculate number of patches in each dimension
    n_patches_h = h // patch_size
    n_patches_w = w // patch_size

    # Draw vertical lines (patch boundaries)
    for i in range(1, n_patches_w):
        x = i * patch_size
        ax.axvline(x=x, color=color, alpha=alpha, linewidth=linewidth, linestyle="-")

    # Draw horizontal lines (patch boundaries)
    for i in range(1, n_patches_h):
        y = i * patch_size
        ax.axhline(y=y, color=color, alpha=alpha, linewidth=linewidth, linestyle="-")

    # Add subtle border around entire image
    ax.axhline(
        y=0, color=color, alpha=alpha * 0.5, linewidth=linewidth * 0.5, linestyle="-"
    )
    ax.axhline(
        y=h - 1,
        color=color,
        alpha=alpha * 0.5,
        linewidth=linewidth * 0.5,
        linestyle="-",
    )
    ax.axvline(
        x=0, color=color, alpha=alpha * 0.5, linewidth=linewidth * 0.5, linestyle="-"
    )
    ax.axvline(
        x=w - 1,
        color=color,
        alpha=alpha * 0.5,
        linewidth=linewidth * 0.5,
        linestyle="-",
    )
