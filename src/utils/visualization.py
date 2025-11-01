import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
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
    dataset: str = "CIFAR10",
    n_images: int = 16,
    outfile: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 12),
) -> None:
    """
    Visualize raw CIFAR-10 images directly from dataset files (before any preprocessing).
    Shows the actual colors and quality as stored in the dataset.
    """
    if dataset != "CIFAR10":
        raise ValueError("show_raw_batch currently only supports CIFAR10 dataset.")

    # Load raw data without transformations
    data_dir = Path("./datasets/cifar-10-batches-py")
    if not data_dir.exists():
        raise FileNotFoundError(
            f"CIFAR-10 dataset not found at {data_dir}. "
            "Please ensure the dataset is downloaded."
        )

    # Load first batch for visualization
    with open(data_dir / "data_batch_1", "rb") as f:
        batch = pickle.load(f, encoding="bytes")
        images = batch[b"data"]
        labels = batch[b"labels"]

    # Take first n_images
    images = images[:n_images]
    labels = labels[:n_images]

    # Calculate grid dimensions
    grid_size = int(math.ceil(math.sqrt(n_images)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)

    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(n_images):
        # Convert flat array to image: (3072,) -> (3, 32, 32) -> (32, 32, 3)
        img_flat = images[i]
        img = img_flat.reshape(3, 32, 32).transpose(1, 2, 0)
        img = img.astype(np.float32) / 255.0  # Scale to [0, 1]

        axes[i].imshow(img)
        axes[i].set_title(f"{CIFAR_CLASSES[labels[i]]}", fontsize=8)
        axes[i].axis("off")

    # Hide empty subplots
    for j in range(n_images, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        print(f"Saved raw dataset visualization to {outfile}")
    else:
        plt.show()
