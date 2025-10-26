import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
from torchvision.datasets import CIFAR10
import os
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union
from pathlib import Path


class CIFAR10Dataset(Dataset):
    """
    Custom dataset for CIFAR-10 data.

    Args:
        data: Numpy array of image data.
        labels: Numpy array of labels.
        transform: Optional transform to apply to samples.
    """

    def __init__(
        self, data: np.ndarray, labels: np.ndarray, transform: Optional[callable] = None
    ):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Reshape to (3, 32, 32) then transpose to (32, 32, 3) for (H, W, C)
        sample = self.data[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        if self.transform:
            sample = self.transform(sample)
        return sample, self.labels[idx]


def load_data(
    dataset: str = "CIFAR10",
    transformation: Optional[callable] = None,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-10 data and return train/test DataLoaders.

    Args:
        dataset: Dataset name
        transformation: Optional transform for data.
        n_train: Number of training samples to use.
        n_test: Number of test samples to use.
        batch_size: Batch size for DataLoader.

    Returns:
        Tuple of (train_generator, test_generator)
    """
    data_dir = f"./data/cifar-10-batches-py"
    if not os.path.exists(data_dir):
        print(f"Data {data_dir} not found. Downloading {dataset} ...")
        CIFAR10(root="./data", train=True, download=True)
        CIFAR10(root="./data", train=False, download=True)
        print(f"Downloaded {dataset} to {data_dir}")

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Failed to download or locate {dataset} data at {data_dir}"
        )

    # Load training data (data_batch 1 to data_batch 5)
    train_data = []
    train_labels = []
    for i in range(1, 6):
        with open(os.path.join(data_dir, f"data_batch_{i}"), "rb") as f:
            batch = pickle.load(f, encoding="bytes")
            train_data.append(batch[b"data"])
            train_labels.extend(batch[b"labels"])

    train_data = np.vstack(train_data)
    train_labels = np.array(train_labels)

    # Load test data
    with open(os.path.join(data_dir, "test_batch"), "rb") as f:
        test_batch = pickle.load(f, encoding="bytes")
        test_data = test_batch[b"data"]
        test_labels = np.array(test_batch[b"labels"])

    # Create datasets (transform applied in __getitem__)
    transform = transformation if transformation else ToTensor()
    train_dataset = CIFAR10Dataset(train_data, train_labels, transform=transform)
    test_dataset = CIFAR10Dataset(test_data, test_labels, transform=transform)

    # Split train set if specified
    total_size = len(train_dataset)
    train_size = n_train if n_train else int(0.8 * total_size)
    test_size = n_test if n_test else len(test_dataset)

    train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(
        test_dataset, range(min(test_size, len(test_dataset)))
    )

    # Generators with lazy loading
    train_generator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_generator = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_generator, test_generator


def show(
    x: Union[np.ndarray, torch.Tensor],
    y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    outfile: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    cmap: str = "gray",
):
    """
    Visualize a batch of images in a grid.

    Args:
        x: Tensor of shape [batch_size, 3, H, W]
        y: Optional tensor of labels corresponding to x
        outfile: Optional file path to save the figure
        figsize: Tuple for figure size (width, height)
        cmap: Colormap for grayscale images
    """
    if isinstance(x, torch.Tensor):
        x = x.numpy()

    batch_size = x.shape[0]
    height, width = x.shape[2], x.shape[3]

    # Create a grid
    grid_size = int(np.ceil(np.sqrt(batch_size)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()

    for i in range(batch_size):
        # Convert CHW to HWC and ensure values are in [0, 1]
        img = x[i].transpose(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {y[i].item()}" if "y" in locals() else f"Img {i}")
        axes[i].axis("off")

    # Remove empty subplots
    for j in range(batch_size, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()


# Test the function
if __name__ == "__main__":
    train_generator, test_generator = load_data(
        dataset="CIFAR10", n_train=40000, n_test=10000, batch_size=32
    )
    for x, y in train_generator:
        print(
            f"Sample shape: {x.shape}, Type: {x.dtype}, Label: {y[0]}, Min/Max: {x.min().item()}/{x.max().item()}"
        )
        show(x, y=y, outfile="visualization.png")
        break
