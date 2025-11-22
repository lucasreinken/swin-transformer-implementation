"""
Data loading utilities for different datasets.
"""

import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
import os
from typing import Optional, Tuple, Callable
from pathlib import Path
from torchvision import datasets

from .datasets import CIFAR10Dataset
from .transforms import get_default_transforms

import logging

logger = logging.getLogger(__name__)


def _load_cifar10_data(
    transformation: Callable,
    use_batch_for_val: bool,
    val_batch: int,
    img_size: int,
) -> Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset
]:
    """Load CIFAR-10 dataset with custom batch splitting logic."""
    from .transforms import get_default_transforms

    data_dir = Path("./datasets/cifar-10-batches-py")
    if not data_dir.exists():
        logger.info(f"Data {data_dir} not found. Downloading CIFAR10 ...")
        datasets.CIFAR10(root="./datasets", train=True, download=True)
        datasets.CIFAR10(root="./datasets", train=False, download=True)
        logger.info(f"Downloaded CIFAR10 to {data_dir}")

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Failed to download or locate CIFAR10 data at {data_dir}"
        )

    # Load training data (data_batch 1 to data_batch 5)
    if use_batch_for_val:
        # Use specified batch for validation, others for training
        train_data = []
        train_labels = []
        val_data = []
        val_labels = []

        for i in range(1, 6):
            with open(os.path.join(data_dir, f"data_batch_{i}"), "rb") as f:
                batch = pickle.load(f, encoding="bytes")

            if i == val_batch:
                # This batch goes to validation
                val_data.append(batch[b"data"])
                val_labels = np.array(batch[b"labels"])
            else:
                # These batches go to training
                train_data.append(batch[b"data"])
                train_labels.extend(batch[b"labels"])

        train_data = np.vstack(train_data)
        train_labels = np.array(train_labels)
        val_data = np.vstack(val_data)

    else:
        # Original approach: combine all batches for training
        train_data = []
        train_labels = []
        for i in range(1, 6):
            with open(os.path.join(data_dir, f"data_batch_{i}"), "rb") as f:
                batch = pickle.load(f, encoding="bytes")
                train_data.append(batch[b"data"])
                train_labels.extend(batch[b"labels"])

        train_data = np.vstack(train_data)
        train_labels = np.array(train_labels)

        # Split training data for validation (simple approach)
        total_size = len(train_data)
        val_size = total_size // 6  # Roughly 1/6 for validation
        train_size = total_size - val_size

        # Simple split (not ideal but maintains compatibility)
        val_data = train_data[-val_size:]
        val_labels = train_labels[-val_size:]
        train_data = train_data[:train_size]
        train_labels = train_labels[:train_size]

    # Load test data (always the official test batch)
    with open(os.path.join(data_dir, "test_batch"), "rb") as f:
        test_batch = pickle.load(f, encoding="bytes")
        test_data = test_batch[b"data"]
        test_labels = np.array(test_batch[b"labels"])

    # Get correct transformations for train/val/test
    train_transform = get_default_transforms(
        "CIFAR10", img_size=img_size, is_training=True
    )
    val_transform = get_default_transforms(
        "CIFAR10", img_size=img_size, is_training=False
    )
    test_transform = get_default_transforms(
        "CIFAR10", img_size=img_size, is_training=False
    )

    # Create datasets
    train_dataset = CIFAR10Dataset(train_data, train_labels, transform=train_transform)
    val_dataset = CIFAR10Dataset(val_data, val_labels, transform=val_transform)
    test_dataset = CIFAR10Dataset(test_data, test_labels, transform=test_transform)

    return train_dataset, val_dataset, test_dataset


def _load_cifar100_data(
    transformation: Callable,
) -> Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset
]:
    """Load CIFAR-100 dataset with validation split."""
    train_dataset = datasets.CIFAR100(
        root="./datasets", train=True, transform=transformation, download=True
    )
    test_dataset = datasets.CIFAR100(
        root="./datasets", train=False, transform=transformation, download=True
    )

    # Split training data for validation
    total_size = len(train_dataset)
    val_size = total_size // 6  # Roughly 1/6 for validation
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    return train_dataset, val_dataset, test_dataset


def _load_imagenet_data(
    transformation: Callable,
) -> Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset
]:
    """Load ImageNet dataset."""
    train_dir = Path("./datasets/imagenet/train")
    val_dir = Path("./datasets/imagenet/val")

    if not Path(train_dir).exists() or not Path(val_dir).exists():
        raise FileNotFoundError(
            "ImageNet data not found in ./datasets. Please download and extract it."
        )

    train_dataset = datasets.ImageFolder(train_dir, transform=transformation)
    val_dataset = datasets.ImageFolder(val_dir, transform=transformation)

    # For ImageNet, we'll use the provided val split, but create a smaller validation set
    # and use part of training for additional validation if needed
    val_size = min(len(val_dataset), 50000)  # Use up to 50K for validation
    if len(val_dataset) > val_size:
        val_dataset, _ = torch.utils.data.random_split(
            val_dataset,
            [val_size, len(val_dataset) - val_size],
            generator=torch.Generator().manual_seed(42),
        )

    # Use the official ImageNet validation set as our test set
    test_dataset = val_dataset

    return train_dataset, val_dataset, test_dataset


def _create_dataloaders(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    worker_init_fn,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoader objects with consistent configuration."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        worker_init_fn=worker_init_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader, test_loader


def _apply_dataset_limits(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    n_train: Optional[int],
    n_test: Optional[int],
) -> Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset
]:
    """Apply size limits to datasets if specified."""
    if n_train is not None and n_train < len(train_dataset):
        train_dataset = torch.utils.data.Subset(train_dataset, range(n_train))

    if n_test is not None and n_test < len(val_dataset):
        val_dataset = torch.utils.data.Subset(val_dataset, range(n_test))

    if n_test is not None and n_test < len(test_dataset):
        test_dataset = torch.utils.data.Subset(test_dataset, range(n_test))

    return train_dataset, val_dataset, test_dataset


def load_data(
    dataset: str = "CIFAR10",
    transformation: Optional[callable] = None,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    use_batch_for_val: bool = False,
    val_batch: int = 5,
    batch_size: int = 32,
    num_workers: int = 4,
    root: str = "./datasets",
    img_size: int = 224,
    worker_init_fn=None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load data and return train/val/test DataLoaders.

    For CIFAR-10, when use_batch_for_val=True, uses one training batch for validation
    to keep the official test set untouched.

    Args:
        dataset: Dataset name
        transformation: Optional transform for data.
        n_train: Number of training samples to use.
        n_test: Number of test samples to use.
        use_batch_for_val: If True, use one CIFAR-10 training batch for validation.
        val_batch: Which training batch to use for validation (1-5).
        batch_size: Batch size for DataLoader.
        num_workers: Number of workers for DataLoader.
        root: Root directory for dataset.
        img_size: Target image size.

    Returns:
        Tuple of (train_generator, val_generator, test_generator)
    """
    # Set default transformation if not provided
    if transformation is None:
        transformation = get_default_transforms(dataset, img_size)

    # Load dataset-specific data
    if dataset == "CIFAR10":
        train_dataset, val_dataset, test_dataset = _load_cifar10_data(
            transformation, use_batch_for_val, val_batch, img_size
        )
    elif dataset == "CIFAR100":
        train_dataset, val_dataset, test_dataset = _load_cifar100_data(transformation)
    elif dataset == "ImageNet":
        train_dataset, val_dataset, test_dataset = _load_imagenet_data(transformation)
    else:
        raise ValueError(f"Dataset {dataset} not supported.")

    # Apply dataset size limits if specified
    train_dataset, val_dataset, test_dataset = _apply_dataset_limits(
        train_dataset, val_dataset, test_dataset, n_train, n_test
    )

    # Create DataLoaders
    return _create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        num_workers,
        worker_init_fn,
    )
