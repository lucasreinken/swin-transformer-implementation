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
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from config import DATA_CONFIG

from .datasets import CIFAR10Dataset, ADE20KDataset
from .transforms import get_default_transforms
from ..utils.seeds import set_worker_seeds

import logging

logger = logging.getLogger(__name__)


def _load_cifar10_data(
    train_transformation: Callable,
    val_transformation: Callable,
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
    train_transform = train_transformation
    val_transform = val_transformation
    test_transform = val_transformation

    # Create datasets
    train_dataset = CIFAR10Dataset(train_data, train_labels, transform=train_transform)
    val_dataset = CIFAR10Dataset(val_data, val_labels, transform=val_transform)
    test_dataset = CIFAR10Dataset(test_data, test_labels, transform=test_transform)

    return train_dataset, val_dataset, test_dataset


def _load_cifar100_data(
    train_transformation: Callable,
    val_transformation: Callable,
) -> Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset
]:
    """Load CIFAR-100 dataset with validation split."""
    train_dataset = datasets.CIFAR100(
        root="./datasets", train=True, transform=train_transformation, download=True
    )
    test_dataset = datasets.CIFAR100(
        root="./datasets", train=False, transform=val_transformation, download=True
    )

    # Create validation dataset with val transform
    val_full_dataset = datasets.CIFAR100(
        root="./datasets", train=True, transform=val_transformation, download=False
    )

    # Split training data for validation
    total_size = len(train_dataset)
    val_size = total_size // 6  # Roughly 1/6 for validation
    train_size = total_size - val_size

    train_dataset, _ = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    _, val_dataset = torch.utils.data.random_split(
        val_full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    return train_dataset, val_dataset, test_dataset


def _download_ade20k(data_dir: Path) -> None:
    """
    Download and extract ADE20K dataset.
    
    Args:
        data_dir: Directory to download and extract dataset to
    """
    import urllib.request
    import zipfile
    import shutil
    
    logger.info(f"Downloading ADE20K dataset to {data_dir}...")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Official ADE20K download URL
    url = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
    zip_path = data_dir / "ADEChallengeData2016.zip"
    
    try:
        # Download with progress
        def _progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            if count % 50 == 0:  # Print every 50 blocks
                logger.info(f"Download progress: {percent}%")
        
        urllib.request.urlretrieve(url, zip_path, _progress_hook)
        logger.info("Download completed. Extracting...")
        
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # ADE20K extracts to ADEChallengeData2016/
        extracted_dir = data_dir / "ADEChallengeData2016"
        if extracted_dir.exists():
            # Move contents to data_dir
            for item in extracted_dir.iterdir():
                shutil.move(str(item), str(data_dir / item.name))
            extracted_dir.rmdir()
        
        # Clean up zip file
        zip_path.unlink()
        logger.info(f"ADE20K dataset successfully downloaded and extracted to {data_dir}")
        
    except Exception as e:
        logger.error(f"Failed to download ADE20K: {e}")
        # Clean up partial downloads
        if zip_path.exists():
            zip_path.unlink()
        raise


def _load_ade20k_data(
    train_transformation: Callable,
    val_transformation: Callable,
    root: str,
) -> Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset
]:
    """
    Load ADE20K dataset with automatic download fallback.
    
    Checks in order:
    1. Shared storage: /home/space/datasets/ade20k
    2. User directory: ~/datasets/ade20k
    3. Auto-download to user directory if not found
    
    Args:
        train_transformation: Transform for training data (should handle image+mask)
        val_transformation: Transform for validation data (should handle image+mask)
        root: Root directory hint (not strictly used, we check multiple locations)
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    
    Note:
        For ADE20K segmentation, transformations must be synchronized transforms
        that process both image and mask together to maintain spatial correspondence.
    """
    # Check multiple possible locations
    shared_path = Path("/home/space/datasets/ade20k")
    user_path = Path.home() / "datasets" / "ade20k"
    local_path = Path(root) / "ade20k"
    
    data_root = None
    
    # Check shared storage first (no download needed)
    if shared_path.exists() and (shared_path / "images").exists():
        data_root = shared_path
        logger.info(f"Using shared ADE20K dataset from {data_root}")
    
    # Check user directory
    elif user_path.exists() and (user_path / "images").exists():
        data_root = user_path
        logger.info(f"Using user ADE20K dataset from {data_root}")
    
    # Check local path (for local development)
    elif local_path.exists() and (local_path / "images").exists():
        data_root = local_path
        logger.info(f"Using local ADE20K dataset from {data_root}")
    
    # Download to user directory if not found anywhere
    else:
        data_root = user_path
        logger.info(f"ADE20K dataset not found. Downloading to {data_root}...")
        _download_ade20k(data_root)
    
    # Create datasets with synchronized transforms
    train_dataset = ADE20KDataset(
        root=data_root,
        split='training',
        transform=train_transformation,
    )
    
    val_dataset = ADE20KDataset(
        root=data_root,
        split='validation',
        transform=val_transformation,
    )
    
    # For ADE20K, use validation set as test set (standard practice)
    test_dataset = ADE20KDataset(
        root=data_root,
        split='validation',
        transform=val_transformation,
    )
    
    logger.info(
        f"Loaded ADE20K data from {data_root}: "
        f"train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
    )
    
    return train_dataset, val_dataset, test_dataset


# src/data/dataloader.py
# ... existing code ...


def _load_imagenet_data(
    transformation: Callable,
    val_transformation: Callable,
    root: str,
) -> Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset
]:
    """Load ImageNet dataset using ImageFolder for squashed filesystem."""
    logger.info(f"Starting ImageNet data loading from root: {root}")

    root_path = Path(root)
    logger.info(f"Resolved root path: {root_path}")
    logger.info(f"Root path exists: {root_path.exists()}")
    logger.info(f"Root path is dir: {root_path.is_dir()}")

    if not root_path.exists():
        raise FileNotFoundError(f"Root path {root} does not exist.")

    contents = list(root_path.iterdir()) if root_path.is_dir() else []
    logger.info(f"Contents of {root}: {[str(p) for p in contents]}")

    # Use ImageFolder for squashed filesystem structure
    train_dir = root_path / "train_set"
    val_dir = root_path / "val_set"
    logger.info(f"Expected train_dir: {train_dir}, exists: {train_dir.exists()}")
    logger.info(f"Expected val_dir: {val_dir}, exists: {val_dir.exists()}")

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"ImageNet data not found in {root}. Expected 'train_set' and 'val_set' subfolders."
        )

    train_dataset = datasets.ImageFolder(train_dir, transform=transformation)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transformation)
    logger.info(
        f"Loaded ImageNet data from {root}: train={len(train_dataset)}, val={len(val_dataset)}"
    )

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


# ... existing code ...


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
        worker_init_fn=set_worker_seeds if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        worker_init_fn=set_worker_seeds if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        worker_init_fn=set_worker_seeds if num_workers > 0 else None,
    )

    return train_loader, val_loader, test_loader


def _subset(dataset, n, stratified, seed=42):
    """
    Return a subset of size n from a dataset,
    optionally preserving class distribution when stratified.
    """
    if not stratified:
        subset, _ = torch.utils.data.random_split(
            dataset,
            [n, len(dataset) - n],
            generator=torch.Generator().manual_seed(seed),
        )
        return subset

    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("Stratified split requires dataset.targets")

    idx = list(range(len(dataset)))
    idx_sub, _ = train_test_split(
        idx,
        train_size=n,
        stratify=targets,
        random_state=seed,
    )
    return Subset(dataset, idx_sub)


def _apply_dataset_limits(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    n_train: Optional[int],
    n_test: Optional[int],
    stratified: bool
) -> Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset
]:
    """Apply size limits to datasets if specified."""
    if n_train is not None and n_train < len(train_dataset):
        train_dataset = _subset(train_dataset, n_train, stratified)

    if n_test is not None and n_test < len(val_dataset):
        val_dataset = _subset(val_dataset, n_test, stratified)

    if n_test is not None and n_test < len(test_dataset):
        test_dataset = _subset(test_dataset, n_test, stratified)

    if stratified:
        logger.info("Dataset limits applied (stratified sampling enabled)")
    else:
        logger.info("Dataset limits applied")

    return train_dataset, val_dataset, test_dataset


def load_data(
    dataset: str = "CIFAR10",
    transformation: Optional[callable] = None,
    val_transformation: Optional[callable] = None,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    stratified: bool = False,
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
        transformation: Optional transform for training data.
        val_transformation: Optional transform for validation/test data.
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
        transformation = get_default_transforms(dataset, img_size, is_training=True)
    if val_transformation is None:
        val_transformation = get_default_transforms(
            dataset, img_size, is_training=False
        )

    # Load dataset-specific data
    if dataset == "CIFAR10":
        train_dataset, val_dataset, test_dataset = _load_cifar10_data(
            transformation, val_transformation, use_batch_for_val, val_batch, img_size
        )
    elif dataset == "CIFAR100":
        train_dataset, val_dataset, test_dataset = _load_cifar100_data(
            transformation, val_transformation
        )
    elif dataset == "ImageNet":
        train_dataset, val_dataset, test_dataset = _load_imagenet_data(
            transformation, val_transformation, root
        )
    elif dataset == "ADE20K":
        train_dataset, val_dataset, test_dataset = _load_ade20k_data(
            transformation, val_transformation, root
        )
    else:
        raise ValueError(f"Dataset {dataset} not supported.")

    # Apply dataset size limits if specified
    train_dataset, val_dataset, test_dataset = _apply_dataset_limits(
        train_dataset, val_dataset, test_dataset, n_train, n_test, stratified
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
