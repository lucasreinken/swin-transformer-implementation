#!/usr/bin/env python3
"""
Quick test script for ImageNet data loading with subset.
"""

import sys
import os

sys.path.append("/home/georg/Desktop/Machine Learning Project")

from src.data.dataloader import load_data
from src.data.transforms import get_default_transforms


def test_imagenet_loading():
    """Test ImageNet data loading with small subset."""
    print("Testing ImageNet data loading...")

    # Use small subset for testing
    n_train = 1000
    n_test = 100

    # Get transforms
    transform = get_default_transforms("ImageNet", img_size=224, is_training=True)

    try:
        train_loader, val_loader, test_loader = load_data(
            dataset="ImageNet",
            transformation=transform,
            n_train=n_train,
            n_test=n_test,
            batch_size=32,
            num_workers=0,  # Avoid multiprocessing issues locally
            root="/mnt/datasets/imagenet/2012",  # This will fail locally, but shows the call
            img_size=224,
        )

        print(f"Train loader: {len(train_loader)} batches")
        print(f"Val loader: {len(val_loader)} batches")
        print(f"Test loader: {len(test_loader)} batches")

        # Try to get one batch
        for batch in train_loader:
            print(f"Batch shape: {batch[0].shape}, labels shape: {batch[1].shape}")
            break

        print("ImageNet loading test passed!")

    except Exception as e:
        print(f"ImageNet loading failed: {e}")
        return False

    return True


if __name__ == "__main__":
    test_imagenet_loading()
