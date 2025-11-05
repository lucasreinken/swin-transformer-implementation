"""
Dataset classes for different datasets.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple
from PIL import Image


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

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Reshape to (3, 32, 32) then transpose to (32, 32, 3) for (H, W, C)
        sample = self.data[idx].reshape(3, 32, 32).transpose(1, 2, 0)

        # Convert numpy array to PIL Image
        sample = (sample * 255).astype(np.uint8)  # Scale to [0, 255] for PIL
        sample = Image.fromarray(sample)

        if self.transform:
            sample = self.transform(sample)
        return sample, self.labels[idx]
