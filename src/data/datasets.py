"""
Dataset classes for different datasets.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple
from PIL import Image
import os
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


class ADE20KDataset(Dataset):
    """
    Custom dataset for ADE20K semantic segmentation.
    
    Args:
        root: Root directory of ADE20K dataset
        split: 'training' or 'validation'
        transform: Optional synchronized transform to apply to both image and mask
                   Should be a callable that takes (image, mask) and returns (image_tensor, mask_tensor)
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'training',
        transform: Optional[callable] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        
        # ADE20K structure: images/training/, annotations/training/
        self.images_dir = self.root / 'images' / split
        self.annotations_dir = self.root / 'annotations' / split
        
        # Get all image files
        self.images = sorted(list(self.images_dir.glob('*.jpg')))
        
        if len(self.images) == 0:
            raise RuntimeError(
                f"No images found in {self.images_dir}. "
                f"Please ensure ADE20K dataset is downloaded correctly."
            )
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load annotation (segmentation mask)
        # ADE20K annotations have same name but .png extension
        ann_path = self.annotations_dir / (img_path.stem + '.png')
        mask = Image.open(ann_path)
        
        # Apply synchronized transform (transforms both image and mask)
        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            # Default: just convert to tensors without resizing
            # This will cause batching errors - transform should always be provided
            from torchvision.transforms import functional as TF
            image = TF.to_tensor(image)
            mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        
        return image, mask
