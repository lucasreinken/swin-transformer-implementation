"""
Data transformation and augmentation utilities.
"""

from torchvision import transforms
from typing import Callable
from config import AUGMENTATION_CONFIG
import random


class RandAugment:
    """Custom RandAugment"""

    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.ops = [
            transforms.ColorJitter(brightness=self._mag(m)),
            transforms.ColorJitter(contrast=self._mag(m)),
            transforms.ColorJitter(saturation=self._mag(m)),
            transforms.ColorJitter(hue=self._mag(m, 0.5)),
            transforms.RandomAffine(degrees=self._mag(m, 30)),
            transforms.GaussianBlur(3),
            transforms.RandomPosterize(bits=int(max(1, 8 - self._mag(m) // 4))),
            transforms.RandomSolarize(threshold=self._mag(m, 256)),
            transforms.RandomEqualize(),
        ]

    def _mag(self, m, max_val=1.0):
        return (m / 30) * max_val

    def __call__(self, img):
        ops = random.sample(self.ops, self.n)
        for op in ops:
            img = op(img)
        return img


def get_basic_transforms(img_size: int) -> Callable:
    """Get basic transforms without augmentation."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=AUGMENTATION_CONFIG["mean"],
                std=AUGMENTATION_CONFIG["std"],
            ),
        ]
    )


def get_cifar_training_transforms(img_size: int) -> Callable:
    """Get training transforms for CIFAR datasets."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=AUGMENTATION_CONFIG["mean"], std=AUGMENTATION_CONFIG["std"]
            ),
            transforms.RandomErasing(p=AUGMENTATION_CONFIG["random_erase_prob"]),
        ]
    )


def get_imagenet_training_transforms(img_size: int) -> Callable:
    """Get training transforms for ImageNet-style datasets."""
    return transforms.Compose(
        [
            transforms.Resize(256),  # Upsample for cropping
            transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
            RandAugment(
                n=AUGMENTATION_CONFIG["rand_augment_n"],
                m=AUGMENTATION_CONFIG["rand_augment_m"],
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=AUGMENTATION_CONFIG["mean"], std=AUGMENTATION_CONFIG["std"]
            ),
            transforms.RandomErasing(p=AUGMENTATION_CONFIG["random_erase_prob"]),
        ]
    )

def get_derm_training_transforms(img_size: int) -> Callable:
    """
    Generic training transforms for dermatologic classification datasets.
    """
    
    # 1. Base Geometric Transforms
    transforms_list = [
        transforms.Resize(256),
        transforms.RandomResizedCrop(
            img_size,
            scale=AUGMENTATION_CONFIG.get("crop_scale", (0.7, 1.0)),
        ),
    ]

    # 2. Flips / Rotation (Skin has no orientation)
    if AUGMENTATION_CONFIG.get("horizontal_flip", False):
        transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if AUGMENTATION_CONFIG.get("vertical_flip", False):
        transforms_list.append(transforms.RandomVerticalFlip(p=0.5))

    rotation_deg = AUGMENTATION_CONFIG.get("rotation", 0)
    if rotation_deg > 0:
        transforms_list.append(transforms.RandomRotation(rotation_deg))

    # 3. Color Augmentation
    if AUGMENTATION_CONFIG.get("color_jitter", 0.0) > 0:
        cj = AUGMENTATION_CONFIG["color_jitter"]
        transforms_list.append(
            transforms.ColorJitter(
                brightness=cj,
                contrast=cj,
                saturation=cj,
                hue=0.0  # Keep hue fixed to preserve diagnostic color features
            )
        )

    # 4. Normalization
    transforms_list += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=AUGMENTATION_CONFIG["mean"],
            std=AUGMENTATION_CONFIG["std"],
        ),
    ]

    return transforms.Compose(transforms_list)


def get_validation_transforms(img_size: int) -> Callable:
    """Get validation/test transforms (same for all datasets)."""
    return transforms.Compose(
        [
            transforms.Resize(256),  # Resize shorter side to 256
            transforms.CenterCrop(img_size),  # Center crop to img_size x img_size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=AUGMENTATION_CONFIG["mean"], std=AUGMENTATION_CONFIG["std"]
            ),
        ]
    )


def get_default_transforms(
    dataset: str, img_size: int = 224, is_training: bool = False
) -> Callable:
    """
    Get default transformations for different datasets.
    CIFAR: 32x32, ImageNet: 224x224, ADE20K: 512x512

    Args:
        dataset: Dataset name like 'CIFAR10', 'ImageNet', or 'ADE20K'
        img_size: Target image size for resizing

    Returns:
        A torchvision.transforms.Compose object with the appropriate transformations.
        For segmentation datasets (ADE20K), returns a synchronized transform that
        handles both image and mask.
    """
    
    # Special handling for segmentation datasets
    if dataset == "ADE20K":
        from .segmentation_transforms import ADE20KTransform
        return ADE20KTransform(
            img_size=img_size,
            is_training=is_training,
            mean=AUGMENTATION_CONFIG["mean"],
            std=AUGMENTATION_CONFIG["std"],
        )

    # Use standard transforms for classification datasets
    if not AUGMENTATION_CONFIG["use_augmentation"]:
        return get_basic_transforms(img_size)

    if is_training:
        if dataset in ["CIFAR10", "CIFAR100"]:
            return get_cifar_training_transforms(img_size)
        if dataset in ["SCIN", "SD198"]:
            return get_derm_training_transforms(img_size)
        else:
            return get_imagenet_training_transforms(img_size)
    else:
        return get_validation_transforms(img_size)
