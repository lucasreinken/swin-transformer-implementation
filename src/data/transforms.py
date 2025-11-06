"""
Data transformation and augmentation utilities.
"""


from torchvision import transforms
from typing import Callable
from config import AUGMENTATION_CONFIG
import random
from src.utils.preprocess import PatchEmbed, normalize_patches
import torch
def get_cifar100_patch_transform(img_size: int, patch_size: int = 4, embed_dim: int = 48) -> Callable:
    """
    Transform for CIFAR-100: ToTensor, Normalize, PatchEmbed, Normalize patches.
    Output: [batch, num_patches, embed_dim]
    """
    patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
    def transform_fn(img):
        x = transforms.ToTensor()(img)
        x = transforms.Normalize(
            mean=AUGMENTATION_CONFIG["mean"], std=AUGMENTATION_CONFIG["std"]
        )(x)
        x = x.unsqueeze(0) if x.dim() == 3 else x  # [C, H, W] -> [1, C, H, W]
        x = patch_embed(x)  # [1, num_patches, embed_dim]
        x = normalize_patches(x)
        return x.squeeze(0)  # [num_patches, embed_dim]
    return transform_fn


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


def get_validation_transforms(img_size: int) -> Callable:
    """Get validation/test transforms (same for all datasets)."""
    return transforms.Compose(
        [
            transforms.Resize(img_size),
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
    CIFAR: 32x32, ImageNet: 224x224

    Args:
        dataset: Dataset name like 'CIFAR10' or 'ImageNet'
        img_size: Target image size for resizing

    Returns:
        A torchvision.transforms.Compose object with the appropriate transformations.
    """

    if dataset == "CIFAR100":
        # Always use patch embedding for CIFAR-100 for Swin prep
        return get_cifar100_patch_transform(img_size)

    if not AUGMENTATION_CONFIG["use_augmentation"]:
        return get_basic_transforms(img_size)

    if is_training:
        if dataset == "CIFAR10":
            return get_cifar_training_transforms(img_size)
        else:
            return get_imagenet_training_transforms(img_size)
    else:
        return get_validation_transforms(img_size)
