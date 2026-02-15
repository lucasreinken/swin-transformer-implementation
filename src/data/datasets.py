"""
Dataset classes for different datasets.
"""

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import numpy as np
from typing import Optional, Tuple, List, Any
from PIL import Image
import os
from pathlib import Path
from io import BytesIO

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

class SCINDataset(Dataset):
    """
    SCIN dataset wrapper that FLATTENS multi-image cases into individual images.

    - Each original SCIN row is a "case" with potentially multiple image columns.
    - This dataset yields (image_tensor, dummy_label) so it works with existing loaders.

    Args:
        hf_dataset: Hugging Face datasets.Dataset for SCIN split (e.g., load_dataset("google/scin")["train"])
        transform: Callable that takes a PIL image and returns a tensor
        return_case_meta: If True, returns (image, case_index, image_key) for debugging
    """

    def __init__(
        self,
        hf_dataset,
        transform: Optional[callable] = None,
        return_case_meta: bool = False,
        image_cols: Optional[List[str]] = None,
    ):
        self.ds = hf_dataset
        self.transform = transform
        self.return_case_meta = return_case_meta

        # Detect image columns if not provided
        if image_cols is None:
            image_cols = []
            for col, feat in self.ds.features.items():
                if "Image" in str(type(feat)):
                    image_cols.append(col)
            if not image_cols:
                # fallback: any column name containing "image"
                image_cols = [c for c in self.ds.column_names if "image" in c.lower()]

        if not image_cols:
            raise ValueError("No SCIN image columns detected. Pass image_cols explicitly.")

        self.image_cols = image_cols

        # Build an index mapping: flat_idx -> (case_idx, image_key)
        self.index: List[Tuple[int, str]] = []
        for case_idx in range(len(self.ds)):
            ex = self.ds[case_idx]
            for k in self.image_cols:
                if k not in ex:
                    continue
                if ex[k] is None:
                    continue
                # Some HF Image objects exist but may fail decode later, keep and skip on __getitem__ if needed.
                self.index.append((case_idx, k))

        if len(self.index) == 0:
            raise RuntimeError("SCIN index is empty. No images found after flattening.")

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _to_pil(img: Any) -> Optional[Image.Image]:
        # PIL
        if isinstance(img, Image.Image):
            return img.convert("RGB")

        # HF dict-like: {"bytes":..., "path":...}
        if isinstance(img, dict):
            if img.get("bytes") is not None:
                return Image.open(BytesIO(img["bytes"])).convert("RGB")
            if img.get("path") is not None:
                return Image.open(img["path"]).convert("RGB")

        # numpy array
        if isinstance(img, np.ndarray):
            if img.dtype == object:
                return None
            try:
                return Image.fromarray(img).convert("RGB")
            except Exception:
                return None

        # best-effort
        try:
            arr = np.array(img)
            if arr.dtype == object:
                return None
            return Image.fromarray(arr).convert("RGB")
        except Exception:
            return None

    def __getitem__(self, idx: int):
        case_idx, image_key = self.index[idx]
        ex = self.ds[case_idx]
        pil = self._to_pil(ex[image_key])

        # If decode fails, retry by skipping forward until a decodable item is found.
        if pil is None:
            j = idx
            while j + 1 < len(self.index):
                j += 1
                case_idx, image_key = self.index[j]
                ex = self.ds[case_idx]
                pil = self._to_pil(ex[image_key])
                if pil is not None:
                    break
            if pil is None:
                raise RuntimeError("Failed to decode any SCIN image from this index onward.")

        if self.transform:
            img_tensor = self.transform(pil)
        else:
            img_tensor = TF.to_tensor(pil)

        if self.return_case_meta:
            return img_tensor, case_idx, image_key

        return img_tensor
