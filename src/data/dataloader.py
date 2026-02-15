"""
Data loading utilities for different datasets.
"""

import torch
from torch.utils.data import DataLoader
import pickle
import zipfile
import numpy as np
import os
import json
import shutil
from typing import Callable, Tuple, Optional, Dict, List
from pathlib import Path
from torchvision import datasets
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from datasets import load_dataset as hf_load_dataset
from datasets.utils.logging import set_verbosity_error, disable_progress_bar
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import logging as hf_logging

from config import DATA_CONFIG, SEED_CONFIG

from .datasets import CIFAR10Dataset, ADE20KDataset, SCINDataset, SD198Dataset
from .transforms import get_default_transforms
from ..utils.seeds import set_worker_seeds

import logging

set_verbosity_error()
disable_progress_bar()
hf_logging.set_verbosity_error()

logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

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
        generator=torch.Generator().manual_seed(SEED_CONFIG.get("seed", 42)),
    )
    _, val_dataset = torch.utils.data.random_split(
        val_full_dataset,
        [train_size, val_size],
       generator=torch.Generator().manual_seed(SEED_CONFIG.get("seed", 42)),
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


def _download_scin(data_dir: Path):
    """
    Force download SCIN into a controlled cache directory.
    """
    logger.info(f"Downloading SCIN dataset to {data_dir}...")
    data_dir.mkdir(parents=True, exist_ok=True)

    # force_download ensures full materialization
    hf_load_dataset(
        "google/scin",
        split="train",
        cache_dir=str(data_dir),
        download_mode="force_redownload"
    )

    logger.info("SCIN dataset download complete.")

def _load_scin_data(
    train_transformation,
    val_transformation,
    root: str,
    val_frac: float = 0.10,
    test_frac: float = 0.0,
):
    """
    Load SCIN with explicit storage control.
    """

    shared_path = Path("/home/space/datasets/scin")
    user_path = Path.home() / "datasets" / "scin"
    local_path = Path(root) / "scin"

    data_root = None

    # 1) Shared storage
    if shared_path.exists():
        data_root = shared_path
        logger.info(f"Using shared SCIN dataset from {data_root}")

    # 2) User directory
    elif user_path.exists():
        data_root = user_path
        logger.info(f"Using user SCIN dataset from {data_root}")

    # 3) Local project directory
    elif local_path.exists():
        data_root = local_path
        logger.info(f"Using local SCIN dataset from {data_root}")

    # 4) Download if none found
    else:
        data_root = user_path
        logger.info(f"SCIN dataset not found. Downloading to {data_root}...")
        _download_scin(data_root)

    # Load without re-downloading
    hf = hf_load_dataset(
        "google/scin",
        split="train",
        cache_dir=str(data_root),
        download_mode="reuse_dataset_if_exists"
    )

    # ---- Case split (no leakage) ----

    n_cases = len(hf)
    idx = np.arange(n_cases)

    if test_frac > 0:
        train_idx, test_idx = train_test_split(idx, test_size=test_frac, random_state=SEED_CONFIG.get("seed", 42))
    else:
        train_idx, test_idx = idx, np.array([], dtype=int)

    if val_frac > 0:
        train_idx, val_idx = train_test_split(train_idx, test_size=val_frac, random_state=SEED_CONFIG.get("seed", 42))
    else:
        val_idx = np.array([], dtype=int)

    hf_train = hf.select(train_idx.tolist())
    hf_val = hf.select(val_idx.tolist()) if len(val_idx) > 0 else hf.select(train_idx[:1].tolist())
    hf_test = hf.select(test_idx.tolist()) if len(test_idx) > 0 else hf.select(val_idx.tolist())

    train_dataset = SCINDataset(hf_train, transform=train_transformation)
    val_dataset = SCINDataset(hf_val, transform=val_transformation)
    test_dataset = SCINDataset(hf_test, transform=val_transformation)

    logger.info(
        f"Loaded SCIN from {data_root}: "
        f"cases train={len(hf_train)}, val={len(hf_val)}, test={len(hf_test)} | "
        f"images train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset


# ---------------------------------------------------------------------
# SD-198 helper functions
# ---------------------------------------------------------------------
def _looks_like_class_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    for f in p.rglob("*"):
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            return True
    return False


def _find_imagefolder_root(root: Path) -> Path:
    """
    Find a directory that looks like an ImageFolder root:
    it has many subdirs, each containing images.
    """
    best = None
    best_score = -1

    for d in root.rglob("*"):
        if not d.is_dir():
            continue

        subdirs = [x for x in d.iterdir() if x.is_dir()]
        if len(subdirs) < 20:
            continue

        # score = how many of the first N subdirs contain images
        good = 0
        probe = subdirs[:200]
        for sd in probe:
            if _looks_like_class_dir(sd):
                good += 1

        if good > best_score:
            best_score = good
            best = d

    if best is None:
        raise RuntimeError(
            "Could not auto-detect SD-198 ImageFolder root. "
            "Inspect sd198_raw/ and set the path manually."
        )

    return best


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    """
    Try symlink first (fast, no duplication). If not possible, copy.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    # If already exists, do nothing
    if dst.exists():
        return

    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _materialize_splits(
    imagefolder_root: Path,
    out_root: Path,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> None:
    """
    Create on-disk splits:
      out_root/train/<class>/*
      out_root/val/<class>/*
      out_root/test/<class>/*
    using per-class stratified splitting (deterministic).
    """

    marker = out_root / ".splits_ok"
    if marker.exists():
        logger.info(f"SD-198 splits already prepared at {out_root}")
        return

    class_dirs = sorted([p for p in imagefolder_root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class folders found in {imagefolder_root}")

    logger.info(f"Preparing SD-198 splits from {len(class_dirs)} classes...")

    # Ensure clean target (avoid partial split states)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        imgs = sorted([p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}])

        if len(imgs) == 0:
            continue

        idx = np.arange(len(imgs))

        # test split
        if test_frac > 0:
            idx_trainval, idx_test = train_test_split(
                idx, test_size=test_frac, random_state=seed, shuffle=True
            )
        else:
            idx_trainval, idx_test = idx, np.array([], dtype=int)

        # val split from trainval
        if val_frac > 0:
            idx_train, idx_val = train_test_split(
                idx_trainval, test_size=val_frac, random_state=seed, shuffle=True
            )
        else:
            idx_train, idx_val = idx_trainval, np.array([], dtype=int)

        # Link/copy files into split dirs
        for split_name, split_idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
            for i in split_idx:
                src = imgs[int(i)]
                dst = out_root / split_name / cls_name / src.name
                _safe_link_or_copy(src, dst)

    # Write marker + metadata
    meta = {
        "imagefolder_root": str(imagefolder_root),
        "val_frac": float(val_frac),
        "test_frac": float(test_frac),
        "seed": int(seed),
    }
    (out_root / "split_meta.json").write_text(json.dumps(meta, indent=2))
    marker.write_text("ok")
    logger.info(f"SD-198 splits prepared at {out_root}")


def _download_sd198(data_dir: Path) -> None:
    """
    Download + extract SD-198 zip into:
      data_dir/sd198_raw/...
    """
    logger.info(f"Downloading SD-198 zip to {data_dir}...")
    data_dir.mkdir(parents=True, exist_ok=True)

    # detect zip file name in repo
    repo_files = list_repo_files(repo_id="resyhgerwshshgdfghsdfgh/SD-198", repo_type="dataset")
    zip_name = None
    for cand in ("sd-198.zip", "sd198.zip", "SD-198.zip", "sd_198.zip"):
        if cand in repo_files:
            zip_name = cand
            break
    if zip_name is None:
        # fallback: first .zip in repo
        zips = [f for f in repo_files if f.lower().endswith(".zip")]
        if not zips:
            raise FileNotFoundError(
                f"No zip found in HF repo {"resyhgerwshshgdfghsdfgh/SD-198"}. Files: {repo_files[:20]}"
            )
        zip_name = zips[0]

    logger.info(f"Detected SD-198 zip in repo: {zip_name}")

    zip_path = hf_hub_download(
        repo_id="resyhgerwshshgdfghsdfgh/SD-198",
        filename=zip_name,
        repo_type="dataset",
    )

    extract_root = data_dir / "sd198_raw"
    extract_root.mkdir(parents=True, exist_ok=True)
    marker = extract_root / ".extracted_ok"

    if not marker.exists():
        logger.info(f"Extracting SD-198 zip to {extract_root}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_root)
        marker.write_text("ok")
        logger.info("Extraction complete.")
    else:
        logger.info(f"Extraction marker found. Reusing existing extraction at {extract_root}.")


def _load_sd198_data(
    train_transformation: Callable,
    val_transformation: Callable,
    root: str,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
) -> Tuple[
    "torch.utils.data.Dataset",
    "torch.utils.data.Dataset",
    "torch.utils.data.Dataset",
]:
    """
    Load SD-198 for classification (filesystem-based).

    Then:
    - extract zip to <data_root>/sd198_raw
    - auto-detect ImageFolder root (198 class dirs)
    - create split folders under <data_root>/sd198/{train,val,test}/<class>
    - return SD198Dataset(train/val/test)
    """

    shared_path = Path("/home/space/datasets/sd198")
    user_path = Path.home() / "datasets" / "sd198"
    local_path = Path(root) / "sd198"

    if shared_path.exists():
        data_root = shared_path
        logger.info(f"Using shared SD-198 dataset from {data_root}")
    elif user_path.exists():
        data_root = user_path
        logger.info(f"Using user SD-198 dataset from {data_root}")
    elif local_path.exists():
        data_root = local_path
        logger.info(f"Using local SD-198 dataset from {data_root}")
    else:
        data_root = user_path
        logger.info(f"SD-198 dataset not found. Downloading to {data_root}...")
        _download_sd198(data_root)

    # Ensure extracted
    _download_sd198(data_root)

    extract_root = data_root / "sd198_raw"
    imagefolder_root = _find_imagefolder_root(extract_root)
    logger.info(f"Detected SD-198 ImageFolder root: {imagefolder_root}")

    # Materialize split folders expected by SD198Dataset
    split_root = data_root / "sd198"
    seed = int(SEED_CONFIG.get("seed", 42))
    _materialize_splits(
        imagefolder_root=imagefolder_root,
        out_root=split_root,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )

    # Build datasets
    train_dataset = SD198Dataset(
        root=str(split_root),
        split="train",
        transform=train_transformation,
    )
    if val_frac > 0.0:
        val_dataset = SD198Dataset(
            root=str(split_root),
            split="val",
            transform=val_transformation,
            class_to_idx=train_dataset.class_to_idx,
        )
    else:
        val_dataset = None
    if test_frac > 0.0:
        test_dataset = SD198Dataset(
            root=str(split_root),
            split="test",
            transform=val_transformation,
            class_to_idx=train_dataset.class_to_idx,
        )
    else:
        test_dataset = None
    
    logger.info(
        f"Loaded SD-198 from {split_root}: "
        f"train={len(train_dataset)}, val=0, test={len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset


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
            generator=torch.Generator().manual_seed(SEED_CONFIG.get("seed", 42)),
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
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            worker_init_fn=set_worker_seeds if num_workers > 0 else None,
        )
    else:
        val_loader = None

    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            worker_init_fn=set_worker_seeds if num_workers > 0 else None,
        )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader


def _subset(dataset, n, stratified):
    """
    Return a subset of size n from a dataset,
    optionally preserving class distribution when stratified.
    """
    if not stratified:
        subset, _ = torch.utils.data.random_split(
            dataset,
            [n, len(dataset) - n],
            generator=torch.Generator().manual_seed(SEED_CONFIG.get("seed", 42)),
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
        random_state=SEED_CONFIG.get("seed", 42),
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
    elif dataset == "SCIN":
        # For SSL, stratified is not used (no labels).
        train_dataset, val_dataset, test_dataset = _load_scin_data(
            transformation,
            val_transformation,
            root,
            val_frac=DATA_CONFIG.get("val_frac", 0.10),
            test_frac=DATA_CONFIG.get("test_frac", 0.00),
        )
    elif dataset == "SD198":
        train_dataset, val_dataset, test_dataset = _load_sd198_data(
            transformation,
            val_transformation,
            root,
            val_frac=DATA_CONFIG.get("val_frac", 0.10),
            test_frac=DATA_CONFIG.get("test_frac", 0.10),
        )
    else:
        raise ValueError(f"Dataset {dataset} not supported.")

    if n_train or n_test:
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
