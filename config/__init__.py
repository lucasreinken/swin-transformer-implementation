"""
Configuration module for the ML pipeline.
"""

import os

# Dataset selection - choose one dataset
# DATASET = "cifar10"
DATASET = "cifar100"
# DATASET = "imagenet"


# Data root configuration - smart detection for cluster environments
def _get_data_root():
    """Smart data root detection that prefers local datasets first."""
    # Priority order: local project -> mounted .sqfs -> user home -> other
    possible_roots = [
        "./datasets",  # Local project datasets (highest priority)
        "/home/space/datasets",  # Mounted .sqfs datasets (read-only)
        "/home/space/datasets-sqfs",  # Alternative .sqfs location
        os.path.expanduser("~/datasets"),  # User home (writable)
    ]


def _get_data_root():
    """Smart data root detection that prefers mounted .sqfs datasets."""
    # Priority order: mounted .sqfs -> user home -> local
    possible_roots = [
        "/home/space/datasets",  # Mounted .sqfs datasets (read-only)
        "/home/space/datasets-sqfs",  # Alternative .sqfs location
        os.path.expanduser("~/datasets"),  # User home (writable)
        "./datasets",  # Local directory
    ]

    # Check for existing datasets in all locations
    for root in possible_roots:
        if os.path.exists(root):
            # Check if it contains expected dataset files
            if DATASET.lower() in ["cifar10", "cifar100"]:
                # Handle CIFAR dataset naming correctly
                if DATASET.lower() == "cifar10":
                    dataset_name = "cifar-10-batches-py"
                    tar_name = "cifar-10-python.tar.gz"
                elif DATASET.lower() == "cifar100":
                    dataset_name = "cifar-100-batches-py"
                    tar_name = "cifar-100-python.tar.gz"

                dataset_dir = os.path.join(root, dataset_name)
                tar_file = os.path.join(root, tar_name)

                # Check for extracted dataset or tar file
                if os.path.exists(dataset_dir):
                    print(f"Found existing {DATASET} dataset at: {dataset_dir}")
                    return root
                elif os.path.exists(tar_file):
                    print(
                        f"Found {DATASET} tar file at: {tar_file} (will extract if needed)"
                    )
                    return root
            elif DATASET.lower() == "imagenet":
                # Check for ImageNet structure
                if os.path.exists(os.path.join(root, "train")) or os.path.exists(
                    os.path.join(root, "val")
                ):
                    print(f"Found existing ImageNet dataset at: {root}")
                    return root

    # Check for .sqfs files that might need mounting
    sqfs_dir = "/home/space/datasets-sqfs"
    if os.path.exists(sqfs_dir):
        # Look for .sqfs files
        for file in os.listdir(sqfs_dir):
            if file.endswith(".sqfs"):
                dataset_name = file.replace(".sqfs", "").lower()
                if dataset_name in [DATASET.lower(), f"cifar{DATASET[-1]}0"]:
                    print(f"Found {DATASET} .sqfs file: {os.path.join(sqfs_dir, file)}")
                    print("Note: You may need to mount this .sqfs file first")
                    # Return the sqfs directory as the root
                    return sqfs_dir

    # Fall back to first writable location
    for root in possible_roots[3:]:  # Skip read-only paths (./datasets, /home/space/*)
        try:
            os.makedirs(root, exist_ok=True)
            print(f"Using writable directory for {DATASET}: {root}")
            return root
        except (OSError, PermissionError):
            continue

    fallback = possible_roots[3]  # Default to user home
    print(f"Falling back to: {fallback}")
    return fallback


DATA_ROOT = _get_data_root()


def check_and_mount_sqfs():
    """Check for .sqfs files and offer to mount them if needed."""
    sqfs_dir = "/home/space/datasets-sqfs"
    mount_base = "/home/space/datasets"

    if not os.path.exists(sqfs_dir):
        return

    # Look for relevant .sqfs files
    for file in os.listdir(sqfs_dir):
        if file.endswith(".sqfs"):
            dataset_name = file.replace(".sqfs", "").lower()
            if dataset_name in [DATASET.lower(), f"cifar{DATASET[-1]}0"]:
                sqfs_path = os.path.join(sqfs_dir, file)
                mount_point = os.path.join(mount_base, dataset_name)

                # Check if already mounted
                if os.path.exists(mount_point) and os.listdir(mount_point):
                    print(f"✓ {dataset_name} already mounted at {mount_point}")
                else:
                    print(f"📦 Found {dataset_name} .sqfs file: {sqfs_path}")
                    print(
                        f"   To mount: sudo mount {sqfs_path} {mount_point} -t squashfs -o loop"
                    )
                    print(
                        f"   Or run: python3 -c \"from config import mount_sqfs_dataset; mount_sqfs_dataset('{sqfs_path}', '{mount_point}')\""
                    )


# Check for .sqfs files on import
check_and_mount_sqfs()


def mount_sqfs_dataset(sqfs_path, mount_point):
    """Helper function to mount a .sqfs dataset file."""
    import subprocess

    try:
        # Create mount point if it doesn't exist
        os.makedirs(mount_point, exist_ok=True)

        # Mount the .sqfs file (requires sudo or appropriate permissions)
        cmd = ["sudo", "mount", sqfs_path, mount_point, "-t", "squashfs", "-o", "loop"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Successfully mounted {sqfs_path} to {mount_point}")
            return True
        else:
            print(f"Failed to mount {sqfs_path}: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error mounting {sqfs_path}: {e}")
        return False


def _load_config():
    """Load the appropriate config based on DATASET environment variable."""
    global AUGMENTATION_CONFIG, DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG
    global VIZ_CONFIG, SEED_CONFIG, SCHEDULER_CONFIG, VALIDATION_CONFIG, SWIN_CONFIG

    if DATASET == "cifar10":
        from .cifar10_config import (
            AUGMENTATION_CONFIG,
            DATA_CONFIG,
            MODEL_CONFIG,
            TRAINING_CONFIG,
            VIZ_CONFIG,
            SEED_CONFIG,
            SCHEDULER_CONFIG,
            VALIDATION_CONFIG,
            SWIN_CONFIG,
        )
    elif DATASET == "cifar100":
        from .cifar100_config import (
            AUGMENTATION_CONFIG,
            DATA_CONFIG,
            MODEL_CONFIG,
            TRAINING_CONFIG,
            VIZ_CONFIG,
            SEED_CONFIG,
            SCHEDULER_CONFIG,
            VALIDATION_CONFIG,
            SWIN_CONFIG,
        )
    elif DATASET == "imagenet":
        from .imagenet_config import (
            AUGMENTATION_CONFIG,
            DATA_CONFIG,
            MODEL_CONFIG,
            TRAINING_CONFIG,
            VIZ_CONFIG,
            SEED_CONFIG,
            SCHEDULER_CONFIG,
            VALIDATION_CONFIG,
            SWIN_CONFIG,
        )
    else:
        raise ValueError(
            f"Unknown dataset: {DATASET}. Choose from: cifar10, cifar100, imagenet"
        )

    # Override data root based on environment
    DATA_CONFIG["root"] = DATA_ROOT


# Load the config
_load_config()

__all__ = [
    "AUGMENTATION_CONFIG",
    "DATA_CONFIG",
    "MODEL_CONFIG",
    "TRAINING_CONFIG",
    "VIZ_CONFIG",
    "SEED_CONFIG",
    "SCHEDULER_CONFIG",
    "VALIDATION_CONFIG",
    "SWIN_CONFIG",
]
