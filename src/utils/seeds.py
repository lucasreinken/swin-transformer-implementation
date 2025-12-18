"""
Random seed management for reproducible experiments.
"""

import logging
import random
import numpy as np
import torch
import os

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set seeds for all relevant components to ensure reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, enables deterministic operations (slower but fully reproducible)
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # DataLoader workers
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    # Environment variables for additional determinism
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # Additional deterministic settings
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    logger.info(f"✅ All seeds set to {seed} (deterministic={deterministic})")


def set_random_seeds(seed: int = 42, deterministic: bool = False) -> None:
    """
    Legacy function - redirects to set_all_seeds for backward compatibility.
    """
    set_all_seeds(seed, deterministic)


def set_worker_seeds(worker_id: int):
    """
    Set seeds for DataLoader workers to ensure reproducible data loading.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_worker_init_fn(seed: int):
    """
    Get worker initialization function for DataLoader reproducibility.
    """

    def worker_init_fn(worker_id: int):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn
