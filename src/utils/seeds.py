"""
Random seed management for reproducible experiments.
"""

import logging
import random
import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducible experiments.

    Args:
        seed: Random seed value.
        deterministic: If True, enables deterministic behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    logger.info(f"Random seeds set to {seed} (deterministic={deterministic})")


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
