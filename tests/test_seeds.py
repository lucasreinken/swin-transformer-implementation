"""
Tests for random seed management utilities.
"""

import random
import numpy as np
import torch
import pytest

from src.utils.seeds import set_random_seeds, get_worker_init_fn


class TestSeedManagement:
    """Test cases for seed management functions."""

    def test_set_random_seeds_basic(self):
        """Test basic seed setting functionality."""
        seed = 123
        set_random_seeds(seed)

        # Check that seeds are set by getting sequences of random numbers
        random_seq1 = [random.random() for _ in range(3)]
        np_seq1 = [np.random.random() for _ in range(3)]
        torch_seq1 = [torch.rand(1).item() for _ in range(3)]

        set_random_seeds(seed)  # Reset seed
        random_seq2 = [random.random() for _ in range(3)]
        np_seq2 = [np.random.random() for _ in range(3)]
        torch_seq2 = [torch.rand(1).item() for _ in range(3)]

        # Should be identical sequences
        assert random_seq1 == random_seq2
        assert np_seq1 == np_seq2
        assert torch_seq1 == torch_seq2

    def test_set_random_seeds_reproducibility(self):
        """Test that setting the same seed produces identical results."""
        seed = 456

        # First run
        set_random_seeds(seed)
        random_vals_1 = [random.random() for _ in range(5)]
        np_vals_1 = [np.random.random() for _ in range(5)]
        torch_vals_1 = [torch.rand(1).item() for _ in range(5)]

        # Second run with same seed
        set_random_seeds(seed)
        random_vals_2 = [random.random() for _ in range(5)]
        np_vals_2 = [np.random.random() for _ in range(5)]
        torch_vals_2 = [torch.rand(1).item() for _ in range(5)]

        # Should be identical
        assert random_vals_1 == random_vals_2
        assert np_vals_1 == np_vals_2
        assert torch_vals_1 == torch_vals_2

    def test_cuda_seeds_if_available(self):
        """Test CUDA seed setting if CUDA is available."""
        if torch.cuda.is_available():
            seed = 789
            set_random_seeds(seed)

            # Check CUDA seeds are set
            assert torch.cuda.initial_seed() == seed
        else:
            # Skip test if no CUDA
            pytest.skip("CUDA not available")

    def test_worker_init_fn(self):
        """Test worker initialization function for DataLoader."""
        seed = 111
        worker_id = 5

        init_fn = get_worker_init_fn(seed)
        init_fn(worker_id)

        # Check that worker seed is set correctly by getting sequences
        random_seq1 = [random.random() for _ in range(3)]
        np_seq1 = [np.random.random() for _ in range(3)]
        torch_seq1 = [torch.rand(1).item() for _ in range(3)]

        init_fn(worker_id)  # Reset with same worker seed
        random_seq2 = [random.random() for _ in range(3)]
        np_seq2 = [np.random.random() for _ in range(3)]
        torch_seq2 = [torch.rand(1).item() for _ in range(3)]

        # Should be identical sequences for same worker
        assert random_seq1 == random_seq2
        assert np_seq1 == np_seq2
        assert torch_seq1 == torch_seq2

    def test_deterministic_mode(self):
        """Test deterministic mode setting."""
        if torch.cuda.is_available():
            set_random_seeds(42, deterministic=True)
            assert torch.backends.cudnn.deterministic == True
            assert torch.backends.cudnn.benchmark == False
        else:
            pytest.skip("CUDA not available for deterministic testing")
