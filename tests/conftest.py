"""
Pytest configuration and shared fixtures.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

# Add src to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    batch_size, channels, height, width = 4, 3, 32, 32
    images = torch.randn(batch_size, channels, height, width)
    labels = torch.randint(0, 10, (batch_size,))
    return images, labels


@pytest.fixture
def device():
    """Get available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
