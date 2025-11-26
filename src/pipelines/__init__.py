"""
Training pipelines for different training modes.

- linear_probing: Compare reference (TIMM) vs custom model with pretrained weights
- from_scratch: Train custom model from random initialization
"""

from .linear_probing import run_linear_probing
from .from_scratch import run_from_scratch

__all__ = ["run_linear_probing", "run_from_scratch"]
