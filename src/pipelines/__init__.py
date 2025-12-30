"""
Training pipelines for different training modes.

- linear_probing: Compare reference (TIMM) vs custom model with pretrained weights
- from_scratch: Train custom model from random initialization
- segmentation: Train Swin-T + UperNet for semantic segmentation on ADE20K
"""

from .linear_probing import run_linear_probing
from .from_scratch import run_from_scratch
from .segmentation import run_segmentation_pipeline

__all__ = [
    "run_linear_probing",
    "run_from_scratch",
    "run_segmentation_pipeline",
]
