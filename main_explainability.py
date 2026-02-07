"""
Main entry point for explainability analysis of Swin Transformer.

This script runs the explainability pipeline to visualize and analyze
attention patterns in Swin Transformer models, comparing W-MSA and SW-MSA.

Usage:
    python main_explainability.py
"""

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import load_data
from src.data.transforms import get_default_transforms
from src.utils.seeds import set_all_seeds, get_worker_init_fn
from src.utils.experiment import setup_run_directory, setup_logging

from config.explainability_config import (
    MODEL_CONFIG,
    DATA_CONFIG,
    VIZ_CONFIG,
    SEED_CONFIG,
    OUTPUT_CONFIG,
)

from src.pipelines.explainability import run_explainability

logger = logging.getLogger(__name__)


def validate_configuration() -> None:
    """Validate explainability configuration parameters."""
    logger.info("Validating explainability configuration...")
    
    # Validate model config
    if not MODEL_CONFIG.get('return_attention_maps', False):
        raise ValueError("MODEL_CONFIG must have 'return_attention_maps': True")
    
    # Validate visualization config
    required_viz_keys = ['num_samples', 'colormap', 'overlay_alpha']
    for key in required_viz_keys:
        if key not in VIZ_CONFIG:
            raise ValueError(f"VIZ_CONFIG missing required key: {key}")
    
    # Validate data config
    if DATA_CONFIG.get('batch_size', 1) != 1:
        logger.warning("Explainability works best with batch_size=1 for individual attention maps")
    
    logger.info("Configuration validation passed!")


def main():
    """Main execution function for explainability pipeline."""
    
    # Set up run directory and logging
    run_dir = setup_run_directory(
        base_dir=OUTPUT_CONFIG.get('base_dir', 'runs'),
        experiment_name=OUTPUT_CONFIG.get('experiment_name', 'explainability')
    )
    setup_logging(run_dir, log_level=logging.INFO)
    
    logger.info("="*80)
    logger.info("SWIN TRANSFORMER EXPLAINABILITY ANALYSIS")
    logger.info("="*80)
    logger.info(f"Run directory: {run_dir}")
    
    # Validate configuration
    validate_configuration()
    
    # Set random seeds for reproducibility
    seed = SEED_CONFIG.get('seed', 42)
    deterministic = SEED_CONFIG.get('deterministic', True)
    set_all_seeds(seed, deterministic=deterministic)
    logger.info(f"Random seed set to {seed} (deterministic={deterministic})")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    
    # Load data
    logger.info("Loading data for explainability analysis...")
    logger.info(f"Dataset: {DATA_CONFIG.get('dataset', 'imagenet')}")
    logger.info(f"Data path: {DATA_CONFIG.get('data_path', 'data')}")
    
    # Get transforms
    _, val_transform = get_default_transforms(
        dataset=DATA_CONFIG.get('dataset', 'imagenet'),
        augmentation_strength=DATA_CONFIG.get('augmentation_strength', 'none'),
        img_size=DATA_CONFIG.get('img_size', 224),
    )
    
    # Load validation data (we'll use validation set for visualization)
    _, val_dataset, _ = load_data(
        dataset_name=DATA_CONFIG.get('dataset', 'imagenet'),
        data_path=DATA_CONFIG.get('data_path', 'data'),
        train_transform=val_transform,
        val_transform=val_transform,
    )
    
    # Create data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=DATA_CONFIG.get('batch_size', 1),
        shuffle=DATA_CONFIG.get('shuffle', False),
        num_workers=DATA_CONFIG.get('num_workers', 4),
        pin_memory=DATA_CONFIG.get('pin_memory', True),
        worker_init_fn=get_worker_init_fn(seed),
    )
    
    logger.info(f"Validation set size: {len(val_dataset)}")
    logger.info(f"Batch size: {DATA_CONFIG.get('batch_size', 1)}")
    
    # Get pretrained weights
    pretrained_weights = MODEL_CONFIG.get('pretrained_weights', None)
    if pretrained_weights:
        logger.info(f"Using pretrained weights: {pretrained_weights}")
    else:
        logger.warning("No pretrained weights specified - using random initialization")
    
    # Run explainability pipeline
    try:
        run_explainability(
            model_config=MODEL_CONFIG,
            pretrained_weights=pretrained_weights,
            data_loader=val_loader,
            viz_config=VIZ_CONFIG,
            device=device,
            run_dir=run_dir,
        )
        
        logger.info("="*80)
        logger.info("EXPLAINABILITY ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info(f"Results saved to: {run_dir}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Explainability pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
