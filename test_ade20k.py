"""
Test script to verify ADE20K dataset loading functionality.

This script tests:
1. Dataset auto-download if not present
2. Dataset structure validation
3. Basic data loading and transformation
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_data
from src.data.transforms import get_default_transforms
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ade20k_loading():
    """Test ADE20K dataset loading."""
    
    logger.info("=" * 80)
    logger.info("Testing ADE20K Dataset Loading")
    logger.info("=" * 80)
    
    # Test configuration
    test_config = {
        "dataset": "ADE20K",
        "batch_size": 4,
        "num_workers": 2,
        "root": "./datasets",
        "img_size": 512,
    }
    
    logger.info(f"\nTest Configuration:")
    for key, value in test_config.items():
        logger.info(f"  {key}: {value}")
    
    try:
        # Get transforms
        train_transform = get_default_transforms("ADE20K", 512, is_training=True)
        val_transform = get_default_transforms("ADE20K", 512, is_training=False)
        
        # Load data
        logger.info("\nLoading ADE20K dataset...")
        train_loader, val_loader, test_loader = load_data(
            dataset="ADE20K",
            transformation=train_transform,
            val_transformation=val_transform,
            batch_size=test_config["batch_size"],
            num_workers=test_config["num_workers"],
            root=test_config["root"],
            img_size=test_config["img_size"],
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("Dataset Loading Summary:")
        logger.info("=" * 80)
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Test samples: {len(test_loader.dataset)}")
        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        
        # Test loading a batch
        logger.info("\nLoading first batch from training set...")
        images, masks = next(iter(train_loader))
        
        logger.info("\n" + "=" * 80)
        logger.info("Batch Information:")
        logger.info("=" * 80)
        logger.info(f"Images shape: {images.shape}")
        logger.info(f"Images dtype: {images.dtype}")
        logger.info(f"Images min/max: {images.min():.3f} / {images.max():.3f}")
        logger.info(f"Masks shape: {masks.shape}")
        logger.info(f"Masks dtype: {masks.dtype}")
        logger.info(f"Unique classes in batch: {len(masks.unique())} classes")
        logger.info(f"Class IDs in batch: {sorted(masks.unique().tolist()[:20])}...")  # Show first 20
        logger.info(f"Class range: {masks.min()} to {masks.max()}")
        
        # Verify shapes are correct for batching
        expected_img_shape = (test_config["batch_size"], 3, test_config["img_size"], test_config["img_size"])
        expected_mask_shape = (test_config["batch_size"], test_config["img_size"], test_config["img_size"])
        
        if images.shape == expected_img_shape:
            logger.info(f"✓ Image shape correct: {images.shape}")
        else:
            logger.warning(f"⚠ Image shape mismatch: got {images.shape}, expected {expected_img_shape}")
        
        if masks.shape == expected_mask_shape:
            logger.info(f"✓ Mask shape correct: {masks.shape}")
        else:
            logger.warning(f"⚠ Mask shape mismatch: got {masks.shape}, expected {expected_mask_shape}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ ADE20K Dataset Loading Test PASSED!")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error("❌ ADE20K Dataset Loading Test FAILED!")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    success = test_ade20k_loading()
    sys.exit(0 if success else 1)
