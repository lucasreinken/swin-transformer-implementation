#!/usr/bin/env python3
"""
Standalone script for zero-shot ImageNet validation.
Compares custom Swin Transformer implementation against timm pretrained model.

Usage:
    python validate_model.py --model swin_tiny_patch4_window7_224 --samples 1000
    python validate_model.py --model swin_base_patch4_window7_224 --samples 5000
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import logging

from src.utils.model_validation import ModelValidator
from src.utils.load_weights import load_pretrained_reference, transfer_weights
from src.models import (
    swin_tiny_patch4_window7_224,
    swin_small_patch4_window7_224,
    swin_base_patch4_window7_224,
    swin_large_patch4_window7_224,
)
from src.utils.seeds import set_random_seeds
from src.utils.experiment import setup_run_directory, setup_logging


# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_model_by_name(model_name: str, device: torch.device):
    """Create custom model by variant name."""
    model_map = {
        "swin_tiny_patch4_window7_224": swin_tiny_patch4_window7_224,
        "swin_small_patch4_window7_224": swin_small_patch4_window7_224,
        "swin_base_patch4_window7_224": swin_base_patch4_window7_224,
        "swin_large_patch4_window7_224": swin_large_patch4_window7_224,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_map.keys())}")
    
    model = model_map[model_name](num_classes=1000)
    return model.to(device)


def main(args):
    # Set random seed
    set_random_seeds(42, deterministic=False)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Setup output directory
    run_dir = setup_run_directory()
    setup_logging(run_dir)
    logger.info(f"Results will be saved to: {run_dir}")
    
    # Load ImageNet validation data
    logger.info(f"Loading ImageNet validation data (using {args.samples} samples)...")
    try:
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader, Subset
        
        # ImageNet normalization
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        
        # Try to load ImageNet dataset - check both local and cluster paths
        imagenet_paths = [
            Path("/home/space/datasets/imagenet/2012/val_set"),  # Cluster path (TU Berlin)
            Path("/home/space/datasets/imagenet"),                # Alternative cluster path
            Path("./datasets/imagenet"),                          # Local path
            Path("/data/imagenet"),                               # Generic cluster path
        ]
        
        imagenet_path = None
        for path in imagenet_paths:
            # Check if path exists and contains class folders
            if path.exists() and any(path.iterdir()):
                # For direct val_set path, use it directly
                if path.name == "val_set":
                    imagenet_path = path
                    logger.info(f"Found ImageNet validation set at: {imagenet_path}")
                # For parent paths, check for val subdirectory
                elif (path / "val").exists():
                    imagenet_path = path
                    logger.info(f"Found ImageNet dataset at: {imagenet_path}")
                break
        
        if imagenet_path is None:
            logger.error(f"ImageNet dataset not found. Checked paths:")
            for path in imagenet_paths:
                logger.error(f"  - {path}")
            logger.info("Expected structure: <path>/val/n01440764/... or direct val_set/n01440764/...")
            return
        
        # Use the found path directly if it's val_set, otherwise append 'val'
        if imagenet_path.name == "val_set":
            val_dataset_path = imagenet_path
        else:
            val_dataset_path = imagenet_path / "val"
        
        val_dataset = datasets.ImageFolder(
            val_dataset_path,
            transform=val_transform
        )
        
        # Use subset if specified
        if args.samples and args.samples < len(val_dataset):
            indices = list(range(args.samples))
            val_dataset = Subset(val_dataset, indices)
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        logger.info(f"Loaded {len(val_dataset)} validation samples")
        
    except Exception as e:
        logger.error(f"Error loading ImageNet data: {e}")
        logger.info("Make sure ImageNet dataset is available in ./datasets/imagenet/")
        return
    
    # Create custom model
    logger.info(f"Creating custom model: {args.model}")
    custom_model = get_model_by_name(args.model, device)
    logger.info(f"Custom model parameters: {sum(p.numel() for p in custom_model.parameters()):,}")
    
    # Load pretrained reference model
    logger.info(f"Loading timm pretrained model: {args.model}")
    pretrained_model = load_pretrained_reference(args.model, device)
    
    if pretrained_model is None:
        logger.error("Failed to load pretrained model from timm")
        return
    
    logger.info(f"Pretrained model parameters: {sum(p.numel() for p in pretrained_model.parameters()):,}")
    
    # Transfer weights if requested
    if args.transfer_weights:
        logger.info("Transferring weights from pretrained to custom model...")
        stats = transfer_weights(
            custom_model,
            pretrained_model,
            encoder_only=False,
            device=str(device)
        )
        logger.info(f"Transferred {stats['transferred']} layers")
        if stats['missing'] > 0:
            logger.warning(f"Missing: {stats['missing']} layers")
        if stats['size_mismatches'] > 0:
            logger.warning(f"Size mismatches: {stats['size_mismatches']} layers")
    
    # Run validation
    logger.info("Starting zero-shot validation...")
    validator = ModelValidator(device=str(device))
    
    results = validator.compare_models(
        custom_model=custom_model,
        pretrained_model=pretrained_model,
        dataloader=val_loader,
        run_dir=run_dir,
        model_name=args.model
    )
    
    # Print summary
    logger.info("")
    logger.info("="*70)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Samples: {results['custom_model']['total_samples']}")
    logger.info(f"Validation: {'PASSED ✓' if results['validation_status']['passed'] else 'FAILED ✗'}")
    logger.info(f"Top-1 Δ: {results['differences']['top1_diff']:+.3f}% (|Δ| = {results['differences']['top1_diff_abs']:.3f}%)")
    logger.info(f"Top-5 Δ: {results['differences']['top5_diff']:+.3f}% (|Δ| = {results['differences']['top5_diff_abs']:.3f}%)")
    logger.info("="*70)
    logger.info("")
    logger.info(f"Results saved to: {run_dir}")
    logger.info(f"  - {run_dir / 'zero_shot_imagenet.log'}")
    logger.info(f"  - {run_dir / 'model_validation_results.json'}")
    logger.info(f"  - results/zero_shot_imagenet.log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-shot ImageNet validation for Swin Transformer"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="swin_tiny_patch4_window7_224",
        choices=[
            "swin_tiny_patch4_window7_224",
            "swin_small_patch4_window7_224",
            "swin_base_patch4_window7_224",
            "swin_large_patch4_window7_224",
        ],
        help="Model variant to validate"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of validation samples to use (default: 5000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)"
    )
    parser.add_argument(
        "--transfer-weights",
        action="store_true",
        default=True,
        help="Transfer weights from pretrained to custom model (default: True)"
    )
    parser.add_argument(
        "--no-transfer-weights",
        action="store_false",
        dest="transfer_weights",
        help="Do not transfer weights (test random initialization)"
    )
    
    args = parser.parse_args()
    main(args)
