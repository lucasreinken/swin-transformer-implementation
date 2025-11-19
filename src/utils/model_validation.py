"""
Model validation utilities for comparing custom implementations with
pretrained models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from typing import Dict, Tuple, Optional
from pathlib import Path

from src.utils.load_weights import (
    load_pretrained_reference,
    transfer_weights,
)

logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Validates custom model implementations against pretrained models.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    @torch.no_grad()
    def evaluate_model(
        self, model: nn.Module, dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on validation data.
        """
        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        for images, targets in dataloader:
            images, targets = images.to(self.device), targets.to(self.device)
            outputs = model(images)

            # Top-1 accuracy
            _, pred1 = outputs.topk(1, dim=1)
            correct_top1 += pred1.eq(targets.view_as(pred1)).sum().item()

            # Top-5 accuracy
            _, pred5 = outputs.topk(5, dim=1)
            correct_top5 += pred5.eq(targets.view(-1, 1).expand_as(pred5)).sum().item()

            total += targets.size(0)

        return {
            "top1_accuracy": 100.0 * correct_top1 / total,
            "top5_accuracy": 100.0 * correct_top5 / total,
            "total_samples": total,
        }

    def compare_models(
        self,
        custom_model: nn.Module,
        pretrained_model: nn.Module,
        dataloader: DataLoader,
        run_dir: Path,
    ) -> Dict[str, Dict[str, float]]:
        """Compare custom model performance with pretrained model."""

        # Evaluate both models
        custom_results = self.evaluate_model(custom_model, dataloader)
        pretrained_results = self.evaluate_model(pretrained_model, dataloader)

        # Calculate differences
        top1_diff = (
            pretrained_results["top1_accuracy"] - custom_results["top1_accuracy"]
        )
        top5_diff = (
            pretrained_results["top5_accuracy"] - custom_results["top5_accuracy"]
        )

        results = {
            "custom_model": custom_results,
            "pretrained_model": pretrained_results,
            "differences": {"top1_diff": top1_diff, "top5_diff": top5_diff},
        }

        # Log results
        logger.info("=== MODEL COMPARISON RESULTS ===")
        logger.info(
            f"Pretrained Model  - Top-1: {pretrained_results['top1_accuracy']:.2f}%, Top-5: {pretrained_results['top5_accuracy']:.2f}%"
        )
        logger.info(
            f"Custom Model      - Top-1: {custom_results['top1_accuracy']:.2f}%, Top-5: {custom_results['top5_accuracy']:.2f}%"
        )
        logger.info(
            f"Differences       - Top-1: {top1_diff:.2f}%, Top-5: {top5_diff:.2f}%"
        )

        # Save results to run directory
        results_file = run_dir / "model_validation_results.json"
        import json

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Validation results saved to {results_file}")

        return results

    def validate_model_implementation(
        self,
        custom_model: nn.Module,
        val_dataloader: DataLoader,
        run_dir: Path,
        validation_config: dict,
    ) -> Optional[Dict]:
        """Complete validation pipeline for custom model implementation."""

        if not validation_config.get("enable_validation", False):
            logger.info("Model validation disabled in config - skipping")
            return None

        logger.info("Starting model implementation validation...")

        # Load pretrained reference model
        pretrained_model = load_pretrained_reference(
            validation_config.get("pretrained_model", "swin_tiny_patch4_window7_224"),
            device=self.device,
        )

        if pretrained_model is None:
            logger.warning("Cannot perform validation - pretrained model unavailable")
            return None

        # Transfer weights if requested
        if validation_config.get("transfer_weights", True):
            logger.info("Transferring weights from pretrained to custom model...")
            transfer_stats = transfer_weights(
                custom_model,
                pretrained_model,
                encoder_only=False,
                device=self.device
                )
            logger.info(f"Weight transfer completed: {transfer_stats}")

        # Create validation dataset
        validation_samples = validation_config.get("validation_samples", 1000)
        if len(val_dataloader.dataset) > validation_samples:
            logger.info(f"Using subset of {validation_samples} samples for validation")
            from torch.utils.data import Subset

            val_subset = Subset(val_dataloader.dataset, range(validation_samples))
            val_loader_limited = DataLoader(
                val_subset,
                batch_size=val_dataloader.batch_size,
                shuffle=False,
                num_workers=val_dataloader.num_workers,
                pin_memory=getattr(val_dataloader, "pin_memory", False),
            )
        else:
            val_loader_limited = val_dataloader

        # Compare models
        comparison_results = self.compare_models(
            custom_model, pretrained_model, val_loader_limited, run_dir
        )

        return comparison_results
