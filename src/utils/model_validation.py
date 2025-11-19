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
from src.training import train_one_epoch, evaluate_model

logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Validates custom model implementations against pretrained models.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def train_linear_head(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
    ) -> nn.Module:
        """
        Train only the linear classification head of a model (linear probing).

        Args:
            model: Model with frozen encoder and trainable head
            train_dataloader: Training data
            val_dataloader: Validation data
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization

        Returns:
            Trained model
        """
        logger.info(f"Training linear head for {num_epochs} epochs...")

        # Freeze all parameters except the head
        for name, param in model.named_parameters():
            if "head" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Training {trainable_params:,} parameters in linear head")

        # Setup training components
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
        )

        model.train()

        # Debug: Check the first batch to understand tensor shapes
        first_batch = True

        for epoch in range(num_epochs):
            running_loss = 0.0

            for batch_idx, (inputs, labels) in enumerate(train_dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if first_batch:
                    logger.info(
                        f"Debug - Input shape: {inputs.shape}, Label shape: {labels.shape}"
                    )
                    first_batch = False

                optimizer.zero_grad()
                outputs = model(inputs)

                if batch_idx == 0 and epoch == 0:
                    logger.info(
                        f"Debug - Model output type: {type(outputs)}, shape: {outputs.shape if hasattr(outputs, 'shape') else 'no shape'}"
                    )

                # Handle potential shape issues with TIMM models
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Take first output if tuple

                # Ensure outputs are the right shape [batch_size, num_classes]
                if outputs.dim() != 2:
                    logger.error(
                        f"Unexpected output shape: {outputs.shape}, expected 2D tensor"
                    )
                    outputs = outputs.view(outputs.size(0), -1)

                # Ensure labels are 1D [batch_size]
                if labels.dim() != 1:
                    logger.error(
                        f"Unexpected label shape: {labels.shape}, expected 1D tensor"
                    )
                    labels = labels.view(-1)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(train_dataloader)

            # Validate
            val_loss, val_accuracy, _ = evaluate_model(
                model, val_dataloader, criterion, self.device
            )

            logger.info(
                f"TIMM Training Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_accuracy:.2f}%"
            )

        logger.info("TIMM linear head training completed")
        return model

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

            # Handle potential shape issues with TIMM models (same as training)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Take first output if tuple

            # Ensure outputs are the right shape [batch_size, num_classes]
            if outputs.dim() != 2:
                outputs = outputs.view(outputs.size(0), -1)

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
        train_dataloader: DataLoader = None,
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

        # Replace the classification head to match our task
        # Get the feature dimension from the original head
        if hasattr(pretrained_model, "head"):
            feature_dim = pretrained_model.head.in_features
            # Replace with CIFAR-100 head (100 classes)
            pretrained_model.head = nn.Linear(feature_dim, 100).to(self.device)
            logger.info(
                f"Replaced TIMM classification head: {feature_dim} -> 100 classes"
            )
        elif hasattr(pretrained_model, "classifier"):
            feature_dim = pretrained_model.classifier.in_features
            pretrained_model.classifier = nn.Linear(feature_dim, 100).to(self.device)
            logger.info(f"Replaced TIMM classifier head: {feature_dim} -> 100 classes")

        # Note: Your custom model already has ImageNet pretrained weights from main training
        # TIMM model now also has ImageNet weights + new 100-class head
        # Both models should have the same pretrained encoder weights already
        logger.info("Both models loaded with ImageNet pretrained encoder weights")
        logger.info("Custom model: weights loaded during main training")
        logger.info("TIMM model: weights loaded with pretrained=True")

        # Train TIMM model's linear head if training data provided
        if train_dataloader is not None:
            logger.info("Training TIMM model's linear head...")
            pretrained_model = self.train_linear_head(
                pretrained_model,
                train_dataloader,
                val_dataloader,
                num_epochs=10,  # Match your training
                learning_rate=0.001,
            )
        else:
            logger.warning(
                "No training data provided - TIMM model head will be untrained"
            )

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
