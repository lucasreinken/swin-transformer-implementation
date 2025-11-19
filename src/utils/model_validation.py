"""
Model validation utilities for comparing custom implementations with
pretrained models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import logging
from typing import Dict, Optional
from pathlib import Path
import json
from datetime import datetime
import time
import shutil

from .load_weights import load_pretrained_reference, transfer_weights

logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Validates custom model implementations against pretrained models.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    @torch.no_grad()
    def evaluate_model(
        self, 
        model: nn.Module, 
        dataloader: DataLoader, 
        log_progress: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on validation data with Top-1 and Top-5 accuracy.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader with validation data
            log_progress: Whether to log progress during evaluation
            
        Returns:
            Dictionary with accuracy metrics and timing information
        """
        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        batch_count = 0
        start_time = time.time()

        try:
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
                batch_count += 1
                
                if log_progress and batch_count % 50 == 0:
                    logger.info(f"Evaluated {total} samples...")

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise

        elapsed_time = time.time() - start_time
        
        return {
            "top1_accuracy": 100.0 * correct_top1 / total,
            "top5_accuracy": 100.0 * correct_top5 / total,
            "total_samples": total,
            "evaluation_time": elapsed_time,
            "samples_per_second": total / elapsed_time if elapsed_time > 0 else 0,
        }

    def compare_models(
        self,
        custom_model: nn.Module,
        pretrained_model: nn.Module,
        dataloader: DataLoader,
        run_dir: Path,
        model_name: str = "swin_base_patch4_window7_224",
    ) -> Dict:
        """
        Compare custom model performance with pretrained model.
        
        Args:
            custom_model: Custom implementation to validate
            pretrained_model: Reference pretrained model
            dataloader: Validation data loader
            run_dir: Directory to save results
            model_name: Name of the model for logging
            
        Returns:
            Dictionary with comparison results and validation status
        """

        logger.info("Evaluating timm pretrained model...")
        pretrained_results = self.evaluate_model(pretrained_model, dataloader, log_progress=True)
        
        logger.info("Evaluating custom implementation...")
        custom_results = self.evaluate_model(custom_model, dataloader, log_progress=True)

        # Calculate differences
        top1_diff = pretrained_results["top1_accuracy"] - custom_results["top1_accuracy"]
        top5_diff = pretrained_results["top5_accuracy"] - custom_results["top5_accuracy"]
        top1_diff_abs = abs(top1_diff)
        top5_diff_abs = abs(top5_diff)

        # Validation status
        validation_passed = top1_diff_abs <= 0.1 and top5_diff_abs <= 0.1

        results = {
            "validation_info": {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": model_name,
                "dataset": "ImageNet-1K (validation subset)",
                "device": str(self.device),
            },
            "custom_model": custom_results,
            "pretrained_model": pretrained_results,
            "differences": {
                "top1_diff": top1_diff,
                "top5_diff": top5_diff,
                "top1_diff_abs": top1_diff_abs,
                "top5_diff_abs": top5_diff_abs,
            },
            "validation_status": {
                "passed": validation_passed,
                "threshold": 0.1,
                "criterion": "Top-1 and Top-5 accuracy within 0.1%",
            },
        }

        # Log results
        logger.info("" )
        logger.info("="*70)
        logger.info("    ZERO-SHOT IMAGENET VALIDATION RESULTS")
        logger.info("="*70)
        logger.info(f"Model: {model_name}")
        logger.info(f"Dataset: ImageNet-1K validation subset ({custom_results['total_samples']} samples)")
        logger.info(f"Device: {self.device}")
        logger.info("-"*70)
        logger.info("TIMM Pretrained Model:")
        logger.info(f"  Top-1 Accuracy: {pretrained_results['top1_accuracy']:.3f}%")
        logger.info(f"  Top-5 Accuracy: {pretrained_results['top5_accuracy']:.3f}%")
        logger.info(f"  Inference Time: {pretrained_results['evaluation_time']:.2f}s")
        logger.info(f"  Speed: {pretrained_results['samples_per_second']:.1f} samples/s")
        logger.info("-"*70)
        logger.info("Custom Implementation:")
        logger.info(f"  Top-1 Accuracy: {custom_results['top1_accuracy']:.3f}%")
        logger.info(f"  Top-5 Accuracy: {custom_results['top5_accuracy']:.3f}%")
        logger.info(f"  Inference Time: {custom_results['evaluation_time']:.2f}s")
        logger.info(f"  Speed: {custom_results['samples_per_second']:.1f} samples/s")
        logger.info("-"*70)
        logger.info("Differences:")
        logger.info(f"  Top-1 Δ: {top1_diff:+.3f}% (|Δ| = {top1_diff_abs:.3f}%)")
        logger.info(f"  Top-5 Δ: {top5_diff:+.3f}% (|Δ| = {top5_diff_abs:.3f}%)")
        logger.info("-"*70)
        status_symbol = "✓" if validation_passed else "✗"
        logger.info(f"Validation: {status_symbol} {'PASSED' if validation_passed else 'FAILED'}")
        logger.info(f"  Criterion: Accuracy within ±0.1%")
        logger.info(f"  Top-1: {'PASS' if top1_diff_abs <= 0.1 else 'FAIL'} ({top1_diff_abs:.3f}% ≤ 0.1%)")
        logger.info(f"  Top-5: {'PASS' if top5_diff_abs <= 0.1 else 'FAIL'} ({top5_diff_abs:.3f}% ≤ 0.1%)")
        logger.info("="*70)
        logger.info("")

        # Save JSON results
        results_file = run_dir / "model_validation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"JSON results saved to: {results_file}")
        
        # Save structured log file for report
        self._save_validation_log(results, run_dir, model_name)

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

        # Create validation dataset subset if needed
        validation_samples = validation_config.get("validation_samples", 1000)
        if len(val_dataloader.dataset) > validation_samples:
            logger.info(f"Using subset of {validation_samples} samples for validation")
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
            custom_model, 
            pretrained_model, 
            val_loader_limited, 
            run_dir,
            model_name=validation_config.get("pretrained_model", "swin_tiny_patch4_window7_224")
        )

        return comparison_results
    
    def _save_validation_log(self, results: Dict, run_dir: Path, model_name: str) -> None:
        """
        Save formatted validation log for milestone report.
        
        Args:
            results: Dictionary containing validation results
            run_dir: Path to run directory
            model_name: Name of the model being validated
        """
        log_file = run_dir / "zero_shot_imagenet.log"
        
        with open(log_file, "w") as f:
            f.write("="*70 + "\n")
            f.write("ZERO-SHOT IMAGENET-1K VALIDATION\n")
            f.write("Implementation Validation Report\n")
            f.write("="*70 + "\n\n")
            
            # Metadata
            f.write("VALIDATION METADATA\n")
            f.write("-"*70 + "\n")
            f.write(f"Date: {results['validation_info']['date']}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Dataset: {results['validation_info']['dataset']}\n")
            f.write(f"Samples: {results['custom_model']['total_samples']}\n")
            f.write(f"Device: {results['validation_info']['device']}\n")
            f.write("\n")
            
            # Results Table
            f.write("RESULTS TABLE\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Model':<30} {'Top-1 Acc':<15} {'Top-5 Acc':<15}\n")
            f.write("-"*70 + "\n")
            f.write(f"{'TIMM Pretrained':<30} "
                   f"{results['pretrained_model']['top1_accuracy']:>13.3f}% "
                   f"{results['pretrained_model']['top5_accuracy']:>13.3f}%\n")
            f.write(f"{'Custom Implementation':<30} "
                   f"{results['custom_model']['top1_accuracy']:>13.3f}% "
                   f"{results['custom_model']['top5_accuracy']:>13.3f}%\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Difference (Δ)':<30} "
                   f"{results['differences']['top1_diff']:>+13.3f}% "
                   f"{results['differences']['top5_diff']:>+13.3f}%\n")
            f.write(f"{'Absolute Difference (|Δ|)':<30} "
                   f"{results['differences']['top1_diff_abs']:>13.3f}% "
                   f"{results['differences']['top5_diff_abs']:>13.3f}%\n")
            f.write("\n")
            
            # LaTeX table for report
            f.write("LATEX TABLE (for report)\n")
            f.write("-"*70 + "\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Zero-Shot ImageNet-1K Validation Results}\n")
            f.write("\\label{tab:zero_shot}\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\toprule\n")
            f.write("Model & Top-1 Acc (\\%) & Top-5 Acc (\\%) \\\\\n")
            f.write("\\midrule\n")
            f.write(f"TIMM Pretrained & {results['pretrained_model']['top1_accuracy']:.2f} & "
                   f"{results['pretrained_model']['top5_accuracy']:.2f} \\\\\n")
            f.write(f"Custom Implementation & {results['custom_model']['top1_accuracy']:.2f} & "
                   f"{results['custom_model']['top5_accuracy']:.2f} \\\\\n")
            f.write("\\midrule\n")
            f.write(f"Difference ($\\Delta$) & {results['differences']['top1_diff']:+.2f} & "
                   f"{results['differences']['top5_diff']:+.2f} \\\\\n")
            f.write(f"Absolute Diff ($|\\Delta|$) & {results['differences']['top1_diff_abs']:.2f} & "
                   f"{results['differences']['top5_diff_abs']:.2f} \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
            f.write("\n")
            
            # Validation Status
            f.write("VALIDATION STATUS\n")
            f.write("-"*70 + "\n")
            status = "PASSED ✓" if results['validation_status']['passed'] else "FAILED ✗"
            f.write(f"Status: {status}\n")
            f.write(f"Criterion: {results['validation_status']['criterion']}\n")
            f.write(f"Threshold: ±{results['validation_status']['threshold']}%\n")
            f.write("\n")
            f.write("Per-Metric Status:\n")
            top1_pass = results['differences']['top1_diff_abs'] <= 0.1
            top5_pass = results['differences']['top5_diff_abs'] <= 0.1
            f.write(f"  Top-1: {'PASS ✓' if top1_pass else 'FAIL ✗'} "
                   f"({results['differences']['top1_diff_abs']:.3f}% ≤ 0.1%)\n")
            f.write(f"  Top-5: {'PASS ✓' if top5_pass else 'FAIL ✗'} "
                   f"({results['differences']['top5_diff_abs']:.3f}% ≤ 0.1%)\n")
            f.write("\n")
            
            # Performance Metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*70 + "\n")
            f.write(f"TIMM Model:\n")
            f.write(f"  Evaluation Time: {results['pretrained_model']['evaluation_time']:.2f}s\n")
            f.write(f"  Throughput: {results['pretrained_model']['samples_per_second']:.1f} samples/s\n")
            f.write(f"\nCustom Model:\n")
            f.write(f"  Evaluation Time: {results['custom_model']['evaluation_time']:.2f}s\n")
            f.write(f"  Throughput: {results['custom_model']['samples_per_second']:.1f} samples/s\n")
            f.write("\n")
            
            # Conclusion
            f.write("CONCLUSION\n")
            f.write("-"*70 + "\n")
            if results['validation_status']['passed']:
                f.write("✓ Implementation validation PASSED\n")
                f.write("  Custom implementation matches timm pretrained model within tolerance.\n")
                f.write("  Model is ready for downstream tasks (linear probing, fine-tuning).\n")
            else:
                f.write("✗ Implementation validation FAILED\n")
                f.write("  Custom implementation does not match timm pretrained model.\n")
                f.write("  Please review weight loading and model architecture.\n")
            f.write("\n")
            f.write("="*70 + "\n")
        
        logger.info(f"Detailed validation log saved to: {log_file}")
        
        # Also save to results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        results_log = results_dir / "zero_shot_imagenet.log"
        
        shutil.copy(log_file, results_log)
        logger.info(f"Log also copied to: {results_log}")
