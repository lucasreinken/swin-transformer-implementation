"""
Experiment management utilities for organizing training runs.
"""

import logging
from pathlib import Path
import json
from typing import Dict, List, Optional, Any
import torch
import pandas as pd
import time


def setup_run_directory():
    """Create and return the next run directory for organizing outputs."""
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)

    # Find the next run number
    existing_runs = [
        d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")
    ]
    run_numbers = []
    for run_dir in existing_runs:
        try:
            num = int(run_dir.name.split("_")[1])
            run_numbers.append(num)
        except (ValueError, IndexError):
            continue

    next_run_num = max(run_numbers) + 1 if run_numbers else 1
    run_dir = runs_dir / f"run_{next_run_num}"
    run_dir.mkdir(exist_ok=True)

    return run_dir


def setup_logging(run_dir):
    """Setup logging to save to both console and run directory."""
    # Clear existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create formatters
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler for run directory
    log_file = run_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Setup root logger
    logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


class ExperimentTracker:
    """
    Tracks and logs all aspects of ML experiments for easy comparison.

    Saves:
    - config.json: All hyperparameters and settings
    - results.json: Final metrics and trianing history
    - metadata.json: Run information and environment details
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.start_time = time.time()
        self.metadata = self._collect_metadata()

    def _collect_metadata(self) -> Dict[str, Any]:
        """Collect run metadata."""
        return {
            "run_id": self.run_dir.name,
            "start_time": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)
            ),
            "hostname": "unknown",
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

    def log_config(self, **configs):
        """Save all configuration dictionaries to config.json."""
        config_data = {}
        for name, config in configs.items():
            # Convert config objects to dictionaries if needed
            if hasattr(config, "__dict__"):
                config_data[name] = config.__dict__
            else:
                config_data[name] = config

        config_file = self.run_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2, default=str)

        logging.info(f"Saved configuration to {config_file}")

    def log_results(
        self,
        variant,
        final_metrics: Dict,
        training_history: Dict,
        validation_results: Optional[Dict] = None,
    ):
        """Save final results and training history to results.json."""
        results_data = {
            "final_metrics": final_metrics,
            "training_history": training_history,
            "validation_results": validation_results,
            "duration_seconds": time.time() - self.start_time,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }

        results_file = self.run_dir / f"results_{variant}.json"
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        logging.info(f"Experiment completed. Results saved to {results_file}")

    def finalize(self, variant):
        """Save metadata to metadata.json."""
        self.metadata["duration_seconds"] = time.time() - self.start_time
        self.metadata["completed_at"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()
        )

        metadata_file = self.run_dir / f"metadata_{variant}.json"
        with open(metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

        logging.info(f"Experiment completed. Metadata saved to {metadata_file}")


def load_experiment_data(run_dir: Path) -> Dict[str, Any]:
    """Load all experiment data from a run directory."""
    data = {"run_id": run_dir.name}

    # Load config.json
    config_file = run_dir / "config.json"
    if config_file.exists():
        with open(config_file, "r") as f:
            data["config"] = json.load(f)

    # Load results
    results_file = run_dir / "results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            data["results"] = json.load(f)

    # Load metadata
    metadata_file = run_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            data["metadata"] = json.load(f)

    return data


def compare_experiments(run_ids: List[str], metric: str) -> pd.DataFrame:
    """Compare multiple experiments by a specific metric."""
    runs_dir = Path("runs")
    results = []

    for run_id in run_ids:
        run_dir = runs_dir / run_id
        if not run_dir.exists():
            logging.warning(f"Run {run_id} not found. Skipping.")
            continue

        data = load_experiment_data(run_dir)

        # Extract key information
        row = {
            "run_id": run_id,
            "dataset": data.get("config", {})
            .get("DATA_CONFIG", {})
            .get("dataset", "unknown"),
            "model": data.get("config", {})
            .get("MODEL_CONFIG", {})
            .get("model_name", "unknown"),
        }

        # Add final metric
        if "results" in data and "final_metrics" in data["results"]:
            final_metrics = data["results"]["final_metrics"]
            row[metric] = final_metrics.get(metric.replace("_", " ").title(), None)

        # Add key hyperparameters
        if "config" in data:
            config = data["config"]
            row["learning_rate"] = config.get("TRAINING_CONFIG", {}).get(
                "learning_rate", None
            )
            row["batch_size"] = config.get("TRAINING_CONFIG", {}).get(
                "batch_size", None
            )
            row["epochs"] = config.get("TRAINING_CONFIG", {}).get("num_epochs", None)
            row["img_size"] = config.get("DATA_CONFIG", {}).get("img_size", None)

        results.append(row)

    return pd.DataFrame(results)


def find_best_experiment(metric: str = "test_accuracy", top_k: int = 5) -> pd.DataFrame:
    """Find the best performing experiments by a metric."""
    runs_dir = Path("runs")
    experiments = []

    # Load all experiments
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue

        data = load_experiment_data(run_dir)

        if "results" in data and "final_metrics" in data["results"]:
            final_metrics = data["results"]["final_metrics"]
            metric_value = final_metrics.get(metric.replace("_", " ").title(), None)

            if metric_value is not None:
                experiments.append(
                    {
                        "run_id": run_dir.name,
                        "metric_value": metric_value,
                        "dataset": data.get("config", {})
                        .get("DATA_CONFIG", {})
                        .get("dataset", "unknown"),
                        "completed_at": data.get("metadata", {}).get(
                            "completed_at", "unknown"
                        ),
                    }
                )

    # Sort and return top k
    experiments.sort(key=lambda x: x["metric_value"], reverse=True)
    return pd.DataFrame(experiments[:top_k])


def generate_experiments_summary() -> pd.DataFrame:
    """Generate a summary of all experiments."""
    runs_dir = Path("runs")
    summaries = []

    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue

        data = load_experiment_data(run_dir)

        summary = {
            "run_id": run_dir.name,
            "dataset": data.get("config", {})
            .get("DATA_CONFIG", {})
            .get("dataset", "unknown"),
            "completed": ("results" in data and "final_metrics" in data["results"]),
        }

        # Add final metrics if available
        if summary["completed"]:
            final_metrics = data["results"]["final_metrics"]
            summary.update(
                {
                    "test_accuracy": final_metrics.get("Test Accuracy", None),
                    "test_loss": final_metrics.get("Loss", None),
                    "precision": final_metrics.get("Precision", None),
                    "recall": final_metrics.get("Recall", None),
                    "f1_score": final_metrics.get("F1 Score", None),
                }
            )

        # Add metadata
        if "metadata" in data:
            summary["duration"] = data["metadata"].get("duration_seconds")
            summary["completed_at"] = data["metadata"].get("completed_at")

        summaries.append(summary)

    return pd.DataFrame(summaries)
