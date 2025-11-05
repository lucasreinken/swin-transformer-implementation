"""
Experiment management utilities for organizing training runs.
"""

import logging
from pathlib import Path


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
