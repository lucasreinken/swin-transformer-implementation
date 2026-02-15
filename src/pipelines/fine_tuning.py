"""
Fine-tuning pipeline: Full model training with Layer-wise Learning Rate Decay (LLRD).

This pipeline:
1. Initializes a custom Swin backbone + Linear Head.
2. Loads weights:
   - If checkpoint_path is PROVIDED: Loads SimMIM weights (robust loader).
   - If checkpoint_path is NONE:     Loads ImageNet baseline (TIMM reference + transfer).
   - If checkpoint_path is "random": Randomly initializes weights.
3. Applies Layer-wise Learning Rate Decay (LLRD) via setup_training_components.
4. Trains with Mixed Precision (AMP) support.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    AUGMENTATION_CONFIG,
    DATA_CONFIG,
    DOWNSTREAM_CONFIG,
    TRAINING_CONFIG,
    SWIN_PRESETS,
    SWIN_CONFIG,
    get_pretrained_swin_name,
)

from src.models import (
    SwinTransformerModel,
    ModelWrapper,
    LinearClassificationHead,
)

from src.training import run_training_loop
from src.utils.experiment import ExperimentTracker
from src.training.checkpoints import load_model_weights

from src.pipelines.utils import (
    validate_pretrained_model_name,
    setup_training_components,
    generate_reports,
    save_final_model,
)

from src.pipelines.linear_probing import create_reference_model, create_custom_model_timm

logger = logging.getLogger(__name__)


def setup_mixed_precision(device: torch.device) -> Tuple[Optional[torch.dtype], Optional[torch.amp.GradScaler]]:
    """
    Configure mixed-precision settings for the current training device.
    """
    use_mp = TRAINING_CONFIG.get("mixed_precision", False)

    if not use_mp:
        logger.info("Mixed precision disabled → training in float32 precision")
        return None, None

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            logger.info("Mixed precision enabled → CUDA bf16 selected (hardware supported)")
            return torch.bfloat16, None

        logger.info("Mixed precision enabled → CUDA float16 selected (bf16 unsupported)")
        return torch.float16, torch.amp.GradScaler(device.type)

    if device.type == "cpu":
        logger.info("Mixed precision enabled → CPU bf16 selected")
        return torch.bfloat16, None

    logger.info("Mixed precision requested, but device unsupported → falling back to float32")
    return None, None


def create_finetuning_model(
    checkpoint_path: str | Path | None,
    model_size: str,
    device: torch.device,
    num_classes: int | None = None,
) -> nn.Module:
    """
    Initialize model for fine-tuning.
    
    Modes:
    1. checkpoint_path IS NONE  -> Loads ImageNet weights (via TIMM reference + transfer).
    2. checkpoint_path PROVIDED -> Loads SimMIM weights (Custom Loader).
    3. checkpoint_path is "random" -> Randomly initializes weights.
    """

    if checkpoint_path == "random":
        logger.info("="*60)
        logger.info(f"NO PRETRAINING -> INITIALIZING RANDOM WEIGHTS (FROM SCRATCH)")
        logger.info("="*60)
        
        # 1. Initialize custom Swin model (Randomly initialized by default)
        encoder = SwinTransformerModel(
            img_size=SWIN_CONFIG["img_size"],
            patch_size=SWIN_CONFIG["patch_size"],
            window_size=SWIN_CONFIG["window_size"],
            embedding_dim=SWIN_CONFIG["embed_dim"],
            depths=SWIN_CONFIG["depths"],
            num_heads=SWIN_CONFIG["num_heads"],
            mlp_ratio=SWIN_CONFIG["mlp_ratio"],
            drop_path_rate=SWIN_CONFIG["drop_path_rate"],
        )
        
        # 2. Initialize Head
        head = LinearClassificationHead(
            num_features=encoder.num_features,
            num_classes=num_classes,
        )
        
        # 3. Wrap
        model = ModelWrapper(
            encoder=encoder,
            pred_head=head,
            freeze=False
        ).to(device)
        
        # 4. Explicit Weight Initialization
        encoder.init_weights()
        
        logger.info("Randomly Initialized Model Ready.")
        return model
    
    elif checkpoint_path is None:
        logger.info("="*60)
        logger.info(f"NO CHECKPOINT PROVIDED -> INITIALIZING IMAGENET BASELINE")
        logger.info("="*60)
        
        try:
            # 1. Get the TIMM model name (e.g., 'swin_tiny_patch4_window7_224')
            pretrained_model_name = get_pretrained_swin_name()
            
            # 2. Create Reference Model (TIMM)
            reference_model = create_reference_model(pretrained_model_name, device)
            
            # 3. Create Custom Model & Transfer Weights
            model = create_custom_model_timm(
                reference_model=reference_model,
                model_size=model_size,
                device=device
            )
            
            # 4. Ensure it's Unfrozen for Fine-Tuning
            model.freeze = False
            for p in model.parameters():
                p.requires_grad = True
                
            logger.info("ImageNet Baseline Model Ready for Fine-Tuning.")
            return model

        except Exception as e:
            logger.error(f"Failed to create ImageNet baseline model: {e}")
            raise RuntimeError(f"ImageNet baseline creation failed: {e}") from e

    else:
        logger.info("="*60)
        logger.info(f"LOADING SIMMIM CHECKPOINT: {checkpoint_path}")
        logger.info("="*60)
        
        try:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            # Input validation
            if model_size not in SWIN_PRESETS:
                raise ValueError(
                    f"Invalid model size '{model_size}'. Available: {list(SWIN_PRESETS.keys())}"
                )
            
            if num_classes is None:
                num_classes = int(DOWNSTREAM_CONFIG["num_classes"])

            # 1. Setup Architecture
            preset = SWIN_PRESETS[model_size]
            logger.info(f"Initializing Swin-{model_size} for fine-tuning...")
            
            drop_path_rate = TRAINING_CONFIG.get("drop_path_rate", 0.1)
            
            encoder = SwinTransformerModel(
                img_size=SWIN_CONFIG["img_size"],
                patch_size=SWIN_CONFIG["patch_size"],
                window_size=SWIN_CONFIG["window_size"],
                embedding_dim=SWIN_CONFIG["embed_dim"],
                depths=SWIN_CONFIG["depths"],
                num_heads=SWIN_CONFIG["num_heads"],
                mlp_ratio=SWIN_CONFIG["mlp_ratio"],
                drop_path_rate=SWIN_CONFIG["drop_path_rate"],
            )

            if not hasattr(encoder, "num_features"):
                raise AttributeError("Encoder missing 'num_features' attribute")

            # 2. Setup Head
            logger.info(f"Creating classification head: {encoder.num_features} -> {num_classes} classes")
            head = LinearClassificationHead(
                num_features=encoder.num_features,
                num_classes=num_classes,
            )

            # 3. Wrap Model (Freeze = False for Fine-tuning)
            model = ModelWrapper(
                encoder=encoder,
                pred_head=head,
                freeze=False,
            ).to(device)

            # 4. Load SimMIM Weights
            model = load_model_weights(
                model=model,
                filepath=str(checkpoint_path),
                device=device,
                encoder_only=True 
            )

            logger.info("SimMIM Fine-Tuning Model Ready.")
            return model

        except Exception as e:
            logger.error(f"Failed to create SimMIM fine-tuning model: {e}")
            raise RuntimeError(f"SimMIM model creation failed: {e}") from e


def _train_finetune_model(
    model: nn.Module,
    train_generator: DataLoader,
    val_generator: DataLoader,
    test_generator: DataLoader,
    total_epochs: int,
    warmup_epochs: int,
    learning_rate: float,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    scaler: Optional[torch.amp.GradScaler],
    run_dir: Path,
) -> Tuple[nn.Module, List[float], Dict[str, List[float]]]:
    """
    Run the fine-tuning training loop with Mixed Precision and LLRD.
    """
    # 1. Setup Components (Optimizer with LLRD logic)
    criterion, optimizer, scheduler = setup_training_components(
        model, total_epochs, warmup_epochs, learning_rate
    )

    metrics_history = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "val_accuracy": [],
        "test_accuracy": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1_per_class": [],
    }

    lr_history = []
    mixup = None

    # 2. Run Training Loop
    run_training_loop(
        model=model,
        train_generator=train_generator,
        val_generator=val_generator,
        test_generator=test_generator,
        num_epochs=total_epochs,
        learning_rate=learning_rate,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics_history=metrics_history,
        lr_history=lr_history,
        mixup=mixup,
        device=device,
        amp_dtype=amp_dtype,
        scaler=scaler,
        run_dir=run_dir,
        checkpoint_frequency=TRAINING_CONFIG.get("checkpoint_frequency", 5)
    )

    return criterion, lr_history, metrics_history


def run_fine_tuning(
    train_generator: DataLoader,
    val_generator: DataLoader,
    test_generator: DataLoader,
    total_epochs: int,
    warmup_epochs: int,
    learning_rate: float,
    device: torch.device,
    run_dir: Path,
) -> None:
    """
    Main entry point for Fine-Tuning pipeline.
    """
    dataset = DATA_CONFIG.get("dataset", "dataset")
    # Get checkpoint path from config (can be None for ImageNet baseline)
    checkpoint_path = DOWNSTREAM_CONFIG.get("pretrained_path", None)
    
    logger.info("="*60)
    logger.info(f"STARTING FINE-TUNING PIPELINE ON {dataset.upper()}")
    if checkpoint_path:
        logger.info(f"Target: SimMIM Finetuning (Checkpoint: {checkpoint_path})")
    else:
        logger.info(f"Target: ImageNet Baseline Finetuning (No Checkpoint)")
    logger.info("="*60)

    # 1. Setup Mixed Precision
    amp_dtype, scaler = setup_mixed_precision(device)

    # 2. Determine Model Size from Config
    pretrained_model_name = get_pretrained_swin_name()
    validate_pretrained_model_name(pretrained_model_name)
    
    model_size = None
    for p in pretrained_model_name.lower().split("_"):
        if p in SWIN_PRESETS:
            model_size = p
            break
            
    if model_size is None:
        raise ValueError(f"Could not determine model size from config name: {pretrained_model_name}")

    # 3. Create Model
    model = create_finetuning_model(
        checkpoint_path=checkpoint_path,
        model_size=model_size,
        device=device,
        num_classes=int(DOWNSTREAM_CONFIG["num_classes"])
    )

    # 4. Initialize Tracker
    tracker = ExperimentTracker(run_dir)

    # 5. Train
    logger.info("Starting fine-tuning loop...")
    criterion, lr_history, metrics_history = _train_finetune_model(
        model=model,
        train_generator=train_generator,
        val_generator=val_generator,
        test_generator=test_generator,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        learning_rate=learning_rate,
        device=device,
        amp_dtype=amp_dtype,
        scaler=scaler,
        run_dir=run_dir
    )
    logger.info("Fine-tuning completed successfully.")

    # 6. Finalize & Report
    logger.info("Generating final reports...")
    final_metrics = generate_reports(
        model,
        "finetuned",
        test_generator,
        criterion,
        lr_history,
        metrics_history,
        device,
        amp_dtype,
        run_dir,
    )

    tracker.log_results(
        "finetuned",
        final_metrics=final_metrics,
        training_history=metrics_history,
    )

    tracker.finalize("finetuned")
    save_final_model(model, "finetuned")
    
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Final Test Accuracy: {final_metrics['accuracy']:.2f}%")