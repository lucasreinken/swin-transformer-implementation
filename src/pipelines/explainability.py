"""
Explainability pipeline: Visualize and analyze Swin Transformer attention patterns.

This pipeline loads a pretrained Swin model with attention capture enabled,
runs inference on sample images, and generates visualizations comparing
W-MSA (Window Multi-head Self-Attention) vs SW-MSA (Shifted Window MSA).
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from config import (
    DATA_CONFIG,
    TRAINING_CONFIG,
)

from src.models import SwinTransformerModel
from src.utils.experiment import ExperimentTracker
from src.utils.load_weights import transfer_weights
from src.utils.attention_visualization import AttentionVisualizer

logger = logging.getLogger(__name__)


def create_explainability_model(
    model_config: Dict,
    pretrained_weights: Optional[str],
    device: torch.device
) -> SwinTransformerModel:
    """
    Create Swin Transformer model with attention capture enabled.
    
    Args:
        model_config: Model configuration dictionary
        pretrained_weights: Path to pretrained weights or TIMM model name
        device: Device to place the model on
    
    Returns:
        SwinTransformerModel with return_attention_maps=True
    """
    logger.info("Creating Swin Transformer model for explainability...")
    
    # Ensure attention capture is enabled
    model_config = model_config.copy()
    model_config['return_attention_maps'] = True
    
    # Log configuration
    logger.info(f"Model architecture: Swin Transformer")
    logger.info(f"Attention capture: ENABLED")
    logger.info(f"Configuration: {model_config}")
    
    # Create model
    model = SwinTransformerModel(**model_config)
    
    # Load pretrained weights if provided
    if pretrained_weights:
        logger.info(f"Loading pretrained weights: {pretrained_weights}")
        
        if pretrained_weights.startswith('swin_'):
            # TIMM model name
            from timm import create_model
            logger.info("Loading weights from TIMM pretrained model...")
            timm_model = create_model(pretrained_weights, pretrained=True)
            transfer_weights(
                source_model=timm_model,
                target_model=model,
                strict=False,
                ignore_head=True
            )
            logger.info("Successfully loaded TIMM pretrained weights")
        else:
            # Local checkpoint
            logger.info(f"Loading weights from checkpoint: {pretrained_weights}")
            checkpoint = torch.load(pretrained_weights, map_location=device)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=False)
            logger.info("Successfully loaded checkpoint weights")
    else:
        logger.warning("No pretrained weights provided - using random initialization")
    
    # Model statistics
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    model = model.to(device)
    model.eval()  # Set to eval mode for visualization
    
    return model


def get_sample_images(
    data_loader: DataLoader,
    num_samples: int,
    device: torch.device,
    strategy: str = 'random'
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Extract sample images from data loader for visualization.
    
    Args:
        data_loader: DataLoader to sample from
        num_samples: Number of samples to extract
        device: Device to place tensors on
        strategy: Sampling strategy ('random', 'first', 'diverse')
    
    Returns:
        Tuple of (images, labels, indices)
    """
    logger.info(f"Extracting {num_samples} sample images using '{strategy}' strategy...")
    
    all_images = []
    all_labels = []
    all_indices = []
    
    if strategy == 'first':
        # Take first N samples
        for idx, (images, labels) in enumerate(data_loader):
            if len(all_images) >= num_samples:
                break
            
            batch_size = images.size(0)
            remaining = num_samples - len(all_images)
            take = min(batch_size, remaining)
            
            all_images.append(images[:take])
            all_labels.append(labels[:take])
            all_indices.extend(range(idx * batch_size, idx * batch_size + take))
    
    elif strategy == 'random':
        # Random sampling
        import random
        dataset_size = len(data_loader.dataset)
        selected_indices = random.sample(range(dataset_size), min(num_samples, dataset_size))
        
        for idx in selected_indices:
            image, label = data_loader.dataset[idx]
            all_images.append(image.unsqueeze(0))
            all_labels.append(torch.tensor([label]))
            all_indices.append(idx)
    
    elif strategy == 'diverse':
        # Try to get diverse classes
        class_samples = {}
        target_per_class = max(1, num_samples // data_loader.dataset.num_classes if hasattr(data_loader.dataset, 'num_classes') else 1)
        
        for idx, (images, labels) in enumerate(data_loader):
            for i in range(images.size(0)):
                label = labels[i].item()
                if label not in class_samples:
                    class_samples[label] = []
                
                if len(class_samples[label]) < target_per_class:
                    class_samples[label].append((images[i], label, idx * data_loader.batch_size + i))
                
                if sum(len(samples) for samples in class_samples.values()) >= num_samples:
                    break
            
            if sum(len(samples) for samples in class_samples.values()) >= num_samples:
                break
        
        # Flatten samples
        for label_samples in class_samples.values():
            for img, lbl, idx in label_samples[:num_samples]:
                all_images.append(img.unsqueeze(0))
                all_labels.append(torch.tensor([lbl]))
                all_indices.append(idx)
    
    # Concatenate
    images = torch.cat(all_images, dim=0).to(device)
    labels = torch.cat(all_labels, dim=0).to(device)
    
    logger.info(f"Extracted {len(all_indices)} samples (shape: {images.shape})")
    
    return images, labels, all_indices


def visualize_attention_patterns(
    model: SwinTransformerModel,
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: List[int],
    viz_config: Dict,
    output_dir: Path,
    device: torch.device
) -> Dict:
    """
    Generate attention visualizations for sample images.
    
    Args:
        model: Swin model with attention capture enabled
        images: Sample images [N, 3, H, W]
        labels: Corresponding labels [N]
        indices: Dataset indices for the samples
        viz_config: Visualization configuration
        output_dir: Directory to save visualizations
        device: Device for computation
    
    Returns:
        Dictionary with visualization statistics
    """
    logger.info("Generating attention visualizations...")
    
    # Create visualizer
    visualizer = AttentionVisualizer(
        model=model,
        device=device,
        colormap=viz_config.get('colormap', 'jet'),
        overlay_alpha=viz_config.get('overlay_alpha', 0.6)
    )
    
    stats = {
        'num_samples': len(images),
        'num_visualizations': 0,
        'stages_analyzed': set(),
        'blocks_analyzed': set(),
    }
    
    # Process each image
    for sample_idx in range(len(images)):
        image = images[sample_idx:sample_idx+1]
        label = labels[sample_idx].item()
        dataset_idx = indices[sample_idx]
        
        logger.info(f"Processing sample {sample_idx + 1}/{len(images)} (dataset idx: {dataset_idx}, label: {label})")
        
        # Extract attention maps
        target_stages = viz_config.get('target_stages', None)
        attention_maps = visualizer.extract_attention_maps(
            image,
            target_stages=target_stages
        )
        
        # Create sample-specific output directory
        sample_dir = output_dir / f"sample_{dataset_idx}_label_{label}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Save original image
        img_np = image[0].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        img_pil.save(sample_dir / "original_image.png")
        
        # Get query positions
        query_strategy = viz_config.get('query_strategy', 'center')
        if query_strategy == 'center':
            H, W = 224, 224  # Input resolution
            query_positions = [(H // 2, W // 2)]
        elif query_strategy == 'grid':
            grid_points = viz_config.get('grid_points', 5)
            H, W = 224, 224
            positions = []
            for i in range(grid_points):
                for j in range(grid_points):
                    h = int((i + 0.5) * H / grid_points)
                    w = int((j + 0.5) * W / grid_points)
                    positions.append((h, w))
            query_positions = positions
        elif query_strategy == 'corners_center':
            query_positions = [
                (56, 56),   # Center
                (14, 14),   # Top-left
                (14, 210),  # Top-right
                (210, 14),  # Bottom-left
                (210, 210), # Bottom-right
            ]
        else:
            query_positions = [(112, 112)]  # Default center
        
        # Generate W-MSA vs SW-MSA comparisons
        stages_to_compare = viz_config.get('comparison_stages', [0, 1, 2])
        for stage_idx in stages_to_compare:
            try:
                comparison_path = sample_dir / f"wmsa_vs_swmsa_stage{stage_idx}_center.png"
                visualizer.compare_w_msa_sw_msa(
                    query_position=query_positions[0],
                    stage_idx=stage_idx,
                    save_path=str(comparison_path)
                )
                stats['num_visualizations'] += 1
                stats['stages_analyzed'].add(stage_idx)
                logger.info(f"  Generated W-MSA vs SW-MSA comparison for stage {stage_idx}")
            except Exception as e:
                logger.warning(f"  Failed to generate comparison for stage {stage_idx}: {e}")
        
        # Generate stage evolution visualization
        try:
            evolution_path = sample_dir / "attention_evolution_across_stages.png"
            visualizer.visualize_stage_evolution(
                query_position=query_positions[0],
                save_path=str(evolution_path)
            )
            stats['num_visualizations'] += 1
            logger.info("  Generated stage evolution visualization")
        except Exception as e:
            logger.warning(f"  Failed to generate stage evolution: {e}")
        
        # Save all attention maps
        if viz_config.get('save_all_maps', False):
            all_maps_dir = sample_dir / "all_attention_maps"
            try:
                visualizer.save_all_attention_maps(
                    output_dir=all_maps_dir,
                    query_positions=query_positions,
                    prefix=f"sample{dataset_idx}_"
                )
                logger.info(f"  Saved all attention maps to {all_maps_dir}")
            except Exception as e:
                logger.warning(f"  Failed to save all attention maps: {e}")
        
        # Compute attention statistics
        try:
            attn_stats = visualizer.compute_attention_statistics()
            stats_path = sample_dir / "attention_statistics.json"
            with open(stats_path, 'w') as f:
                # Convert to JSON-serializable format
                json_stats = {
                    key: [float(v) if isinstance(v, (np.floating, float)) else v for v in values]
                    for key, values in attn_stats.items()
                }
                import json
                json.dump(json_stats, f, indent=2)
            logger.info(f"  Saved attention statistics to {stats_path}")
        except Exception as e:
            logger.warning(f"  Failed to compute statistics: {e}")
    
    # Convert sets to lists for JSON serialization
    stats['stages_analyzed'] = sorted(list(stats['stages_analyzed']))
    stats['blocks_analyzed'] = sorted(list(stats['blocks_analyzed']))
    
    logger.info(f"Generated {stats['num_visualizations']} visualizations for {stats['num_samples']} samples")
    
    return stats


def run_explainability(
    model_config: Dict,
    pretrained_weights: Optional[str],
    data_loader: DataLoader,
    viz_config: Dict,
    device: torch.device,
    run_dir: Path,
) -> None:
    """
    Run explainability pipeline: visualize attention patterns.
    
    Args:
        model_config: Model configuration
        pretrained_weights: Path to pretrained weights or TIMM model name
        data_loader: DataLoader for sample images
        viz_config: Visualization configuration
        device: Device to run on
        run_dir: Directory to save results
    """
    logger.info("="*80)
    logger.info("EXPLAINABILITY PIPELINE: Swin Transformer Attention Analysis")
    logger.info("="*80)
    
    # Create model
    model = create_explainability_model(
        model_config=model_config,
        pretrained_weights=pretrained_weights,
        device=device
    )
    
    # Get sample images
    num_samples = viz_config.get('num_samples', 5)
    sampling_strategy = viz_config.get('sampling_strategy', 'first')
    
    images, labels, indices = get_sample_images(
        data_loader=data_loader,
        num_samples=num_samples,
        device=device,
        strategy=sampling_strategy
    )
    
    # Create output directory
    viz_dir = run_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Visualizations will be saved to: {viz_dir}")
    
    # Generate visualizations
    stats = visualize_attention_patterns(
        model=model,
        images=images,
        labels=labels,
        indices=indices,
        viz_config=viz_config,
        output_dir=viz_dir,
        device=device
    )
    
    # Save summary
    summary = {
        'model_config': model_config,
        'pretrained_weights': pretrained_weights,
        'num_samples': num_samples,
        'sampling_strategy': sampling_strategy,
        'visualization_config': viz_config,
        'statistics': stats,
    }
    
    summary_path = run_dir / "explainability_summary.json"
    with open(summary_path, 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    logger.info("="*80)
    logger.info("EXPLAINABILITY PIPELINE COMPLETED")
    logger.info(f"Generated {stats['num_visualizations']} visualizations")
    logger.info(f"Analyzed stages: {stats['stages_analyzed']}")
    logger.info(f"Results saved to: {run_dir}")
    logger.info("="*80)
