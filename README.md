# Swin Transformer Ablation Studies & Model Comparison

Train and compare Swin Transformers, Vision Transformers (ViT), and ResNet models from scratch on CIFAR-10, CIFAR-100, ImageNet, and ADE20K. Includes comprehensive ablation studies for Swin Transformer architectural components.

## 🚀 Quick Setup

### 1. Choose Dataset
Edit `config/__init__.py`:
```python
# DATASET = "cifar10"    
# DATASET = "cifar100"
# DATASET = "ade20k"      # Semantic segmentation
DATASET = "imagenet"     # ← Change this line
```

### 2. Choose Model & Training Settings
Edit the corresponding config file:

**For CIFAR-100** → Edit `config/cifar100_config.py`:
```python
SWIN_CONFIG = {
    "variant": "tiny",  # Options: "tiny", "small", "base", "large"
}

TRAINING_CONFIG = {
    "learning_rate": 0.001,
    "num_epochs": 50,        # ← Change epochs here
    "warmup_epochs": 2,
}
```

**For CIFAR-10** → Edit `config/cifar10_config.py`  
**For ImageNet** → Edit `config/imagenet_config.py`:

#### Model Selection
Choose which model to train from scratch:
```python
MODEL_TYPE = "swin"  # Options: "swin", "vit", "resnet"
```

#### Ablation Studies (Swin Transformer Only)
Enable/disable architectural components for ablation experiments:
```python
MODEL_CONFIGS = {
    "swin": {
        "variant": "tiny",
        # Core Swin components
        "use_shifted_window": True,        # Enable shifted window attention
        "use_relative_bias": True,         # Use relative position bias
        "use_absolute_pos_embed": False,   # Use absolute position embedding
        "use_hierarchical_merge": False,   # Use hierarchical feature merging
        
        # Memory optimization
        "use_gradient_checkpointing": True,  # Trade compute for memory (recommended)
    }
}
```

#### Training Configuration
```python
TRAINING_CONFIG = {
    "learning_rate": 3e-4,    # Scaled for batch_size=128, 50 epochs
    "num_epochs": 50,         # Full training duration
    "warmup_epochs": 3,       # Learning rate warmup
    "batch_size": 128,        # Optimized with gradient checkpointing
}
```

#### Common Ablation Configurations

**Baseline Swin (all features enabled):**
```python
"use_shifted_window": True,
"use_relative_bias": True,
"use_absolute_pos_embed": False,
"use_hierarchical_merge": False,
```

**Ablate shifted windows:**
```python
"use_shifted_window": False,  # Standard self-attention
"use_relative_bias": True,
```

**Ablate relative position bias:**
```python
"use_shifted_window": True,
"use_relative_bias": False,   # Absolute position bias only
```

**Test absolute position embedding:**
```python
"use_absolute_pos_embed": True,  # Enable absolute pos embedding
"use_relative_bias": False,       # Disable relative bias
```

**Memory optimization:**
```python
"use_gradient_checkpointing": True,  # Enable for large batches/low memory
```

### 3. Set Data Path
In `config/__init__.py`:
```python
# Local:
# DATA_ROOT = "./datasets"

# Cluster:
DATA_ROOT = "/home/space/datasets"  # ← Uncomment for cluster
```

## 🏃 Running

### Local
```bash
python main.py
```

### Cluster
```bash
sbatch job.slurm
squeue -u $USER  # Check status
```

## 🎯 Model Variants

| Variant | Parameters | Use Case |
|---------|------------|----------|
| `tiny`  | 29M        | Quick experiments |
| `small` | 50M        | Balanced performance |
| `base`  | 88M        | Full experiments |
| `large` | 197M       | Maximum accuracy |

**To switch models**: Just change `"variant": "tiny"` to `"variant": "base"` etc. in your config file.

## 🔬 Ablation Studies

The ImageNet configuration supports systematic ablation of Swin Transformer components:

| Component | Description | Default | Ablation Impact |
|-----------|-------------|---------|-----------------|
| `use_shifted_window` | Shifted window attention | `True` | Compare vs standard self-attention |
| `use_relative_bias` | Relative position bias | `True` | Test absolute-only positioning |
| `use_absolute_pos_embed` | Absolute position embedding | `False` | Enable for ViT-style positioning |
| `use_hierarchical_merge` | Hierarchical feature merging | `False` | Test different merge strategies |
| `use_gradient_checkpointing` | Memory optimization | `True` | Trade compute for memory |

## 📁 Output

Results saved to `runs/run_XX/`:
```
├── config.json                    # Your settings
├── training.log                   # Full logs  
├── training_curves_*.png          # Loss/accuracy plots
├── confusion_matrix_*.png         # Test results
└── results_*.json                 # Final metrics
```

