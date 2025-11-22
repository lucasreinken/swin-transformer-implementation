# Linear Probing with Swin Transformers

Compare custom Swin Transformer implementations against TIMM reference models on CIFAR-10, CIFAR-100, and ImageNet.

## 🚀 Quick Setup

### 1. Choose Dataset
Edit `config/__init__.py`:
```python
# DATASET = "cifar10"    
DATASET = "cifar100"     # ← Change this line
# DATASET = "imagenet"   
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
**For ImageNet** → Edit `config/imagenet_config.py`

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
apptainer run --nv pml.sif python main.py
```

## 🎯 Model Variants

| Variant | Parameters | Use Case |
|---------|------------|----------|
| `tiny`  | 29M        | Quick experiments |
| `small` | 50M        | Balanced performance |
| `base`  | 88M        | Full experiments |
| `large` | 197M       | Maximum accuracy |

**To switch models**: Just change `"variant": "tiny"` to `"variant": "base"` etc. in your config file.

## 📊 What You Get

The system automatically:
- Downloads TIMM pretrained models
- Creates matching custom Swin architecture  
- Transfers weights between models
- Trains both models with linear probing
- Compares final accuracies


## 📁 Output

Results saved to `runs/run_XX/`:
```
├── config.json                    # Your settings
├── training.log                   # Full logs  
├── training_curves_*.png          # Loss/accuracy plots
├── confusion_matrix_*.png         # Test results
└── results_*.json                 # Final metrics
```

