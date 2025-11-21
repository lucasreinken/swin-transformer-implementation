# Linear Probing with Swin Transformers

A comprehensive implementation for linear probing experiments comparing custom Swin Transformer implementations against TIMM reference models on CIFAR-10, CIFAR-100, and ImageNet datasets.

## 🚀 Quick Start

### 1. Dataset Selection
Edit `config/__init__.py` to choose your dataset:
```python
# Choose ONE dataset (uncomment the desired option)
# DATASET = "cifar10"    # CIFAR-10: 10 classes, 32x32 → 224x224
DATASET = "cifar100"     # CIFAR-100: 100 classes, 32x32 → 224x224  
# DATASET = "imagenet"   # ImageNet-1K: 1000 classes, 224x224
```

### 2. Environment Setup
Choose your data directory in `config/__init__.py`:
```python
# For local development:
DATA_ROOT = "./datasets"

# For cluster (Uncomment for cluster use):
# DATA_ROOT = "/home/space/datasets"
```

### 3. Model Configuration
Edit the appropriate config file based on your dataset choice:

#### For CIFAR-100 (edit `config/cifar100_config.py`):
```python
# Swin Transformer Model Selection
SWIN_CONFIG = {
    "variant": "tiny",  # Options: "tiny", "small", "base", "large"
    # ... other settings
}

# Training Parameters  
TRAINING_CONFIG = {
    "learning_rate": 0.001,
    "num_epochs": 50,
    "warmup_epochs": 2,
    "weight_decay": 1e-4,
}
```

#### For CIFAR-10 (edit `config/cifar10_config.py`):
```python
SWIN_CONFIG = {
    "variant": "base",  # Options: "tiny", "small", "base", "large" 
}

TRAINING_CONFIG = {
    "learning_rate": 0.001,
    "num_epochs": 20,
    "warmup_epochs": 2,
}
```

#### For ImageNet (edit `config/imagenet_config.py`):
```python
SWIN_CONFIG = {
    "variant": "base",  # Options: "tiny", "small", "base", "large"
}

TRAINING_CONFIG = {
    "learning_rate": 0.0001,
    "num_epochs": 90,
    "warmup_epochs": 2,
}
```

## 🔧 Installation

### Local Development
```bash
# Clone repository
git clone <repository-url>
cd Machine-Learning-Project

# Install dependencies
pip install -r requirements.txt

# For GPU support (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Cluster Setup (Apptainer/Singularity)
The project includes a pre-configured container setup:
```bash
# Container is already built: pml.sif
# Job submission script is ready: job.slurm
```

## 🏃‍♂️ Running Experiments

### Local Execution
```bash
# Run linear probing experiment
python main.py
```

### Cluster Execution (SLURM)
```bash
# Submit job to cluster
sbatch job.slurm

# Monitor job status  
squeue -u $USER

# Check job output
cat logs/<job_id>.out
cat logs/<job_id>.err
```

### SLURM Job Configuration
Edit `job.slurm` for different requirements:
```bash
#!/bin/bash
#SBATCH --partition=gpu-teaching-2h  # or gpu for longer jobs
#SBATCH --gpus=1                     # Number of GPUs
#SBATCH --time=02:00:00              # Time limit (HH:MM:SS)
#SBATCH --output=logs/%j.out         # Output file
#SBATCH --error=logs/%j.err          # Error file  
#SBATCH --job-name=ML_Train          # Job name

mkdir -p logs
apptainer run --nv pml.sif python main.py "$@"
```

**Time Recommendations:**
- CIFAR-10: `--time=01:00:00` (1 hour)
- CIFAR-100: `--time=02:00:00` (2 hours)  
- ImageNet: `--time=12:00:00` (12 hours)

## 🔍 Model Architecture Options

### Swin Transformer Variants
| Variant | Parameters | Embed Dim | Depths | Num Heads | TIMM Model |
|---------|-----------|-----------|---------|-----------|------------|
| `tiny`  | 29M       | 96        | [2,2,6,2] | [3,6,12,24] | `swin_tiny_patch4_window7_224` |
| `small` | 50M       | 96        | [2,2,18,2] | [3,6,12,24] | `swin_small_patch4_window7_224` |
| `base`  | 88M       | 128       | [2,2,18,2] | [4,8,16,32] | `swin_base_patch4_window7_224` |  
| `large` | 197M      | 192       | [2,2,18,2] | [6,12,24,48] | `swin_large_patch4_window7_224` |

### What Happens When You Change Variants:
1. **Custom Model Architecture**: Automatically uses the correct embed_dim, depths, and num_heads
2. **TIMM Reference Model**: Automatically loads the corresponding pretrained model
3. **Weight Transfer**: Automatically transfers weights from TIMM model to custom model
4. **Linear Probing**: Both models frozen except for classification head

## 📊 Expected Results

### CIFAR-100 (50 epochs, ResNet50 baseline)
- **Reference (TIMM)**: ~76.43%
- **Custom (Ours)**: ~76.36%  
- **Difference**: <0.1% (validates implementation)

### Output Structure
```
runs/
├── run_XX/                    # Experiment directory
│   ├── config.json           # Saved configuration
│   ├── training_curves_reference.png
│   ├── training_curves_custom.png
│   ├── confusion_matrix_reference.png
│   ├── confusion_matrix_custom.png
│   └── logs/                 # Training logs
trained_models/
├── CIFAR100_final_model_reference_weights.pth
└── CIFAR100_final_model_custom_weights.pth
```

## ⚙️ Configuration Details

### Dataset-Specific Settings
Each dataset has its optimized configuration:

**CIFAR-10/100:**
- Input: 32×32 → resized to 224×224
- Batch size: 32
- Learning rate: 0.001
- Image normalization: ImageNet stats

**ImageNet:**
- Input: 224×224 (native)
- Batch size: 128  
- Learning rate: 0.0001
- Image normalization: ImageNet stats

### Training Configuration
```python
# All configurable in respective config files
TRAINING_CONFIG = {
    "learning_rate": 0.001,      # AdamW learning rate
    "num_epochs": 50,            # Total training epochs
    "warmup_epochs": 2,          # Linear warmup epochs
    "weight_decay": 1e-4,        # L2 regularization
}
```

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
```python
# Reduce batch size in config
DATA_CONFIG = {
    "batch_size": 16,  # Reduce from 32
}
```

**2. Dataset Not Found**
```bash
# Check data path in config/__init__.py
DATA_ROOT = "/correct/path/to/datasets"
```

**3. TIMM Model Loading Issues**  
```bash
# Install/update timm
pip install timm>=0.9.0
```

**4. Cluster Container Issues**
```bash
# Rebuild container if needed
apptainer build pml.sif pml.def
```

### Performance Tips

**Speed Optimization:**
- Use `num_workers=4` for data loading
- Enable `deterministic=False` in SEED_CONFIG for faster training
- Use smaller models (tiny/small) for quick experiments

**Memory Optimization:**
- Reduce batch size for larger models
- Use gradient accumulation if needed
- Monitor GPU memory with `nvidia-smi`

## 📈 Monitoring Progress

### Real-time Monitoring
```bash
# Watch job logs
tail -f logs/<job_id>.out

# Monitor GPU usage  
watch nvidia-smi
```

### Key Metrics to Watch
- **Training Loss**: Should decrease steadily
- **Validation Accuracy**: Should increase and plateau
- **Weight Transfer**: Should show >95% parameter matches
- **Final Comparison**: Custom vs Reference accuracy difference <0.5%

## 🎯 Next Steps

After successful runs, you can:
1. **Experiment with different Swin variants**: Change `variant` in config
2. **Try different datasets**: Switch `DATASET` in `config/__init__.py`
3. **Adjust hyperparameters**: Modify `TRAINING_CONFIG` values
4. **Scale to full ImageNet**: Use longer training times and larger models

## 📚 Project Structure

```
├── main.py                 # Main training script
├── job.slurm              # SLURM job submission script  
├── requirements.txt       # Python dependencies
├── config/               # Configuration files
│   ├── __init__.py       # Dataset selection & loading
│   ├── cifar10_config.py # CIFAR-10 settings
│   ├── cifar100_config.py# CIFAR-100 settings
│   └── imagenet_config.py# ImageNet settings
├── src/                  # Source code
│   ├── models/          # Model implementations
│   ├── data/            # Data loading & preprocessing
│   ├── training/        # Training utilities
│   └── utils/           # Helper functions
└── runs/                # Experiment outputs
```

---

**Ready to run? Just configure your dataset and model, then execute!** 🚀
