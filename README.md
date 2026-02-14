# Swin Transformer Ablation Studies & Model Comparison

Train and compare Swin Transformers, Vision Transformers (ViT), and ResNet models from scratch on CIFAR-10, CIFAR-100, ImageNet, and ADE20K. Includes comprehensive ablation studies for Swin Transformer architectural components.

> **Also included:** Object Detection on COCO 2017 using MMDetection (Cascade Mask R-CNN + FPN), supporting Swin and ResNet backbones.

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
MODEL_TYPE = "swin"  # Options: "swin", "swin_hybrid", "swin_improved", "vit", "resnet"
```

**Model Descriptions:**
- `"swin"`: Baseline Swin Transformer (vanilla implementation)
- `"swin_hybrid"`: Hybrid CNN-Swin with early fusion (Milestone 3 - first improvement)
- `"swin_improved"`: **NEW** Advanced Swin with conv stem + inverted residual FFN (Milestone 3 - second improvement)
- `"vit"`: Vision Transformer baseline
- `"resnet"`: ResNet-50 baseline

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
    "learning_rate": 2e-4,    # Conservative LR for testing
    "num_epochs": 5,          # Quick test run (5 hours)
    "warmup_epochs": 1,       # Learning rate warmup (~20%)
    "batch_size": 128,        # Optimized with gradient checkpointing
}
```

#### Hybrid CNN-Swin Model (Milestone 3)
The `swin_hybrid` model implements **early fusion** between a lightweight CNN stem and Swin Transformer:

```python
MODEL_TYPE = "swin_hybrid"  # Enable hybrid architecture
```

**Architecture:**
- **CNN Stem**: 3 convolutional layers (32→64→96 channels) with BatchNorm + GELU
- **Fusion**: Cascaded (CNN output replaces vanilla patch embedding)
- **Swin Backbone**: Standard hierarchical transformer with shifted windows
- **Expected Gain**: +1-4% top-1 accuracy through better local inductive biases

**Configuration:**
```python
"cnn_stem_config": {
    "channels": [32, 64],      # Intermediate channels
    "kernel_size": 3,          # 3×3 convolutions
    "stride": 2,               # Downsampling by 8× total
    "activation": "gelu",      # GELU activation
    "use_batch_norm": True,    # Batch normalization
}
```

#### Advanced Swin Model (Milestone 3 - Second Improvement)
The `swin_improved` model implements **convolutional stem + inverted residual FFN** for enhanced local feature modeling:

```python
MODEL_TYPE = "swin_improved"  # Enable advanced architecture
```

**Architecture Components:**
- **Convolutional Stem**: Overlapping conv layers (4×4 → 3×3) with BatchNorm + GELU
- **Inverted Residual FFN**: MobileNetV2-style FFN with depthwise conv and residual connections
- **Expansion Ratio**: 4× channel expansion with efficient depthwise mixing
- **Residual Connections**: Around entire FFN blocks for better gradient flow

**Configuration:**
```python
"conv_stem_config": {
    "channels": [48, 96],        # Overlapping conv channels
    "kernel_sizes": [4, 3],      # First conv kernel 4, second 3
    "strides": [4, 2],           # Progressive downsampling
    "activation": "gelu",        # GELU activation
    "use_batch_norm": True,      # Batch normalization
},
"ffn_config": {
    "expand_ratio": 4,           # Channel expansion ratio
    "use_depthwise_conv": True,  # Enable depthwise convolution
    "activation": "gelu",        # Activation function
}
```

**Expected Gain**: +1-3% top-1 accuracy through stronger local inductive biases and better feature mixing.

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

## 🧩 Object Detection on COCO (MMDetection)

This project extends the Swin Transformer analysis to object detection on the **COCO 2017** dataset. We utilize **MMDetection** to implement a **Cascade Mask R-CNN** with a Feature Pyramid Network (FPN), supporting both **Swin Transformer** and **ResNet** backbones.

> **Note:** Because MMDetection requires a specific dependency stack (including a different PyTorch version than the main classification pipeline), this component runs in a separate, dedicated container.

---

### 1. Requirements & Setup

#### Dataset (COCO 2017)
You must download and extract the COCO 2017 dataset beforehand. The directory structure should look like this:

```text
coco/
├── annotations/
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   └── ...
├── train2017/
│   ├── 000000000009.jpg
│   └── ...
└── val2017/
    ├── 000000000139.jpg
    └── ...
```

#### Container Environment
We provide a Singularity/Apptainer definition file to build the required environment:
* **File:** `od_mmdet.def`
* **Contents:** MMDetection, mmcv-full, and compatible PyTorch/CUDA versions.

---

### 2. Configuration

Detection-specific settings are located in `config/od_config.py`. You must set the paths to match your environment.

**Key Parameters:**

| Parameter | Description | Options |
| :--- | :--- | :--- |
| `MODEL_TYPE` | Backbone architecture | `"swin"` or `"resnet"` |
| `SWIN_VARIANT` | Swin size (if Swin is used) | `"tiny"`, `"small"`, `"base"`, `"large"` |
| `PROJECT_ROOT` | Absolute path to this repo | `"/home/user/Machine-Learning-Project"` |
| `DATA_ROOT` | Path containing the `coco` folder | `"/home/user/datasets"` |

---

### 3. Execution

#### Option A: Running on Cluster (Slurm)
Once the `od_mmdet.sif` container is built, submit the job script:

```bash
sbatch job_od.slurm
```

#### Option B: Running Locally
1.  Install the dependencies listed in `requirements_mmdet.txt`.
2.  Run the detection entry point:

```bash
python main_od.py
```

---

### 4. Output & Artifacts

Detection runs are saved in an MMDetection-style structure under the `runs/` directory.

**Directory Naming Convention:**
`cascade_mask_rcnn_{MODEL_TYPE}_{VARIANT}_fpn_coco`

**Example Structure:**
```text
runs/
└── cascade_mask_rcnn_swin_tiny_fpn_coco/
    ├── od_config.py              # Your user config copy
    ├── last_checkpoint           # Symlink to latest weight
    └── 20260104_220010/          # Timestamped run folder
        ├── 20260104_220010.log       # Training logs
        └── vis_data/                 # Visualization data
            ├── scalars.json          # Loss metrics for plotting
            ├── config.py             # Final resolved MMDetection config
            └── 20260104_220010.json
```

---

## 🎨 Semantic Segmentation on ADE20K (UPerNet)

This project extends the analysis to semantic segmentation on **ADE20K** (150 classes, 512×512 resolution). We implement **UPerNet** with **Pyramid Pooling Module (PPM)** and **Feature Pyramid Network (FPN)**, supporting **Swin Transformer**, **ResNet**, and **DeiT** backbones with ImageNet-1K pretrained weights.

> **Note:** Uses a separate entry point (`main_segmentation.py`) and configuration (`config/ade20k_config.py`) to avoid interference with classification.

---

### 1. Requirements & Setup

#### Dataset (ADE20K)
Auto-downloaded on first run. Expected structure:

```text
ADE20K/
├── images/training/        # 20,210 images
├── images/validation/      # 2,000 images
└── annotations/            # Segmentation masks
```

---

### 2. Configuration

Segmentation settings in `config/ade20k_config.py`. Choose backbone in `main_segmentation.py`:

```python
ENCODER_TYPE = "swin"  # Options: "swin", "resnet", "deit"
```

**Key Parameters:**

| Parameter | Description | Options |
| :--- | :--- | :--- |
| `ENCODER_TYPE` | Backbone architecture | `"swin"`, `"resnet"`, `"deit"` |
| `SWIN_CONFIG["variant"]` | Swin size (if Swin) | `"tiny"` (60M encoder params) |
| `RESNET_CONFIG["variant"]` | ResNet depth (if ResNet) | `"resnet101"` (86M encoder params) |
| `DEIT_CONFIG["variant"]` | DeiT model (if DeiT) | `"deit_small_patch16_224"` (22M encoder params) |
| `batch_size` | Training batch size | `8` (RTX 4070 12GB), `16` (A100 40GB+) |
| `num_epochs` | Training duration | `160` (paper setting) |
| `learning_rate` | AdamW learning rate | `6e-5` (paper setting) |

**Training Configuration:**
```python
TRAINING_CONFIG = {
    "learning_rate": 6e-5,
    "num_epochs": 160,
    "warmup_epochs": 2,
    "weight_decay": 0.01,
    "mixed_precision": True,  # bf16 on modern GPUs
}

DATA_CONFIG = {
    "batch_size": 8,          # 12GB GPU constraint
    "img_size": 512,          # ADE20K standard
}
```

---

### 3. Execution

#### Local
```bash
python main_segmentation.py
```

#### Cluster (Slurm)
```bash
sbatch job_segmentation.slurm
squeue -u $USER
```

---

### 4. Output & Artifacts

Runs saved to `runs/Semantic_segmentation_{backbone}_upernet/`:

**Directory Structure:**
```text
runs/
└── Semantic_segmentation_swin_upernet/
    ├── training.log              # Epoch metrics (mIoU, Pixel Acc)
    ├── config.json               # Configuration snapshot
    ├── best_checkpoint.pth       # Best mIoU weights
    └── results.json              # Final metrics
```

**Training Log Example:**
```
Epoch 127/160 - Train Loss: 0.3142, Val Loss: 0.4851
→ New best mIoU: 44.41% (prev: 44.38%)
Pixel Accuracy: 80.51%
Checkpoint saved: best_checkpoint.pth
```

---

### 5. Benchmark Results

Performance on **ADE20K validation** (ImageNet-1K pretrained backbones, UPerNet decoder):

| Backbone | mIoU (%) | Pixel Acc (%) | Params (Encoder + Head) |
|----------|----------|---------------|-------------------------|
| **Swin-Tiny** | **44.41** | **80.51** | 60M + 30M |
| DeiT-Small | 41.34 | 79.66 | 22M + 30M |
| ResNet-101 | 40.39 | 78.95 | 86M + 30M |

**Observations:**
- Swin-T achieves best performance with hierarchical features
- DeiT-S requires MultiLevelNeck for pseudo-hierarchical features
- ResNet-101 provides strong convolutional baseline