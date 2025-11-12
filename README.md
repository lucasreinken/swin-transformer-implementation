# Machine-Learning-Project

## Configuration

### Dataset Selection
Edit `config/__init__.py` to choose your dataset:
```python
# Choose one dataset
DATASET = "cifar10"    # CIFAR-10 (10 classes)
# DATASET = "cifar100"  # CIFAR-100 (100 classes)
# DATASET = "imagenet"  # ImageNet-1K (1000 classes)
```

### Data Root Configuration
Edit `config/__init__.py` to set your data directory:
```python
# For local development:
DATA_ROOT = "./datasets"

# For cluster (comment out local, uncomment cluster):
# DATA_ROOT = "/home/space/datasets"
```

### Running the Project
```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python main.py
```
