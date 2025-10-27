"""
Main orchestration file for the machine learning pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from src.data import load_data
from src.models import SimpleModel
from src.training import train_one_epoch, evaluate_model
from src.utils import show
from config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, VIZ_CONFIG


def main():
    """Main training pipeline."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data with proper normalization
    print("Loading data...")
    # CIFAR-10 mean and std for normalization
    normalize_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_generator, test_generator = load_data(
        dataset=DATA_CONFIG["dataset"],
        transformation=normalize_transform,
        n_train=DATA_CONFIG["n_train"],
        n_test=DATA_CONFIG["n_test"],
        batch_size=DATA_CONFIG["batch_size"],
    )

    # Visualize first batch
    print("Visualizing first batch...")
    for x, y in train_generator:
        print(
            f"Sample shape: {x.shape}, Type: {x.dtype}, "
            f"Label: {y[0]}, Min/Max: {x.min().item()}/{x.max().item()}"
        )
        show(x, y=y, outfile=VIZ_CONFIG["output_file"], figsize=VIZ_CONFIG["figsize"])
        break

    # Initialize model
    print("Initializing model...")
    model = SimpleModel(
        input_dim=MODEL_CONFIG["input_dim"],
        hidden_dims=MODEL_CONFIG["hidden_dims"],
        num_classes=MODEL_CONFIG["num_classes"],
        dropout_rate=MODEL_CONFIG["dropout_rate"],
        use_batch_norm=MODEL_CONFIG["use_batch_norm"],
    ).to(device)

    # Print model architecture
    print(
        f"Model architecture: Input({MODEL_CONFIG['input_dim']}) -> "
        f"Hidden{MODEL_CONFIG['hidden_dims']} -> Output({MODEL_CONFIG['num_classes']})"
    )
    print(
        f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
    )

    # Training loop
    print("Starting training...")
    for epoch in range(TRAINING_CONFIG["num_epochs"]):
        train_loss = train_one_epoch(
            model, train_generator, criterion, optimizer, device
        )
        test_loss, accuracy = evaluate_model(model, test_generator, criterion, device)

        print(
            f"Epoch {epoch+1}/{TRAINING_CONFIG['num_epochs']}: "
            f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
            f"Accuracy: {accuracy:.2f}%"
        )

    print("Training completed!")


if __name__ == "__main__":
    main()
