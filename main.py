"""
Main orchestration file for the machine learning pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from src.data import load_data
from src.models import SimpleModel
from src.training import train_one_epoch, evaluate_model
from src.training.checkpoints import save_checkpoint, save_model_weights
from src.utils.visualization import show
from src.utils.seeds import set_random_seeds, get_worker_init_fn

from config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, VIZ_CONFIG, SEED_CONFIG


def main():
    """Main training pipeline."""
    # Set random seeds first to ensure reproducibility
    set_random_seeds(
        seed=SEED_CONFIG["seed"], deterministic=SEED_CONFIG["deterministic"]
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data with proper normalization
    print("Loading data...")
    train_generator, val_generator, test_generator = load_data(
        dataset=DATA_CONFIG["dataset"],
        n_train=DATA_CONFIG.get("n_train"),
        n_test=DATA_CONFIG.get("n_test"),
        use_batch_for_val=DATA_CONFIG.get("use_batch_for_val", False),
        val_batch=DATA_CONFIG.get("val_batch", 5),
        batch_size=DATA_CONFIG["batch_size"],
        num_workers=DATA_CONFIG["num_workers"],
        root=DATA_CONFIG["root"],
        img_size=DATA_CONFIG["img_size"],
        worker_init_fn=get_worker_init_fn(SEED_CONFIG["seed"]),
    )

    # Visualize first batch
    print("Visualizing first batch...")
    show(
        dataset=DATA_CONFIG["dataset"],
        n_images=16,
        outfile=f"{VIZ_CONFIG['output_file']}",
        figsize=VIZ_CONFIG["figsize"],
    )

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

        # Validate every epoch
        val_loss, val_accuracy = evaluate_model(model, val_generator, criterion, device)

        # Test periodically (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            test_loss, test_accuracy = evaluate_model(
                model, test_generator, criterion, device
            )
            print(
                f"Epoch {epoch+1}/{TRAINING_CONFIG['num_epochs']}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%"
            )
        else:
            print(
                f"Epoch {epoch+1}/{TRAINING_CONFIG['num_epochs']}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
            )

        if (epoch + 1) % 10 == 0:
            print(f"Saving checkpoint for epoch {epoch+1}...")
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                train_loss,
                f"checkpoints/checkpoint_epoch_{epoch+1}.pth",
            )

    print("Training completed!")

    # Final evaluation on test set
    print("Performing final evaluation on test set...")
    final_test_loss, final_test_accuracy = evaluate_model(
        model, test_generator, criterion, device
    )
    print(
        f"Final Test Results: Loss: {final_test_loss:.4f}, Accuracy: {final_test_accuracy:.2f}%"
    )

    # Save final model weights
    save_model_weights(
        model, f"trained_models/{DATA_CONFIG['dataset']}_final_model_weights.pth"
    )


if __name__ == "__main__":
    main()
