"""
Quick test to verify swin_deformable model can be instantiated and run forward pass.
"""

import torch
import sys

sys.path.insert(0, ".")

from config.imagenet_config import MODEL_CONFIGS
from src.models.model_factory import create_model


def test_deformable_model():
    """Test deformable model instantiation and forward pass."""
    print("=" * 80)
    print("Testing Swin Deformable Model")
    print("=" * 80)

    # Get deformable config
    config = MODEL_CONFIGS["swin_deformable"]
    print(f"\nModel config: {config['type']}")
    print(f"Embed dim: {config['embed_dim']}")
    print(f"Depths: {config['depths']}")
    print(f"Num heads: {config['num_heads']}")
    print(f"Use deformable attention: {config.get('use_deformable_attn', False)}")
    print(
        f"Num sampling points: {config.get('deformable_config', {}).get('num_points', 4)}"
    )

    # Create model
    print("\n" + "-" * 80)
    print("Creating model...")
    try:
        model = create_model(config)
        print("✓ Model created successfully")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test forward pass
    print("\n" + "-" * 80)
    print("Testing forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            # Create dummy input: batch_size=2, channels=3, height=224, width=224
            x = torch.randn(2, 3, 224, 224)
            print(f"Input shape: {x.shape}")

            # Forward pass
            output = model(x)
            print(f"Output shape: {output.shape}")
            print(f"✓ Forward pass successful")

            # Check output shape
            assert output.shape == (2, 1000), f"Expected (2, 1000), got {output.shape}"
            print(f"✓ Output shape correct: {output.shape}")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_deformable_model()
    sys.exit(0 if success else 1)
