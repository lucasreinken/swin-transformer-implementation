"""
Tests for complete Swin Transformer model.
"""

import pytest
import torch
import torch.nn as nn

from src.models.swin import (
    SwinTransformerModel,
    swin_tiny_patch4_window7_224,
    swin_small_patch4_window7_224,
    swin_base_patch4_window7_224,
    swin_large_patch4_window7_224,
)


class TestSwinTransformerModel:
    """Test complete Swin Transformer model."""

    def test_swin_model_instantiation(self):
        """Test basic model creation."""
        model = SwinTransformerModel(
            img_size=224,
            patch_size=4,
            embedding_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            num_classes=1000,
        )

        assert model.num_layers == 4
        assert model.num_features == 768  # 96 * 2^(4-1)
        assert model.patches_resolution == [56, 56]  # 224/4

    def test_swin_model_forward_pass(self):
        """Test complete forward pass."""
        model = SwinTransformerModel(
            img_size=32,  # CIFAR-10 size
            patch_size=4,
            embedding_dim=96,
            depths=[2, 2, 2, 2],  # Fewer stages for small images
            num_heads=[3, 6, 12, 24],
            window_size=4,  # Smaller windows
            num_classes=10,
        )

        # CIFAR-10 batch
        x = torch.randn(2, 3, 32, 32)
        output = model(x)

        assert output.shape == (2, 10), f"Expected [2, 10], got {output.shape}"

    def test_swin_model_features_extraction(self):
        """Test feature extraction without classification head."""
        model = SwinTransformerModel(
            img_size=32,
            patch_size=4,
            embedding_dim=96,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            num_classes=10,
            window_size=4,
        )

        x = torch.randn(2, 3, 32, 32)
        features = model.forward_features(x)

        # Should be [B, num_patches, embed_dim] after last stage
        expected_patches = (32 // 4) // (2**3)  # 8 -> 4 -> 2 -> 1
        expected_dim = 96 * (2**3)  # 96 -> 192 -> 384 -> 768

        assert features.shape == (2, expected_patches * expected_patches, expected_dim)

    def test_swin_model_config_storage(self):
        """Test that configuration is properly stored."""
        config = {
            "img_size": 224,
            "patch_size": 4,
            "embedding_dim": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
            "num_classes": 1000,
        }

        model = SwinTransformerModel(**config)

        assert model.config["img_size"] == 224
        assert model.config["embedding_dim"] == 96
        assert model.config["depths"] == [2, 2, 6, 2]

    def test_swin_model_info(self):
        """Test model information retrieval."""
        model = SwinTransformerModel(
            img_size=32,
            patch_size=4,
            embedding_dim=96,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            num_classes=10,
            window_size=4,
        )

        info = model.get_model_info()

        assert info["model_type"] == "SwinTransformer"
        assert "parameter_count" in info
        assert isinstance(info["parameter_count"], int)
        assert info["parameter_count"] > 0

    def test_swin_tiny_variant(self):
        """Test Swin-Tiny variant creation."""
        model = swin_tiny_patch4_window7_224(num_classes=1000)

        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        output = model(x)

        assert output.shape == (1, 1000)

        # Check configuration
        assert model.config["embedding_dim"] == 96
        assert model.config["depths"] == [2, 2, 6, 2]

    def test_swin_base_variant(self):
        """Test Swin-Base variant creation."""
        model = swin_base_patch4_window7_224(num_classes=1000)

        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        output = model(x)

        assert output.shape == (1, 1000)

        # Check configuration
        assert model.config["embedding_dim"] == 128
        assert model.config["depths"] == [2, 2, 18, 2]

    def test_swin_model_parameter_count(self):
        """Test that parameter counts are reasonable."""
        # Tiny model
        tiny = swin_tiny_patch4_window7_224(num_classes=1000)
        tiny_params = sum(p.numel() for p in tiny.parameters())

        # Should be around 28M parameters for Swin-Tiny
        assert (
            20_000_000 < tiny_params < 35_000_000
        ), f"Tiny model has {tiny_params:,} parameters"

        # Base model should have more parameters
        base = swin_base_patch4_window7_224(num_classes=1000)
        base_params = sum(p.numel() for p in base.parameters())

        assert (
            base_params > tiny_params
        ), "Base model should have more parameters than Tiny"

    def test_swin_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = SwinTransformerModel(
            img_size=32,
            patch_size=4,
            embedding_dim=96,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            num_classes=10,
            window_size=4,
        )

        x = torch.randn(2, 3, 32, 32)
        target = torch.randint(0, 10, (2,))

        # Forward pass
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, target)

        # Backward pass
        loss.backward()

        # Check that gradients exist
        assert model.head.weight.grad is not None
        assert model.patch_embedding.proj.weight.grad is not None

    def test_swin_model_different_input_sizes(self):
        """Test model with different input sizes."""
        model = SwinTransformerModel(
            img_size=224,
            patch_size=4,
            embedding_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            num_classes=1000,
        )

        # Test different batch sizes
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, 224, 224)
            output = model(x)
            assert output.shape == (batch_size, 1000)


class TestSwinModelIntegration:
    """Integration tests combining multiple components."""

    def test_swin_model_with_config(self):
        """Test model creation with config dictionary."""
        from config import SWIN_CONFIG

        # Modify config for testing
        test_config = SWIN_CONFIG.copy()
        test_config.update(
            {
                "img_size": 32,
                "depths": [2, 2, 2, 2],  # Adjust for small images
                "window_size": 4,
            }
        )

        model = SwinTransformerModel(**test_config)

        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        output = model(x)

        assert output.shape == (2, test_config["num_classes"])

    def test_swin_model_training_mode(self):
        """Test model behavior in training vs eval mode."""
        model = SwinTransformerModel(
            img_size=32,
            patch_size=4,
            embedding_dim=96,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            drop_path_rate=0.1,  # Should affect training mode
            num_classes=10,
            window_size=4,
        )

        x = torch.randn(2, 3, 32, 32)

        # Training mode
        model.train()
        out_train = model(x)

        # Eval mode
        model.eval()
        out_eval = model(x)

        # Outputs should be different due to drop path
        assert not torch.allclose(out_train, out_eval, atol=1e-6)
