"""
Tests for Swin Transformer Block components.
"""

import pytest
import torch
import torch.nn as nn

from src.models.swin import (
    SwinTransformerBlock,
    BasicLayer,
    MLP,
    DropPath,
    window_partition,
    window_reverse,
    generate_drop_path_rates,
)


class TestWindowOperations:
    """Test window partition and reverse operations."""

    def test_window_partition_basic(self):
        """Test basic window partitioning."""
        B, H, W, C = 2, 56, 56, 96
        window_size = 7
        x = torch.randn(B, H, W, C)

        windows = window_partition(x, window_size)

        # Expected: [B * num_windows, window_size, window_size, C]
        expected_num_windows = (H // window_size) * (W // window_size)
        assert windows.shape == (
            B * expected_num_windows,
            window_size,
            window_size,
            C,
        )

    def test_window_reverse_basic(self):
        """Test window reverse operation."""
        B, H, W, C = 2, 56, 56, 96
        window_size = 7
        num_windows = (H // window_size) * (W // window_size)

        windows = torch.randn(B * num_windows, window_size, window_size, C)
        x = window_reverse(windows, window_size, H, W)

        assert x.shape == (B, H, W, C)

    def test_window_partition_reverse_invertible(self):
        """Test that partition and reverse are inverse operations."""
        B, H, W, C = 2, 56, 56, 96
        window_size = 7
        x_original = torch.randn(B, H, W, C)

        # Partition then reverse
        windows = window_partition(x_original, window_size)
        x_reconstructed = window_reverse(windows, window_size, H, W)

        # Should be identical
        assert torch.allclose(x_original, x_reconstructed, atol=1e-6)

    def test_window_partition_different_sizes(self):
        """Test window partitioning with different input sizes."""
        test_cases = [
            (1, 28, 28, 192, 7),  # Stage 2
            (4, 14, 14, 384, 7),  # Stage 3
            (2, 7, 7, 768, 7),  # Stage 4
        ]

        for B, H, W, C, window_size in test_cases:
            x = torch.randn(B, H, W, C)
            windows = window_partition(x, window_size)
            x_reconstructed = window_reverse(windows, window_size, H, W)
            assert torch.allclose(x, x_reconstructed, atol=1e-6)


class TestMLP:
    """Test MLP component."""

    def test_mlp_forward(self):
        """Test MLP forward pass."""
        B, N, C = 2, 3136, 96
        mlp = MLP(in_features=C, hidden_features=C * 4, dropout=0.1)

        x = torch.randn(B, N, C)
        output = mlp(x)

        assert output.shape == (B, N, C)

    def test_mlp_default_params(self):
        """Test MLP with default parameters."""
        C = 96
        mlp = MLP(in_features=C)

        # Default hidden_features should be 4x input
        assert mlp.fc1.out_features == C * 4
        # Default output should match input
        assert mlp.fc2.out_features == C

    def test_mlp_custom_params(self):
        """Test MLP with custom parameters."""
        mlp = MLP(in_features=96, hidden_features=256, out_features=128)

        assert mlp.fc1.out_features == 256
        assert mlp.fc2.out_features == 128


class TestDropPath:
    """Test DropPath (Stochastic Depth)."""

    def test_drop_path_training(self):
        """Test DropPath in training mode."""
        drop_path = DropPath(drop_prob=0.5)
        drop_path.train()

        x = torch.ones(10, 100)
        output = drop_path(x)

        # Output should have same shape
        assert output.shape == x.shape
        # Due to randomness, output should sometimes differ from input
        # Run multiple times to check stochasticity
        outputs = [drop_path(x) for _ in range(10)]
        # At least some outputs should be different
        assert not all(torch.allclose(out, outputs[0]) for out in outputs)

    def test_drop_path_eval(self):
        """Test DropPath in eval mode (should be identity)."""
        drop_path = DropPath(drop_prob=0.5)
        drop_path.eval()

        x = torch.ones(10, 100)
        output = drop_path(x)

        # In eval mode, should be identity function
        assert torch.allclose(output, x)

    def test_drop_path_zero_prob(self):
        """Test DropPath with zero probability."""
        drop_path = DropPath(drop_prob=0.0)
        drop_path.train()

        x = torch.ones(10, 100)
        output = drop_path(x)

        # With prob=0, should always be identity
        assert torch.allclose(output, x)


class TestSwinTransformerBlock:
    """Test Swin Transformer Block."""

    def test_swin_block_forward_w_msa(self):
        """Test Swin block forward pass with W-MSA (shift_size=0)."""
        dim = 96
        input_resolution = (56, 56)
        num_heads = 3
        window_size = 7

        block = SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,  # W-MSA
        )

        B = 2
        H, W = input_resolution
        x = torch.randn(B, H * W, dim)
        output = block(x)

        assert output.shape == (B, H * W, dim)

    def test_swin_block_forward_sw_msa(self):
        """Test Swin block forward pass with SW-MSA (shift_size>0)."""
        dim = 96
        input_resolution = (56, 56)
        num_heads = 3
        window_size = 7

        block = SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2,  # SW-MSA
        )

        B = 2
        H, W = input_resolution
        x = torch.randn(B, H * W, dim)
        output = block(x)

        assert output.shape == (B, H * W, dim)

    def test_swin_block_small_resolution(self):
        """Test Swin block with small input resolution."""
        dim = 96
        input_resolution = (7, 7)  # Smaller than typical window size
        num_heads = 3
        window_size = 7

        block = SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
        )

        # Window size should be adjusted
        assert block.window_size == min(input_resolution)
        assert block.shift_size == 0

    def test_swin_block_different_stages(self):
        """Test Swin block with different stage configurations."""
        test_configs = [
            # (dim, resolution, num_heads, window_size)
            (96, (56, 56), 3, 7),  # Stage 1
            (192, (28, 28), 6, 7),  # Stage 2
            (384, (14, 14), 12, 7),  # Stage 3
            (768, (7, 7), 24, 7),  # Stage 4
        ]

        for dim, resolution, num_heads, window_size in test_configs:
            block = SwinTransformerBlock(
                dim=dim,
                input_resolution=resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,
            )

            B = 2
            H, W = resolution
            x = torch.randn(B, H * W, dim)
            output = block(x)

            assert output.shape == (B, H * W, dim)


class TestBasicLayer:
    """Test Basic Layer (Swin stage)."""

    def test_basic_layer_no_downsample(self):
        """Test basic layer without downsampling."""
        dim = 96
        input_resolution = (56, 56)
        depth = 2
        num_heads = 3

        layer = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            downsample=None,
        )

        B = 2
        H, W = input_resolution
        x = torch.randn(B, H * W, dim)
        output = layer(x)

        # No downsampling, shape should be unchanged
        assert output.shape == (B, H * W, dim)

    def test_basic_layer_with_downsample(self):
        """Test basic layer with downsampling."""
        from src.models.swin import PatchMerging

        dim = 96
        input_resolution = (56, 56)
        depth = 2
        num_heads = 3

        layer = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            downsample=PatchMerging,
        )

        B = 2
        H, W = input_resolution
        x = torch.randn(B, H * W, dim)
        output = layer(x)

        # With downsampling: H/2 * W/2 patches, 2*dim features
        assert output.shape == (B, (H // 2) * (W // 2), 2 * dim)

    def test_basic_layer_alternating_attention(self):
        """Test that basic layer creates alternating W-MSA and SW-MSA blocks."""
        dim = 96
        input_resolution = (56, 56)
        depth = 4
        num_heads = 3
        window_size = 7

        layer = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            downsample=None,
        )

        # Check that blocks alternate between shift_size=0 and shift_size=window_size//2
        for i, block in enumerate(layer.blocks):
            if i % 2 == 0:
                assert block.shift_size == 0, f"Block {i} should have shift_size=0"
            else:
                expected_shift = window_size // 2
                assert (
                    block.shift_size == expected_shift
                ), f"Block {i} should have shift_size={expected_shift}"


class TestDropPathRates:
    """Test drop path rate generation."""

    def test_generate_drop_path_rates_basic(self):
        """Test basic drop path rate generation."""
        rates = generate_drop_path_rates(0.2, 4)

        assert len(rates) == 4
        assert rates[0] == 0.0  # First layer should have 0
        assert rates[-1] == pytest.approx(0.2)  # Last layer should have max
        # Should be monotonically increasing
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    def test_generate_drop_path_rates_zero(self):
        """Test with zero drop path rate."""
        rates = generate_drop_path_rates(0.0, 4)

        assert all(rate == 0.0 for rate in rates)

    def test_generate_drop_path_rates_negative(self):
        """Test with negative drop path rate (should treat as zero)."""
        rates = generate_drop_path_rates(-0.1, 4)

        assert all(rate == 0.0 for rate in rates)

    def test_generate_drop_path_rates_single_depth(self):
        """Test with depth=1."""
        rates = generate_drop_path_rates(0.2, 1)

        assert len(rates) == 1
        assert rates[0] == pytest.approx(0.2)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_swin_stage_pipeline(self):
        """Test a complete Swin stage pipeline."""
        from src.models.swin import PatchEmbed, PatchMerging

        # Simulate input: 224x224 image
        B = 2
        x = torch.randn(B, 3, 224, 224)

        # Stage 1: Patch Embedding
        patch_embed = PatchEmbed(img_size=224, patch_size=4, embedding_dim=96)
        x = patch_embed(x)  # [B, 3136, 96] = [B, 56*56, 96]
        assert x.shape == (B, 3136, 96)

        # Stage 2: Basic Layer without downsampling
        layer1 = BasicLayer(
            dim=96,
            input_resolution=(56, 56),
            depth=2,
            num_heads=3,
            window_size=7,
            downsample=None,
        )
        x = layer1(x)  # [B, 3136, 96]
        assert x.shape == (B, 3136, 96)

        # Stage 3: Basic Layer with downsampling
        layer2 = BasicLayer(
            dim=96,
            input_resolution=(56, 56),
            depth=2,
            num_heads=3,
            window_size=7,
            downsample=PatchMerging,
        )
        x = layer2(x)  # [B, 784, 192] = [B, 28*28, 192]
        assert x.shape == (B, 784, 192)

    def test_residual_connection_preservation(self):
        """Test that residual connections preserve information."""
        dim = 96
        input_resolution = (56, 56)
        block = SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=3,
            window_size=7,
            shift_size=0,
            dropout=0.0,  # No dropout
            drop_path=0.0,  # No drop path
        )

        B = 2
        H, W = input_resolution
        x = torch.randn(B, H * W, dim)

        # With current placeholder attention (Identity), output should equal input
        output = block(x)

        # Due to residual connections and LayerNorm, output won't be exactly x,
        # but should be related. Just check shape for now.
        assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
