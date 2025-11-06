"""
Test script for Swin Transformer Patch Embedding layer.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import torch
from src.models.swin.patch_embedding import PatchEmbed


def test_swin_patch_embedding():
    print("Testing Swin Transformer Patch Embedding...")
    patch_embedding = PatchEmbed(
        img_size=224,  # ImageNet standard size
        patch_size=4,  # 4x4 patches
        in_channels=3,  # RGB images
        embedding_dim=96,  # Swin-Tiny embedding dimension
    )

    # Test with dummy input
    test_input = torch.randn(1, 3, 224, 224)  # [B, C, H, W]

    print(f"Input shape: {test_input.shape}")

    # Forward pass
    with torch.no_grad():
        output = patch_embedding(test_input)

    print(f"Output shape: {output.shape}")

    # Verify expected output shapes
    expected_num_patches = (224 // 4) ** 2  # 56 * 56 = 3136
    expected_shape = (1, expected_num_patches, 96)  # [B, num_patches, embedding_dim]

    assert (
        output.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {output.shape}"

    assert (
        patch_embedding.num_patches == expected_num_patches
    ), f"Wrong num_patches: {patch_embedding.num_patches}"
    assert output.shape[0] == test_input.shape[0], "Batch size mismatch"
    assert output.shape[2] == 96, f"Wrong embedding dimension: {output.shape[2]}"

    print("All shape verifications passed.")
    print(f"    - {expected_num_patches} patches created form 224x224 image")
    print(f"    - Each patch embedded to 96-dim vector")
    print(f"    - Output: {output.shape} (batch, sequence_length, embedding_dim)")

    # Test parameter count
    total_params = sum(p.numel() for p in patch_embedding.parameters())
    print(f"    - Total parameters in Patch Embedding: {total_params:,}")


def test_different_image_sizes():
    """Test Patch Embedding with different image sizes."""

    sizes = [224, 384]

    for img_size in sizes:
        patch_embedding = PatchEmbed(
            img_size=img_size, patch_size=4, in_channels=3, embedding_dim=96
        )
        x = torch.randn(1, 3, img_size, img_size)

        with torch.no_grad():
            output = patch_embedding(x)

        expected_patches = (img_size // 4) ** 2
        expected_shape = (1, expected_patches, 96)

        assert (
            output.shape == expected_shape
        ), f"Failed for size {img_size}: {output.shape} vs {expected_shape}"
        print(f"{img_size}x{img_size} -> {expected_patches} patches")
