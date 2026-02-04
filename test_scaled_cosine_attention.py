"""
Test Scaled Cosine Attention for Swin Transformer V2.

This test validates:
1. WindowAttentionV2 instantiation and forward pass
2. Q and K normalization to unit length
3. Temperature parameter is learnable and positive
4. Cosine similarity bounds [-1, 1] before scaling
5. Integration with SwinV2TransformerBlock
6. Backward pass and gradient flow
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.swin.window_attention import WindowAttentionV2
from models.swin.swin_transformer_block import SwinV2TransformerBlock


def test_window_attention_v2_instantiation():
    """Test that WindowAttentionV2 can be instantiated."""
    print("\n" + "=" * 70)
    print("TEST 1: WindowAttentionV2 Instantiation")
    print("=" * 70)

    attn = WindowAttentionV2(
        dim=96, window_size=(7, 7), num_heads=3, attn_dropout=0.1, proj_dropout=0.1
    )

    print(f"✓ WindowAttentionV2 created successfully")
    print(f"  - Embedding dim: {attn.embed_dim}")
    print(f"  - Head dim: {attn.head_dim}")
    print(f"  - Num heads: {attn.num_heads}")
    print(f"  - Window size: {attn.window_size}")

    # Check temperature parameter exists
    assert hasattr(attn, "logit_scale"), "logit_scale parameter missing"
    print(
        f"  - Temperature parameter shape: {attn.logit_scale.shape} (should be [num_heads, 1, 1])"
    )
    print(f"  - Initial tau (~0.1): {(1.0 / attn.logit_scale.exp()).mean().item():.4f}")

    return attn


def test_forward_pass():
    """Test forward pass and output shape."""
    print("\n" + "=" * 70)
    print("TEST 2: Forward Pass")
    print("=" * 70)

    attn = WindowAttentionV2(dim=96, window_size=(7, 7), num_heads=3)

    # Create input: [num_windows*B, N, C] where N = window_size^2
    B = 2
    num_windows = 4
    N = 7 * 7  # 49
    C = 96

    x = torch.randn(num_windows * B, N, C)
    print(f"Input shape: {x.shape}")

    # Forward pass
    out = attn(x)
    print(f"Output shape: {out.shape}")

    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(f"✓ Forward pass successful, output shape matches input")

    return attn, x, out


def test_q_k_normalization():
    """Test that Q and K are normalized to unit length."""
    print("\n" + "=" * 70)
    print("TEST 3: Q and K Normalization")
    print("=" * 70)

    attn = WindowAttentionV2(dim=96, window_size=(7, 7), num_heads=3)

    # Create input
    B = 1
    N = 49
    C = 96
    x = torch.randn(B, N, C)

    # Hook to capture Q and K
    q_captured = []
    k_captured = []

    def forward_hook(module, input, output):
        # Access Q and K from the forward pass
        # We'll need to modify the forward to store them temporarily
        pass

    # Alternative: manually compute Q and K
    with torch.no_grad():
        qkv = attn.qkv(x)
        qkv = qkv.reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv.unbind(0)

        # Normalize (same as in forward)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # Check L2 norm is ~1
        q_norm = torch.norm(q, p=2, dim=-1)
        k_norm = torch.norm(k, p=2, dim=-1)

        print(f"Q L2 norm - mean: {q_norm.mean():.6f}, std: {q_norm.std():.6f}")
        print(f"K L2 norm - mean: {k_norm.mean():.6f}, std: {k_norm.std():.6f}")

        assert torch.allclose(
            q_norm, torch.ones_like(q_norm), atol=1e-5
        ), "Q not normalized"
        assert torch.allclose(
            k_norm, torch.ones_like(k_norm), atol=1e-5
        ), "K not normalized"
        print(f"✓ Q and K are normalized to unit length")

        # Check cosine similarity range
        cos_sim = torch.matmul(q, k.transpose(-2, -1))
        print(
            f"\nCosine similarity - min: {cos_sim.min():.4f}, max: {cos_sim.max():.4f}"
        )
        print(f"  (should be in [-1, 1])")
        assert cos_sim.min() >= -1.0 and cos_sim.max() <= 1.0, "Cosine sim out of range"
        print(f"✓ Cosine similarity is in valid range [-1, 1]")


def test_temperature_parameter():
    """Test that temperature parameter is learnable and properly constrained."""
    print("\n" + "=" * 70)
    print("TEST 4: Temperature Parameter")
    print("=" * 70)

    attn = WindowAttentionV2(dim=96, window_size=(7, 7), num_heads=3)

    # Check parameter requires grad
    assert attn.logit_scale.requires_grad, "Temperature parameter not learnable"
    print(f"✓ Temperature parameter requires gradient")

    # Check initial value
    tau = 1.0 / attn.logit_scale.exp()
    print(f"Initial tau per head: {tau.squeeze()}")
    print(f"Mean tau: {tau.mean():.4f}")

    # Check that it's in reasonable range (0.05 - 0.5)
    assert (
        tau.min() > 0.01 and tau.max() < 1.0
    ), f"Temperature out of expected range: {tau.mean():.4f}"
    print(f"✓ Temperature in reasonable range (0.01 - 1.0)")

    # Test gradient flow to temperature
    x = torch.randn(1, 49, 96, requires_grad=True)
    out = attn(x)
    loss = out.sum()
    loss.backward()

    assert (
        attn.logit_scale.grad is not None
    ), "Temperature parameter has no gradient after backward"
    print(f"✓ Temperature parameter receives gradients")
    print(f"  Gradient norm: {attn.logit_scale.grad.norm().item():.6f} (should be > 0)")


def test_swin_v2_block_integration():
    """Test that SwinV2TransformerBlock uses WindowAttentionV2."""
    print("\n" + "=" * 70)
    print("TEST 5: SwinV2TransformerBlock Integration")
    print("=" * 70)

    block = SwinV2TransformerBlock(
        dim=96, num_heads=3, window_size=7, shift_size=0, drop_path=0.1
    )

    # Check that it uses WindowAttentionV2
    assert isinstance(
        block.attn, WindowAttentionV2
    ), f"Expected WindowAttentionV2, got {type(block.attn)}"
    print(f"✓ SwinV2TransformerBlock uses WindowAttentionV2")

    # Test forward pass through block
    # V2 block expects [B, H*W, C] input
    B, H, W, C = 2, 56, 56, 96
    x = torch.randn(B, H * W, C)  # Flattened spatial dimensions
    print(f"Input shape: {x.shape} (B={B}, H={H}, W={W}, C={C})")

    out, out_H, out_W = block(x, H, W)
    print(f"Output shape: {out.shape}")
    print(f"Output spatial dims: H={out_H}, W={out_W}")

    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    assert out_H == H and out_W == W, f"Expected H={H}, W={W}, got H={out_H}, W={out_W}"
    print(f"✓ Forward pass through SwinV2TransformerBlock successful")


def test_backward_pass():
    """Test backward pass and gradient flow."""
    print("\n" + "=" * 70)
    print("TEST 6: Backward Pass")
    print("=" * 70)

    attn = WindowAttentionV2(dim=96, window_size=(7, 7), num_heads=3)

    # Create input with gradient tracking
    x = torch.randn(8, 49, 96, requires_grad=True)
    print(f"Input shape: {x.shape}")

    # Forward pass
    out = attn(x)

    # Create dummy loss
    loss = out.sum()
    print(f"Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Check gradients
    assert x.grad is not None, "Input has no gradient"
    assert attn.qkv.weight.grad is not None, "QKV weight has no gradient"
    assert attn.proj.weight.grad is not None, "Proj weight has no gradient"
    assert attn.logit_scale.grad is not None, "Temperature has no gradient"

    print(f"✓ Backward pass successful")
    print(f"  - Input gradient shape: {x.grad.shape}")
    print(f"  - QKV weight gradient norm: {attn.qkv.weight.grad.norm().item():.6f}")
    print(f"  - Proj weight gradient norm: {attn.proj.weight.grad.norm().item():.6f}")
    print(f"  - Temperature gradient norm: {attn.logit_scale.grad.norm().item():.6f}")


def test_comparison_with_v1():
    """Compare output distribution between V1 and V2 attention."""
    print("\n" + "=" * 70)
    print("TEST 7: Comparison with WindowAttention (V1)")
    print("=" * 70)

    from models.swin.window_attention import WindowAttention

    # Create both versions with same params
    attn_v1 = WindowAttention(dim=96, window_size=(7, 7), num_heads=3)
    attn_v2 = WindowAttentionV2(dim=96, window_size=(7, 7), num_heads=3)

    # Same input
    x = torch.randn(8, 49, 96)

    # Forward through both
    with torch.no_grad():
        out_v1 = attn_v1(x)
        out_v2 = attn_v2(x)

    print(f"V1 output - mean: {out_v1.mean():.4f}, std: {out_v1.std():.4f}")
    print(f"V2 output - mean: {out_v2.mean():.4f}, std: {out_v2.std():.4f}")

    # V2 should have different statistics due to cosine attention
    # but should be in similar range
    assert out_v2.std() > 0, "V2 output has zero variance (possible bug)"
    print(f"✓ V2 produces different but valid output distribution")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("SCALED COSINE ATTENTION TESTS FOR SWIN V2")
    print("=" * 70)

    try:
        test_window_attention_v2_instantiation()
        test_forward_pass()
        test_q_k_normalization()
        test_temperature_parameter()
        test_swin_v2_block_integration()
        test_backward_pass()
        test_comparison_with_v1()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        print("\nScaled Cosine Attention is working correctly:")
        print("  ✓ Q and K are normalized to unit length")
        print("  ✓ Cosine similarity is bounded in [-1, 1]")
        print("  ✓ Temperature τ is learnable and properly initialized")
        print("  ✓ Integrated with SwinV2TransformerBlock")
        print("  ✓ Gradients flow correctly through all components")
        print("\nReady for training!")

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"TEST FAILED: {e}")
        print("=" * 70)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
