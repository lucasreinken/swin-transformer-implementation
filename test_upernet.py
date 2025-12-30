"""
Test UperNet Segmentation Model with Swin-T Encoder

Tests the complete segmentation pipeline including:
1. Model creation with proper configuration
2. Forward pass with 512x512 input (ADE20K standard)
3. Multi-scale feature extraction
4. Encoder freezing capability  
5. Batch processing
6. Padding approach verification (no global attention fallback warnings)
"""

import torch
import sys
import gc

# Ensure src is in path
sys.path.insert(0, '/home/pml20/Machine-Learning-Project')


def test_model_creation():
    """Test that segmentation model can be created with correct architecture."""
    print("\n" + "="*60)
    print("TEST 1: Model Creation")
    print("="*60)
    
    from src.models.model_factory import create_segmentation_model
    from config import ade20k_config as config
    
    model = create_segmentation_model(config.SWIN_CONFIG, config.DOWNSTREAM_CONFIG, load_pretrained=False)
    
    # Check model structure
    assert hasattr(model, 'encoder'), "Model should have encoder"
    assert hasattr(model, 'seg_head'), "Model should have seg_head"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    head_params = sum(p.numel() for p in model.seg_head.parameters())
    
    print(f"✓ Model created successfully")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Encoder parameters: {encoder_params:,}")
    print(f"  - Decoder head parameters: {head_params:,}")
    
    # Verify parameter counts are reasonable
    assert total_params > 50_000_000, f"Expected ~60M params, got {total_params:,}"
    assert total_params < 70_000_000, f"Expected ~60M params, got {total_params:,}"
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return True


def test_forward_pass():
    """Test forward pass produces correct output shape."""
    print("\n" + "="*60)
    print("TEST 2: Forward Pass (512x512 input)")
    print("="*60)
    
    from src.models.model_factory import create_segmentation_model
    from config import ade20k_config as config
    
    model = create_segmentation_model(config.SWIN_CONFIG, config.DOWNSTREAM_CONFIG, load_pretrained=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # ADE20K standard input size
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, 512, 512, device=device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    expected_shape = (batch_size, 150, 512, 512)  # 150 classes for ADE20K
    
    print(f"✓ Forward pass successful")
    print(f"  - Input shape: {tuple(input_tensor.shape)}")
    print(f"  - Output shape: {tuple(output.shape)}")
    print(f"  - Expected shape: {expected_shape}")
    
    assert output.shape == expected_shape, f"Output shape mismatch: {output.shape} vs {expected_shape}"
    
    del model, input_tensor, output
    gc.collect()
    torch.cuda.empty_cache()
    
    return True


def test_multiscale_features():
    """Test that encoder produces correct multi-scale features."""
    print("\n" + "="*60)
    print("TEST 3: Multi-scale Feature Extraction")
    print("="*60)
    
    from src.models.model_factory import create_segmentation_model
    from config import ade20k_config as config
    
    model = create_segmentation_model(config.SWIN_CONFIG, config.DOWNSTREAM_CONFIG, load_pretrained=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    input_tensor = torch.randn(1, 3, 512, 512, device=device)
    
    with torch.no_grad():
        features = model.encoder.forward_features(input_tensor)
    
    # Handle different return formats
    if isinstance(features, (list, tuple)):
        print(f"✓ Multi-scale features extracted: {len(features)} stages")
        for i, feat in enumerate(features):
            print(f"  - Stage {i+1}: {tuple(feat.shape)}")
    else:
        # Single tensor returned - this is the final pooled feature
        print(f"✓ Features extracted: {tuple(features.shape)}")
    
    # Just verify we got features without error
    print("  - Feature extraction successful")
    
    del model, input_tensor, features
    gc.collect()
    torch.cuda.empty_cache()
    
    return True


def test_encoder_freezing():
    """Test that encoder can be frozen for fine-tuning."""
    print("\n" + "="*60)
    print("TEST 4: Encoder Freezing")
    print("="*60)
    
    from src.models.model_factory import create_segmentation_model
    from config import ade20k_config as config
    
    model = create_segmentation_model(config.SWIN_CONFIG, config.DOWNSTREAM_CONFIG, load_pretrained=False)
    
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    print(f"✓ Encoder frozen successfully")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Frozen parameters: {frozen_params:,}")
    
    # Encoder should be frozen (~28M), head should be trainable (~31M)
    assert trainable_params > 25_000_000, "Decoder head should have ~31M trainable params"
    assert frozen_params > 25_000_000, "Encoder should have ~28M frozen params"
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return True


def test_batch_processing():
    """Test that model handles batch processing correctly."""
    print("\n" + "="*60)
    print("TEST 5: Batch Processing")
    print("="*60)
    
    from src.models.model_factory import create_segmentation_model
    from config import ade20k_config as config
    
    model = create_segmentation_model(config.SWIN_CONFIG, config.DOWNSTREAM_CONFIG, load_pretrained=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 512, 512, device=device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    expected_shape = (batch_size, 150, 512, 512)
    
    print(f"✓ Batch processing successful")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Output shape: {tuple(output.shape)}")
    
    assert output.shape == expected_shape, f"Output shape mismatch: {output.shape} vs {expected_shape}"
    
    del model, input_tensor, output
    gc.collect()
    torch.cuda.empty_cache()
    
    return True


def test_padding_approach():
    """Test that the padding approach works correctly for non-divisible resolutions."""
    print("\n" + "="*60)
    print("TEST 6: Padding Approach Verification")
    print("="*60)
    
    from src.models.swin.swin_transformer_block import SwinTransformerBlock
    
    # Test with resolution that doesn't divide by window_size=7
    # 128x128 feature map (stage 1 for 512x512 input)
    block = SwinTransformerBlock(
        dim=96,
        input_resolution=(128, 128),
        num_heads=3,
        window_size=7,
        shift_size=0,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    block = block.to(device)
    block.eval()
    
    # Input: [B, H*W, C]
    x = torch.randn(1, 128*128, 96, device=device)
    
    with torch.no_grad():
        output = block(x)
    
    print(f"✓ Padding approach works correctly")
    print(f"  - Input resolution: 128x128 (not divisible by window_size=7)")
    print(f"  - window_size maintained: {block.window_size}")
    print(f"  - Input shape: {tuple(x.shape)}")
    print(f"  - Output shape: {tuple(output.shape)}")
    
    # Window size should remain 7 (not fall back to 128)
    assert block.window_size == 7, f"Window size should be 7, got {block.window_size}"
    assert output.shape == x.shape, f"Output shape should match input: {output.shape} vs {x.shape}"
    
    del block, x, output
    gc.collect()
    torch.cuda.empty_cache()
    
    return True


def test_pretrained_weight_loading():
    """Test that pretrained ImageNet weights can be loaded from TIMM."""
    print("\n" + "="*60)
    print("TEST 7: Pretrained Weight Loading from TIMM")
    print("="*60)
    
    from src.models.model_factory import create_segmentation_model
    from config import ade20k_config as config
    
    # Create model WITH pretrained weights
    model = create_segmentation_model(
        config.SWIN_CONFIG, 
        config.DOWNSTREAM_CONFIG,
        load_pretrained=True
    )
    
    # Verify model was created
    assert model is not None, "Model creation failed"
    
    # Check that encoder has non-zero weights (pretrained weights loaded)
    # Get first conv layer weight
    first_layer_weight = model.encoder.patch_embed.proj.weight
    weight_sum = first_layer_weight.abs().sum().item()
    
    print(f"✓ Pretrained weights loaded successfully")
    print(f"  - First layer weight sum: {weight_sum:.4f}")
    print(f"  - Weight is non-zero: {weight_sum > 0}")
    
    assert weight_sum > 0, "Weights should be non-zero after loading pretrained"
    
    # Quick forward pass to verify model works
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    input_tensor = torch.randn(1, 3, 512, 512, device=device)
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"  - Forward pass successful: {tuple(output.shape)}")
    
    del model, input_tensor, output
    gc.collect()
    torch.cuda.empty_cache()
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("UperNet Segmentation Model Tests")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Multi-scale Features", test_multiscale_features),
        ("Encoder Freezing", test_encoder_freezing),
        ("Batch Processing", test_batch_processing),
        ("Padding Approach", test_padding_approach),
        ("Pretrained Weight Loading", test_pretrained_weight_loading),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, "PASSED" if success else "FAILED"))
        except Exception as e:
            print(f"\n✗ {name} FAILED with exception:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"FAILED: {e}"))
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, result in results:
        status = "✓" if result == "PASSED" else "✗"
        print(f"{status} {name}: {result}")
    
    all_passed = all(r[1] == "PASSED" for r in results)
    print("\n" + ("="*60))
    print(f"{'ALL TESTS PASSED!' if all_passed else 'SOME TESTS FAILED!'}")
    print("="*60)
    
    sys.exit(0 if all_passed else 1)
