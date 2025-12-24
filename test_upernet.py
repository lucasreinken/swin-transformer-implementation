"""
Simple test for UperNet segmentation model.
Tests model creation, forward pass, and multi-scale features.
"""

import torch
from config.ade20k_config import SWIN_CONFIG, DOWNSTREAM_CONFIG
from src.models import create_segmentation_model


def main():
    print("=" * 70)
    print("Testing UperNet Segmentation Model")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Test 1: Create model
    print("\n1. Creating segmentation model...")
    model = create_segmentation_model(SWIN_CONFIG, DOWNSTREAM_CONFIG)
    model = model.to(device)
    model.eval()
    
    params = model.get_num_params()
    print(f"   Total parameters: {params['total']:,}")
    print(f"   Trainable: {params['trainable']:,}")
    
    # Test 2: Forward pass
    print("\n2. Testing forward pass (batch=1, 512x512)...")
    with torch.no_grad():
        x = torch.randn(1, 3, 512, 512, device=device)
        output = model(x)
        
        expected_shape = (1, 150, 512, 512)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"   ✓ Input: {tuple(x.shape)} → Output: {tuple(output.shape)}")
        
        del x, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Test 3: Multi-scale features
    print("\n3. Testing multi-scale features...")
    with torch.no_grad():
        x = torch.randn(1, 3, 512, 512, device=device)
        features = model.encoder(x, return_multi_scale=True)
        
        assert len(features) == 4, f"Expected 4 scales, got {len(features)}"
        print(f"   ✓ Feature scales: {len(features)}")
        for i, feat in enumerate(features):
            print(f"     Stage {i+1}: {tuple(feat.shape)}")
        
        del x, features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Test 4: Encoder freezing
    print("\n4. Testing encoder freezing...")
    frozen_model = create_segmentation_model(
        SWIN_CONFIG,
        {**DOWNSTREAM_CONFIG, "freeze_encoder": True}
    )
    frozen_model = frozen_model.to(device)
    
    encoder_frozen = all(not p.requires_grad for p in frozen_model.encoder.parameters())
    head_trainable = any(p.requires_grad for p in frozen_model.seg_head.parameters())
    
    assert encoder_frozen, "Encoder should be frozen"
    assert head_trainable, "Head should be trainable"
    print(f"   ✓ Encoder frozen: {encoder_frozen}")
    print(f"   ✓ Head trainable: {head_trainable}")
    
    del frozen_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
