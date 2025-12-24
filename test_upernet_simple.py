"""
Minimal UperNet test for CPU environments - tests basic functionality only.
"""
import torch
from config.ade20k_config import SWIN_CONFIG, DOWNSTREAM_CONFIG
from src.models import create_segmentation_model

def main():
    print("=" * 70)
    print("Minimal UperNet Test (CPU-friendly)")
    print("=" * 70)
    
    # Use CPU explicitly
    device = torch.device('cpu')
    
    # Test 1: Model creation
    print("\n✓ Test 1: Creating segmentation model...")
    try:
        model = create_segmentation_model(SWIN_CONFIG, DOWNSTREAM_CONFIG)
        model = model.to(device)
        model.eval()
        print("  SUCCESS: Model created")
    except Exception as e:
        print(f"  FAILED: {e}")
        return
    
    # Test 2: Single small forward pass (minimal memory)
    print("\n✓ Test 2: Testing forward pass (batch=1, 256x256)...")
    try:
        # Use smaller resolution to save memory
        x = torch.randn(1, 3, 256, 256, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        expected_shape = (1, 150, 256, 256)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"  SUCCESS: Output shape {output.shape}")
        
        # Clean up
        del x, output
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"  FAILED: {e}")
        return
    
    # Test 3: Verify multi-scale features work
    print("\n✓ Test 3: Testing multi-scale feature extraction...")
    try:
        x = torch.randn(1, 3, 256, 256, device=device)
        
        with torch.no_grad():
            # Access encoder directly
            features = model.encoder.forward_features(x, return_multi_scale=True)
        
        assert len(features) == 4, f"Expected 4 feature scales, got {len(features)}"
        print(f"  SUCCESS: Got {len(features)} feature scales")
        
        # Print feature shapes
        for i, feat in enumerate(features):
            print(f"    Stage {i}: {feat.shape}")
        
        del x, features
        
    except Exception as e:
        print(f"  FAILED: {e}")
        return
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
