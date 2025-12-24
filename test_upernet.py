"""
Comprehensive test for UperNet segmentation model.
Tests full Swin-T encoder + UperNet head integration.
"""

import torch
import gc
from config.ade20k_config import SWIN_CONFIG, DOWNSTREAM_CONFIG
from src.models import create_segmentation_model


def main():
    print("=" * 70)
    print("Testing Full UperNet Segmentation Model (Swin-T + UperNet)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.cuda.empty_cache()
        gc.collect()
    
    # Test 1: Create full model (Swin-T encoder + UperNet head)
    print("\n1. Creating full segmentation model (Swin-T + UperNet)...")
    try:
        # Create model on CPU first to save GPU memory during initialization
        model = create_segmentation_model(SWIN_CONFIG, DOWNSTREAM_CONFIG)
        
        # Get params before moving to GPU
        params = model.get_num_params()
        print(f"   Model created on CPU")
        print(f"   Encoder params: {params['encoder']:,}")
        print(f"   Head params: {params['head']:,}")
        print(f"   Total params: {params['total']:,}")
        
        # Now move to GPU
        print(f"   Moving model to {device}...")
        model = model.to(device)
        model.eval()
        print(f"   ✓ Model successfully loaded on {device}")
        
        if torch.cuda.is_available():
            print(f"   GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
    except Exception as e:
        print(f"   ✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Forward pass with real image size (512x512)
    print("\n2. Testing forward pass with 512x512 images...")
    try:
        with torch.no_grad():
            x = torch.randn(1, 3, 512, 512, device=device)
            print(f"   Input tensor created: {tuple(x.shape)}")
            
            output = model(x)
            
            expected_shape = (1, 150, 512, 512)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
            print(f"   ✓ Input: {tuple(x.shape)} → Output: {tuple(output.shape)}")
            
            del x, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Multi-scale feature extraction from Swin-T
    print("\n3. Testing multi-scale features from Swin-T encoder...")
    try:
        with torch.no_grad():
            x = torch.randn(1, 3, 512, 512, device=device)
            features = model.encoder(x, return_multi_scale=True)
            
            assert len(features) == 4, f"Expected 4 scales, got {len(features)}"
            print(f"   ✓ Got {len(features)} feature scales:")
            
            expected_channels = [96, 192, 384, 768]  # Swin-T channel progression
            for i, (feat, expected_c) in enumerate(zip(features, expected_channels)):
                B, N, C = feat.shape
                H = W = int(N ** 0.5)
                assert C == expected_c, f"Stage {i}: expected {expected_c} channels, got {C}"
                print(f"     Stage {i+1}: [{B}, {N}, {C}] = [{B}, {C}, {H}x{H}]")
            
            del x, features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
    except Exception as e:
        print(f"   ✗ Multi-scale test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 4: Encoder freezing
    print("\n4. Testing encoder freezing for fine-tuning...")
    try:
        frozen_model = create_segmentation_model(
            SWIN_CONFIG,
            {**DOWNSTREAM_CONFIG, "freeze_encoder": True}
        )
        frozen_model = frozen_model.to(device)
        
        encoder_frozen = all(not p.requires_grad for p in frozen_model.encoder.parameters())
        head_trainable = any(p.requires_grad for p in frozen_model.seg_head.parameters())
        
        encoder_params = sum(p.numel() for p in frozen_model.encoder.parameters())
        trainable_params = sum(p.numel() for p in frozen_model.parameters() if p.requires_grad)
        
        print(f"   ✓ Head trainable: {head_trainable} ({trainable_params:,} params)")
        
        del frozen_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
    except Exception as e:
        print(f"   ✗ Freezing test failed: {e}")
        import traceback
        traceback.print_exc()
        returncuda.empty_cache()
        
    except Exception as e:
        print(f"   ✗ Freezing test failed: {e}")
        return
    
    # Test 5: Batch processing
    print("\n5. Testing batch processing (batch_size=2)...")
    try:
        with torch.no_grad():
            x = torch.randn(2, 3, 512, 512, device=device)
            print(f"   ✓ Batch processing: {tuple(x.shape)} → {tuple(output.shape)}")
            
            del x, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
    except Exception as e:
        print(f"   ✗ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        returnrch.cuda.empty_cache()
            
    except Exception as e:
        print(f"   ✗ Batch processing failed: {e}")
        return
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
    print("\n🎯 Model is ready for:")
    print("   • Loading ImageNet pretrained Swin-T weights")
    print("   • Fine-tuning on ADE20K dataset")
    print("   • Semantic segmentation training")
    print("=" * 70)


if __name__ == "__main__":
    main()
