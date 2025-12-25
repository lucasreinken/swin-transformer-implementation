"""
Model factory for creating different architectures for comparison experiments.
"""

import torch
from torch.utils.checkpoint import checkpoint
import timm
import torchvision.models as models
from .swin.swin_transformer_model import SwinTransformerModel
from .model_wrapper import ModelWrapper
from .segmentation_wrapper import SegmentationModelWrapper
from .heads import LinearClassificationHead, UperNetHead


def create_model(config):
    """
    Create model based on configuration type.

    Args:
        config: MODEL_CONFIG dictionary

    Returns:
        ModelWrapper containing the model
    """
    model_type = config["type"]

    if model_type == "swin":
        return create_swin_model(config)
    elif model_type == "vit":
        return create_vit_model(config)
    elif model_type == "resnet":
        return create_resnet_model(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_swin_model(config):
    """Create Swin Transformer model (existing implementation)."""
    encoder = SwinTransformerModel(
        img_size=224,  # Fixed for ImageNet
        patch_size=config["patch_size"],
        embedding_dim=config["embed_dim"],
        depths=config["depths"],
        num_heads=config["num_heads"],
        window_size=config["window_size"],
        mlp_ratio=config["mlp_ratio"],
        dropout_rate=config["dropout"],
        attention_dropout_rate=config["attention_dropout"],
        projection_dropout_rate=config["projection_dropout"],
        drop_path_rate=config["drop_path_rate"],
        use_shifted_window=config["use_shifted_window"],
        use_relative_bias=config["use_relative_bias"],
        use_absolute_pos_embed=config["use_absolute_pos_embed"],
        use_hierarchical_merge=config["use_hierarchical_merge"],
        use_gradient_checkpointing=config.get("use_gradient_checkpointing", False),
    )

    pred_head = LinearClassificationHead(
        num_features=encoder.num_features,
        num_classes=1000,  # ImageNet
    )

    return ModelWrapper(
        encoder=encoder,
        pred_head=pred_head,
        freeze=False,  # From scratch training
    )


def create_vit_model(config):
    """Create ViT model using timm library with gradient checkpointing."""
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=config["num_classes"],
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        drop_path_rate=0.1,  # Add some dropout for regularization
    )

    # Enable gradient checkpointing for memory efficiency if requested
    if config.get("use_gradient_checkpointing", False):
        if hasattr(model, "grad_checkpointing"):
            model.grad_checkpointing = True
        elif hasattr(model, "blocks"):
            # Manually enable gradient checkpointing for blocks
            original_forward = model.blocks.forward

            def checkpointed_forward(x):
                if model.training:
                    for block in model.blocks:
                        x = checkpoint(block, x, use_reentrant=False)
                    return x
                else:
                    return original_forward(x)

            model.blocks.forward = checkpointed_forward

    return model


def create_resnet_model(config):
    """Create ResNet model using torchvision with gradient checkpointing."""
    # Use layers config to choose ResNet variant
    layers = config.get("layers", [3, 4, 6, 3])

    if layers == [3, 4, 6, 3]:
        model = models.resnet50(pretrained=False, num_classes=config["num_classes"])
    elif layers == [3, 4, 23, 3]:
        model = models.resnet101(pretrained=False, num_classes=config["num_classes"])
    else:
        # Default to ResNet-50
        model = models.resnet50(pretrained=False, num_classes=config["num_classes"])

    # Enable gradient checkpointing for memory efficiency if requested
    if config.get("use_gradient_checkpointing", False):
        if (
            hasattr(model, "layer1")
            and hasattr(model, "layer2")
            and hasattr(model, "layer3")
            and hasattr(model, "layer4")
        ):
            # Wrap each ResNet layer with gradient checkpointing
            original_layer1_forward = model.layer1.forward
            original_layer2_forward = model.layer2.forward
            original_layer3_forward = model.layer3.forward
            original_layer4_forward = model.layer4.forward

            def checkpointed_layer1(x):
                return (
                    checkpoint(original_layer1_forward, x, use_reentrant=False)
                    if model.training
                    else original_layer1_forward(x)
                )

            def checkpointed_layer2(x):
                return (
                    checkpoint(original_layer2_forward, x, use_reentrant=False)
                    if model.training
                    else original_layer2_forward(x)
                )

            def checkpointed_layer3(x):
                return (
                    checkpoint(original_layer3_forward, x, use_reentrant=False)
                    if model.training
                    else original_layer3_forward(x)
                )

            def checkpointed_layer4(x):
                return (
                    checkpoint(original_layer4_forward, x, use_reentrant=False)
                    if model.training
                    else original_layer4_forward(x)
                )

            model.layer1.forward = checkpointed_layer1
            model.layer2.forward = checkpointed_layer2
            model.layer3.forward = checkpointed_layer3
            model.layer4.forward = checkpointed_layer4

    return model


def create_segmentation_model(swin_config, downstream_config, load_pretrained=True):
    """
    Create segmentation model with Swin encoder + UperNet head.
    
    Args:
        swin_config: SWIN_CONFIG dictionary from config
        downstream_config: DOWNSTREAM_CONFIG dictionary from config
        load_pretrained: If True, load ImageNet pretrained weights for encoder
    
    Returns:
        SegmentationModelWrapper containing encoder + segmentation head
    """
    from src.utils.load_weights import load_pretrained_reference, transfer_weights
    
    # Create Swin Transformer encoder
    encoder = SwinTransformerModel(
        img_size=swin_config["img_size"],  # 512 for ADE20K
        patch_size=swin_config["patch_size"],
        embedding_dim=swin_config["embed_dim"],
        depths=swin_config["depths"],
        num_heads=swin_config["num_heads"],
        window_size=swin_config["window_size"],
        mlp_ratio=swin_config["mlp_ratio"],
        dropout_rate=swin_config["dropout"],
        attention_dropout_rate=swin_config["attention_dropout"],
        projection_dropout_rate=swin_config["projection_dropout"],
        drop_path_rate=swin_config["drop_path_rate"],
        use_shifted_window=swin_config.get("use_shifted_window", True),
        use_relative_bias=swin_config.get("use_relative_bias", True),
        use_absolute_pos_embed=swin_config.get("use_absolute_pos_embed", False),
        use_hierarchical_merge=swin_config.get("use_hierarchical_merge", False),
        use_gradient_checkpointing=swin_config.get("use_gradient_checkpointing", False),
    )
    
    # Calculate in_channels for each stage
    # Swin-T: [96, 192, 384, 768]
    embed_dim = swin_config["embed_dim"]
    in_channels = [int(embed_dim * (2 ** i)) for i in range(len(swin_config["depths"]))]
    
    # Create UperNet segmentation head
    seg_head = UperNetHead(
        in_channels=in_channels,
        num_classes=downstream_config["num_classes"],
        channels=512,  # FPN channels (standard for UperNet)
        pool_scales=(1, 2, 3, 6),  # PPM scales from paper
        dropout=0.1,  # Dropout before classifier
    )
    
    # Combine encoder + head
    model = SegmentationModelWrapper(
        encoder=encoder,
        seg_head=seg_head,
        freeze_encoder=downstream_config.get("freeze_encoder", False),
    )
    
    # Load pretrained ImageNet weights for encoder
    if load_pretrained and downstream_config.get("use_pretrained", True):
        pretrained_model_name = "swin_tiny_patch4_window7_224"  # TIMM model name
        print(f"Loading pretrained weights from TIMM: {pretrained_model_name}")
        
        pretrained_model = load_pretrained_reference(
            model_name=pretrained_model_name,
            device="cpu",  # Load to CPU first, then move to device
        )
        
        if pretrained_model is not None:
            stats = transfer_weights(
                custom_model=model,
                pretrained_model=pretrained_model,
                encoder_only=True,  # Only transfer encoder weights
            )
            print(f"Weight transfer complete: {stats['transferred']} layers transferred, "
                  f"{stats['missing']} missing, {stats['size_mismatches']} size mismatches")
        else:
            print("Warning: Could not load pretrained weights. Training from scratch.")
    
    return model
