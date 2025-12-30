import logging
from typing import Optional, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from timm import create_model
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning(
        "timm library not found. Cannot transfer weights."
    )


def load_pretrained_reference(
    model_name: str = "swin_tiny_patch4_window7_224",
    device: str = "cuda",
) -> Optional[nn.Module]:
    """
    Load a pretrained Swin model from timm.

    Args:
        model_name: Name of the timm model to load. Default: "swin_tiny_patch4_window7_224".
        device: Target device. Default: "cuda".

    Returns:
        Pretrained model set to eval mode, or None if loading fails or timm is unavailable.
    """
    if not TIMM_AVAILABLE:
        logger.error(
            "timm library is not available. Cannot load pretrained models."
        )
        return None

    try:
        model = create_model(model_name, pretrained=True, num_classes=1000)
        model.eval()
        logger.info(f"Loaded pretrained model: {model_name} from timm.")
        return model.to(device)
    except Exception as e:
        logger.error(f"Failed to load pretrained model: {e}")
        return None


def transfer_weights(
    custom_model: nn.Module,
    pretrained_model: nn.Module = None,
    encoder_only: bool = True,
    model_name: str = None,
    device: str = "cuda",
) -> Dict[str, int]:
    """
    Transfer pretrained weights into a custom model.

    Args:
        custom_model: Target model whose weights will be updated.
        pretrained_model: Preloaded pretrained model with a state_dict to copy from.
            Default: None (must be provided if model_name is not used).
        encoder_only: If True, transfer weights into the encoder only.
            Default: True.
        model_name: Optional timm model name to load if pretrained_model is None.
            Default: None.
        device: Target device ("cuda" or "cpu") for loading the reference model.
            Default: "cuda".

    Returns:
        Dictionary with counts for transferred layers, missing keys,
        and size mismatches.
    """

    if model_name and not pretrained_model:
        pretrained_model = load_pretrained_reference(model_name, device)

    pretrained_state = pretrained_model.state_dict()

    # Get the target module (encoder only or full model)
    if encoder_only:
        target_module = custom_model.encoder
    else:
        target_module = custom_model
    
    custom_state = target_module.state_dict()

    transferred = 0
    missing = []
    size_mismatches = []
    new_state_dict = {}

    for name, param in custom_state.items():
        if name in pretrained_state:
            pretrained_param = pretrained_state[name]
            if param.shape == pretrained_param.shape:
                new_state_dict[name] = pretrained_param.clone()
                transferred += 1
            else:
                size_mismatches.append(
                    f"{name}: {param.shape} vs {pretrained_param.shape}"
                )
                new_state_dict[name] = param  # Keep original
        else:
            missing.append(name)
            new_state_dict[name] = param  # Keep original

    # Load the updated state dict into the target module
    target_module.load_state_dict(new_state_dict)

    logger.info(f"Weight transfer: {transferred} layers transferred.")
    if missing:
        logger.warning(
            f"Missing weights: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if size_mismatches:
        logger.warning(
            f"Size mismatches: {size_mismatches[:3]}{'...' if len(size_mismatches) > 3 else ''}"
        )

    return {
        "transferred": transferred,
        "missing": len(missing),
        "size_mismatches": len(size_mismatches),
    }
