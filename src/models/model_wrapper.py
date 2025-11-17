import torch.nn as nn
from src.models import (
    SwinTransformerModel,
    LinearClassificationHead,
)

class ModelWrapper(nn.Module):
    """
    General wrapper for an encoder and a prediction head.
    """

    def __init__(
            self,
            encoder: nn.Module,
            pred_head: nn.Module,
            freeze: bool = True
            ):
        """
        Initialize Model Wrapper.

        Args:
            encoder: Feature extractor backbone returning feature vectors.
            pred_head: Prediction head operating on the encoder features.
            freeze: Whether to freeze the encoder parameters (e.g., for linear probing). (default: True)
        """
        super().__init__()
        self.encoder = encoder
        self.pred_head = pred_head

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)
        logits = self.pred_head(features)
        return logits


def create_linear_classification_model(
    encoder: SwinTransformerModel,
    num_classes: int = 1000,
    freeze: bool = True,
) -> ModelWrapper:
    """
    Create a ModelWrapper with a linear classification head on top of an encoder.

    Args:
        encoder: Backbone model returning feature vectors.
        num_classes: Number of output classes. (default = 1000)
        freeze: Whether to freeze the encoder parameters (linear probing if True). (default = True)
    """

    pred_head = LinearClassificationHead(
        num_features=encoder.num_features,
        num_classes=num_classes,
    )

    return ModelWrapper(
        encoder=encoder,
        pred_head=pred_head,
        freeze=freeze,
    )

def swin_tiny_patch4_window7_224(**kwargs) -> ModelWrapper:
    encoder = SwinTransformerModel(
        img_size=224,
        patch_size=4,
        embedding_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        **kwargs
    )

    model = create_linear_classification_model(
        encoder=encoder,
        freeze=False
    )

    return model


def swin_small_patch4_window7_224(**kwargs) -> ModelWrapper:
    encoder = SwinTransformerModel(
        img_size=224,
        patch_size=4,
        embedding_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        **kwargs
    )

    model = create_linear_classification_model(
        encoder=encoder,
        freeze=False
    )

    return model


def swin_base_patch4_window7_224(**kwargs) -> ModelWrapper:
    encoder = SwinTransformerModel(
        img_size=224,
        patch_size=4,
        embedding_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        **kwargs
    )

    model = create_linear_classification_model(
        encoder=encoder,
        freeze=False
    )

    return model


def swin_large_patch4_window7_224(**kwargs) -> ModelWrapper:
    encoder = SwinTransformerModel(
        img_size=224,
        patch_size=4,
        embedding_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        **kwargs
    )

    model = create_linear_classification_model(
        encoder=encoder,
        freeze=False
    )

    return model