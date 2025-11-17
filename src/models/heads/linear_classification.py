import torch
import torch.nn as nn

class LinearClassificationHead(nn.Module):
    """
    Simple linear classification head.
    """
    def __init__(
            self,
            num_features: int = 96,
            num_classes: int = 1000,    
            norm_layer: nn.Module = nn.LayerNorm,
            ):
        """
        Initialize Linear Classification Head.

        Args:
            in_dim: Size of the incoming feature vector.
            num_classes: Number of output classes returned as logits.
        """
        super().__init__()

        self.norm = norm_layer(num_features)  # Feature normalization
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # Average pooling
        
        # Match timm's head structure
        self.head = nn.ModuleDict({"fc": nn.Linear(num_features, num_classes)})

    def forward(self, x):
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)

        x = self.head["fc"](x)
        return x