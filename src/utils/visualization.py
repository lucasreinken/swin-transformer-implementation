import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union


def show(
    x: Union[np.ndarray, torch.Tensor],
    y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    outfile: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    cmap: str = "gray",
) -> None:
    """
    Visualize a batch of images in a grid.

    Args:
        x: Tensor of shape [batch_size, 3, H, W]
        y: Optional tensor of labels corresponding to x
        outfile: Optional file path to save the figure
        figsize: Tuple for figure size (width, height)
        cmap: Colormap for grayscale images
    """
    if isinstance(x, torch.Tensor):
        x = x.numpy()

    batch_size = x.shape[0]

    # Create a grid
    grid_size = int(np.ceil(np.sqrt(batch_size)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()

    for i in range(batch_size):
        # Convert CHW to HWC and ensure values are in [0, 1]
        img = x[i].transpose(1, 2, 0)
        axes[i].imshow(img)

        if y is not None:
            axes[i].set_title(f"Label: {y[i].item()}")
        else:
            axes[i].set_title(f"Img {i}")
        axes[i].axis("off")

    # Remove empty subplots
    for j in range(batch_size, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()
