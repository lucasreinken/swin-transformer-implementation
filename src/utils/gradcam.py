"""
Grad-CAM implementation for Swin Transformer.

Computes class-discriminative localisation heatmaps using gradients from a
classification head with respect to intermediate Swin Transformer stage
activations (Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization", ICCV 2017).

This module is self-contained: it creates a temporary classification head,
loads pretrained TIMM weights for it, and produces per-stage Grad-CAM
heatmaps without modifying the existing attention-based pipeline.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# ImageNet normalisation constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert an ImageNet-normalised [1, 3, H, W] tensor to [H, W, 3] numpy in [0, 1]."""
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    img = image_tensor.cpu().permute(1, 2, 0).numpy()
    img = IMAGENET_STD * img + IMAGENET_MEAN
    return np.clip(img, 0, 1)


# =====================================================================
# Core Grad-CAM class
# =====================================================================

class SwinGradCAM:
    """
    Grad-CAM for Swin Transformer.

    Registers forward / backward hooks on the last ``SwinTransformerBlock``
    of each requested stage, performs a single forward + backward pass through
    the encoder *and* a classification head, and produces per-stage heatmaps.

    Usage::

        gradcam = SwinGradCAM.create_from_timm(encoder, 'swin_tiny_patch4_window7_224', device)
        heatmaps, pred_cls = gradcam.compute(image, stage_indices=[0, 1, 2])
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        head: torch.nn.Module,
        device: str = 'cuda',
    ):
        """
        Args:
            encoder: ``SwinTransformerModel`` (bare encoder, **no** classification head).
            head: ``LinearClassificationHead`` producing ``[B, num_classes]`` logits.
            device: Computation device.
        """
        self.encoder = encoder
        self.head = head
        self.device = device

        # Storage filled by hooks during forward / backward
        self._activations: Dict[int, torch.Tensor] = {}
        self._gradients: Dict[int, torch.Tensor] = {}
        self._resolutions: Dict[int, Tuple[int, int]] = {}
        self._hooks: list = []

    # -----------------------------------------------------------------
    # Hook management
    # -----------------------------------------------------------------

    def _register_hooks(self, stage_indices: List[int]) -> None:
        """Register forward and backward hooks on the last block of each stage."""
        self._remove_hooks()
        self._activations.clear()
        self._gradients.clear()
        self._resolutions.clear()

        for stage_idx in stage_indices:
            if stage_idx >= len(self.encoder.layers):
                logger.warning(
                    f"Stage {stage_idx} out of range "
                    f"(model has {len(self.encoder.layers)} stages), skipping"
                )
                continue

            last_block = self.encoder.layers[stage_idx].blocks[-1]

            # Closures must capture *stage_idx* by value, not by reference.
            def _make_fwd_hook(idx: int):
                def hook(module, inp, output):
                    # output: tensor [B, H*W, C]  (SwinTransformerBlock returns a tensor)
                    self._activations[idx] = output
                    # inp = (x, H, W) — capture spatial resolution
                    self._resolutions[idx] = (int(inp[1]), int(inp[2]))
                return hook

            def _make_bwd_hook(idx: int):
                def hook(module, grad_input, grad_output):
                    # grad_output[0]: gradient w.r.t. block output [B, H*W, C]
                    self._gradients[idx] = grad_output[0]
                return hook

            fh = last_block.register_forward_hook(_make_fwd_hook(stage_idx))
            bh = last_block.register_full_backward_hook(_make_bwd_hook(stage_idx))
            self._hooks.extend([fh, bh])

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # -----------------------------------------------------------------
    # Core computation
    # -----------------------------------------------------------------

    def compute(
        self,
        image: torch.Tensor,
        stage_indices: List[int] = (0, 1, 2),
        target_class: Optional[int] = None,
        img_size: int = 224,
    ) -> Tuple[Dict[int, np.ndarray], int]:
        """
        Compute Grad-CAM heatmaps for the requested stages.

        Args:
            image: Input image tensor ``[1, 3, H, W]``.
            stage_indices: Stage indices to produce heatmaps for.
            target_class: Class to compute gradients for (``None`` → predicted class).
            img_size: Target spatial size for the upsampled heatmaps.

        Returns:
            ``(heatmaps, predicted_class)`` where *heatmaps* maps
            ``stage_idx → numpy array [img_size, img_size]`` in ``[0, 1]``.
        """
        stage_indices = list(stage_indices)
        self._register_hooks(stage_indices)

        # Temporarily disable attention-map collection (saves memory)
        prev_flag = self.encoder.return_attention_maps
        self.encoder.return_attention_maps = False

        try:
            # ---------- forward + backward WITH gradients ----------
            with torch.enable_grad():
                x = image.clone().to(self.device)

                features = self.encoder(x)      # [B, H*W, C]  (e.g. [1, 49, 768])
                logits = self.head(features)     # [B, num_classes]

                if target_class is None:
                    target_class = int(logits.argmax(dim=1).item())

                self.encoder.zero_grad()
                self.head.zero_grad()
                logits[0, target_class].backward()

            # ---------- build heatmaps ----------
            heatmaps: Dict[int, np.ndarray] = {}

            for sid in stage_indices:
                if sid not in self._activations or sid not in self._gradients:
                    logger.warning(f"Stage {sid}: activation or gradient missing — skipped")
                    continue

                act = self._activations[sid].detach()    # [B, H*W, C]
                grad = self._gradients[sid].detach()      # [B, H*W, C]

                # α_c = global-average-pool of gradients over spatial dimension
                weights = grad.mean(dim=1, keepdim=True)  # [B, 1, C]

                # Weighted combination → ReLU (class-discriminative saliency)
                cam = (act * weights).sum(dim=-1)          # [B, H*W]
                cam = F.relu(cam)

                H, W = self._resolutions[sid]
                cam = cam.view(1, 1, H, W)
                cam = F.interpolate(
                    cam, size=(img_size, img_size),
                    mode='bilinear', align_corners=False,
                )
                cam = cam.squeeze().cpu().numpy()

                lo, hi = cam.min(), cam.max()
                if (hi - lo) > 1e-8:
                    cam = (cam - lo) / (hi - lo)
                else:
                    cam = np.zeros_like(cam)

                heatmaps[sid] = cam
                logger.info(
                    f"  Grad-CAM stage {sid}: {H}×{W} → {img_size}×{img_size}, "
                    f"class={target_class}"
                )

            return heatmaps, target_class

        finally:
            # Always restore state and clean up
            self.encoder.return_attention_maps = prev_flag
            self._remove_hooks()
            self._activations.clear()
            self._gradients.clear()
            self._resolutions.clear()

    # -----------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------

    @staticmethod
    def create_from_timm(
        encoder: torch.nn.Module,
        timm_model_name: str,
        device: str = 'cuda',
    ) -> 'SwinGradCAM':
        """
        Build a ``SwinGradCAM`` by loading TIMM classification-head weights
        into a new ``LinearClassificationHead``.

        The TIMM pretrained model is loaded once to copy the head parameters
        (``norm.weight``, ``norm.bias``, ``head.fc.weight``, ``head.fc.bias``)
        and then discarded.

        Args:
            encoder: ``SwinTransformerModel`` (bare encoder).
            timm_model_name: TIMM model identifier (e.g. ``'swin_tiny_patch4_window7_224'``).
            device: Computation device.

        Returns:
            Configured ``SwinGradCAM`` instance ready for ``.compute()``.
        """
        from src.models.heads.linear_classification import LinearClassificationHead
        import timm

        num_features = encoder.num_features  # 768 for Swin-Tiny
        head = LinearClassificationHead(num_features=num_features, num_classes=1000)

        # ---- transfer head weights from TIMM pretrained model ----
        timm_model = timm.create_model(timm_model_name, pretrained=True)
        timm_state = timm_model.state_dict()

        head_state = head.state_dict()
        transferred = 0

        for key in list(head_state.keys()):
            # Direct match (norm.weight, norm.bias, head.fc.weight, head.fc.bias)
            if key in timm_state and head_state[key].shape == timm_state[key].shape:
                head_state[key] = timm_state[key].clone()
                transferred += 1
            # Fallback: TIMM may store head.weight instead of head.fc.weight
            elif key.startswith('head.fc.'):
                alt_key = key.replace('head.fc.', 'head.')
                if alt_key in timm_state and head_state[key].shape == timm_state[alt_key].shape:
                    head_state[key] = timm_state[alt_key].clone()
                    transferred += 1

        head.load_state_dict(head_state)
        head = head.to(device)
        head.eval()

        del timm_model  # free memory

        logger.info(
            f"Grad-CAM head: {transferred}/{len(head_state)} weight tensors loaded from TIMM"
        )
        return SwinGradCAM(encoder=encoder, head=head, device=device)


# =====================================================================
# Visualisation functions
# =====================================================================

# Readable resolution strings per stage (for Swin-Tiny with patch_size=4)
_STAGE_RESOLUTION_STR = {0: '56×56', 1: '28×28', 2: '14×14', 3: '7×7'}


def visualize_gradcam_multistage(
    image_np: np.ndarray,
    heatmaps: Dict[int, np.ndarray],
    stages: List[int],
    predicted_class: int,
    colormap: str = 'jet',
    overlay_alpha: float = 0.6,
    save_path: Optional[str] = None,
) -> Image.Image:
    """
    Create a 1 × (1 + N) panel figure: [Original | Grad-CAM S0 | S1 | S2].

    Args:
        image_np: Denormalised image ``[H, W, 3]`` in ``[0, 1]``.
        heatmaps: ``{stage_idx: heatmap}`` — each heatmap is ``[H, W]`` in ``[0, 1]``.
        stages: Ordered list of stage indices to display.
        predicted_class: Predicted ImageNet class index (shown in title).
        colormap: Matplotlib colourmap name.
        overlay_alpha: Heatmap overlay transparency.
        save_path: If given, save the figure as PNG.

    Returns:
        PIL Image of the rendered figure.
    """
    n_panels = 1 + len(stages)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]

    # Panel 0: original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image', fontsize=10)
    axes[0].axis('off')

    # Panels 1–N: Grad-CAM overlays
    for i, sid in enumerate(stages):
        axes[i + 1].imshow(image_np)
        if sid in heatmaps:
            im = axes[i + 1].imshow(
                heatmaps[sid], cmap=colormap, alpha=overlay_alpha,
                vmin=0, vmax=1,
            )
            plt.colorbar(im, ax=axes[i + 1], fraction=0.046, pad=0.04)
        res = _STAGE_RESOLUTION_STR.get(sid, '')
        axes[i + 1].set_title(f'Grad-CAM Stage {sid}\n{res}', fontsize=10)
        axes[i + 1].axis('off')

    fig.suptitle(
        f'Grad-CAM Multi-Stage Analysis  (predicted class {predicted_class})',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Render to PIL
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    pil_img = Image.frombytes('RGBA', (w, h), fig.canvas.buffer_rgba()).convert('RGB')

    if save_path:
        pil_img.save(save_path, dpi=(300, 300))
        logger.info(f"Saved Grad-CAM multi-stage → {save_path}")

    plt.close(fig)
    return pil_img


def visualize_attention_vs_gradcam(
    image_np: np.ndarray,
    attention_heatmap: np.ndarray,
    gradcam_heatmap: np.ndarray,
    stage_idx: int,
    colormap: str = 'jet',
    overlay_alpha: float = 0.6,
    save_path: Optional[str] = None,
) -> Image.Image:
    """
    Create a 1 × 3 panel figure: [Original | Self-Attention | Grad-CAM].

    Args:
        image_np: Denormalised image ``[H, W, 3]`` in ``[0, 1]``.
        attention_heatmap: Attention heatmap ``[H, W]`` in ``[0, 1]``.
        gradcam_heatmap: Grad-CAM heatmap ``[H, W]`` in ``[0, 1]``.
        stage_idx: Stage index (used for panel titles).
        colormap: Matplotlib colourmap name.
        overlay_alpha: Heatmap overlay transparency.
        save_path: If given, save the figure as PNG.

    Returns:
        PIL Image of the rendered figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    # Panel 0: original
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image', fontsize=10)
    axes[0].axis('off')

    # Panel 1: self-attention
    axes[1].imshow(image_np)
    im1 = axes[1].imshow(
        attention_heatmap, cmap=colormap, alpha=overlay_alpha, vmin=0, vmax=1,
    )
    axes[1].set_title(f'Self-Attention (Stage {stage_idx})', fontsize=10)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 2: Grad-CAM
    axes[2].imshow(image_np)
    im2 = axes[2].imshow(
        gradcam_heatmap, cmap=colormap, alpha=overlay_alpha, vmin=0, vmax=1,
    )
    axes[2].set_title(f'Grad-CAM (Stage {stage_idx})', fontsize=10)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f'Self-Attention vs Grad-CAM — Stage {stage_idx}',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Render to PIL
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    pil_img = Image.frombytes('RGBA', (w, h), fig.canvas.buffer_rgba()).convert('RGB')

    if save_path:
        pil_img.save(save_path, dpi=(300, 300))
        logger.info(f"Saved attention vs Grad-CAM → {save_path}")

    plt.close(fig)
    return pil_img
