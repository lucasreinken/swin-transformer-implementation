import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttention(nn.Module):
    """
    Window-based Multi-Head Self-Attention (W-MSA) module with relative position bias.

    This module implements local self-attention as used in the Swin Transformer.
    Instead of attending globally over all tokens, attention is computed within
    non-overlapping windows (e.g., 7×7 patches). Each head learns an additive
    relative position bias to encode spatial relationships inside the window.

    Architecture:
    Input → Linear(QKV projection) → Scaled Dot-Product Attention (per window & head)
        → Add relative position bias → Softmax → Dropout
        → Weighted sum of Values → Linear projection → Dropout

    Supports:
    - Multi-head attention across fixed-size windows
    - Learnable relative position bias per attention head
    - Optional attention mask for shifted windows (SW-MSA)
    - Dropout for regularization

    """

    def __init__(
        self,
        dim: int,
        window_size: tuple[int],
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_relative_bias: bool = True,  # Ablation flag: True for learned bias, False for zero bias
        use_absolute_pos_embed: bool = False,  # Ablation flag: True for absolute pos embed (ViT-style)
    ):
        """
        Initialize W-MSA (/ SW-MSA).

        Args:
            dim: Input feature dimension
            window_size: The height and width of the window.
            num_heads: Number of attention heads.
            attn_dropout: Attention dropout rate. Default: 0.0
            proj_dropout: Projection dropout rate. Default: 0.0
        """

        super().__init__()

        assert (
            dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"

        self.embed_dim = dim
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.num_heads = num_heads
        self.use_relative_bias = use_relative_bias
        self.use_absolute_pos_embed = use_absolute_pos_embed

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        # Relative postion bias as learnable parameter
        self.relative_position_bias_table = nn.Parameter(
            torch.empty(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                self.num_heads,
            )
        )  # 2*Wh-1 * 2*Ww-1, nH

        # Random initialization to break symmetry
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        relative_position_index = self._get_relative_position_index(self.window_size)
        self.register_buffer(
            "relative_position_index", relative_position_index, persistent=False
        )

        # Optimized qkv layer (normally multiple linear layers)
        self.qkv = nn.Linear(dim, dim * 3)

        # Final projection layer
        self.proj = nn.Linear(dim, dim)

    def _get_relative_position_index(
        self, window_size: tuple[int, int]
    ) -> torch.Tensor:
        """Compute pair-wise relative position index for tokens in a window."""
        Wh, Ww = window_size

        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)  # (2, Wh*Ww)

        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)
        relative_coords[:, :, 0] += Wh - 1
        relative_coords[:, :, 1] += Ww - 1
        relative_coords[:, :, 0] *= 2 * Ww - 1

        relative_position_index = relative_coords.sum(-1).to(torch.long)  # (N, N)

        return relative_position_index

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [num_windows*B, N, C]
            attn_mask: boolean mask [num_windows, Wh*Ww, Wh*Ww]. Default = None

        Returns:
            Output tensor [num_windows*B, N, C]
        """
        wB, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(wB, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # [3, wB, nH, N, head_dim]
        q, k, v = qkv.unbind(0)  # each: [wB, nH, N, head_dim]

        # Scale dot product
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)

        # Relative position bias: [nH, N, N] -> broadcast to [wB, nH, N, N]
        if self.use_relative_bias:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            )  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()  # nH, Wh*Ww, Wh*Ww
            # Add learnable relative postition biases to scores (attention matrix)
            scores = scores + relative_position_bias.unsqueeze(0)
        # Note: Relative bias can now be combined with absolute position embeddings

        # Masking mechanism (for SW-MSA)
        if attn_mask is not None:
            nW = attn_mask.shape[0]  # num_windows
            scores = scores.view(-1, nW, self.num_heads, N, N)  # [B, nW, nH, N, N]
            scores = scores.masked_fill(
                attn_mask.unsqueeze(1).unsqueeze(0) != 0, -100.0
            )  # [1, nW, 1, N, N]; -100 -> 0 after softmax
            scores = scores.view(-1, self.num_heads, N, N)  # [wB, nH, N, N]

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        attn_out = torch.matmul(attn, v)  # [wB, nH, N, head_dim]

        attn_out = attn_out.transpose(1, 2).contiguous().view(wB, N, C)  # [wB, N, C]

        out = self.proj(attn_out)
        out = self.proj_dropout(out)

        return out
