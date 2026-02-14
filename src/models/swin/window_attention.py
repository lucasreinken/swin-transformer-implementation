import torch
import torch.nn as nn
import torch.nn.functional as F


class LogCPB(nn.Module):
    """
    Log-Spaced Continuous Position Bias (Log-CPB) for Swin Transformer V2.

    Replaces the discrete relative position bias table with a continuous parameterization
    that generalizes better to different window sizes and resolutions.

    Key features:
    - Uses log-spaced coordinates: log(|Δx| + 1), log(|Δy| + 1)
    - Small MLP generates bias values dynamically
    - Smooth interpolation for unseen relative positions
    - Better transfer to different resolutions without retraining

    Reference: Swin Transformer V2 (Liu et al., 2022), Section 3.2
    """

    def __init__(
        self, num_heads: int, window_size: tuple[int, int], hidden_dim: int = 512
    ):
        """
        Initialize Log-CPB module.

        Args:
            num_heads: Number of attention heads
            window_size: Window size (Wh, Ww) - used to pre-compute coordinate grid
            hidden_dim: Hidden dimension of the MLP (default: 512)
        """
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size

        # Small MLP to generate bias from log-spaced coordinates
        # Input: 2D (log-spaced relative coordinates)
        # Output: num_heads (bias per head)
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_heads, bias=False),
        )

        # Pre-compute relative coordinate grid for the given window size
        # This generates all possible (Δx, Δy) pairs in a window
        self._make_coords(window_size)

    def _make_coords(self, window_size: tuple[int, int]):
        """
        Pre-compute relative coordinate grid for the window.

        Creates a grid of all possible relative positions:
        - Range: [-(W-1), W-1] for both x and y
        - Total: (2*W-1) × (2*W-1) positions
        """
        Wh, Ww = window_size

        # Create coordinate grid for relative positions
        # coords_h, coords_w: each is (2*W-1,)
        coords_h = torch.arange(-(Wh - 1), Wh, dtype=torch.float32)
        coords_w = torch.arange(-(Ww - 1), Ww, dtype=torch.float32)

        # Create meshgrid and flatten
        # coords: (2*Wh-1, 2*Ww-1, 2) - all relative position pairs
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.reshape(-1, 2)  # [(2*Wh-1)*(2*Ww-1), 2]

        # Register as buffer (not a parameter, moves with model to device)
        self.register_buffer("relative_coords_table", coords, persistent=False)

    def forward(self, window_size: tuple[int, int] = None) -> torch.Tensor:
        """
        Generate continuous position bias for the given window size.

        Args:
            window_size: Optional window size to generate bias for.
                        If None, uses the pre-computed window size from init.

        Returns:
            Bias tensor of shape [1, num_heads, Wh*Ww, Wh*Ww]
            Ready to be added to attention scores.
        """
        # Use provided window size or default to initialization size
        if window_size is None:
            window_size = self.window_size

        # If window size changed, recompute coords (for dynamic resolution)
        if window_size != self.window_size:
            self._make_coords(window_size)

        # Get relative coordinates
        rel_coords = self.relative_coords_table  # [(2*W-1)^2, 2]

        # Apply log-spacing: log(|Δx| + 1), log(|Δy| + 1)
        # Adding 1 prevents log(0), and log-spacing gives better extrapolation
        log_coords = torch.sign(rel_coords) * torch.log(1.0 + rel_coords.abs())

        # Generate bias using MLP
        # bias: [(2*W-1)^2, num_heads]
        bias = self.mlp(log_coords)

        # Reshape to spatial grid: [2*Wh-1, 2*Ww-1, num_heads]
        Wh, Ww = window_size
        bias = bias.reshape(2 * Wh - 1, 2 * Ww - 1, self.num_heads)

        # Permute to [num_heads, 2*Wh-1, 2*Ww-1] and add batch dim
        bias = bias.permute(2, 0, 1).unsqueeze(0)

        return bias

    def get_bias_for_window(self, window_size: tuple[int, int] = None) -> torch.Tensor:
        """
        Get bias table indexed for attention computation.

        This method generates the full bias table and extracts only the needed
        indices for the actual window positions.

        Returns:
            Bias tensor of shape [num_heads, Wh*Ww, Wh*Ww]
        """
        if window_size is None:
            window_size = self.window_size

        Wh, Ww = window_size

        # Generate full bias table
        bias_table = self.forward(window_size).squeeze(0)  # [num_heads, 2*Wh-1, 2*Ww-1]

        # Create index mapping from window positions to bias table
        # Same logic as V1's relative_position_index
        coords_h = torch.arange(Wh, device=bias_table.device)
        coords_w = torch.arange(Ww, device=bias_table.device)
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij")
        )  # [2, Wh, Ww]
        coords_flatten = coords.reshape(2, -1)  # [2, Wh*Ww]

        # Compute relative coordinates between all pairs
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # [2, Wh*Ww, Wh*Ww]
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # [Wh*Ww, Wh*Ww, 2]

        # Shift to positive indices
        relative_coords[:, :, 0] += Wh - 1
        relative_coords[:, :, 1] += Ww - 1

        # Flatten to 1D index
        relative_coords[:, :, 0] *= 2 * Ww - 1
        relative_position_index = relative_coords.sum(-1)  # [Wh*Ww, Wh*Ww]

        # Index into bias table
        # bias_table is [num_heads, 2*Wh-1, 2*Ww-1], flatten to [num_heads, (2*Wh-1)*(2*Ww-1)]
        bias_table_flat = bias_table.reshape(
            self.num_heads, -1
        )  # [num_heads, (2*Wh-1)*(2*Ww-1)]

        # Gather bias values
        # relative_position_index: [Wh*Ww, Wh*Ww] -> [Wh*Ww*Wh*Ww]
        idx = relative_position_index.reshape(-1)
        bias = bias_table_flat[:, idx]  # [num_heads, Wh*Ww*Wh*Ww]
        bias = bias.reshape(self.num_heads, Wh * Ww, Wh * Ww)

        return bias


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
        return_attention: bool = False,  # Explainability flag: True to capture attention weights
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
        self.return_attention = return_attention
        
        # Storage for attention weights (only used when return_attention=True)
        self._last_attention_weights = None

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
        
        # Store attention weights if requested (for visualization)
        if self.return_attention:
            self._last_attention_weights = attn.detach().clone()
        
        attn = self.attn_dropout(attn)
        attn_out = torch.matmul(attn, v)  # [wB, nH, N, head_dim]

        attn_out = attn_out.transpose(1, 2).contiguous().view(wB, N, C)  # [wB, N, C]

        out = self.proj(attn_out)
        out = self.proj_dropout(out)

        return out


class WindowAttentionV2(nn.Module):
    """
    Window-based Multi-Head Self-Attention (W-MSA) for Swin V2 with Scaled Cosine Attention.

    Key differences from WindowAttention (V1):
    1. Uses cosine similarity instead of dot-product: cos(q, k) = (q·k) / (‖q‖·‖k‖)
    2. Scaled by learnable temperature: sim = cos(q, k) / τ
    3. Temperature τ is learned per layer (> 0.01 to avoid numerical issues)

    This provides:
    - Better stability: cosine is bounded in [-1, 1], less sensitive to input scale
    - Improved training: milder attention logits, smoother gradients
    - Small accuracy gains: typically +0.5-1% when combined with post-norm

    Reference: Swin Transformer V2 (Liu et al., 2022)
    """

    def __init__(
        self,
        dim: int,
        window_size: tuple[int],
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_relative_bias: bool = True,
        use_absolute_pos_embed: bool = False,
        use_log_cpb: bool = True,  # Use Log-CPB by default for V2
    ):
        """
        Initialize Scaled Cosine Attention for Swin V2.

        Args:
            dim: Input feature dimension
            window_size: The height and width of the window (Wh, Ww)
            num_heads: Number of attention heads
            attn_dropout: Attention dropout rate. Default: 0.0
            proj_dropout: Projection dropout rate. Default: 0.0
            use_relative_bias: Whether to use relative position bias (default: True)
            use_absolute_pos_embed: Whether to use absolute position embedding (default: False)
            use_log_cpb: Whether to use Log-CPB (True) or discrete table (False). Default: True
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
        self.use_log_cpb = use_log_cpb

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        # Learnable temperature parameter (per layer)
        # Initialize to ~0.1-0.3, constrained to be > 0.01
        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1)))
        )  # log(10) ≈ 2.3, so tau ≈ 0.1

        # ─────────────────────────────────────────────────────────────
        # Relative Position Bias: Log-CPB (V2) or Discrete Table (V1)
        # ─────────────────────────────────────────────────────────────

        if use_log_cpb:
            # V2: Continuous Position Bias with log-spaced coordinates
            self.cpb = LogCPB(num_heads, window_size, hidden_dim=512)
        else:
            # V1: Discrete lookup table (for ablation/comparison)
            self.relative_position_bias_table = nn.Parameter(
                torch.empty(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                    self.num_heads,
                )
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

            # Relative position index
            relative_position_index = self._get_relative_position_index(
                self.window_size
            )
            self.register_buffer(
                "relative_position_index", relative_position_index, persistent=False
            )

        # QKV projection (combined for efficiency)
        self.qkv = nn.Linear(dim, dim * 3)

        # Output projection
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
        Forward pass with scaled cosine attention.

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

        # ─────────────────────────────────────────────────────────────
        # Scaled Cosine Attention (V2 improvement)
        # ─────────────────────────────────────────────────────────────

        # L2 normalize Q and K to unit length
        q = F.normalize(q, dim=-1)  # [wB, nH, N, head_dim]
        k = F.normalize(k, dim=-1)  # [wB, nH, N, head_dim]

        # Cosine similarity (dot product of normalized vectors)
        scores = torch.matmul(q, k.transpose(-2, -1))  # [wB, nH, N, N], range [-1, 1]

        # Scale by learnable temperature τ
        # logit_scale is log(1/τ), so exp(logit_scale) = 1/τ
        # We clamp to avoid extreme values (max 100 = 1/0.01)
        logit_scale = torch.clamp(
            self.logit_scale,
            max=torch.log(torch.tensor(100.0, device=self.logit_scale.device)),
        )
        scores = scores * logit_scale.exp()  # [wB, nH, N, N]

        # ─────────────────────────────────────────────────────────────
        # Relative Position Bias: Log-CPB (V2) or Discrete Table (V1)
        # ─────────────────────────────────────────────────────────────

        if self.use_relative_bias:
            if self.use_log_cpb:
                # V2: Generate continuous bias from Log-CPB
                relative_position_bias = self.cpb.get_bias_for_window(
                    self.window_size
                )  # [nH, Wh*Ww, Wh*Ww]
            else:
                # V1: Lookup from discrete table
                relative_position_bias = self.relative_position_bias_table[
                    self.relative_position_index.view(-1)
                ].view(
                    self.window_size[0] * self.window_size[1],
                    self.window_size[0] * self.window_size[1],
                    -1,
                )  # Wh*Ww, Wh*Ww, nH
                relative_position_bias = relative_position_bias.permute(
                    2, 0, 1
                ).contiguous()  # nH, Wh*Ww, Wh*Ww

            scores = scores + relative_position_bias.unsqueeze(0)

        # ─────────────────────────────────────────────────────────────
        # Masking (for shifted window attention)
        # ─────────────────────────────────────────────────────────────

        if attn_mask is not None:
            nW = attn_mask.shape[0]  # num_windows
            scores = scores.view(-1, nW, self.num_heads, N, N)  # [B, nW, nH, N, N]
            scores = scores.masked_fill(
                attn_mask.unsqueeze(1).unsqueeze(0) != 0, -100.0
            )
            scores = scores.view(-1, self.num_heads, N, N)  # [wB, nH, N, N]

        # ─────────────────────────────────────────────────────────────
        # Softmax and apply to values (same as V1)
        # ─────────────────────────────────────────────────────────────

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        attn_out = torch.matmul(attn, v)  # [wB, nH, N, head_dim]

        attn_out = attn_out.transpose(1, 2).contiguous().view(wB, N, C)  # [wB, N, C]

        out = self.proj(attn_out)
        out = self.proj_dropout(out)

        return out
