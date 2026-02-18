"""Dynamic Multi-Head Attention with growable projections.

All Q/K/V/Out projections use GrowableLinear, enabling the attention
mechanism to expand its embedding dimension during training while
preserving learned representations.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from morphllm.layers import GrowableLinear


class DynamicMHA(nn.Module):
    """Multi-Head Attention with runtime-expandable embedding dimension.

    Uses separate GrowableLinear projections for Q, K, V, and output,
    allowing independent control over width expansion. Growth keeps
    n_head constant and increases head_dim.

    Args:
        d_model: Model embedding dimension.
        n_head: Number of attention heads.
        dropout: Attention dropout probability.
        device: Target device.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float = 0.0,
        device: Optional[str] = None,
    ):
        super().__init__()
        assert d_model % n_head == 0, (
            f"d_model ({d_model}) must be divisible by n_head ({n_head})"
        )

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout = dropout

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Separate projections for flexible resizing
        self.q_proj = GrowableLinear(d_model, d_model, device=device)
        self.k_proj = GrowableLinear(d_model, d_model, device=device)
        self.v_proj = GrowableLinear(d_model, d_model, device=device)
        self.out_proj = GrowableLinear(d_model, d_model, device=device)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with scaled dot-product attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional causal mask of shape (seq_len, seq_len).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        B, T, C = x.shape

        # Project to Q, K, V and reshape for multi-head
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        if mask is not None:
            att = att.masked_fill(mask[:T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)

        if self.dropout > 0 and self.training:
            att = F.dropout(att, p=self.dropout)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(y)

    def grow_embedding(self, new_d_model: int) -> None:
        """Expand the embedding dimension (width growth).

        Uses zero-init for new dimensions: new input columns are zero
        (so new dims don't contribute) and new output rows are zero
        (so old behavior preserved for old dims, zeros for new dims).

        Args:
            new_d_model: Target embedding dimension. Must be divisible
                by n_head.
        """
        if new_d_model <= self.d_model:
            return

        assert new_d_model % self.n_head == 0, (
            f"new_d_model ({new_d_model}) must be divisible by n_head ({self.n_head})"
        )

        # All projections: expand input (zero cols) and output (zero rows)
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            proj.expand_output_zero(new_d_model)
            proj.expand_input_zero(new_d_model)

        self.d_model = new_d_model
        self.head_dim = new_d_model // self.n_head
