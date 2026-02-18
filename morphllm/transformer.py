"""Dynamic Transformer Block with growable attention and FFN.

Supports both width growth (zero-init expansion) and depth growth (Net2DeeperNet).
Width growth uses zero-initialized new dimensions so the model's output is
minimally perturbed. Depth growth creates identity blocks that act as
pass-throughs until trained.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from morphllm.attention import DynamicMHA
from morphllm.layers import GrowableLinear


class GrowableLayerNorm(nn.Module):
    """LayerNorm that can expand its normalized dimension.

    When grown, the new dimensions get default parameters (weight=1, bias=0)
    and the old dimensions keep learned statistics.
    """

    def __init__(self, d_model: int, device: Optional[str] = None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model, device=device))
        self.bias = nn.Parameter(torch.zeros(d_model, device=device))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.d_model,), self.weight, self.bias, self.eps)

    def grow(self, new_d_model: int) -> None:
        """Expand the normalized dimension, preserving learned affine params.

        New dimensions get weight=0, bias=0. This ensures the LN output
        is exactly 0 for new dims (0 * normalized + 0 = 0), regardless
        of the renormalization effect. This minimizes perturbation to
        downstream layers during width growth.
        """
        if new_d_model <= self.d_model:
            return
        with torch.no_grad():
            # weight=0 for new dims: kills new-dim signal after LN
            new_weight = torch.zeros(new_d_model, device=self.device)
            new_weight[: self.d_model] = self.weight
            new_bias = torch.zeros(new_d_model, device=self.device)
            new_bias[: self.d_model] = self.bias
            self.weight = nn.Parameter(new_weight)
            self.bias = nn.Parameter(new_bias)
            self.d_model = new_d_model


class DynamicTransformerBlock(nn.Module):
    """A single Transformer block with growable attention and feed-forward layers.

    Architecture: Pre-Norm Transformer
        x -> LayerNorm -> MHA -> + residual -> LayerNorm -> FFN -> + residual

    The FFN hidden dimension is ``ffn_ratio * d_model`` (default 4×).

    Args:
        d_model: Model embedding dimension.
        n_head: Number of attention heads.
        dropout: Dropout probability.
        ffn_ratio: FFN hidden dim multiplier.
        device: Target device.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float = 0.0,
        ffn_ratio: int = 4,
        device: Optional[str] = None,
    ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.d_model = d_model
        self.n_head = n_head
        self.ffn_ratio = ffn_ratio

        self.attn = DynamicMHA(d_model, n_head, dropout=dropout, device=device)
        self.norm1 = GrowableLayerNorm(d_model, device=device)

        d_ff = ffn_ratio * d_model
        self.fc1 = GrowableLinear(d_model, d_ff, device=device)
        self.fc2 = GrowableLinear(d_ff, d_model, device=device)
        self.norm2 = GrowableLayerNorm(d_model, device=device)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # MoE support: set to True after convert_block_to_moe()
        self._use_moe = False

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pre-Norm Transformer forward pass with residual connections.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            mask: Optional causal mask.

        Returns:
            Output of shape (batch, seq_len, d_model).
            If MoE is active, also sets self._moe_loss for auxiliary loss.
        """
        # Attention sub-layer with residual
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))

        # FFN sub-layer with residual (dense or MoE)
        if self._use_moe:
            moe_out, moe_loss = self.moe(self.norm2(x))
            x = x + self.dropout(moe_out)
            self._moe_loss = moe_loss
        else:
            x = x + self.dropout(self.fc2(F.gelu(self.fc1(self.norm2(x)))))
            self._moe_loss = torch.tensor(0.0)
        return x

    def grow(self, new_d_model: int) -> None:
        """Expand all sub-modules to a new embedding dimension (width growth).

        Uses zero-init for all new dimensions. The attention sublayer's
        new output rows are zero, the FFN sublayer's new output rows are
        zero, so the residual is minimally perturbed.

        Args:
            new_d_model: Target embedding dimension. Must be divisible by n_head.
        """
        if new_d_model <= self.d_model:
            return

        # 1. LayerNorm expansion (weight=1, bias=0 for new dims)
        self.norm1.grow(new_d_model)
        self.norm2.grow(new_d_model)

        # 2. Attention expansion (zero-init for new dims)
        self.attn.grow_embedding(new_d_model)

        # 3. FFN expansion
        # FC1: zero input cols, expand hidden dim proportionally
        self.fc1.expand_input_zero(new_d_model)
        new_d_ff = self.ffn_ratio * new_d_model
        self.fc1.expand_output_zero(new_d_ff)

        # FC2: zero input cols (from FC1 growth), zero output rows (new d_model dims)
        self.fc2.expand_input_zero(new_d_ff)
        self.fc2.expand_output_zero(new_d_model)

        self.d_model = new_d_model

    @classmethod
    def create_identity_block(
        cls,
        d_model: int,
        n_head: int,
        dropout: float = 0.0,
        ffn_ratio: int = 4,
        device: Optional[str] = None,
    ) -> "DynamicTransformerBlock":
        """Create a new block initialized as an identity function (Net2DeeperNet).

        The FC2 weights are set to zero so the FFN output is zero, and the
        attention output projection weights are set to zero. This makes the
        block a pure pass-through via the residual connections:
            output = x + 0 + 0 = x

        Over training, gradients will perturb these zero weights, gradually
        introducing the new block's capacity.

        Args:
            d_model: Embedding dimension.
            n_head: Number of heads.
            dropout: Dropout probability.
            ffn_ratio: FFN multiplier.
            device: Target device.

        Returns:
            A new DynamicTransformerBlock initialized as identity.
        """
        block = cls(d_model, n_head, dropout=dropout, ffn_ratio=ffn_ratio, device=device)

        with torch.no_grad():
            # Zero out FC2 so FFN contributes nothing
            block.fc2.weight.zero_()
            if block.fc2.bias is not None:
                block.fc2.bias.zero_()

            # Zero out attention output projection so MHA contributes nothing
            block.attn.out_proj.weight.zero_()
            if block.attn.out_proj.bias is not None:
                block.attn.out_proj.bias.zero_()

        return block
