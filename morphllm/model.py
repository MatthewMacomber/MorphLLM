"""MorphLLM — The full adaptive language model.

Assembles GrowableEmbedding, N DynamicTransformerBlocks, and a GrowableLinear
LM head into a complete causal language model that supports:
  - Width growth (Net2WiderNet): expand d_model across all components
  - Depth growth (Net2DeeperNet): insert identity-initialized blocks
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from morphllm.layers import GrowableEmbedding, GrowableLinear
from morphllm.transformer import DynamicTransformerBlock, GrowableLayerNorm


@dataclass
class MorphConfig:
    """Configuration for a MorphLLM seed model.

    Attributes:
        vocab_size: Number of tokens in the vocabulary.
        d_model: Embedding dimension.
        n_head: Number of attention heads.
        n_layers: Number of transformer blocks.
        max_seq_len: Maximum sequence length.
        dropout: Dropout probability.
        ffn_ratio: FFN hidden dim = ffn_ratio * d_model.
        device: Target device (None = auto-detect).
    """
    vocab_size: int = 8000
    d_model: int = 64
    n_head: int = 4
    n_layers: int = 2
    max_seq_len: int = 512
    dropout: float = 0.0
    ffn_ratio: int = 4
    device: Optional[str] = None


class MorphLLM(nn.Module):
    """Adaptive Morphogenic Engine — a language model that grows.

    Architecture (Pre-Norm GPT-style):
        Token Embedding + Positional Embedding
        → N × DynamicTransformerBlock
        → LayerNorm
        → Linear LM Head (tied weights with token embedding)

    Growth operations:
        ``grow_width(new_d_model)``: Expand embedding dimension everywhere.
        ``grow_depth(position)``: Insert identity block at a given position.

    Args:
        config: MorphConfig with model hyperparameters.
    """

    def __init__(self, config: MorphConfig):
        super().__init__()
        self.config = config

        device = config.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_str = device

        # Token embedding
        self.tok_emb = GrowableEmbedding(
            config.vocab_size, config.d_model, device=device
        )

        # Learned positional embedding
        self.pos_emb = GrowableEmbedding(
            config.max_seq_len, config.d_model, device=device
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DynamicTransformerBlock(
                config.d_model,
                config.n_head,
                dropout=config.dropout,
                ffn_ratio=config.ffn_ratio,
                device=device,
            )
            for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_f = GrowableLayerNorm(config.d_model, device=device)

        # LM head (projects back to vocab)
        self.lm_head = GrowableLinear(
            config.d_model, config.vocab_size, bias=False, device=device
        )

        # Causal mask: upper-triangular = 0 (masked), lower-triangular = 1
        causal = torch.tril(
            torch.ones(config.max_seq_len, config.max_seq_len, device=device)
        )
        self.register_buffer("causal_mask", causal)

        # Track architecture state
        self._d_model = config.d_model
        self._n_layers = config.n_layers
        self._n_head = config.n_head

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass producing logits and optional loss.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            targets: Target token IDs for CE loss, shape (batch, seq_len).

        Returns:
            Tuple of (logits, loss). Loss is None if targets not provided.
            Logits shape: (batch, seq_len, vocab_size).
        """
        B, T = input_ids.shape

        # Embeddings
        tok = self.tok_emb(input_ids)
        pos = self.pos_emb(
            torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        )
        x = tok + pos

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask=self.causal_mask)

        # Final norm + LM head
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    # ------------------------------------------------------------------
    # Growth Operations
    # ------------------------------------------------------------------

    def grow_width(self, new_d_model: int) -> None:
        """Expand the embedding dimension across the entire model.

        Uses zero-init for all new dimensions: new embedding dims are 0,
        attention/FFN new output rows are 0, new input cols are 0.
        This is approximately function-preserving — the only perturbation
        comes from LayerNorm renormalization over the wider dimension,
        which is small and corrected within a few training steps.

        Depth growth (grow_depth) remains exactly function-preserving.

        Args:
            new_d_model: Target embedding dimension. Must be divisible by n_head.
        """
        if new_d_model <= self._d_model:
            return

        assert new_d_model % self._n_head == 0, (
            f"new_d_model ({new_d_model}) must be divisible by "
            f"n_head ({self._n_head})"
        )

        # 1. Expand embeddings (zero-fill new dims)
        self.tok_emb.expand_dim(new_d_model, noise_std=0.0)
        self.pos_emb.expand_dim(new_d_model, noise_std=0.0)

        # 2. Expand each transformer block (zero-init new dims)
        for block in self.blocks:
            block.grow(new_d_model)

        # 3. Expand final layer norm
        self.ln_f.grow(new_d_model)

        # 4. Expand LM head input (zero columns — new dims don't affect logits)
        self.lm_head.expand_input_zero(new_d_model)

        # Update config
        self._d_model = new_d_model
        self.config.d_model = new_d_model

    def grow_depth(self, position: Optional[int] = None) -> None:
        """Insert an identity-initialized transformer block (Net2DeeperNet).

        The new block acts as a pure pass-through (output = input) via
        zero-initialized output projections. Training will gradually
        introduce its capacity.

        Args:
            position: Where to insert the new block (0-indexed). If None,
                appends to the end (before final LN).
        """
        new_block = DynamicTransformerBlock.create_identity_block(
            self._d_model,
            self._n_head,
            dropout=self.config.dropout,
            ffn_ratio=self.config.ffn_ratio,
            device=self.device_str,
        )

        if position is None:
            position = len(self.blocks)

        self.blocks.insert(position, new_block)
        self._n_layers = len(self.blocks)
        self.config.n_layers = self._n_layers

    def expand_vocab(self, new_vocab_size: int) -> None:
        """Expand the vocabulary size.

        Adds new randomly-initialized token embeddings and expands the
        LM head output dimension.

        Args:
            new_vocab_size: Target vocabulary size.
        """
        if new_vocab_size <= self.config.vocab_size:
            return

        self.tok_emb.expand_vocab(new_vocab_size)
        self.lm_head.expand_output(new_vocab_size, noise_std=0.0)
        self.config.vocab_size = new_vocab_size

    def architecture_summary(self) -> dict:
        """Return a summary of the current model architecture."""
        return {
            "vocab_size": self.config.vocab_size,
            "d_model": self._d_model,
            "n_head": self._n_head,
            "n_layers": self._n_layers,
            "max_seq_len": self.config.max_seq_len,
            "ffn_ratio": self.config.ffn_ratio,
            "num_parameters": self.num_parameters,
            "param_size_mb": self.num_parameters * 4 / (1024 * 1024),
        }
