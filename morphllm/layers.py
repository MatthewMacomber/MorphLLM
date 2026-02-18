"""Growable neural network layers with function-preserving expansion.

Implements Net2WiderNet transformations that allow layers to increase
their dimensions during training without disrupting learned representations.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class GrowableLinear(nn.Module):
    """A Linear layer that can expand its input and output dimensions dynamically.

    Implements Net2Net initialization for function preservation. When expanded,
    the new weights are copies of existing neurons with symmetry-breaking noise,
    ensuring the network's output is unchanged at the moment of expansion.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Resolve device: prefer explicit, then CUDA if available, else CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Initialize parameters
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming uniform initialization (PyTorch default for nn.Linear)."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)

    def expand_output(
        self,
        new_out_features: int,
        noise_std: float = 1e-4,
        forced_indices: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Net2WiderNet: Expand the output dimension (width).

        Copies weights from randomly selected existing neurons to fill new
        positions, then adds small noise for symmetry breaking.

        Args:
            new_out_features: Target output dimension (must be > current).
            noise_std: Std dev of Gaussian noise for symmetry breaking.
            forced_indices: If provided, use these indices instead of
                randomly selecting which neurons to clone. Used for
                coordinated growth where all layers must replicate
                the same dimensions.

        Returns:
            Tensor of mapping indices (which parent each new neuron was copied
            from), or None if no growth was needed. The downstream layer must
            use these indices in its ``expand_input`` call.
        """
        if new_out_features <= self.out_features:
            return None  # No growth needed

        n_new = new_out_features - self.out_features

        # Use forced indices or randomly select
        if forced_indices is not None:
            indices = forced_indices
        else:
            indices = torch.randint(0, self.out_features, (n_new,))

        with torch.no_grad():
            # Build new weight tensor: [old_weights; cloned_weights + noise]
            new_weight = torch.zeros(
                (new_out_features, self.in_features), device=self.device
            )
            new_weight[: self.out_features] = self.weight
            new_weight[self.out_features :] = self.weight[indices]

            # Symmetry-breaking noise on cloned portion
            if noise_std > 0:
                noise = torch.normal(
                    0,
                    noise_std,
                    size=new_weight[self.out_features :].shape,
                    device=self.device,
                )
                new_weight[self.out_features :] += noise

            self.weight = nn.Parameter(new_weight)

            if self.bias is not None:
                new_bias = torch.zeros(new_out_features, device=self.device)
                new_bias[: self.out_features] = self.bias
                new_bias[self.out_features :] = self.bias[indices]
                self.bias = nn.Parameter(new_bias)

            self.out_features = new_out_features

        return indices

    def expand_input(
        self,
        new_in_features: int,
        mapping_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """Expand the input dimension to match a preceding layer's output growth.

        When the previous layer gains new output neurons via ``expand_output``,
        this layer must gain corresponding input channels. Weights are copied
        from the parent neurons and scaled by the replication factor to preserve
        the dot product magnitude (Net2WiderNet normalization).

        Args:
            new_in_features: Target input dimension.
            mapping_indices: Indices from the preceding layer's ``expand_output``
                indicating which parent neuron each new input came from.
        """
        if new_in_features <= self.in_features:
            return

        old_in = self.in_features

        with torch.no_grad():
            new_weight = torch.zeros(
                (self.out_features, new_in_features), device=self.device
            )
            new_weight[:, :old_in] = self.weight

            if mapping_indices is not None:
                # Copy incoming weights for the new input channels
                new_weight[:, old_in:] = self.weight[:, mapping_indices]

                # Net2WiderNet normalization: count replication factor per parent
                # and scale both original and clone weights to preserve sums.
                replication_count = torch.ones(old_in, device=self.device)
                for idx in mapping_indices:
                    replication_count[idx.item()] += 1

                # Scale columns for parents that were replicated
                for i, idx in enumerate(mapping_indices):
                    parent = idx.item()
                    factor = 1.0 / replication_count[parent]
                    new_weight[:, parent] = self.weight[:, parent] * factor
                    new_weight[:, old_in + i] = (
                        self.weight[:, parent] * factor
                    )

            self.weight = nn.Parameter(new_weight)
            self.in_features = new_in_features

    def expand_output_zero(self, new_out_features: int) -> None:
        """Expand output dimension with zero-initialized new rows.

        Unlike ``expand_output``, no neuron copying — simply appends
        zero rows. Used for width growth where new dimensions should
        not contribute to the output until trained.

        Args:
            new_out_features: Target output dimension.
        """
        if new_out_features <= self.out_features:
            return

        with torch.no_grad():
            new_weight = torch.zeros(
                (new_out_features, self.in_features), device=self.device
            )
            new_weight[: self.out_features] = self.weight
            self.weight = nn.Parameter(new_weight)

            if self.bias is not None:
                new_bias = torch.zeros(new_out_features, device=self.device)
                new_bias[: self.out_features] = self.bias
                self.bias = nn.Parameter(new_bias)

            self.out_features = new_out_features

    def expand_input_zero(self, new_in_features: int) -> None:
        """Expand input dimension with zero-initialized new columns.

        Unlike ``expand_input``, no weight copying or scaling — simply
        appends zero columns. New input dimensions will not contribute
        to the output until trained.

        Args:
            new_in_features: Target input dimension.
        """
        if new_in_features <= self.in_features:
            return

        with torch.no_grad():
            new_weight = torch.zeros(
                (self.out_features, new_in_features), device=self.device
            )
            new_weight[:, : self.in_features] = self.weight
            self.weight = nn.Parameter(new_weight)
            self.in_features = new_in_features

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


class GrowableEmbedding(nn.Module):
    """An embedding layer that can expand its vocabulary size and embedding dimension.

    Supports two types of growth:
    - Vocabulary expansion: adds new token embeddings (randomly initialized).
    - Dimension expansion: widens the embedding vectors, copying existing values
      and adding noise for the new dimensions.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return nn.functional.embedding(
            input_ids, self.weight, padding_idx=self.padding_idx
        )

    def expand_vocab(self, new_num_embeddings: int) -> None:
        """Add new token embeddings (randomly initialized).

        Args:
            new_num_embeddings: Target vocabulary size.
        """
        if new_num_embeddings <= self.num_embeddings:
            return

        n_new = new_num_embeddings - self.num_embeddings

        with torch.no_grad():
            new_weight = torch.empty(
                (new_num_embeddings, self.embedding_dim), device=self.device
            )
            new_weight[: self.num_embeddings] = self.weight
            # Initialize new embeddings with same distribution
            nn.init.normal_(new_weight[self.num_embeddings :])

            self.weight = nn.Parameter(new_weight)
            self.num_embeddings = new_num_embeddings

    def expand_dim(
        self, new_embedding_dim: int, noise_std: float = 1e-4
    ) -> None:
        """Widen the embedding dimension.

        Preserves existing values and fills new dimensions with small noise.
        This is NOT strictly function-preserving by itself — the downstream
        layers must also be expanded to maintain invariance.

        Args:
            new_embedding_dim: Target embedding dimension.
            noise_std: Std dev for initializing new dimensions.
        """
        if new_embedding_dim <= self.embedding_dim:
            return

        with torch.no_grad():
            new_weight = torch.zeros(
                (self.num_embeddings, new_embedding_dim), device=self.device
            )
            new_weight[:, : self.embedding_dim] = self.weight
            # Fill new dimensions with small noise
            noise = torch.normal(
                0,
                noise_std,
                size=(self.num_embeddings, new_embedding_dim - self.embedding_dim),
                device=self.device,
            )
            new_weight[:, self.embedding_dim :] = noise

            if self.padding_idx is not None:
                new_weight[self.padding_idx].fill_(0)

            self.weight = nn.Parameter(new_weight)
            self.embedding_dim = new_embedding_dim

    def expand_dim_net2wider(
        self, new_embedding_dim: int, noise_std: float = 1e-4
    ) -> Optional[torch.Tensor]:
        """Net2WiderNet-style dimension expansion.

        Instead of filling new dimensions with zeros, copies existing dimensions
        to the new positions. This ensures that LayerNorm (which normalizes
        across all dimensions) produces unchanged output, preserving the
        function-preserving invariant through the entire model chain.

        Returns mapping indices so downstream layers can adjust their input
        weights to compensate for the replication.

        Args:
            new_embedding_dim: Target embedding dimension.
            noise_std: Noise for symmetry breaking.

        Returns:
            Mapping indices (which parent dim each new dim was copied from),
            or None if no growth needed.
        """
        if new_embedding_dim <= self.embedding_dim:
            return None

        n_new = new_embedding_dim - self.embedding_dim
        indices = torch.randint(0, self.embedding_dim, (n_new,))

        with torch.no_grad():
            new_weight = torch.empty(
                (self.num_embeddings, new_embedding_dim), device=self.device
            )
            new_weight[:, : self.embedding_dim] = self.weight
            new_weight[:, self.embedding_dim :] = self.weight[:, indices]

            # Symmetry breaking noise
            if noise_std > 0:
                noise = torch.normal(
                    0,
                    noise_std,
                    size=(self.num_embeddings, n_new),
                    device=self.device,
                )
                new_weight[:, self.embedding_dim :] += noise

            if self.padding_idx is not None:
                new_weight[self.padding_idx].fill_(0)

            self.weight = nn.Parameter(new_weight)
            self.embedding_dim = new_embedding_dim

        return indices

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"padding_idx={self.padding_idx}"
        )
