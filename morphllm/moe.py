"""Dynamic Mixture-of-Experts with expert spawning from dense FFN.

Implements sparse scaling via MoE: when the growth controller decides
SPAWN_MOE, the dense FFN in a transformer block is converted into a
MoE layer with top-k token routing.

Components:
- **GatingNetwork**: Learned linear router with top-k selection + load
  balancing loss for even expert utilization.
- **Expert**: A single FFN expert (fc1 → GELU → fc2).
- **DynamicMoELayer**: Holds N experts + gating. Supports spawning new
  experts by cloning existing ones. Forward pass uses top-k routing
  with expert-parallel dispatch.
"""

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from morphllm.layers import GrowableLinear


# ------------------------------------------------------------------
# Expert (single FFN)
# ------------------------------------------------------------------


class Expert(nn.Module):
    """A single FFN expert: fc1 → GELU → fc2.

    This mirrors the FFN structure inside DynamicTransformerBlock.

    Args:
        d_model: Input/output dimension.
        d_ff: Hidden dimension.
        device: Target device.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: Optional[str] = None,
    ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.d_model = d_model
        self.d_ff = d_ff

        self.fc1 = GrowableLinear(d_model, d_ff, device=device)
        self.fc2 = GrowableLinear(d_ff, d_model, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: [batch*tokens, d_model] → [batch*tokens, d_model]."""
        return self.fc2(F.gelu(self.fc1(x)))

    @classmethod
    def from_ffn(
        cls,
        fc1: nn.Module,
        fc2: nn.Module,
        device: Optional[str] = None,
    ) -> "Expert":
        """Create an expert from existing FFN weights (for MoE conversion).

        Args:
            fc1: Existing first linear layer.
            fc2: Existing second linear layer.
            device: Target device.

        Returns:
            Expert initialized with the given weights.
        """
        d_model = fc1.weight.shape[1]
        d_ff = fc1.weight.shape[0]
        expert = cls(d_model, d_ff, device=device)

        with torch.no_grad():
            expert.fc1.weight.copy_(fc1.weight)
            if fc1.bias is not None and expert.fc1.bias is not None:
                expert.fc1.bias.copy_(fc1.bias)
            expert.fc2.weight.copy_(fc2.weight)
            if fc2.bias is not None and expert.fc2.bias is not None:
                expert.fc2.bias.copy_(fc2.bias)

        return expert


# ------------------------------------------------------------------
# Gating Network (Token Router)
# ------------------------------------------------------------------


class GatingNetwork(nn.Module):
    """Top-k gating with load-balancing auxiliary loss.

    Routes each token to the top-k experts based on learned affinity
    scores. Includes a load-balancing loss that penalizes uneven
    expert utilization to prevent "expert collapse" (all tokens
    routed to one expert).

    Args:
        d_model: Input token dimension.
        num_experts: Number of experts to route to.
        top_k: Number of experts each token is sent to.
        noise_std: Gaussian noise added during training for exploration.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std

        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self._load_balance_loss: torch.Tensor = torch.tensor(0.0)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute gating weights for each token.

        Args:
            x: Token embeddings [batch*seq_len, d_model].

        Returns:
            Tuple of:
                - gate_weights: Top-k softmax weights [num_tokens, top_k].
                - expert_indices: Which experts selected [num_tokens, top_k].
                - load_balance_loss: Scalar auxiliary loss.
        """
        # Compute logits
        logits = self.gate(x)  # [num_tokens, num_experts]

        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(
            logits, self.top_k, dim=-1
        )  # [num_tokens, top_k]

        # Softmax over selected experts only
        gate_weights = F.softmax(top_k_logits, dim=-1)  # [num_tokens, top_k]

        # Load-balancing loss (Switch Transformer style)
        # Encourages uniform expert usage
        self._load_balance_loss = self._compute_load_balance_loss(logits)

        return gate_weights, top_k_indices, self._load_balance_loss

    def _compute_load_balance_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute load-balancing auxiliary loss.

        From Switch Transformers (Fedus et al., 2022):
        L_balance = N * sum_i(f_i * P_i)
        where f_i = fraction of tokens routed to expert i
              P_i = mean routing probability for expert i
        """
        num_tokens = logits.shape[0]
        if num_tokens == 0:
            return torch.tensor(0.0, device=logits.device)

        # Routing probabilities (full softmax)
        probs = F.softmax(logits, dim=-1)  # [num_tokens, num_experts]

        # Which expert each token would go to (top-1 for balance calculation)
        assignments = torch.argmax(logits, dim=-1)  # [num_tokens]

        # f_i: fraction of tokens assigned to each expert
        # Use float scatter for differentiable-ish computation
        f = torch.zeros(self.num_experts, device=logits.device)
        for i in range(self.num_experts):
            f[i] = (assignments == i).float().mean()

        # P_i: mean probability for each expert
        p = probs.mean(dim=0)  # [num_experts]

        # Loss: N * dot(f, P)
        return self.num_experts * (f * p).sum()

    def grow(self, new_num_experts: int) -> None:
        """Expand the gate to accommodate more experts.

        New expert columns are initialized with small random weights.
        """
        if new_num_experts <= self.num_experts:
            return

        old_weight = self.gate.weight.data  # [old_num_experts, d_model]
        new_weight = torch.zeros(
            new_num_experts, self.d_model, device=old_weight.device
        )
        new_weight[: self.num_experts] = old_weight

        # Initialize new expert gate weights with small random values
        # so they have a chance of being selected
        nn.init.xavier_uniform_(
            new_weight[self.num_experts:]
        )

        self.gate = nn.Linear(
            self.d_model, new_num_experts, bias=False
        ).to(old_weight.device)
        self.gate.weight = nn.Parameter(new_weight)
        self.num_experts = new_num_experts


# ------------------------------------------------------------------
# Dynamic MoE Layer
# ------------------------------------------------------------------


class DynamicMoELayer(nn.Module):
    """Mixture-of-Experts layer with dynamic expert spawning.

    Replaces the dense FFN in a transformer block. Each token is routed
    to top-k experts, and the weighted outputs are combined.

    Supports:
    - Converting a dense FFN to MoE (first expert = original FFN).
    - Spawning new experts by cloning existing ones.
    - Growing all experts when d_model increases (width growth).

    Args:
        d_model: Model dimension.
        d_ff: FFN hidden dimension.
        num_experts: Initial number of experts.
        top_k: Number of experts per token.
        noise_std: Gating noise for training exploration.
        device: Target device.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 2,
        top_k: int = 2,
        noise_std: float = 0.1,
        device: Optional[str] = None,
    ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.d_model = d_model
        self.d_ff = d_ff
        self.top_k = min(top_k, num_experts)

        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, device=device) for _ in range(num_experts)
        ])
        self.gate = GatingNetwork(
            d_model, num_experts, top_k=self.top_k, noise_std=noise_std
        )

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """MoE forward pass with top-k expert routing.

        Args:
            x: Input [batch, seq_len, d_model].

        Returns:
            Tuple of:
                - output: [batch, seq_len, d_model]
                - load_balance_loss: Scalar for auxiliary training objective.
        """
        batch_size, seq_len, d_model = x.shape
        # Flatten to [num_tokens, d_model]
        x_flat = x.view(-1, d_model)

        # Get gating decisions
        gate_weights, expert_indices, lb_loss = self.gate(x_flat)
        # gate_weights: [num_tokens, top_k]
        # expert_indices: [num_tokens, top_k]

        # Dispatch tokens to experts and combine
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_idx = expert_indices[:, k]  # [num_tokens]
            weight = gate_weights[:, k].unsqueeze(-1)  # [num_tokens, 1]

            for e_idx in range(self.num_experts):
                # Mask: which tokens go to this expert for this k-slot
                mask = expert_idx == e_idx  # [num_tokens]
                if not mask.any():
                    continue

                expert_input = x_flat[mask]  # [n_selected, d_model]
                expert_output = self.experts[e_idx](expert_input)
                output[mask] += weight[mask] * expert_output

        output = output.view(batch_size, seq_len, d_model)
        return output, lb_loss

    def spawn_expert(self, clone_from: int = 0, noise_std: float = 1e-4) -> int:
        """Spawn a new expert by cloning an existing one.

        Small noise is added to break symmetry so the new expert
        diverges from its parent during training.

        Args:
            clone_from: Index of the expert to clone.
            noise_std: Standard deviation of noise for symmetry breaking.

        Returns:
            Index of the newly spawned expert.
        """
        if clone_from >= self.num_experts:
            raise IndexError(
                f"clone_from={clone_from} but only {self.num_experts} experts"
            )

        # Deep copy the source expert
        new_expert = copy.deepcopy(self.experts[clone_from])

        # Add noise for symmetry breaking
        with torch.no_grad():
            for param in new_expert.parameters():
                param.add_(torch.randn_like(param) * noise_std)

        self.experts.append(new_expert)

        # Expand gating network to include new expert
        self.gate.grow(self.num_experts)

        new_idx = self.num_experts - 1
        return new_idx

    @classmethod
    def from_dense_ffn(
        cls,
        fc1: nn.Module,
        fc2: nn.Module,
        top_k: int = 2,
        num_initial_experts: int = 2,
        noise_std: float = 1e-4,
        device: Optional[str] = None,
    ) -> "DynamicMoELayer":
        """Convert a dense FFN into a MoE layer.

        The first expert inherits the original FFN weights. Additional
        experts are clones with noise for symmetry breaking.

        This is the key function called during a SPAWN_MOE growth event.

        Args:
            fc1: Original first linear layer.
            fc2: Original second linear layer.
            top_k: Experts per token.
            num_initial_experts: Total experts to create.
            noise_std: Noise for cloned experts.
            device: Target device.

        Returns:
            DynamicMoELayer initialized from the dense FFN.
        """
        d_model = fc1.weight.shape[1]
        d_ff = fc1.weight.shape[0]

        if device is None:
            device = fc1.weight.device

        moe = cls.__new__(cls)
        nn.Module.__init__(moe)
        moe.device = device
        moe.d_model = d_model
        moe.d_ff = d_ff
        moe.top_k = min(top_k, num_initial_experts)

        # First expert = original FFN
        first_expert = Expert.from_ffn(fc1, fc2, device=device)
        moe.experts = nn.ModuleList([first_expert])

        # Clone additional experts with noise
        for _ in range(num_initial_experts - 1):
            cloned = copy.deepcopy(first_expert)
            with torch.no_grad():
                for param in cloned.parameters():
                    param.add_(torch.randn_like(param) * noise_std)
            moe.experts.append(cloned)

        moe.gate = GatingNetwork(
            d_model, num_initial_experts, top_k=moe.top_k
        )

        return moe


# ------------------------------------------------------------------
# Helper: Convert a transformer block's FFN to MoE
# ------------------------------------------------------------------


def convert_block_to_moe(
    block,
    top_k: int = 2,
    num_experts: int = 2,
    noise_std: float = 1e-4,
) -> DynamicMoELayer:
    """Replace a DynamicTransformerBlock's dense FFN with a MoE layer.

    After calling this, the block's fc1/fc2 are replaced by a single
    `moe` attribute, and the forward method should be updated to use it.

    Args:
        block: A DynamicTransformerBlock.
        top_k: Experts per token.
        num_experts: Number of initial experts.
        noise_std: Clone noise.

    Returns:
        The created DynamicMoELayer (also attached to block as block.moe).
    """
    moe_layer = DynamicMoELayer.from_dense_ffn(
        fc1=block.fc1,
        fc2=block.fc2,
        top_k=top_k,
        num_initial_experts=num_experts,
        noise_std=noise_std,
        device=getattr(block, "device", None),
    )

    # Replace FFN with MoE
    block.moe = moe_layer  # type: ignore[attr-defined]
    # Remove dense FFN to free memory
    del block.fc1
    del block.fc2
    block._use_moe = True  # type: ignore[attr-defined]

    return moe_layer
