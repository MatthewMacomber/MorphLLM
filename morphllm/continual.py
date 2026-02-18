"""Continual learning utilities for anti-forgetting during growth.

Implements three strategies inspired by the SDFT paper (arXiv:2601.19897v1):

1. **RehearsalBuffer** — Self-Synthesized Rehearsal (SSR): stores past
   training examples in a fixed-size circular buffer and mixes them
   into new training batches to prevent catastrophic forgetting.

2. **SelfDistillationLoss** — On-policy self-distillation via reverse KL
   divergence. The model acts as its own teacher (via EMA weights),
   generating training signals that preserve prior capabilities.

3. **FreezeExpandTune** — Structural regularization cycle:
   Freeze old params → Expand model → Tune new params → Unfreeze all.
   This protects learned representations during the initial adaptation
   of newly grown capacity.
"""

import copy
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# 1. Rehearsal Buffer (Self-Synthesized Rehearsal)
# ------------------------------------------------------------------


class RehearsalBuffer:
    """Fixed-size circular buffer for storing past training examples.

    During training, a fraction of each batch comes from this buffer
    (rehearsal) to prevent forgetting earlier data. This implements
    Self-Synthesized Rehearsal (SSR) — the model's own outputs on
    earlier data are replayed alongside new data.

    Args:
        max_size: Maximum number of examples to store.
        mix_ratio: Fraction of each batch that should come from the
            buffer (0.0 = no rehearsal, 1.0 = all rehearsal).
    """

    def __init__(self, max_size: int = 10000, mix_ratio: float = 0.2):
        if not 0.0 <= mix_ratio <= 1.0:
            raise ValueError(f"mix_ratio must be in [0, 1], got {mix_ratio}")
        self.max_size = max_size
        self.mix_ratio = mix_ratio
        self._buffer: deque[Dict[str, torch.Tensor]] = deque(maxlen=max_size)

    def add(self, input_ids: torch.Tensor, targets: torch.Tensor) -> None:
        """Store training examples in the buffer.

        Args:
            input_ids: Token IDs, shape [batch, seq_len].
            targets: Target IDs, shape [batch, seq_len].
        """
        # Store individual examples (detached, on CPU to save GPU mem)
        for i in range(input_ids.size(0)):
            self._buffer.append({
                "input_ids": input_ids[i].detach().cpu(),
                "targets": targets[i].detach().cpu(),
            })

    def sample(
        self,
        n: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n examples from the buffer.

        Args:
            n: Number of examples to sample.
            device: Device to place tensors on.

        Returns:
            Tuple of (input_ids, targets), each shape [n, seq_len].
        """
        n = min(n, len(self._buffer))
        samples = random.sample(list(self._buffer), n)
        input_ids = torch.stack([s["input_ids"] for s in samples]).to(device)
        targets = torch.stack([s["targets"] for s in samples]).to(device)
        return input_ids, targets

    def mix_batch(
        self,
        new_input_ids: torch.Tensor,
        new_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mix new training data with rehearsal examples.

        Replaces a fraction (mix_ratio) of the batch with buffer samples.
        If the buffer is empty or mix_ratio is 0, returns the batch unchanged.

        Args:
            new_input_ids: New batch input IDs [batch, seq_len].
            new_targets: New batch targets [batch, seq_len].

        Returns:
            Mixed (input_ids, targets) with rehearsal examples swapped in.
        """
        if self.mix_ratio == 0.0 or len(self._buffer) == 0:
            return new_input_ids, new_targets

        batch_size = new_input_ids.size(0)
        n_rehearsal = max(1, int(batch_size * self.mix_ratio))
        n_new = batch_size - n_rehearsal

        # Sample from buffer
        rehearsal_ids, rehearsal_tgt = self.sample(
            n_rehearsal, device=new_input_ids.device
        )

        # Truncate or pad rehearsal to match seq_len
        seq_len = new_input_ids.size(1)
        rehearsal_ids = _pad_or_truncate(rehearsal_ids, seq_len)
        rehearsal_tgt = _pad_or_truncate(rehearsal_tgt, seq_len)

        # Combine: keep first n_new from new, append rehearsal
        mixed_ids = torch.cat([new_input_ids[:n_new], rehearsal_ids], dim=0)
        mixed_tgt = torch.cat([new_targets[:n_new], rehearsal_tgt], dim=0)

        return mixed_ids, mixed_tgt

    @property
    def size(self) -> int:
        """Number of examples currently in the buffer."""
        return len(self._buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()


# ------------------------------------------------------------------
# 2. Self-Distillation Loss (SDFT-inspired)
# ------------------------------------------------------------------


class EMATeacher:
    """Exponential Moving Average teacher for self-distillation.

    Maintains a shadow copy of the student model's parameters,
    updated via EMA. The teacher generates "soft targets" that
    anchor the student to its prior behavior, reducing forgetting.

    Inspired by SDFT (arXiv:2601.19897v1), Section 3.

    Args:
        model: The student model.
        decay: EMA decay factor (0.999 = slow update, more stable).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        # Deep copy parameters (not the model structure, just state)
        self._shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            self._shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        """Update shadow parameters with EMA.

        Call this after each optimizer step.
        """
        for name, param in model.named_parameters():
            if name in self._shadow:
                self._shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )

    def apply_shadow(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Temporarily replace model params with shadow params.

        Returns the original parameters so they can be restored.
        """
        originals = {}
        for name, param in model.named_parameters():
            if name in self._shadow:
                originals[name] = param.data.clone()
                param.data.copy_(self._shadow[name])
        return originals

    def restore(self, model: nn.Module, originals: Dict[str, torch.Tensor]) -> None:
        """Restore original parameters after teacher inference."""
        for name, param in model.named_parameters():
            if name in originals:
                param.data.copy_(originals[name])

    def grow_shadow(self, model: nn.Module) -> None:
        """Re-sync shadow after model growth (new params added/reshaped)."""
        self._shadow = {}
        for name, param in model.named_parameters():
            self._shadow[name] = param.data.clone()


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Reverse KL divergence loss for self-distillation.

    Implements DKL(student || teacher) at the token level.
    The student tries to match the teacher's distribution,
    anchoring it to prior behavior.

    From SDFT paper, Equation 1-2.

    Args:
        student_logits: Student output logits [batch, seq_len, vocab].
        teacher_logits: Teacher output logits [batch, seq_len, vocab].
        temperature: Softmax temperature for smoothing distributions.

    Returns:
        Scalar loss (mean reverse KL over all tokens).
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # Reverse KL: DKL(student || teacher)
    # = sum_x student(x) * (log student(x) - log teacher(x))
    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
        log_target=False,
    )

    # Scale by T^2 to match the gradient magnitude of cross-entropy
    return kl * (temperature ** 2)


# ------------------------------------------------------------------
# 3. Freeze-Expand-Tune Cycle
# ------------------------------------------------------------------


@dataclass
class FreezeState:
    """Tracks which parameters are frozen vs. tunable."""
    frozen_names: List[str] = field(default_factory=list)
    is_frozen: bool = False


class FreezeExpandTune:
    """Manages the Freeze-Expand-Tune structural regularization cycle.

    After a growth event:
    1. **Freeze** — Lock all pre-existing parameters.
    2. **Tune** — Train only the newly added parameters for N steps.
    3. **Unfreeze** — Unlock all parameters for full fine-tuning.

    This protects learned representations during the critical initial
    adaptation period after growth.

    Args:
        warmup_steps: Number of steps to keep old params frozen.
    """

    def __init__(self, warmup_steps: int = 100):
        self.warmup_steps = warmup_steps
        self._state = FreezeState()
        self._steps_frozen = 0

    def freeze_old_params(
        self,
        model: nn.Module,
        old_param_names: List[str],
    ) -> None:
        """Freeze pre-existing parameters after growth.

        Args:
            model: The model after growth.
            old_param_names: Names of parameters that existed before growth.
        """
        self._state.frozen_names = []
        for name, param in model.named_parameters():
            if name in old_param_names:
                param.requires_grad = False
                self._state.frozen_names.append(name)
        self._state.is_frozen = True
        self._steps_frozen = 0

    def step(self, model: nn.Module) -> bool:
        """Called after each training step during the frozen phase.

        Returns:
            True if unfreezing just happened (training should continue).
        """
        if not self._state.is_frozen:
            return False

        self._steps_frozen += 1
        if self._steps_frozen >= self.warmup_steps:
            self.unfreeze_all(model)
            return True
        return False

    def unfreeze_all(self, model: nn.Module) -> None:
        """Unfreeze all parameters for full fine-tuning."""
        for param in model.parameters():
            param.requires_grad = True
        self._state.frozen_names = []
        self._state.is_frozen = False
        self._steps_frozen = 0

    @property
    def is_frozen(self) -> bool:
        """Whether old params are currently frozen."""
        return self._state.is_frozen

    @property
    def steps_remaining(self) -> int:
        """Steps remaining in the freeze phase."""
        if not self._state.is_frozen:
            return 0
        return max(0, self.warmup_steps - self._steps_frozen)

    @property
    def n_frozen(self) -> int:
        """Number of currently frozen parameters."""
        return len(self._state.frozen_names)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _pad_or_truncate(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad or truncate tensor along dim=1 to target_len."""
    current_len = tensor.size(1)
    if current_len == target_len:
        return tensor
    if current_len > target_len:
        return tensor[:, :target_len]
    # Pad with zeros
    pad_size = target_len - current_len
    return F.pad(tensor, (0, pad_size), value=0)
