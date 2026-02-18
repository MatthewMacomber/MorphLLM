"""Optimizer state migration utilities for model growth.

When a model grows (width or depth), the optimizer (Adam/AdamW) holds
state tensors (exp_avg, exp_avg_sq) that no longer match the new
parameter shapes. This module migrates those states safely.

Also provides memory defragmentation and warm-restart scheduling.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def migrate_optimizer_state(
    optimizer: Optimizer,
    old_param_refs: Dict[str, nn.Parameter],
    old_param_shapes: Dict[str, torch.Size],
    model: nn.Module,
) -> None:
    """Migrate Adam/AdamW optimizer state after model growth.

    Growth operations replace nn.Parameter objects (e.g., expand_input_zero
    creates a new tensor). The optimizer state dict is keyed by the OLD
    param object, so we must re-key it to the NEW param object and
    expand momentum buffers with zeros for new dimensions.

    Args:
        optimizer: The optimizer whose state needs migration.
        old_param_refs: Mapping of param name → old Parameter object
            (captured BEFORE growth via capture_param_state).
        old_param_shapes: Mapping of param name → old shape.
        model: The model after growth (provides new param references).
    """
    # Build name → new parameter mapping
    new_params = {name: p for name, p in model.named_parameters()}

    for name, old_shape in old_param_shapes.items():
        new_param = new_params.get(name)
        old_param = old_param_refs.get(name)
        if new_param is None or old_param is None:
            continue

        new_shape = new_param.shape

        # Get optimizer state keyed by the OLD param object
        state = optimizer.state.pop(old_param, None)
        if state is None:
            continue

        # Expand momentum buffers if shape changed
        if old_shape != new_shape:
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key not in state:
                    continue
                old_buffer = state[key]
                if old_buffer.shape == new_shape:
                    continue
                new_buffer = torch.zeros(
                    new_shape, dtype=old_buffer.dtype, device=old_buffer.device
                )
                slices = tuple(slice(0, s) for s in old_buffer.shape)
                new_buffer[slices] = old_buffer
                state[key] = new_buffer

        # Re-key state to the NEW param object
        optimizer.state[new_param] = state

    # Update param group references to point to new param objects
    _update_param_groups(optimizer, old_param_refs, new_params)


def migrate_optimizer_for_new_params(
    optimizer: Optimizer,
    new_params: List[nn.Parameter],
) -> None:
    """Add new parameters (from depth growth) to the optimizer.

    When new layers are inserted, their parameters aren't tracked by
    the optimizer yet. This adds them to the first param group.

    Args:
        optimizer: The optimizer to update.
        new_params: Parameters from the newly inserted layer(s).
    """
    for param in new_params:
        if param not in optimizer.state:
            # Add to first param group
            optimizer.param_groups[0]["params"].append(param)


def capture_param_state(
    model: nn.Module,
) -> Tuple[Dict[str, nn.Parameter], Dict[str, torch.Size]]:
    """Capture parameter references and shapes before a growth event.

    Call this BEFORE growing the model, then pass both dicts to
    migrate_optimizer_state() after growth.

    Returns:
        Tuple of (param_refs, param_shapes) where:
        - param_refs: name → Parameter object
        - param_shapes: name → shape
    """
    refs = {name: p for name, p in model.named_parameters()}
    shapes = {name: p.shape for name, p in model.named_parameters()}
    return refs, shapes


def capture_param_shapes(model: nn.Module) -> Dict[str, torch.Size]:
    """Capture current parameter shapes (convenience wrapper).

    For full migration, use capture_param_state() instead.
    """
    return {name: p.shape for name, p in model.named_parameters()}


def defragment_memory(
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
) -> None:
    """Defragment GPU memory after a growth event.

    Moves model (and optionally optimizer) to CPU, clears CUDA cache,
    then moves everything back. This consolidates fragmented allocations.

    Args:
        model: The model to defragment.
        optimizer: Optional optimizer (its tensors are on GPU too).
    """
    if not torch.cuda.is_available():
        return  # Nothing to defragment on CPU

    device = next(model.parameters()).device
    if device.type != "cuda":
        return

    # Move to CPU
    model.cpu()
    if optimizer is not None:
        _move_optimizer_to_device(optimizer, torch.device("cpu"))

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Move back
    model.to(device)
    if optimizer is not None:
        _move_optimizer_to_device(optimizer, device)


def create_warm_restart_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    base_lr: Optional[float] = None,
) -> LambdaLR:
    """Create a linear warmup LR scheduler for post-growth recovery.

    After growth, new neurons need a brief warmup period. This creates
    a scheduler that linearly increases LR from 0 to base_lr over
    warmup_steps, then holds constant.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of steps for linear warmup.
        base_lr: Target LR (default: use optimizer's current LR).

    Returns:
        LambdaLR scheduler.
    """
    if warmup_steps <= 0:
        return LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)
        return 1.0

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _move_optimizer_to_device(
    optimizer: Optimizer,
    device: torch.device,
) -> None:
    """Move all optimizer state tensors to the specified device."""
    for state in optimizer.state.values():
        for key, val in state.items():
            if isinstance(val, torch.Tensor):
                state[key] = val.to(device)


def _update_param_groups(
    optimizer: Optimizer,
    old_param_refs: Dict[str, nn.Parameter],
    new_params: Dict[str, nn.Parameter],
) -> None:
    """Replace old parameter references in optimizer param_groups.

    After growth, the Parameter objects change identity. The optimizer's
    param_groups still hold references to the OLD objects, so we swap
    them out for the new ones.
    """
    # Build old-id → new-param mapping
    old_to_new = {}
    for name, old_p in old_param_refs.items():
        new_p = new_params.get(name)
        if new_p is not None and old_p is not new_p:
            old_to_new[id(old_p)] = new_p

    if not old_to_new:
        return

    for group in optimizer.param_groups:
        new_param_list = []
        for p in group["params"]:
            replacement = old_to_new.get(id(p))
            new_param_list.append(replacement if replacement is not None else p)
        group["params"] = new_param_list
