"""Training pipeline for MorphLLM with integrated growth hooks.

Features:
- **Growth Loop**: Periodically checks the GrowthController for decisions.
- **Anti-Forgetting**: Integrates RehearsalBuffer, EMATeacher, and
  FreezeExpandTune.
- **Optimizer Migration**: Handles state transfer when parameters change.
- **MoE Support**: Handles auxiliary loss and expert spawning.
- **Checkpointing**: Saves/loads model, optimizer, and training state.
"""

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from morphllm.continual import (
    EMATeacher,
    FreezeExpandTune,
    RehearsalBuffer,
    compute_distillation_loss,
)
from morphllm.controller import GrowthAction, GrowthController
from morphllm.model import MorphLLM
from morphllm.moe import convert_block_to_moe
from morphllm.optimizer_utils import (
    capture_param_state,
    create_warm_restart_scheduler,
    defragment_memory,
    migrate_optimizer_state,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for the training loop."""

    max_steps: int = 1000
    learning_rate: float = 1e-4
    batch_size: int = 32
    check_growth_every_n_steps: int = 50
    checkpoint_every_n_steps: int = 500
    warmup_steps_after_growth: int = 100
    rehearsal_ratio: float = 0.2
    distillation_temp: float = 2.0
    distillation_weight: float = 1.0
    moe_loss_weight: float = 0.01
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "checkpoints"


class Trainer:
    """Manages the training of a growing MorphLLM."""

    def __init__(
        self,
        model: MorphLLM,
        config: TrainerConfig,
        controller: Optional[GrowthController] = None,
        optimizer: Optional[Optimizer] = None,
    ):
        self.model = model.to(config.device)
        self.config = config
        self.controller = controller or GrowthController()
        
        # Optimizer (default to AdamW if not provided)
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate
        )

        # Continual Learning Components
        self.rehearsal = RehearsalBuffer(mix_ratio=config.rehearsal_ratio)
        self.teacher = EMATeacher(self.model)
        self.freezer = FreezeExpandTune(warmup_steps=config.warmup_steps_after_growth)

        # State
        self.global_step = 0
        self.current_loss = 0.0

        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)

    def train_step(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Perform a single training step.

        Args:
            input_ids: Input tokens [batch, seq_len].
            targets: Target tokens [batch, seq_len].

        Returns:
            Dictionary of loss components.
        """
        self.model.train()
        
        # 1. Mix with Rehearsal Buffer (SSR)
        input_ids, targets = self.rehearsal.mix_batch(input_ids, targets)
        input_ids = input_ids.to(self.config.device)
        targets = targets.to(self.config.device)

        # 2. Add current batch to buffer (detached, on CPU)
        self.rehearsal.add(input_ids, targets)

        # 3. Forward Pass
        logits, mlm_loss = self.model(input_ids, targets=targets)
        loss = mlm_loss

        metrics = {"mlm_loss": mlm_loss.item()}

        # 4. Self-Distillation Loss (SDFT)
        # Teacher predicts on same input (no grad)
        with torch.no_grad():
            teacher_logits, _ = self.model(input_ids)  # Current model as teacher
            # But we want the EMA weights...
            # The EMATeacher stores shadow params. We can't easily run forward
            # with shadow params without swapping them in.
            # Optimization: Only swap if distillation_weight > 0
        
        if self.config.distillation_weight > 0:
            # Swap in teacher params
            originals = self.teacher.apply_shadow(self.model)
            with torch.no_grad():
                self.model.eval()
                teacher_logits, _ = self.model(input_ids)
            # Swap back student params
            self.teacher.restore(self.model, originals)
            self.model.train()

            distill_loss = compute_distillation_loss(
                logits, teacher_logits, temperature=self.config.distillation_temp
            )
            loss = loss + self.config.distillation_weight * distill_loss
            metrics["distill_loss"] = distill_loss.item()

        # 5. MoE Load Balancing Loss
        moe_loss = sum(
            block._moe_loss for block in self.model.blocks if hasattr(block, "_moe_loss")
        )
        if isinstance(moe_loss, torch.Tensor) and moe_loss.item() > 0:
             loss = loss + self.config.moe_loss_weight * moe_loss
             metrics["moe_loss"] = moe_loss.item()

        # 6. Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 7. Post-step updates
        self.teacher.update(self.model)
        self.controller.record_loss(metrics["mlm_loss"])
        
        # Handle freeze/unfreeze cycle
        if self.freezer.is_frozen:
            just_unfrozen = self.freezer.step(self.model)
            if just_unfrozen:
                logger.info(f"Step {self.global_step}: Unfreezing all parameters.")

        self.global_step += 1
        self.current_loss = loss.item()
        metrics["total_loss"] = self.current_loss
        
        # 8. Check for growth
        if self.global_step % self.config.check_growth_every_n_steps == 0:
            self._maybe_grow()

        # 9. Checkpoint
        if self.global_step % self.config.checkpoint_every_n_steps == 0:
            self.save_checkpoint()

        return metrics

    def train_loop(self, dataloader: DataLoader) -> None:
        """Run the training loop for max_steps."""
        logger.info("Starting training loop...")
        iterator = iter(dataloader)
        
        while self.global_step < self.config.max_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)
            
            input_ids, targets = batch
            metrics = self.train_step(input_ids, targets)
            
            if self.global_step % 10 == 0:
                log_str = f"Step {self.global_step}: " + ", ".join(
                    f"{k}={v:.4f}" for k, v in metrics.items()
                )
                logger.info(log_str)

    def _maybe_grow(self) -> None:
        """Check controller and potentially execute model growth."""
        decision = self.controller.decide(
            current_d_model=self.model.d_model,
            current_n_layers=self.model.n_layers,
            current_n_head=self.model.config.n_head,
            num_params=self.model.num_parameters,
        )

        if decision.action == GrowthAction.WAIT:
            return

        logger.info(f"Step {self.global_step}: Growth Decision: {decision.action} - {decision.reason}")

        if decision.action == GrowthAction.NO_BUDGET:
             return # Just log and continue

        # 1. Capture state before growth (references & shapes)
        # Needed for optimizer migration
        old_param_refs, old_param_shapes = capture_param_state(self.model)
        old_param_names = list(old_param_refs.keys())

        # 2. Execute Growth
        if decision.action == GrowthAction.GROW_WIDTH:
            if decision.new_d_model:
                self.model.grow_width(decision.new_d_model)
                logger.info(f"Grown width to d_model={decision.new_d_model}")
        
        elif decision.action == GrowthAction.GROW_DEPTH:
            self.model.grow_depth(decision.depth_position)
            logger.info(f"Grown depth to n_layers={self.model.n_layers}")
            
            # Start freeze cycle for depth growth (Net2DeeperNet)
            self.freezer.freeze_old_params(self.model, old_param_names)
            logger.info("Freezing old parameters for adaptation phase.")

        elif decision.action == GrowthAction.SPAWN_MOE:
            if decision.moe_block_idx is not None:
                block = self.model.blocks[decision.moe_block_idx]
                # Check if already MoE
                if getattr(block, "_use_moe", False):
                     # Already MoE, spawn new expert
                     block.moe.spawn_expert(clone_from=0)
                     logger.info(f"Spawned new expert in block {decision.moe_block_idx}")
                else:
                    # Convert to MoE
                    convert_block_to_moe(block)
                    logger.info(f"Converted block {decision.moe_block_idx} to MoE")

        # 3. Post-Growth Updates
        
        # Migrate Optimizer State
        # Re-map state from old params to new params, expand momentum buffers
        migrate_optimizer_state(
            self.optimizer, old_param_refs, old_param_shapes, self.model
        )
        
        # Clean up memory (fragmentation from resizing)
        defragment_memory(self.model)
        
        # Warmup Scheduler (to stabilize training after structural change)
        # We replace the LR scheduler for a short period
        # (For simplicity here, we assume the user might be using a scheduler external to this class,
        # but if we owned the scheduler we'd wrap it. 
        # For now, let's just reset the optimizer's internal step count or similar if needed.
        # Actually, standard Adam doesn't need reset, but a warmup is good practice.)
        # TODO: Implement proper scheduler integration.
        
        # Update Teacher (resize shadow params)
        self.teacher.grow_shadow(self.model)
        
        # Notify Controller
        self.controller.notify_growth_completed()

    def save_checkpoint(self, path: Optional[str] = None) -> None:
        """Save training state."""
        if path is None:
            path = os.path.join(
                self.config.output_dir, f"checkpoint_{self.global_step}.pt"
            )
        
        state = {
            "global_step": self.global_step,
            "model_config": self.model.config,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            # We don't save controller/rehearsal state for simplicity in this prototype,
            # but in production we should.
        }
        torch.save(state, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training state."""
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        
        # 1. Restore Model
        # Note: We must reconstruct the model with the correct config first!
        # The user of this class must have already instantiated self.model 
        # with the correct structure or we need to rebuild it here.
        # For this implementation, we assume self.model matches the checkpoints architecture
        # OR we rely on state_dict loading to work if shapes match.
        # If the checkpoint has a different structure (e.g. more layers), loading will fail
        # unless we resize self.model first.
        
        # Basic check: resize model if config differs?
        # A robust implementation would rebuild the model from 'model_config'.
        # Here we just try to load state_dict.
        
        try:
            self.model.load_state_dict(checkpoint["model_state"])
        except RuntimeError as e:
            logger.error("Failed to load state dict - model architecture mismatch?")
            raise e
            
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.global_step = checkpoint["global_step"]
        logger.info(f"Loaded checkpoint from {path}")

