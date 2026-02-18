"""Growth controller: decision logic for when and how to grow.

Implements the decision matrix:

| Resource State         | Loss Trend  | Action       |
|------------------------|-------------|--------------|
| High VRAM, Low Compute | Plateau     | Net2Wider    |
| High VRAM, High Compute| Plateau     | MoE Spawn    |
| Low VRAM, Any          | Plateau     | Prune/Quantize|
| Any                    | Decreasing  | Wait         |
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from morphllm.monitor import ResourceMonitor, ResourceSnapshot


class GrowthAction(Enum):
    """Possible actions the controller can recommend."""

    WAIT = auto()           # Loss still decreasing — do nothing
    GROW_WIDTH = auto()     # Expand d_model (Net2WiderNet)
    GROW_DEPTH = auto()     # Insert identity block (Net2DeeperNet)
    SPAWN_MOE = auto()      # Convert FFN → MoE (high compute utilization)
    PRUNE = auto()          # Out of VRAM — prune or quantize
    NO_BUDGET = auto()      # Would grow, but not enough VRAM


@dataclass
class GrowthConfig:
    """Tunable hyperparameters for growth decisions.

    Attributes:
        vram_ceiling_pct: Max VRAM usage before blocking growth (default 85%).
        vram_safety_pct: VRAM margin to reserve during growth (default 10%).
        ram_ceiling_pct: Max RAM usage before blocking growth.
        compute_threshold_pct: GPU util above this → "high compute" (60%).
        plateau_window: Number of recent losses for plateau detection.
        plateau_threshold: Max std-dev of loss window to count as plateau.
        min_steps_between_growths: Cooldown steps after a growth event.
        prefer_width_over_depth: If True, prefer width growth over depth
            when both are possible. Default True (width first).
        max_d_model: Upper bound on d_model growth.
        max_n_layers: Upper bound on depth growth.
        width_growth_factor: Multiplier for width growth (e.g., 1.25 = +25%).
    """

    vram_ceiling_pct: float = 85.0
    vram_safety_pct: float = 10.0
    ram_ceiling_pct: float = 80.0
    compute_threshold_pct: float = 60.0
    plateau_window: int = 50
    plateau_threshold: float = 0.01
    min_steps_between_growths: int = 200
    prefer_width_over_depth: bool = True
    max_d_model: int = 2048
    max_n_layers: int = 48
    width_growth_factor: float = 1.25


@dataclass
class GrowthDecision:
    """A concrete growth recommendation from the controller.

    Attributes:
        action: The recommended action.
        reason: Human-readable explanation.
        new_d_model: Proposed new d_model (for GROW_WIDTH).
        depth_position: Where to insert a new block (for GROW_DEPTH).
        moe_block_idx: Which block to convert to MoE (for SPAWN_MOE).
    """

    action: GrowthAction
    reason: str
    new_d_model: Optional[int] = None
    depth_position: Optional[int] = None
    moe_block_idx: Optional[int] = None


class LossTracker:
    """Tracks training loss history for plateau detection.

    A loss plateau is detected when the standard deviation of recent
    losses falls below a threshold, indicating training has stalled.
    """

    def __init__(self, window_size: int = 50, threshold: float = 0.01):
        self.window_size = window_size
        self.threshold = threshold
        self._losses: deque[float] = deque(maxlen=window_size)

    def record(self, loss: float) -> None:
        """Record a new loss value."""
        self._losses.append(loss)

    @property
    def count(self) -> int:
        """Number of recorded losses."""
        return len(self._losses)

    @property
    def is_full(self) -> bool:
        """Whether the window has enough data for plateau detection."""
        return len(self._losses) >= self.window_size

    @property
    def mean(self) -> float:
        """Mean of recent losses."""
        if not self._losses:
            return float("inf")
        return sum(self._losses) / len(self._losses)

    @property
    def std(self) -> float:
        """Standard deviation of recent losses."""
        if len(self._losses) < 2:
            return float("inf")
        m = self.mean
        variance = sum((x - m) ** 2 for x in self._losses) / len(self._losses)
        return variance ** 0.5

    @property
    def is_plateau(self) -> bool:
        """Detect if loss has plateaued (low variance over window)."""
        if not self.is_full:
            return False
        return self.std < self.threshold

    @property
    def is_decreasing(self) -> bool:
        """Detect if loss is trending downward.

        Compares first half mean to second half mean.
        """
        if not self.is_full:
            return False  # Not enough data to tell
        losses = list(self._losses)
        mid = len(losses) // 2
        first_half = sum(losses[:mid]) / mid
        second_half = sum(losses[mid:]) / (len(losses) - mid)
        return second_half < first_half

    def clear(self) -> None:
        """Reset loss history (e.g., after a growth event)."""
        self._losses.clear()


class GrowthController:
    """Decides when and how to grow the model.

    Combines resource monitoring (VRAM, RAM, compute utilization)
    with loss plateau detection to make growth decisions.

    Args:
        monitor: ResourceMonitor for checking system resources.
        config: Growth configuration hyperparameters.
    """

    def __init__(
        self,
        monitor: Optional[ResourceMonitor] = None,
        config: Optional[GrowthConfig] = None,
    ):
        self.monitor = monitor or ResourceMonitor()
        self.config = config or GrowthConfig()
        self.loss_tracker = LossTracker(
            window_size=self.config.plateau_window,
            threshold=self.config.plateau_threshold,
        )
        self._steps_since_growth = 0
        self._total_growths = 0

    def record_loss(self, loss: float) -> None:
        """Record a training loss value."""
        self.loss_tracker.record(loss)
        self._steps_since_growth += 1

    def decide(
        self,
        current_d_model: int,
        current_n_layers: int,
        current_n_head: int,
        num_params: int,
        snapshot: Optional[ResourceSnapshot] = None,
    ) -> GrowthDecision:
        """Evaluate whether the model should grow, and how.

        This implements the decision matrix from the design doc.

        Args:
            current_d_model: Current embedding dimension.
            current_n_layers: Current number of transformer blocks.
            current_n_head: Number of attention heads.
            num_params: Current total parameter count.
            snapshot: Optional pre-captured resource snapshot.
                If None, a fresh snapshot is taken.

        Returns:
            GrowthDecision with the recommended action and details.
        """
        snap = snapshot or self.monitor.snapshot()

        # Rule 0: Cooldown after recent growth
        if self._steps_since_growth < self.config.min_steps_between_growths:
            return GrowthDecision(
                action=GrowthAction.WAIT,
                reason=(
                    f"Cooldown: {self._steps_since_growth}/"
                    f"{self.config.min_steps_between_growths} steps since last growth"
                ),
            )

        # Rule 1: If loss is still decreasing, wait
        if self.loss_tracker.is_decreasing:
            return GrowthDecision(
                action=GrowthAction.WAIT,
                reason="Loss is still decreasing — no growth needed",
            )

        # Rule 2: If not enough data for plateau detection, wait
        if not self.loss_tracker.is_full:
            return GrowthDecision(
                action=GrowthAction.WAIT,
                reason=(
                    f"Insufficient data ({self.loss_tracker.count}/"
                    f"{self.config.plateau_window} steps) for plateau detection"
                ),
            )

        # Rule 3: Check for loss plateau
        if not self.loss_tracker.is_plateau:
            return GrowthDecision(
                action=GrowthAction.WAIT,
                reason=(
                    f"Loss not plateaued (std={self.loss_tracker.std:.4f} "
                    f"> threshold={self.config.plateau_threshold})"
                ),
            )

        # --- Loss has plateaued — determine resource state ---

        # Classify resource availability
        has_vram_budget = self._has_vram_budget(snap)
        high_compute = self._is_high_compute(snap)
        low_vram = self._is_low_vram(snap)

        # Decision matrix
        if low_vram:
            return GrowthDecision(
                action=GrowthAction.PRUNE,
                reason=(
                    f"Loss plateaued but VRAM critically low "
                    f"({snap.vram_used_pct:.1f}% used > "
                    f"{self.config.vram_ceiling_pct}% ceiling). "
                    f"Consider pruning or quantization."
                ),
            )

        if has_vram_budget and high_compute:
            return GrowthDecision(
                action=GrowthAction.SPAWN_MOE,
                reason=(
                    f"Loss plateaued, VRAM available, GPU compute high "
                    f"({snap.gpu_utilization_pct:.0f}%). "
                    f"MoE spawning avoids compute bottleneck."
                ),
                moe_block_idx=current_n_layers - 1,  # Convert last block
            )

        if has_vram_budget:
            # Choose between width and depth
            return self._choose_dense_growth(
                current_d_model, current_n_layers, current_n_head, num_params
            )

        # Has VRAM but not enough budget for growth
        return GrowthDecision(
            action=GrowthAction.NO_BUDGET,
            reason=(
                f"Loss plateaued but insufficient VRAM budget for growth. "
                f"VRAM: {snap.vram_used_pct:.1f}% used."
            ),
        )

    def notify_growth_completed(self) -> None:
        """Call after a growth event to reset cooldown and loss history."""
        self._steps_since_growth = 0
        self._total_growths += 1
        self.loss_tracker.clear()

    @property
    def total_growths(self) -> int:
        """Total number of growth events that have occurred."""
        return self._total_growths

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _has_vram_budget(self, snap: ResourceSnapshot) -> bool:
        """Check if we have VRAM headroom for growth."""
        if not snap.has_gpu:
            return True  # CPU mode — assume RAM is sufficient
        if snap.vram_used_pct is None:
            return True
        return snap.vram_used_pct < self.config.vram_ceiling_pct

    def _is_high_compute(self, snap: ResourceSnapshot) -> bool:
        """Check if GPU compute utilization is high."""
        if snap.gpu_utilization_pct is None:
            return False  # Can't tell — assume low
        return snap.gpu_utilization_pct >= self.config.compute_threshold_pct

    def _is_low_vram(self, snap: ResourceSnapshot) -> bool:
        """Check if VRAM is critically low (above ceiling)."""
        if not snap.has_gpu or snap.vram_used_pct is None:
            return False  # CPU mode — not a VRAM problem
        return snap.vram_used_pct >= self.config.vram_ceiling_pct

    def _choose_dense_growth(
        self,
        current_d_model: int,
        current_n_layers: int,
        current_n_head: int,
        num_params: int,
    ) -> GrowthDecision:
        """Choose between width and depth growth.

        Prefers width when below max and config says so,
        falls back to depth if width is maxed out.
        """
        can_grow_width = current_d_model < self.config.max_d_model
        can_grow_depth = current_n_layers < self.config.max_n_layers

        if not can_grow_width and not can_grow_depth:
            return GrowthDecision(
                action=GrowthAction.WAIT,
                reason=(
                    f"Loss plateaued but model at max size "
                    f"(d_model={current_d_model}, n_layers={current_n_layers})"
                ),
            )

        if self.config.prefer_width_over_depth and can_grow_width:
            new_d = self._compute_new_d_model(current_d_model, current_n_head)
            return GrowthDecision(
                action=GrowthAction.GROW_WIDTH,
                reason=(
                    f"Loss plateaued, VRAM available, low compute. "
                    f"Growing width {current_d_model} → {new_d}."
                ),
                new_d_model=new_d,
            )

        if can_grow_depth:
            return GrowthDecision(
                action=GrowthAction.GROW_DEPTH,
                reason=(
                    f"Loss plateaued, VRAM available. "
                    f"Growing depth {current_n_layers} → {current_n_layers + 1}."
                ),
                depth_position=current_n_layers,  # append at end
            )

        # Width maxed, depth available
        new_d = self._compute_new_d_model(current_d_model, current_n_head)
        return GrowthDecision(
            action=GrowthAction.GROW_WIDTH,
            reason=(
                f"Loss plateaued. Depth maxed, growing width "
                f"{current_d_model} → {new_d}."
            ),
            new_d_model=new_d,
        )

    def _compute_new_d_model(
        self, current_d_model: int, n_head: int
    ) -> int:
        """Compute new d_model, ensuring divisibility by n_head and cap."""
        raw = int(current_d_model * self.config.width_growth_factor)
        # Round up to nearest multiple of n_head
        new_d = ((raw + n_head - 1) // n_head) * n_head
        return min(new_d, self.config.max_d_model)
