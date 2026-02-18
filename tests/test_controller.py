"""Tests for GrowthController, LossTracker, and the decision matrix."""

import pytest

from morphllm.controller import (
    GrowthAction,
    GrowthConfig,
    GrowthController,
    GrowthDecision,
    LossTracker,
)
from morphllm.monitor import ResourceMonitor, ResourceSnapshot


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_snapshot(
    vram_used_pct: float = 50.0,
    gpu_util: float = 30.0,
    has_gpu: bool = True,
) -> ResourceSnapshot:
    """Create a synthetic ResourceSnapshot for testing."""
    vram_total = 8_000_000_000  # 8 GB
    vram_used = int(vram_total * vram_used_pct / 100)
    return ResourceSnapshot(
        ram_total_bytes=64_000_000_000,
        ram_used_bytes=16_000_000_000,
        ram_used_pct=25.0,
        ram_free_bytes=48_000_000_000,
        vram_total_bytes=vram_total if has_gpu else None,
        vram_used_bytes=vram_used if has_gpu else None,
        vram_used_pct=vram_used_pct if has_gpu else None,
        vram_free_bytes=(vram_total - vram_used) if has_gpu else None,
        gpu_utilization_pct=gpu_util if has_gpu else None,
        has_gpu=has_gpu,
    )


def fill_plateau(controller: GrowthController, loss: float = 1.0, n: int = 50):
    """Fill loss tracker with constant losses to trigger plateau."""
    for _ in range(n):
        controller.record_loss(loss)


# ------------------------------------------------------------------
# LossTracker Tests
# ------------------------------------------------------------------

class TestLossTracker:
    """Unit tests for LossTracker plateau detection."""

    def test_empty_tracker(self):
        tracker = LossTracker(window_size=10)
        assert tracker.count == 0
        assert not tracker.is_full
        assert not tracker.is_plateau
        assert not tracker.is_decreasing

    def test_plateau_detection_constant_loss(self):
        """Constant loss → std ≈ 0 → plateau detected."""
        tracker = LossTracker(window_size=10, threshold=0.01)
        for _ in range(10):
            tracker.record(1.0)
        assert tracker.is_full
        assert tracker.is_plateau
        assert tracker.std < 0.001

    def test_no_plateau_with_varying_loss(self):
        """Varying loss → high std → no plateau."""
        tracker = LossTracker(window_size=10, threshold=0.01)
        for i in range(10):
            tracker.record(1.0 + (i % 2) * 0.5)  # Alternates 1.0 and 1.5
        assert tracker.is_full
        assert not tracker.is_plateau

    def test_decreasing_loss(self):
        """Steadily decreasing loss is detected."""
        tracker = LossTracker(window_size=10, threshold=0.5)
        for i in range(10):
            tracker.record(10.0 - i * 0.5)  # 10.0, 9.5, ..., 5.5
        assert tracker.is_decreasing

    def test_not_decreasing_constant(self):
        """Constant loss is NOT classified as decreasing (second half ≈ first)."""
        tracker = LossTracker(window_size=10, threshold=0.01)
        for _ in range(10):
            tracker.record(1.0)
        assert not tracker.is_decreasing

    def test_mean_and_std(self):
        tracker = LossTracker(window_size=4)
        for v in [2.0, 4.0, 6.0, 8.0]:
            tracker.record(v)
        assert tracker.mean == pytest.approx(5.0)
        assert tracker.std == pytest.approx((5.0) ** 0.5, rel=0.01)

    def test_clear_resets(self):
        tracker = LossTracker(window_size=5)
        for _ in range(5):
            tracker.record(1.0)
        assert tracker.is_full
        tracker.clear()
        assert tracker.count == 0
        assert not tracker.is_full


# ------------------------------------------------------------------
# GrowthController Decision Matrix Tests
# ------------------------------------------------------------------

class TestGrowthControllerDecisionMatrix:
    """Tests verifying the decision matrix from Table 1."""

    def make_controller(self, **config_overrides) -> GrowthController:
        """Create a controller with custom config."""
        defaults = dict(
            vram_ceiling_pct=85.0,
            compute_threshold_pct=60.0,
            plateau_window=10,
            plateau_threshold=0.01,
            min_steps_between_growths=0,  # No cooldown for testing
            max_d_model=2048,
            max_n_layers=48,
            width_growth_factor=1.25,
        )
        defaults.update(config_overrides)
        config = GrowthConfig(**defaults)
        return GrowthController(monitor=ResourceMonitor(), config=config)

    def test_wait_when_loss_decreasing(self):
        """Rule: Loss decreasing → WAIT regardless of resources."""
        ctrl = self.make_controller()
        # Feed decreasing loss
        for i in range(10):
            ctrl.record_loss(10.0 - i * 0.5)

        snap = make_snapshot(vram_used_pct=30.0, gpu_util=10.0)
        decision = ctrl.decide(
            current_d_model=32, current_n_layers=2,
            current_n_head=4, num_params=50000, snapshot=snap,
        )
        assert decision.action == GrowthAction.WAIT

    def test_wait_when_insufficient_data(self):
        """Rule: Not enough loss data → WAIT."""
        ctrl = self.make_controller()
        ctrl.record_loss(1.0)  # Only 1 sample, need 10

        snap = make_snapshot(vram_used_pct=30.0)
        decision = ctrl.decide(
            current_d_model=32, current_n_layers=2,
            current_n_head=4, num_params=50000, snapshot=snap,
        )
        assert decision.action == GrowthAction.WAIT

    def test_grow_width_on_plateau_with_vram(self):
        """Rule: Plateau + VRAM available + low compute → GROW_WIDTH."""
        ctrl = self.make_controller()
        fill_plateau(ctrl, loss=1.0, n=10)

        snap = make_snapshot(vram_used_pct=50.0, gpu_util=20.0)
        decision = ctrl.decide(
            current_d_model=32, current_n_layers=2,
            current_n_head=4, num_params=50000, snapshot=snap,
        )
        assert decision.action == GrowthAction.GROW_WIDTH
        assert decision.new_d_model is not None
        assert decision.new_d_model > 32
        # Must be divisible by n_head
        assert decision.new_d_model % 4 == 0

    def test_spawn_moe_on_plateau_high_compute(self):
        """Rule: Plateau + VRAM available + high compute → SPAWN_MOE."""
        ctrl = self.make_controller()
        fill_plateau(ctrl, loss=1.0, n=10)

        snap = make_snapshot(vram_used_pct=50.0, gpu_util=80.0)
        decision = ctrl.decide(
            current_d_model=32, current_n_layers=2,
            current_n_head=4, num_params=50000, snapshot=snap,
        )
        assert decision.action == GrowthAction.SPAWN_MOE

    def test_prune_on_plateau_low_vram(self):
        """Rule: Plateau + VRAM critically low → PRUNE."""
        ctrl = self.make_controller()
        fill_plateau(ctrl, loss=1.0, n=10)

        snap = make_snapshot(vram_used_pct=90.0, gpu_util=20.0)
        decision = ctrl.decide(
            current_d_model=32, current_n_layers=2,
            current_n_head=4, num_params=50000, snapshot=snap,
        )
        assert decision.action == GrowthAction.PRUNE

    def test_prune_overrides_high_compute(self):
        """Rule: Even with high compute, low VRAM → PRUNE (safety first)."""
        ctrl = self.make_controller()
        fill_plateau(ctrl, loss=1.0, n=10)

        snap = make_snapshot(vram_used_pct=95.0, gpu_util=90.0)
        decision = ctrl.decide(
            current_d_model=32, current_n_layers=2,
            current_n_head=4, num_params=50000, snapshot=snap,
        )
        assert decision.action == GrowthAction.PRUNE

    def test_cpu_mode_no_gpu(self):
        """CPU mode (no GPU): should still recommend growth on plateau."""
        ctrl = self.make_controller()
        fill_plateau(ctrl, loss=1.0, n=10)

        snap = make_snapshot(has_gpu=False)
        decision = ctrl.decide(
            current_d_model=32, current_n_layers=2,
            current_n_head=4, num_params=50000, snapshot=snap,
        )
        # Should recommend width growth (CPU mode has unlimited "VRAM")
        assert decision.action == GrowthAction.GROW_WIDTH

    def test_depth_growth_when_width_maxed(self):
        """When d_model is at max, fall back to depth growth."""
        ctrl = self.make_controller(max_d_model=32)
        fill_plateau(ctrl, loss=1.0, n=10)

        snap = make_snapshot(vram_used_pct=50.0, gpu_util=20.0)
        decision = ctrl.decide(
            current_d_model=32, current_n_layers=2,
            current_n_head=4, num_params=50000, snapshot=snap,
        )
        assert decision.action == GrowthAction.GROW_DEPTH
        assert decision.depth_position is not None


class TestGrowthControllerCooldown:
    """Tests for cooldown behavior between growth events."""

    def test_cooldown_prevents_immediate_growth(self):
        config = GrowthConfig(
            min_steps_between_growths=100,
            plateau_window=10,
            plateau_threshold=0.01,
        )
        ctrl = GrowthController(config=config)

        # Fill plateau
        for _ in range(10):
            ctrl.record_loss(1.0)

        # Only 10 steps — cooldown of 100 not met
        snap = make_snapshot(vram_used_pct=50.0, gpu_util=20.0)
        decision = ctrl.decide(
            current_d_model=32, current_n_layers=2,
            current_n_head=4, num_params=50000, snapshot=snap,
        )
        assert decision.action == GrowthAction.WAIT
        assert "Cooldown" in decision.reason

    def test_notify_growth_resets_cooldown(self):
        config = GrowthConfig(
            min_steps_between_growths=5,
            plateau_window=5,
            plateau_threshold=0.01,
        )
        ctrl = GrowthController(config=config)

        # Fill enough steps
        for _ in range(10):
            ctrl.record_loss(1.0)

        # Should be able to grow now
        snap = make_snapshot(vram_used_pct=50.0, gpu_util=20.0)
        decision = ctrl.decide(
            current_d_model=32, current_n_layers=2,
            current_n_head=4, num_params=50000, snapshot=snap,
        )
        assert decision.action == GrowthAction.GROW_WIDTH

        # Notify growth → resets cooldown
        ctrl.notify_growth_completed()
        assert ctrl.total_growths == 1

        # Immediately after → cooldown blocks growth
        ctrl.record_loss(1.0)
        decision = ctrl.decide(
            current_d_model=40, current_n_layers=2,
            current_n_head=4, num_params=60000, snapshot=snap,
        )
        assert decision.action == GrowthAction.WAIT


class TestNewDModelComputation:
    """Tests for width growth factor and n_head divisibility."""

    def test_growth_factor_25pct(self):
        config = GrowthConfig(
            width_growth_factor=1.25,
            min_steps_between_growths=0,
            plateau_window=5,
            plateau_threshold=0.01,
        )
        ctrl = GrowthController(config=config)
        for _ in range(5):
            ctrl.record_loss(1.0)

        snap = make_snapshot(vram_used_pct=50.0, gpu_util=20.0)
        decision = ctrl.decide(
            current_d_model=32, current_n_layers=2,
            current_n_head=4, num_params=50000, snapshot=snap,
        )
        assert decision.action == GrowthAction.GROW_WIDTH
        # 32 * 1.25 = 40, divisible by 4
        assert decision.new_d_model == 40

    def test_growth_factor_respects_n_head(self):
        """New d_model must be divisible by n_head."""
        config = GrowthConfig(
            width_growth_factor=1.1,  # 32 * 1.1 = 35.2 → round up to 36
            min_steps_between_growths=0,
            plateau_window=5,
            plateau_threshold=0.01,
        )
        ctrl = GrowthController(config=config)
        for _ in range(5):
            ctrl.record_loss(1.0)

        snap = make_snapshot(vram_used_pct=50.0, gpu_util=20.0)
        decision = ctrl.decide(
            current_d_model=32, current_n_layers=2,
            current_n_head=4, num_params=50000, snapshot=snap,
        )
        assert decision.new_d_model is not None
        assert decision.new_d_model % 4 == 0
        assert decision.new_d_model >= 36  # ceil(35.2) → 36

    def test_growth_capped_at_max(self):
        config = GrowthConfig(
            width_growth_factor=2.0,
            max_d_model=48,
            min_steps_between_growths=0,
            plateau_window=5,
            plateau_threshold=0.01,
        )
        ctrl = GrowthController(config=config)
        for _ in range(5):
            ctrl.record_loss(1.0)

        snap = make_snapshot(vram_used_pct=50.0, gpu_util=20.0)
        decision = ctrl.decide(
            current_d_model=32, current_n_layers=2,
            current_n_head=4, num_params=50000, snapshot=snap,
        )
        assert decision.new_d_model == 48  # capped
