"""Tests for optimizer state migration after model growth.

These tests verify the critical property: after growing a model and
migrating optimizer state, training can continue without shape mismatches
or crashes, and momentum is preserved for original parameters.
"""

import torch
import torch.nn as nn

from morphllm.model import MorphConfig, MorphLLM
from morphllm.optimizer_utils import (
    capture_param_shapes,
    capture_param_state,
    create_warm_restart_scheduler,
    migrate_optimizer_for_new_params,
    migrate_optimizer_state,
)

DEVICE = "cpu"


def make_model_and_optimizer():
    """Create a small seed model with AdamW optimizer."""
    config = MorphConfig(
        vocab_size=100,
        d_model=32,
        n_head=4,
        n_layers=2,
        max_seq_len=64,
        dropout=0.0,
        device=DEVICE,
    )
    model = MorphLLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


class TestMigrateOptimizerStateWidth:
    """Tests for optimizer migration after width growth."""

    def test_training_step_after_width_growth(self):
        """CRITICAL: A full training step succeeds after growth + migration."""
        torch.manual_seed(42)
        model, optimizer = make_model_and_optimizer()
        ids = torch.randint(0, 100, (2, 10), device=DEVICE)
        targets = torch.randint(0, 100, (2, 10), device=DEVICE)

        # Step 1: Take a few training steps to build optimizer state
        for _ in range(3):
            logits, loss = model(ids, targets=targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Step 2: Capture shapes, grow, migrate
        old_refs, old_shapes = capture_param_state(model)
        model.grow_width(64)
        migrate_optimizer_state(optimizer, old_refs, old_shapes, model)

        # Step 3: Training should continue without error
        logits, loss = model(ids, targets=targets)
        loss.backward()
        optimizer.step()  # <-- This crashes if migration is wrong
        optimizer.zero_grad()

        assert logits.shape == (2, 10, 100)

    def test_momentum_preserved_for_original_params(self):
        """Optimizer momentum for original neurons is preserved after migration."""
        torch.manual_seed(42)
        model, optimizer = make_model_and_optimizer()
        ids = torch.randint(0, 100, (2, 10), device=DEVICE)
        targets = torch.randint(0, 100, (2, 10), device=DEVICE)

        # Build up optimizer state
        for _ in range(5):
            logits, loss = model(ids, targets=targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Capture pre-growth momentum for a specific parameter
        lm_head = model.lm_head
        state_before = optimizer.state[lm_head.weight]
        exp_avg_before = state_before["exp_avg"].clone()

        # Grow and migrate
        old_refs, old_shapes = capture_param_state(model)
        model.grow_width(64)
        migrate_optimizer_state(optimizer, old_refs, old_shapes, model)

        # Check momentum is preserved for old dimensions
        state_after = optimizer.state[lm_head.weight]
        exp_avg_after = state_after["exp_avg"]

        # LM head was [100, 32] → [100, 64]
        # Old momentum should be in [:, :32]
        old_momentum_slice = exp_avg_after[:, :32]
        torch.testing.assert_close(
            old_momentum_slice, exp_avg_before, atol=1e-7, rtol=1e-7
        )

    def test_new_dimensions_have_zero_momentum(self):
        """New dimensions should have zero momentum after migration."""
        torch.manual_seed(42)
        model, optimizer = make_model_and_optimizer()
        ids = torch.randint(0, 100, (2, 10), device=DEVICE)
        targets = torch.randint(0, 100, (2, 10), device=DEVICE)

        # Build optimizer state
        for _ in range(3):
            logits, loss = model(ids, targets=targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        old_refs, old_shapes = capture_param_state(model)
        model.grow_width(64)
        migrate_optimizer_state(optimizer, old_refs, old_shapes, model)

        # Check that new dimensions have zero momentum
        lm_head = model.lm_head
        state = optimizer.state[lm_head.weight]
        exp_avg = state["exp_avg"]
        # LM head: [100, 64], new dims are [:, 32:]
        assert torch.all(exp_avg[:, 32:] == 0.0)

    def test_multiple_growth_steps(self):
        """Multiple growth + migration cycles don't accumulate errors."""
        torch.manual_seed(42)
        model, optimizer = make_model_and_optimizer()
        ids = torch.randint(0, 100, (2, 10), device=DEVICE)
        targets = torch.randint(0, 100, (2, 10), device=DEVICE)

        for new_d in [40, 48, 64]:
            # Train a bit
            for _ in range(2):
                logits, loss = model(ids, targets=targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Grow + migrate
            old_refs, old_shapes = capture_param_state(model)
            model.grow_width(new_d)
            migrate_optimizer_state(optimizer, old_refs, old_shapes, model)

        # Final training step should work
        logits, loss = model(ids, targets=targets)
        loss.backward()
        optimizer.step()
        assert logits.shape == (2, 10, 100)


class TestMigrateOptimizerStateDepth:
    """Tests for optimizer migration after depth growth."""

    def test_training_step_after_depth_growth(self):
        """Training step succeeds after adding identity block + new params."""
        torch.manual_seed(42)
        model, optimizer = make_model_and_optimizer()
        ids = torch.randint(0, 100, (2, 10), device=DEVICE)
        targets = torch.randint(0, 100, (2, 10), device=DEVICE)

        # Build optimizer state
        for _ in range(3):
            logits, loss = model(ids, targets=targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Grow depth — new block's params aren't in optimizer yet
        old_params = set(model.parameters())
        model.grow_depth()
        new_params = [p for p in model.parameters() if p not in old_params]

        # Add new params to optimizer
        migrate_optimizer_for_new_params(optimizer, new_params)

        # Training should work
        logits, loss = model(ids, targets=targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert logits.shape == (2, 10, 100)
        assert model.n_layers == 3

    def test_combined_width_and_depth_growth(self):
        """Width + depth growth with proper migration."""
        torch.manual_seed(42)
        model, optimizer = make_model_and_optimizer()
        ids = torch.randint(0, 100, (2, 10), device=DEVICE)
        targets = torch.randint(0, 100, (2, 10), device=DEVICE)

        # Train
        for _ in range(3):
            logits, loss = model(ids, targets=targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Width growth + migration
        old_refs, old_shapes = capture_param_state(model)
        model.grow_width(64)
        migrate_optimizer_state(optimizer, old_refs, old_shapes, model)

        # Depth growth + new params
        old_params = set(model.parameters())
        model.grow_depth()
        new_params = [p for p in model.parameters() if p not in old_params]
        migrate_optimizer_for_new_params(optimizer, new_params)

        # Train more
        for _ in range(3):
            logits, loss = model(ids, targets=targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert model.d_model == 64
        assert model.n_layers == 3


class TestWarmRestartScheduler:
    """Tests for post-growth LR warmup."""

    def test_linear_warmup(self):
        """LR increases linearly during warmup, then holds."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = create_warm_restart_scheduler(optimizer, warmup_steps=10)

        lrs = []
        for step in range(20):
            lrs.append(optimizer.param_groups[0]["lr"])
            # Simulate a training step
            optimizer.step()
            scheduler.step()

        # LR should start near 0 and reach full LR by step 10
        assert lrs[0] < 0.0002  # Near zero at start
        assert abs(lrs[10] - 0.001) < 1e-6  # Full LR at warmup end
        assert abs(lrs[15] - 0.001) < 1e-6  # Stays at full LR

    def test_zero_warmup_is_noop(self):
        """Zero warmup steps = constant LR."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = create_warm_restart_scheduler(optimizer, warmup_steps=0)

        optimizer.step()
        scheduler.step()
        assert abs(optimizer.param_groups[0]["lr"] - 0.001) < 1e-6


class TestCaptureParamShapes:
    """Tests for capture_param_shapes utility."""

    def test_captures_all_params(self):
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))
        shapes = capture_param_shapes(model)

        assert "0.weight" in shapes
        assert shapes["0.weight"] == torch.Size([20, 10])
        assert "1.weight" in shapes
        assert shapes["1.weight"] == torch.Size([5, 20])

    def test_captures_morph_model(self):
        config = MorphConfig(
            vocab_size=100, d_model=32, n_head=4, n_layers=2,
            max_seq_len=64, dropout=0.0, device=DEVICE,
        )
        model = MorphLLM(config)
        shapes = capture_param_shapes(model)

        assert "lm_head.weight" in shapes
        assert shapes["lm_head.weight"] == torch.Size([100, 32])
