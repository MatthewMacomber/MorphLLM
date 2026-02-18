"""Tests for the full MorphLLM model — the most critical test suite.

Width growth is approximately function-preserving (small LN perturbation).
Depth growth is exactly function-preserving (identity blocks).
"""

import torch
import pytest

from morphllm.model import MorphConfig, MorphLLM

DEVICE = "cpu"


def make_seed_model(**overrides) -> MorphLLM:
    """Create a small seed model for testing."""
    defaults = dict(
        vocab_size=100,
        d_model=32,
        n_head=4,
        n_layers=2,
        max_seq_len=64,
        dropout=0.0,
        device=DEVICE,
    )
    defaults.update(overrides)
    config = MorphConfig(**defaults)
    return MorphLLM(config)


class TestMorphLLMBasic:
    """Basic forward pass and config tests."""

    def test_forward_shape(self):
        """Logits have shape (batch, seq_len, vocab_size)."""
        model = make_seed_model()
        ids = torch.randint(0, 100, (2, 10), device=DEVICE)
        logits, loss = model(ids)
        assert logits.shape == (2, 10, 100)
        assert loss is None

    def test_forward_with_targets(self):
        """Forward with targets returns a scalar loss."""
        model = make_seed_model()
        ids = torch.randint(0, 100, (2, 10), device=DEVICE)
        targets = torch.randint(0, 100, (2, 10), device=DEVICE)
        logits, loss = model(ids, targets=targets)
        assert logits.shape == (2, 10, 100)
        assert loss is not None
        assert loss.ndim == 0  # scalar

    def test_architecture_summary(self):
        """architecture_summary returns expected keys."""
        model = make_seed_model()
        summary = model.architecture_summary()
        assert summary["d_model"] == 32
        assert summary["n_head"] == 4
        assert summary["n_layers"] == 2
        assert summary["vocab_size"] == 100
        assert summary["num_parameters"] > 0
        assert summary["param_size_mb"] > 0

    def test_seed_model_is_small(self):
        """Seed model has a modest parameter count."""
        model = make_seed_model()
        # A 32-dim, 2-layer model should be well under 1M params
        assert model.num_parameters < 500_000


class TestMorphLLMWidthGrowth:
    """Tests for grow_width.

    Width growth is approximately function-preserving: the output changes
    slightly due to LayerNorm renormalization, but the perturbation is small.
    """

    def test_approximately_preserving_grow_width(self):
        """Logits close after width growth (small LN perturbation)."""
        torch.manual_seed(42)
        model = make_seed_model()
        ids = torch.randint(0, 100, (2, 10), device=DEVICE)

        with torch.no_grad():
            logits_before, _ = model(ids)

        model.grow_width(64)

        with torch.no_grad():
            logits_after, _ = model(ids)

        # Perturbation from LN renormalization — should be small but not zero
        diff = (logits_before - logits_after).abs().max().item()
        assert diff < 2.0, f"Width growth perturbation too large: {diff}"

    def test_grow_width_updates_architecture(self):
        """grow_width correctly updates model config and dimensions."""
        model = make_seed_model()
        params_before = model.num_parameters

        model.grow_width(64)

        assert model.d_model == 64
        assert model.config.d_model == 64
        assert model.num_parameters > params_before

    def test_grow_width_noop(self):
        """grow_width with smaller/equal size does nothing."""
        model = make_seed_model()
        params = model.num_parameters
        model.grow_width(32)
        assert model.num_parameters == params

    def test_grow_width_bad_divisibility(self):
        """grow_width rejects sizes not divisible by n_head."""
        model = make_seed_model()
        with pytest.raises(AssertionError):
            model.grow_width(33)

    def test_forward_after_grow_width(self):
        """Forward pass works with correct logits shape after growth."""
        model = make_seed_model()
        model.grow_width(64)
        ids = torch.randint(0, 100, (2, 10), device=DEVICE)
        logits, _ = model(ids)
        assert logits.shape == (2, 10, 100)

    def test_multiple_width_growths(self):
        """Multiple width growths produce valid forward passes."""
        torch.manual_seed(42)
        model = make_seed_model()

        model.grow_width(48)
        ids = torch.randint(0, 100, (2, 10), device=DEVICE)
        logits1, _ = model(ids)
        assert logits1.shape == (2, 10, 100)

        model.grow_width(64)
        logits2, _ = model(ids)
        assert logits2.shape == (2, 10, 100)


class TestMorphLLMDepthGrowth:
    """Tests for grow_depth (Net2DeeperNet) — EXACTLY function-preserving."""

    def test_function_preserving_grow_depth(self):
        """CRITICAL: Logits unchanged after inserting an identity block."""
        torch.manual_seed(42)
        model = make_seed_model()
        ids = torch.randint(0, 100, (2, 10), device=DEVICE)

        with torch.no_grad():
            logits_before, _ = model(ids)

        model.grow_depth()  # Append at end

        with torch.no_grad():
            logits_after, _ = model(ids)

        torch.testing.assert_close(
            logits_before, logits_after, atol=1e-5, rtol=1e-5
        )

    def test_grow_depth_at_position(self):
        """Inserting at a specific position preserves function."""
        torch.manual_seed(42)
        model = make_seed_model()
        ids = torch.randint(0, 100, (2, 10), device=DEVICE)

        with torch.no_grad():
            logits_before, _ = model(ids)

        model.grow_depth(position=1)  # Insert between block 0 and 1

        with torch.no_grad():
            logits_after, _ = model(ids)

        assert model.n_layers == 3
        torch.testing.assert_close(
            logits_before, logits_after, atol=1e-5, rtol=1e-5
        )

    def test_grow_depth_updates_layer_count(self):
        """grow_depth increments n_layers."""
        model = make_seed_model()
        assert model.n_layers == 2
        model.grow_depth()
        assert model.n_layers == 3
        model.grow_depth()
        assert model.n_layers == 4

    def test_multiple_depth_growths_preserve_function(self):
        """Multiple depth insertions all preserve function."""
        torch.manual_seed(42)
        model = make_seed_model()
        ids = torch.randint(0, 100, (2, 10), device=DEVICE)

        with torch.no_grad():
            logits_original, _ = model(ids)

        # Insert 3 identity blocks at various positions
        model.grow_depth(position=0)
        model.grow_depth(position=2)
        model.grow_depth()

        with torch.no_grad():
            logits_grown, _ = model(ids)

        assert model.n_layers == 5
        torch.testing.assert_close(
            logits_original, logits_grown, atol=1e-4, rtol=1e-4
        )


class TestMorphLLMVocabGrowth:
    """Tests for expand_vocab."""

    def test_expand_vocab(self):
        """Vocabulary expansion updates config and allows new token IDs."""
        model = make_seed_model()
        model.expand_vocab(200)
        assert model.config.vocab_size == 200

        # Can now forward with token IDs up to 199
        ids = torch.randint(0, 200, (2, 10), device=DEVICE)
        logits, _ = model(ids)
        assert logits.shape == (2, 10, 200)

    def test_expand_vocab_noop(self):
        """Expansion with smaller/equal size does nothing."""
        model = make_seed_model()
        model.expand_vocab(100)
        assert model.config.vocab_size == 100


class TestMorphLLMCombinedGrowth:
    """Tests combining width and depth growth together."""

    def test_grow_width_then_depth(self):
        """Width then depth growth both produce valid outputs."""
        torch.manual_seed(42)
        model = make_seed_model()
        ids = torch.randint(0, 100, (2, 10), device=DEVICE)

        model.grow_width(64)
        logits1, _ = model(ids)
        assert logits1.shape == (2, 10, 100)

        model.grow_depth()
        logits2, _ = model(ids)
        assert logits2.shape == (2, 10, 100)

    def test_grow_depth_then_width(self):
        """Depth then width growth both produce valid outputs."""
        torch.manual_seed(42)
        model = make_seed_model()
        ids = torch.randint(0, 100, (2, 10), device=DEVICE)

        model.grow_depth()
        logits1, _ = model(ids)
        assert logits1.shape == (2, 10, 100)

        model.grow_width(64)
        logits2, _ = model(ids)
        assert logits2.shape == (2, 10, 100)

    def test_training_step_after_growth(self):
        """A gradient step succeeds after combined growth (no shape crash)."""
        model = make_seed_model()
        model.grow_width(64)
        model.grow_depth()

        ids = torch.randint(0, 100, (2, 10), device=DEVICE)
        targets = torch.randint(0, 100, (2, 10), device=DEVICE)

        logits, loss = model(ids, targets=targets)
        assert loss is not None

        loss.backward()
        # Verify gradients have correct shapes
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                assert p.grad.shape == p.shape
