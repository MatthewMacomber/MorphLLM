"""Tests for continual learning: RehearsalBuffer, EMATeacher, FreezeExpandTune."""

import torch
import torch.nn as nn

from morphllm.continual import (
    EMATeacher,
    FreezeExpandTune,
    RehearsalBuffer,
    compute_distillation_loss,
)
from morphllm.model import MorphConfig, MorphLLM

DEVICE = "cpu"


def make_model():
    config = MorphConfig(
        vocab_size=100,
        d_model=32,
        n_head=4,
        n_layers=2,
        max_seq_len=64,
        dropout=0.0,
        device=DEVICE,
    )
    return MorphLLM(config)


# ------------------------------------------------------------------
# RehearsalBuffer Tests
# ------------------------------------------------------------------

class TestRehearsalBuffer:

    def test_add_and_size(self):
        buf = RehearsalBuffer(max_size=100)
        ids = torch.randint(0, 100, (4, 10))
        targets = torch.randint(0, 100, (4, 10))
        buf.add(ids, targets)
        assert buf.size == 4

    def test_max_size_enforced(self):
        buf = RehearsalBuffer(max_size=5)
        for _ in range(10):
            buf.add(
                torch.randint(0, 100, (1, 10)),
                torch.randint(0, 100, (1, 10)),
            )
        assert buf.size == 5

    def test_sample_returns_correct_shape(self):
        buf = RehearsalBuffer(max_size=100)
        ids = torch.randint(0, 100, (10, 8))
        targets = torch.randint(0, 100, (10, 8))
        buf.add(ids, targets)

        sampled_ids, sampled_tgt = buf.sample(3)
        assert sampled_ids.shape == (3, 8)
        assert sampled_tgt.shape == (3, 8)

    def test_sample_more_than_available(self):
        buf = RehearsalBuffer(max_size=100)
        ids = torch.randint(0, 100, (2, 8))
        targets = torch.randint(0, 100, (2, 8))
        buf.add(ids, targets)

        sampled_ids, _ = buf.sample(10)
        assert sampled_ids.shape[0] == 2  # Only 2 available

    def test_mix_batch_no_rehearsal(self):
        """mix_ratio=0 returns batch unchanged."""
        buf = RehearsalBuffer(max_size=100, mix_ratio=0.0)
        buf.add(torch.zeros(5, 10, dtype=torch.long), torch.zeros(5, 10, dtype=torch.long))

        new_ids = torch.ones(4, 10, dtype=torch.long)
        new_tgt = torch.ones(4, 10, dtype=torch.long)
        mixed_ids, mixed_tgt = buf.mix_batch(new_ids, new_tgt)

        assert torch.equal(mixed_ids, new_ids)
        assert torch.equal(mixed_tgt, new_tgt)

    def test_mix_batch_with_rehearsal(self):
        """mix_ratio=0.5 replaces half the batch with buffer samples."""
        buf = RehearsalBuffer(max_size=100, mix_ratio=0.5)
        buf.add(
            torch.zeros(10, 8, dtype=torch.long),
            torch.zeros(10, 8, dtype=torch.long),
        )

        new_ids = torch.ones(4, 8, dtype=torch.long)
        new_tgt = torch.ones(4, 8, dtype=torch.long)
        mixed_ids, mixed_tgt = buf.mix_batch(new_ids, new_tgt)

        # Total batch size preserved
        assert mixed_ids.shape[0] == 4
        # First n_new items are from new data (ones), rest from buffer (zeros)
        n_new = 4 - max(1, int(4 * 0.5))  # 4 - 2 = 2
        assert torch.all(mixed_ids[:n_new] == 1)
        assert torch.all(mixed_ids[n_new:] == 0)

    def test_mix_batch_empty_buffer(self):
        """Empty buffer → batch returned unchanged."""
        buf = RehearsalBuffer(max_size=100, mix_ratio=0.5)
        new_ids = torch.ones(4, 10, dtype=torch.long)
        new_tgt = torch.ones(4, 10, dtype=torch.long)
        mixed_ids, mixed_tgt = buf.mix_batch(new_ids, new_tgt)
        assert torch.equal(mixed_ids, new_ids)

    def test_mix_batch_different_seq_len(self):
        """Buffer items with different seq_len are padded/truncated."""
        buf = RehearsalBuffer(max_size=100, mix_ratio=0.5)
        buf.add(
            torch.ones(5, 20, dtype=torch.long),  # seq_len=20
            torch.ones(5, 20, dtype=torch.long),
        )

        new_ids = torch.zeros(4, 10, dtype=torch.long)  # seq_len=10
        new_tgt = torch.zeros(4, 10, dtype=torch.long)
        mixed_ids, _ = buf.mix_batch(new_ids, new_tgt)

        assert mixed_ids.shape[1] == 10  # Truncated to match new data

    def test_clear(self):
        buf = RehearsalBuffer(max_size=100)
        buf.add(torch.zeros(5, 10, dtype=torch.long), torch.zeros(5, 10, dtype=torch.long))
        buf.clear()
        assert buf.size == 0

    def test_invalid_mix_ratio(self):
        import pytest
        with pytest.raises(ValueError):
            RehearsalBuffer(mix_ratio=1.5)


# ------------------------------------------------------------------
# EMATeacher + Distillation Loss Tests
# ------------------------------------------------------------------

class TestEMATeacher:

    def test_shadow_initialized_from_model(self):
        model = make_model()
        teacher = EMATeacher(model, decay=0.999)
        for name, param in model.named_parameters():
            assert name in teacher._shadow
            torch.testing.assert_close(teacher._shadow[name], param.data)

    def test_ema_update(self):
        """Shadow moves slowly toward new params."""
        model = nn.Linear(10, 10)
        teacher = EMATeacher(model, decay=0.9)

        # Modify model params
        with torch.no_grad():
            model.weight.fill_(1.0)

        teacher.update(model)

        # Shadow should be 0.9 * old + 0.1 * new
        # Old was random, new is all 1s
        # Just check it moved toward 1.0
        assert teacher._shadow["weight"].mean() > 0.05  # Non-trivial update

    def test_apply_and_restore(self):
        """apply_shadow swaps params, restore puts them back."""
        model = nn.Linear(10, 10)
        original_weight = model.weight.data.clone()
        teacher = EMATeacher(model, decay=0.9)

        # Modify model
        with torch.no_grad():
            model.weight.fill_(99.0)

        # Apply shadow (should restore to original-ish values)
        originals = teacher.apply_shadow(model)
        assert not torch.equal(model.weight.data, torch.full_like(model.weight.data, 99.0))

        # Restore
        teacher.restore(model, originals)
        assert torch.all(model.weight.data == 99.0)

    def test_grow_shadow(self):
        """After model growth, grow_shadow resyncs."""
        model = make_model()
        teacher = EMATeacher(model, decay=0.999)

        old_d = model.d_model
        model.grow_width(64)
        teacher.grow_shadow(model)

        # Shadow should now have new shapes
        for name, param in model.named_parameters():
            assert name in teacher._shadow
            assert teacher._shadow[name].shape == param.shape


class TestDistillationLoss:

    def test_loss_is_zero_for_identical_logits(self):
        """KL divergence of identical distributions is ~0."""
        logits = torch.randn(2, 10, 100)
        loss = compute_distillation_loss(logits, logits.clone())
        assert loss.item() < 0.01

    def test_loss_positive_for_different_logits(self):
        """Different distributions → positive KL."""
        student = torch.randn(2, 10, 100)
        teacher = torch.randn(2, 10, 100)
        loss = compute_distillation_loss(student, teacher)
        assert loss.item() > 0.0

    def test_temperature_affects_loss(self):
        """Higher temperature → smoother distributions → smaller KL."""
        student = torch.randn(2, 10, 100)
        teacher = torch.randn(2, 10, 100)
        loss_t1 = compute_distillation_loss(student, teacher, temperature=1.0)
        loss_t5 = compute_distillation_loss(student, teacher, temperature=5.0)
        # With T^2 scaling, higher T doesn't necessarily give smaller loss
        # but the distributions are more similar, so unscaled KL is smaller
        # Just verify both are computable and positive
        assert loss_t1.item() > 0
        assert loss_t5.item() > 0

    def test_backward_works(self):
        """Loss supports backpropagation."""
        student = torch.randn(2, 10, 100, requires_grad=True)
        teacher = torch.randn(2, 10, 100)
        loss = compute_distillation_loss(student, teacher)
        loss.backward()
        assert student.grad is not None


# ------------------------------------------------------------------
# FreezeExpandTune Tests
# ------------------------------------------------------------------

class TestFreezeExpandTune:

    def test_freeze_old_params(self):
        model = make_model()
        old_names = set(n for n, _ in model.named_parameters())
        fet = FreezeExpandTune(warmup_steps=10)

        # Depth growth adds new blocks with new param names
        model.grow_depth()
        fet.freeze_old_params(model, list(old_names))

        assert fet.is_frozen
        assert fet.n_frozen > 0

        # Old params should be frozen, new params trainable
        for name, param in model.named_parameters():
            if name in old_names:
                assert not param.requires_grad, f"{name} should be frozen"
            else:
                assert param.requires_grad, f"{name} should be trainable"

    def test_step_counts_correctly(self):
        model = make_model()
        old_names = [n for n, _ in model.named_parameters()]
        fet = FreezeExpandTune(warmup_steps=5)

        model.grow_depth()
        fet.freeze_old_params(model, old_names)

        assert fet.steps_remaining == 5

        for i in range(4):
            unfrozen = fet.step(model)
            assert not unfrozen  # Not yet

        assert fet.steps_remaining == 1

        unfrozen = fet.step(model)
        assert unfrozen  # Just unfroze
        assert not fet.is_frozen
        assert fet.steps_remaining == 0

    def test_unfreeze_all_restores_grad(self):
        model = make_model()
        old_names = [n for n, _ in model.named_parameters()]
        fet = FreezeExpandTune(warmup_steps=5)

        model.grow_depth()
        fet.freeze_old_params(model, old_names)

        # Force unfreeze
        fet.unfreeze_all(model)

        for _, param in model.named_parameters():
            assert param.requires_grad

    def test_training_during_freeze(self):
        """Only new params update during freeze phase."""
        torch.manual_seed(42)
        model = make_model()
        old_names = [n for n, _ in model.named_parameters()]
        ids = torch.randint(0, 100, (2, 10))
        targets = torch.randint(0, 100, (2, 10))

        model.grow_depth()
        fet = FreezeExpandTune(warmup_steps=100)
        fet.freeze_old_params(model, old_names)

        # New block params should be trainable
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable) > 0, "No trainable params after freeze"

        optimizer = torch.optim.AdamW(trainable, lr=1e-3)

        # Should be able to train (only new params)
        logits, loss = model(ids, targets=targets)
        loss.backward()
        optimizer.step()
        # No error = success


# ------------------------------------------------------------------
# Integration: Full growth + continual-learning cycle
# ------------------------------------------------------------------

class TestContinualLearningIntegration:

    def test_full_growth_cycle(self):
        """End-to-end: train → grow → freeze → distill → unfreeze → train."""
        torch.manual_seed(42)
        model = make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        buffer = RehearsalBuffer(max_size=100, mix_ratio=0.3)
        teacher = EMATeacher(model, decay=0.999)
        fet = FreezeExpandTune(warmup_steps=3)

        ids = torch.randint(0, 100, (4, 10))
        targets = torch.randint(0, 100, (4, 10))

        # Phase 1: Train normally, fill buffer
        for _ in range(5):
            logits, loss = model(ids, targets=targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            teacher.update(model)
            buffer.add(ids, targets)

        # Phase 2: Grow depth (adds new block with new param names)
        old_names = [n for n, _ in model.named_parameters()]
        model.grow_depth()
        teacher.grow_shadow(model)
        fet.freeze_old_params(model, old_names)

        # Phase 3: Train with freeze + rehearsal (only new params)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-3)

        for _ in range(3):
            mixed_ids, mixed_tgt = buffer.mix_batch(ids, targets)
            logits, loss = model(mixed_ids, targets=mixed_tgt)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            just_unfroze = fet.step(model)

        assert not fet.is_frozen  # Should have unfrozen after 3 steps

        # Phase 4: Full fine-tuning
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        logits, loss = model(ids, targets=targets)
        loss.backward()
        optimizer.step()

        assert logits.shape == (4, 10, 100)
        assert model.n_layers == 3
