"""Tests for MoE: GatingNetwork, Expert, DynamicMoELayer, and block conversion."""

import torch
import torch.nn as nn

from morphllm.moe import (
    DynamicMoELayer,
    Expert,
    GatingNetwork,
    convert_block_to_moe,
)
from morphllm.model import MorphConfig, MorphLLM
from morphllm.transformer import DynamicTransformerBlock

DEVICE = "cpu"


# ------------------------------------------------------------------
# Expert Tests
# ------------------------------------------------------------------

class TestExpert:

    def test_forward_shape(self):
        expert = Expert(d_model=32, d_ff=128, device=DEVICE)
        x = torch.randn(10, 32)  # [num_tokens, d_model]
        out = expert(x)
        assert out.shape == (10, 32)

    def test_from_ffn(self):
        """Expert.from_ffn copies weights from existing linear layers."""
        from morphllm.layers import GrowableLinear
        fc1 = GrowableLinear(32, 128, device=DEVICE)
        fc2 = GrowableLinear(128, 32, device=DEVICE)

        expert = Expert.from_ffn(fc1, fc2, device=DEVICE)

        torch.testing.assert_close(expert.fc1.weight, fc1.weight)
        torch.testing.assert_close(expert.fc2.weight, fc2.weight)


# ------------------------------------------------------------------
# GatingNetwork Tests
# ------------------------------------------------------------------

class TestGatingNetwork:

    def test_forward_shapes(self):
        gate = GatingNetwork(d_model=32, num_experts=4, top_k=2)
        x = torch.randn(20, 32)  # 20 tokens

        weights, indices, lb_loss = gate(x)
        assert weights.shape == (20, 2)
        assert indices.shape == (20, 2)
        assert lb_loss.dim() == 0  # scalar

    def test_weights_sum_to_one(self):
        """Softmax gate weights for each token sum to ~1."""
        gate = GatingNetwork(d_model=32, num_experts=4, top_k=2)
        gate.eval()  # No noise
        x = torch.randn(10, 32)
        weights, _, _ = gate(x)
        sums = weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(10), atol=1e-5, rtol=1e-5)

    def test_load_balance_loss_positive(self):
        gate = GatingNetwork(d_model=32, num_experts=4, top_k=2)
        x = torch.randn(100, 32)
        _, _, lb_loss = gate(x)
        assert lb_loss.item() > 0

    def test_top_k_indices_in_range(self):
        gate = GatingNetwork(d_model=32, num_experts=4, top_k=2)
        x = torch.randn(20, 32)
        _, indices, _ = gate(x)
        assert indices.min() >= 0
        assert indices.max() < 4

    def test_grow_adds_expert_slots(self):
        gate = GatingNetwork(d_model=32, num_experts=2, top_k=2)
        gate.grow(4)
        assert gate.num_experts == 4
        assert gate.gate.weight.shape[0] == 4

        # Old weights preserved
        x = torch.randn(10, 32)
        weights, indices, _ = gate(x)
        assert indices.max() < 4


# ------------------------------------------------------------------
# DynamicMoELayer Tests
# ------------------------------------------------------------------

class TestDynamicMoELayer:

    def test_forward_shape(self):
        moe = DynamicMoELayer(d_model=32, d_ff=128, num_experts=2, top_k=2, device=DEVICE)
        x = torch.randn(2, 10, 32)  # [batch, seq, d_model]
        out, lb_loss = moe(x)
        assert out.shape == (2, 10, 32)
        assert lb_loss.dim() == 0

    def test_num_experts(self):
        moe = DynamicMoELayer(d_model=32, d_ff=128, num_experts=4, device=DEVICE)
        assert moe.num_experts == 4

    def test_spawn_expert(self):
        moe = DynamicMoELayer(d_model=32, d_ff=128, num_experts=2, device=DEVICE)
        new_idx = moe.spawn_expert(clone_from=0)
        assert new_idx == 2
        assert moe.num_experts == 3
        assert moe.gate.num_experts == 3

    def test_spawn_expert_out_of_range(self):
        import pytest
        moe = DynamicMoELayer(d_model=32, d_ff=128, num_experts=2, device=DEVICE)
        with pytest.raises(IndexError):
            moe.spawn_expert(clone_from=5)

    def test_forward_after_spawn(self):
        """Forward still works after spawning a new expert."""
        moe = DynamicMoELayer(d_model=32, d_ff=128, num_experts=2, top_k=2, device=DEVICE)
        moe.spawn_expert(clone_from=0)
        x = torch.randn(2, 10, 32)
        out, _ = moe(x)
        assert out.shape == (2, 10, 32)

    def test_from_dense_ffn(self):
        """Convert dense FFN to MoE."""
        from morphllm.layers import GrowableLinear
        fc1 = GrowableLinear(32, 128, device=DEVICE)
        fc2 = GrowableLinear(128, 32, device=DEVICE)

        moe = DynamicMoELayer.from_dense_ffn(fc1, fc2, num_initial_experts=3)
        assert moe.num_experts == 3

        # First expert should have original weights
        torch.testing.assert_close(
            moe.experts[0].fc1.weight, fc1.weight
        )

        x = torch.randn(2, 10, 32)
        out, _ = moe(x)
        assert out.shape == (2, 10, 32)

    def test_backward_works(self):
        """MoE layer supports backpropagation."""
        moe = DynamicMoELayer(d_model=32, d_ff=128, num_experts=2, top_k=2, device=DEVICE)
        x = torch.randn(2, 10, 32, requires_grad=True)
        out, lb_loss = moe(x)
        loss = out.sum() + lb_loss
        loss.backward()
        assert x.grad is not None


# ------------------------------------------------------------------
# Block Conversion Tests
# ------------------------------------------------------------------

class TestConvertBlockToMoE:

    def test_convert_block(self):
        """Convert a transformer block's FFN to MoE."""
        block = DynamicTransformerBlock(
            d_model=32, n_head=4, device=DEVICE
        )
        assert not block._use_moe

        moe_layer = convert_block_to_moe(block, num_experts=2)

        assert block._use_moe
        assert hasattr(block, "moe")
        assert not hasattr(block, "fc1")
        assert not hasattr(block, "fc2")
        assert isinstance(moe_layer, DynamicMoELayer)

    def test_forward_after_conversion(self):
        """Block forward works after MoE conversion."""
        block = DynamicTransformerBlock(
            d_model=32, n_head=4, device=DEVICE
        )
        x = torch.randn(2, 10, 32)

        # Forward before conversion
        out_before = block(x).detach()

        convert_block_to_moe(block, num_experts=2)

        # Forward after conversion
        out_after = block(x)
        assert out_after.shape == (2, 10, 32)
        assert block._moe_loss.dim() == 0


# ------------------------------------------------------------------
# Full Model Integration
# ------------------------------------------------------------------

class TestMoEModelIntegration:

    def test_training_with_moe_block(self):
        """Full training step after converting a block to MoE."""
        torch.manual_seed(42)
        config = MorphConfig(
            vocab_size=100, d_model=32, n_head=4, n_layers=2,
            max_seq_len=64, dropout=0.0, device=DEVICE,
        )
        model = MorphLLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        ids = torch.randint(0, 100, (2, 10))
        targets = torch.randint(0, 100, (2, 10))

        # Train normally first
        logits, loss = model(ids, targets=targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Convert block 0 to MoE
        block = model.blocks[0]
        convert_block_to_moe(block, num_experts=2, top_k=2)

        # Re-create optimizer with new params
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Train with MoE
        logits, loss = model(ids, targets=targets)

        # Add load-balance loss from MoE blocks
        moe_loss = sum(
            b._moe_loss for b in model.blocks if hasattr(b, "_moe_loss")
        )
        total_loss = loss + 0.01 * moe_loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert logits.shape == (2, 10, 100)

    def test_spawn_expert_during_training(self):
        """Spawn expert mid-training and continue."""
        torch.manual_seed(42)
        config = MorphConfig(
            vocab_size=100, d_model=32, n_head=4, n_layers=2,
            max_seq_len=64, dropout=0.0, device=DEVICE,
        )
        model = MorphLLM(config)

        ids = torch.randint(0, 100, (2, 10))
        targets = torch.randint(0, 100, (2, 10))

        # Convert to MoE
        block = model.blocks[0]
        convert_block_to_moe(block, num_experts=2, top_k=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Train
        logits, loss = model(ids, targets=targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Spawn a 3rd expert
        block.moe.spawn_expert(clone_from=0)
        assert block.moe.num_experts == 3

        # Re-create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Continue training
        logits, loss = model(ids, targets=targets)
        loss.backward()
        optimizer.step()

        assert logits.shape == (2, 10, 100)
