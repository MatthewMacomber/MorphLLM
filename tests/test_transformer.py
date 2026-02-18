"""Tests for DynamicTransformerBlock and GrowableLayerNorm."""

import torch

from morphllm.transformer import DynamicTransformerBlock, GrowableLayerNorm

DEVICE = "cpu"


class TestGrowableLayerNorm:
    """Tests for GrowableLayerNorm."""

    def test_forward_shape(self):
        ln = GrowableLayerNorm(32, device=DEVICE)
        x = torch.randn(2, 10, 32, device=DEVICE)
        y = ln(x)
        assert y.shape == (2, 10, 32)

    def test_grow_preserves_old_params(self):
        """Growth preserves learned affine parameters for old dimensions."""
        ln = GrowableLayerNorm(32, device=DEVICE)
        # Set some non-default values
        with torch.no_grad():
            ln.weight.fill_(2.0)
            ln.bias.fill_(0.5)

        ln.grow(64)
        assert ln.d_model == 64
        assert ln.weight.shape == (64,)
        # Old params preserved
        assert torch.all(ln.weight[:32] == 2.0)
        assert torch.all(ln.bias[:32] == 0.5)
        # New params: weight=0 (kills new-dim signal), bias=0
        assert torch.all(ln.weight[32:] == 0.0)
        assert torch.all(ln.bias[32:] == 0.0)


class TestDynamicTransformerBlock:
    """Tests for the DynamicTransformerBlock."""

    def test_forward_shape(self):
        block = DynamicTransformerBlock(32, n_head=4, device=DEVICE)
        x = torch.randn(2, 10, 32, device=DEVICE)
        y = block(x)
        assert y.shape == (2, 10, 32)

    def test_forward_with_mask(self):
        block = DynamicTransformerBlock(32, n_head=4, device=DEVICE)
        x = torch.randn(2, 10, 32, device=DEVICE)
        mask = torch.tril(torch.ones(10, 10, device=DEVICE))
        y = block(x, mask=mask)
        assert y.shape == (2, 10, 32)

    def test_grow_updates_all_dimensions(self):
        """Grow correctly updates d_model across all sub-modules."""
        block = DynamicTransformerBlock(32, n_head=4, device=DEVICE)
        block.grow(64)

        assert block.d_model == 64
        assert block.attn.d_model == 64
        assert block.norm1.d_model == 64
        assert block.norm2.d_model == 64
        assert block.fc1.in_features == 64
        assert block.fc1.out_features == 64 * 4  # ffn_ratio=4
        assert block.fc2.in_features == 64 * 4
        assert block.fc2.out_features == 64

    def test_grow_noop(self):
        block = DynamicTransformerBlock(32, n_head=4, device=DEVICE)
        block.grow(32)
        assert block.d_model == 32

    def test_forward_after_growth(self):
        """Forward pass works with correct shapes after growth."""
        block = DynamicTransformerBlock(32, n_head=4, device=DEVICE)
        block.grow(64)
        x = torch.randn(2, 10, 64, device=DEVICE)
        y = block(x)
        assert y.shape == (2, 10, 64)

    def test_identity_block_is_passthrough(self):
        """Identity block acts as a pure pass-through (output ≈ input)."""
        torch.manual_seed(42)
        block = DynamicTransformerBlock.create_identity_block(
            d_model=32, n_head=4, device=DEVICE
        )
        x = torch.randn(2, 10, 32, device=DEVICE)

        with torch.no_grad():
            y = block(x)

        # Output should be very close to input (identity via residual)
        # Not exact due to LayerNorm, but the residual of the zero-init
        # sublayers should make it very close.
        torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)

    def test_identity_block_zero_weights(self):
        """Identity block has zero weights in FC2 and attn.out_proj."""
        block = DynamicTransformerBlock.create_identity_block(
            d_model=32, n_head=4, device=DEVICE
        )
        assert torch.all(block.fc2.weight == 0)
        assert torch.all(block.fc2.bias == 0)
        assert torch.all(block.attn.out_proj.weight == 0)
        assert torch.all(block.attn.out_proj.bias == 0)
