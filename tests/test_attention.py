"""Tests for DynamicMHA attention module."""

import torch
import pytest

from morphllm.attention import DynamicMHA

DEVICE = "cpu"


class TestDynamicMHA:
    """Tests for the DynamicMHA module."""

    def test_forward_shape(self):
        """Output shape matches input shape."""
        mha = DynamicMHA(d_model=32, n_head=4, device=DEVICE)
        x = torch.randn(2, 10, 32, device=DEVICE)
        y = mha(x)
        assert y.shape == (2, 10, 32)

    def test_forward_with_causal_mask(self):
        """Forward works with causal masking."""
        mha = DynamicMHA(d_model=32, n_head=4, device=DEVICE)
        x = torch.randn(2, 10, 32, device=DEVICE)
        mask = torch.tril(torch.ones(10, 10, device=DEVICE))
        y = mha(x, mask=mask)
        assert y.shape == (2, 10, 32)

    def test_grow_embedding_updates_dimensions(self):
        """grow_embedding correctly updates d_model and head_dim."""
        mha = DynamicMHA(d_model=32, n_head=4, device=DEVICE)
        assert mha.head_dim == 8

        mha.grow_embedding(64)
        assert mha.d_model == 64
        assert mha.head_dim == 16
        assert mha.q_proj.in_features == 64
        assert mha.q_proj.out_features == 64

    def test_grow_embedding_noop(self):
        """grow_embedding with smaller/equal size does nothing."""
        mha = DynamicMHA(d_model=32, n_head=4, device=DEVICE)
        mha.grow_embedding(32)
        assert mha.d_model == 32
        mha.grow_embedding(16)
        assert mha.d_model == 32

    def test_grow_embedding_bad_divisibility(self):
        """grow_embedding rejects sizes not divisible by n_head."""
        mha = DynamicMHA(d_model=32, n_head=4, device=DEVICE)
        with pytest.raises(AssertionError):
            mha.grow_embedding(33)

    def test_forward_after_growth(self):
        """Forward pass works with correct shapes after growth."""
        mha = DynamicMHA(d_model=32, n_head=4, device=DEVICE)
        mha.grow_embedding(64)
        x = torch.randn(2, 10, 64, device=DEVICE)
        y = mha(x)
        assert y.shape == (2, 10, 64)
