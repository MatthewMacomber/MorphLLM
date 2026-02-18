"""Tests for GrowableLinear and GrowableEmbedding function-preserving properties."""

import pytest
import torch

from morphllm.layers import GrowableEmbedding, GrowableLinear

# All tests run on CPU with small tensors for speed.
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# GrowableLinear Tests
# ---------------------------------------------------------------------------


class TestGrowableLinear:
    """Tests for the GrowableLinear layer."""

    def test_forward_basic(self):
        """Basic forward pass produces correct output shape."""
        layer = GrowableLinear(16, 32, device=DEVICE)
        x = torch.randn(2, 5, 16, device=DEVICE)
        y = layer(x)
        assert y.shape == (2, 5, 32)

    def test_expand_output_noop(self):
        """expand_output with smaller/equal size returns None and does nothing."""
        layer = GrowableLinear(16, 32, device=DEVICE)
        result = layer.expand_output(32)
        assert result is None
        assert layer.out_features == 32

        result = layer.expand_output(16)
        assert result is None
        assert layer.out_features == 32

    def test_expand_output_changes_dimensions(self):
        """expand_output correctly updates out_features and weight shape."""
        layer = GrowableLinear(16, 32, device=DEVICE)
        indices = layer.expand_output(64)
        assert layer.out_features == 64
        assert layer.weight.shape == (64, 16)
        assert indices is not None
        assert indices.shape == (32,)  # 64 - 32 = 32 new neurons

    def test_expand_input_changes_dimensions(self):
        """expand_input correctly updates in_features and weight shape."""
        layer = GrowableLinear(16, 32, device=DEVICE)
        layer.expand_input(24)
        assert layer.in_features == 24
        assert layer.weight.shape == (32, 24)

    def test_expand_input_noop(self):
        """expand_input with smaller/equal size does nothing."""
        layer = GrowableLinear(16, 32, device=DEVICE)
        layer.expand_input(16)
        assert layer.in_features == 16
        layer.expand_input(8)
        assert layer.in_features == 16

    def test_function_preserving_width_expansion(self):
        """CRITICAL: Forward output unchanged after width expansion.

        This tests the core Net2WiderNet invariant:
          layer2(layer1(x))  ==  layer2'(layer1'(x))

        where layer1' and layer2' are the expanded versions.
        We use noise_std=0 to get exact preservation.
        """
        torch.manual_seed(42)
        in_dim, hidden_dim, out_dim = 8, 16, 4
        new_hidden = 32

        layer1 = GrowableLinear(in_dim, hidden_dim, device=DEVICE)
        layer2 = GrowableLinear(hidden_dim, out_dim, device=DEVICE)

        x = torch.randn(3, 5, in_dim, device=DEVICE)

        # Output before growth
        with torch.no_grad():
            y_before = layer2(layer1(x))

        # Grow: widen layer1 output, then widen layer2 input to match
        indices = layer1.expand_output(new_hidden, noise_std=0.0)
        assert indices is not None
        layer2.expand_input(new_hidden, mapping_indices=indices)

        # Output after growth
        with torch.no_grad():
            y_after = layer2(layer1(x))

        torch.testing.assert_close(y_before, y_after, atol=1e-5, rtol=1e-5)

    def test_function_preserving_with_bias(self):
        """Function preservation holds when bias is enabled."""
        torch.manual_seed(123)
        layer1 = GrowableLinear(8, 12, bias=True, device=DEVICE)
        layer2 = GrowableLinear(12, 4, bias=True, device=DEVICE)

        x = torch.randn(2, 3, 8, device=DEVICE)
        with torch.no_grad():
            y_before = layer2(layer1(x))

        indices = layer1.expand_output(20, noise_std=0.0)
        layer2.expand_input(20, mapping_indices=indices)

        with torch.no_grad():
            y_after = layer2(layer1(x))

        torch.testing.assert_close(y_before, y_after, atol=1e-5, rtol=1e-5)

    def test_function_preserving_no_bias(self):
        """Function preservation holds when bias is disabled."""
        torch.manual_seed(999)
        layer1 = GrowableLinear(8, 12, bias=False, device=DEVICE)
        layer2 = GrowableLinear(12, 4, bias=False, device=DEVICE)

        x = torch.randn(2, 3, 8, device=DEVICE)
        with torch.no_grad():
            y_before = layer2(layer1(x))

        indices = layer1.expand_output(20, noise_std=0.0)
        layer2.expand_input(20, mapping_indices=indices)

        with torch.no_grad():
            y_after = layer2(layer1(x))

        torch.testing.assert_close(y_before, y_after, atol=1e-5, rtol=1e-5)

    def test_expand_output_with_noise_breaks_symmetry(self):
        """With noise_std > 0, the new neurons should diverge slightly."""
        torch.manual_seed(42)
        layer = GrowableLinear(8, 16, device=DEVICE)
        indices = layer.expand_output(24, noise_std=0.01)
        assert indices is not None

        # Check that cloned neurons are NOT identical to parents (noise was added)
        for i, parent_idx in enumerate(indices):
            new_idx = 16 + i
            parent_w = layer.weight[parent_idx.item()]
            new_w = layer.weight[new_idx]
            # They should be close but not exactly equal
            assert not torch.equal(parent_w, new_w)
            assert torch.allclose(parent_w, new_w, atol=0.1)

    def test_bias_expanded_correctly(self):
        """Bias values are correctly copied during expand_output."""
        layer = GrowableLinear(8, 4, bias=True, device=DEVICE)
        with torch.no_grad():
            layer.bias.fill_(1.0)

        indices = layer.expand_output(8, noise_std=0.0)
        assert indices is not None
        # All new bias values should be 1.0 (copied from parents who are all 1.0)
        assert torch.allclose(layer.bias[4:], torch.ones(4, device=DEVICE))

    def test_multiple_expansions(self):
        """Layer can be expanded multiple times sequentially."""
        layer1 = GrowableLinear(8, 16, device=DEVICE)
        layer2 = GrowableLinear(16, 4, device=DEVICE)

        x = torch.randn(1, 3, 8, device=DEVICE)
        with torch.no_grad():
            y_original = layer2(layer1(x))

        # First expansion
        idx1 = layer1.expand_output(24, noise_std=0.0)
        layer2.expand_input(24, mapping_indices=idx1)

        # Second expansion
        idx2 = layer1.expand_output(32, noise_std=0.0)
        layer2.expand_input(32, mapping_indices=idx2)

        with torch.no_grad():
            y_grown = layer2(layer1(x))

        torch.testing.assert_close(y_original, y_grown, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# GrowableEmbedding Tests
# ---------------------------------------------------------------------------


class TestGrowableEmbedding:
    """Tests for the GrowableEmbedding layer."""

    def test_forward_basic(self):
        """Basic forward pass produces correct output shape."""
        emb = GrowableEmbedding(100, 32, device=DEVICE)
        ids = torch.randint(0, 100, (2, 10), device=DEVICE)
        out = emb(ids)
        assert out.shape == (2, 10, 32)

    def test_expand_vocab(self):
        """Vocabulary expansion preserves existing embeddings."""
        emb = GrowableEmbedding(100, 32, device=DEVICE)
        old_weight = emb.weight.data.clone()

        emb.expand_vocab(200)
        assert emb.num_embeddings == 200
        assert emb.weight.shape == (200, 32)

        # Old embeddings unchanged
        torch.testing.assert_close(emb.weight[:100], old_weight)

    def test_expand_vocab_noop(self):
        """Vocab expansion with smaller/equal size does nothing."""
        emb = GrowableEmbedding(100, 32, device=DEVICE)
        emb.expand_vocab(100)
        assert emb.num_embeddings == 100
        emb.expand_vocab(50)
        assert emb.num_embeddings == 100

    def test_expand_dim(self):
        """Dimension expansion preserves existing embedding values."""
        emb = GrowableEmbedding(100, 32, device=DEVICE)
        old_weight = emb.weight.data.clone()

        emb.expand_dim(64)
        assert emb.embedding_dim == 64
        assert emb.weight.shape == (100, 64)

        # Old dimensions preserved
        torch.testing.assert_close(
            emb.weight[:, :32], old_weight, atol=1e-6, rtol=1e-6
        )

    def test_expand_dim_noop(self):
        """Dim expansion with smaller/equal size does nothing."""
        emb = GrowableEmbedding(100, 32, device=DEVICE)
        emb.expand_dim(32)
        assert emb.embedding_dim == 32
        emb.expand_dim(16)
        assert emb.embedding_dim == 32

    def test_padding_idx_preserved(self):
        """Padding index stays zero after expansions."""
        emb = GrowableEmbedding(100, 32, padding_idx=0, device=DEVICE)
        assert torch.all(emb.weight[0] == 0)

        emb.expand_vocab(200)
        assert torch.all(emb.weight[0] == 0)

        emb.expand_dim(64)
        assert torch.all(emb.weight[0] == 0)

    def test_expand_dim_forward_preserves_old_dims(self):
        """Forward pass embedding[:, :old_dim] unchanged after dim expansion."""
        emb = GrowableEmbedding(100, 32, device=DEVICE)
        ids = torch.randint(0, 100, (2, 5), device=DEVICE)

        y_before = emb(ids).clone()
        emb.expand_dim(64, noise_std=0.0)
        y_after = emb(ids)

        # First 32 dims should be identical
        torch.testing.assert_close(
            y_after[:, :, :32], y_before, atol=1e-6, rtol=1e-6
        )
        # New dims should be zero (noise_std=0)
        assert torch.all(y_after[:, :, 32:] == 0)
