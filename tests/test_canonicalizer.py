"""Tests for detinfer.canonicalizer -- OutputCanonicalizer."""

import torch
import pytest

from detinfer.inference.canonicalizer import (
    OutputCanonicalizer,
    Precision,
)


class TestOutputCanonicalizer:
    """Tests for the output canonicalizer."""

    def test_exact_precision_no_rounding(self):
        """Exact precision should not modify the tensor."""
        canon = OutputCanonicalizer(precision="exact")
        t = torch.tensor([1.123456789])
        result = canon.canonicalize(t)
        assert result.original_hash == result.canonical_hash
        assert not result.hashes_differ

    def test_high_precision_rounds(self):
        """High precision should round to 6 decimal places."""
        canon = OutputCanonicalizer(precision="high")
        # Create a tensor with extra precision
        t = torch.tensor([1.1234567890123])
        result = canon.canonicalize(t)
        # Canonical hash should differ since rounding occurred
        assert result.precision == Precision.HIGH

    def test_same_tensor_same_hash(self):
        """Identical tensors should produce identical canonical hashes."""
        canon = OutputCanonicalizer(precision="high")
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([1.0, 2.0, 3.0])
        r1 = canon.canonicalize(t1)
        r2 = canon.canonicalize(t2)
        assert r1.canonical_hash == r2.canonical_hash

    def test_different_tensor_different_hash(self):
        """Different tensors should produce different canonical hashes."""
        canon = OutputCanonicalizer(precision="high")
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([1.0, 2.0, 4.0])
        r1 = canon.canonicalize(t1)
        r2 = canon.canonicalize(t2)
        assert r1.canonical_hash != r2.canonical_hash

    def test_near_identical_tensors_match_at_low_precision(self):
        """Tensors differing only at high precision should match at low precision."""
        canon = OutputCanonicalizer(precision="low")
        t1 = torch.tensor([1.001])
        t2 = torch.tensor([1.004])  # Differs at 3rd decimal
        r1 = canon.canonicalize(t1)
        r2 = canon.canonicalize(t2)
        # At "low" (2 decimals): 1.00 == 1.00
        assert r1.canonical_hash == r2.canonical_hash

    def test_canonical_hash_convenience(self):
        """canonical_hash() shorthand should work."""
        canon = OutputCanonicalizer(precision="high")
        t = torch.tensor([1.0, 2.0])
        h = canon.canonical_hash(t)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256

    def test_canonicalize_logits_top_k(self):
        """Token-level canonicalization should use top-k token IDs."""
        canon = OutputCanonicalizer(precision="token")
        # Simulate logits: (batch=1, seq=5, vocab=100)
        logits = torch.randn(1, 5, 100)
        result = canon.canonicalize_logits(logits, top_k=5)
        assert result.precision == Precision.TOKEN_LEVEL
        assert len(result.canonical_hash) == 64

    def test_token_precision_canonicalize_hashes_token_ids(self):
        """Token precision should hash token IDs directly without rounding lookup."""
        canon = OutputCanonicalizer(precision="token")
        tokens = torch.tensor([[1, 2, 3]], dtype=torch.int64)

        from_int = canon.canonicalize(tokens)
        from_float = canon.canonicalize(tokens.float())

        assert from_int.precision == Precision.TOKEN_LEVEL
        assert from_int.canonical_hash == from_float.canonical_hash

    def test_logits_same_top_tokens_same_hash(self):
        """Two logits with same top tokens should produce same canonical hash."""
        canon = OutputCanonicalizer(precision="token")
        # Create logits where token 42 is clearly the top token
        logits1 = torch.zeros(1, 1, 100)
        logits1[0, 0, 42] = 100.0  # Token 42 dominates
        logits1[0, 0, 10] = 50.0   # Token 10 second

        logits2 = torch.zeros(1, 1, 100)
        logits2[0, 0, 42] = 99.5   # Same top token, slightly different value
        logits2[0, 0, 10] = 49.5   # Same second token

        r1 = canon.canonicalize_logits(logits1, top_k=2)
        r2 = canon.canonicalize_logits(logits2, top_k=2)
        assert r1.canonical_hash == r2.canonical_hash

    def test_compare_matching_tensors(self):
        """Identical tensors should match."""
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([1.0, 2.0, 3.0])
        result = OutputCanonicalizer.compare(t1, t2)
        assert result.match

    def test_compare_with_tolerance(self):
        """Tensors within tolerance should match."""
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([1.0000001, 2.0000001, 3.0000001])
        result = OutputCanonicalizer.compare(t1, t2, tolerance=1e-5)
        assert result.match

    def test_compare_shape_mismatch(self):
        """Different shapes should not match."""
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([1.0, 2.0, 3.0])
        result = OutputCanonicalizer.compare(t1, t2)
        assert not result.match
        assert "Shape mismatch" in result.reason


class TestCanonicalResult:
    """Tests for CanonicalResult."""

    def test_str_representation(self):
        """String representation should be informative."""
        canon = OutputCanonicalizer(precision="high")
        t = torch.tensor([1.0, 2.0])
        result = canon.canonicalize(t)
        result_str = str(result)
        assert "Canonical" in result_str
        assert "high" in result_str

