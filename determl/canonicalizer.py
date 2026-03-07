"""
determl.canonicalizer -- Cross-Hardware Output Normalization

The reason two identical models on A100 vs V100 give different outputs
is floating-point ordering. The canonicalizer normalizes outputs so
they produce identical hashes across different hardware.

This is what makes determl actually useful for decentralized AI --
not just "same output on same machine" but "provably same output
across different nodes."
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from determl.utils import hash_tensor, hash_string


class Precision(str, Enum):
    """Canonicalization precision levels."""
    EXACT = "exact"          # No rounding — requires identical hardware
    HIGH = "high"            # Round to 6 decimal places
    MEDIUM = "medium"        # Round to 4 decimal places
    LOW = "low"              # Round to 2 decimal places
    TOKEN_LEVEL = "token"    # For LLMs: compare token IDs, not logits


# Decimal places for each precision level
_PRECISION_DECIMALS: dict[Precision, int] = {
    Precision.EXACT: -1,   # No rounding
    Precision.HIGH: 6,
    Precision.MEDIUM: 4,
    Precision.LOW: 2,
}


@dataclass
class CanonicalResult:
    """Result of canonicalization."""
    original_hash: str
    canonical_hash: str
    precision: Precision
    tensor_shape: tuple[int, ...]
    dtype: str

    @property
    def hashes_differ(self) -> bool:
        """True if canonicalization changed the hash (rounding had effect)."""
        return self.original_hash != self.canonical_hash

    def __str__(self) -> str:
        status = "modified" if self.hashes_differ else "unchanged"
        return (
            f"Canonical ({self.precision.value}, {status})\n"
            f"  Original:    {self.original_hash[:32]}...\n"
            f"  Canonical:   {self.canonical_hash[:32]}...\n"
            f"  Shape: {self.tensor_shape}, dtype: {self.dtype}"
        )


class OutputCanonicalizer:
    """Normalizes model outputs for cross-hardware consistency.

    Different GPUs (A100, V100, RTX 4090) can produce slightly different
    floating-point results due to different FMA units and reduction orders.
    The canonicalizer rounds outputs to a specified precision so that small
    hardware-induced differences are eliminated before hashing.

    Usage:
        canonicalizer = OutputCanonicalizer(precision="high")
        canonical = canonicalizer.canonicalize(output_tensor)
        print(canonical.canonical_hash)  # Same across hardware

    For LLMs, use token-level comparison:
        canonicalizer = OutputCanonicalizer(precision="token")
        canonical = canonicalizer.canonicalize_logits(logits)
    """

    def __init__(self, precision: str | Precision = Precision.HIGH):
        if isinstance(precision, str):
            precision = Precision(precision)
        self.precision = precision

    def canonicalize(self, tensor: torch.Tensor) -> CanonicalResult:
        """Canonicalize a tensor by rounding to the configured precision.

        Args:
            tensor: Any output tensor from a model.

        Returns:
            CanonicalResult with both original and canonical hashes.
        """
        original_hash = hash_tensor(tensor)

        if self.precision == Precision.EXACT:
            canonical_tensor = tensor
        else:
            decimals = _PRECISION_DECIMALS[self.precision]
            canonical_tensor = self._round_tensor(tensor, decimals)

        canonical_hash = hash_tensor(canonical_tensor)

        return CanonicalResult(
            original_hash=original_hash,
            canonical_hash=canonical_hash,
            precision=self.precision,
            tensor_shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
        )

    def canonicalize_logits(
        self, logits: torch.Tensor, top_k: int = 10
    ) -> CanonicalResult:
        """Canonicalize LLM logits using top-k token agreement.

        Instead of comparing raw floating-point logits (which differ
        across hardware), this extracts the top-k token IDs. Two models
        agree if they predict the same top tokens in the same order.

        This is the most robust canonicalization for LLMs because token
        IDs are integers — no floating-point noise.

        Args:
            logits: Model output logits, shape (batch, seq_len, vocab_size).
            top_k: Number of top tokens to compare.

        Returns:
            CanonicalResult based on top-k token IDs.
        """
        original_hash = hash_tensor(logits)

        # Extract top-k token IDs (integers — hardware-independent)
        if logits.dim() == 3:
            # (batch, seq_len, vocab) -> take last token position
            last_logits = logits[:, -1, :]
        elif logits.dim() == 2:
            last_logits = logits
        else:
            last_logits = logits.view(-1)

        top_k_ids = torch.topk(last_logits, k=min(top_k, last_logits.shape[-1])).indices
        canonical_hash = hash_tensor(top_k_ids.to(torch.int64))

        return CanonicalResult(
            original_hash=original_hash,
            canonical_hash=canonical_hash,
            precision=Precision.TOKEN_LEVEL,
            tensor_shape=tuple(logits.shape),
            dtype=str(logits.dtype),
        )

    def canonical_hash(self, tensor: torch.Tensor) -> str:
        """Convenience: get just the canonical hash string."""
        return self.canonicalize(tensor).canonical_hash

    @staticmethod
    def compare(
        output_a: torch.Tensor,
        output_b: torch.Tensor,
        tolerance: float = 1e-6,
    ) -> ComparisonResult:
        """Compare two tensors with tolerance for floating-point differences.

        Args:
            output_a: First output tensor.
            output_b: Second output tensor.
            tolerance: Maximum allowed absolute difference.

        Returns:
            ComparisonResult with match status and statistics.
        """
        if output_a.shape != output_b.shape:
            return ComparisonResult(
                match=False,
                reason=f"Shape mismatch: {output_a.shape} vs {output_b.shape}",
                max_diff=float("inf"),
                mean_diff=float("inf"),
                tolerance=tolerance,
            )

        a = output_a.detach().cpu().float()
        b = output_b.detach().cpu().float()
        diff = torch.abs(a - b)

        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        match = max_diff <= tolerance

        return ComparisonResult(
            match=match,
            reason="" if match else f"Max diff {max_diff:.2e} exceeds tolerance {tolerance:.2e}",
            max_diff=max_diff,
            mean_diff=mean_diff,
            tolerance=tolerance,
        )

    @staticmethod
    def _round_tensor(tensor: torch.Tensor, decimals: int) -> torch.Tensor:
        """Round tensor to N decimal places."""
        t = tensor.detach().cpu().float()
        scale = 10.0 ** decimals
        return torch.round(t * scale) / scale


@dataclass
class ComparisonResult:
    """Result of comparing two model outputs."""
    match: bool
    reason: str
    max_diff: float
    mean_diff: float
    tolerance: float

    def __str__(self) -> str:
        if self.match:
            return (
                f"MATCH (tolerance={self.tolerance:.0e})\n"
                f"  Max diff: {self.max_diff:.2e}, Mean diff: {self.mean_diff:.2e}"
            )
        else:
            return (
                f"MISMATCH: {self.reason}\n"
                f"  Max diff: {self.max_diff:.2e}, Mean diff: {self.mean_diff:.2e}"
            )
