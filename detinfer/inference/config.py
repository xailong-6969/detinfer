"""
detinfer.config — Deterministic Configuration

Locks down ALL sources of randomness in one call: PyTorch, NumPy,
Python's random module, CUDA seeds, cuDNN flags, and cuBLAS workspace.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class DeterministicConfig:
    """Configuration that locks down all sources of randomness.

    Usage:
        config = DeterministicConfig(seed=42)
        config.apply()   # Everything is now deterministic
        config.snapshot() # Returns dict of current state

    Args:
        seed: Master random seed applied to all RNGs.
        warn_only: If True, enable deterministic algorithms with warn-only
                   behavior for unsupported ops (logs warnings instead of
                   raising errors). Useful when some layers don't have
                   deterministic GPU implementations.
    """

    seed: int = 42
    warn_only: bool = False
    _applied: bool = field(default=False, repr=False, init=False)
    _cublas_was_preset: bool = field(default=False, repr=False, init=False)

    def apply(self) -> DeterministicConfig:
        """Lock every source of randomness. Returns self for chaining.

        Sets:
            - torch.manual_seed
            - torch.cuda.manual_seed_all
            - random.seed
            - numpy.random.seed
            - PYTHONHASHSEED env var
            - torch.use_deterministic_algorithms(True)
            - torch.backends.cudnn.deterministic = True
            - torch.backends.cudnn.benchmark = False
            - CUBLAS_WORKSPACE_CONFIG env var (required for CUDA determinism)
        """
        # Step 1: cuBLAS workspace — set BEFORE any CUDA interaction
        self._cublas_was_preset = bool(
            os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        )
        if not self._cublas_was_preset and torch.cuda.is_available():
            import warnings
            warnings.warn(
                "CUBLAS_WORKSPACE_CONFIG was not set before process start. "
                "detinfer set it at runtime (best-effort). For strongest "
                "guarantees, set CUBLAS_WORKSPACE_CONFIG=:4096:8 in your "
                "environment before launching Python.",
                stacklevel=2,
            )
        # Always set/enforce the correct value
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # Step 2: Python built-in
        random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

        # Step 3: NumPy
        np.random.seed(self.seed)

        # Step 4: PyTorch CPU + GPU seeds
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Step 5: PyTorch deterministic mode
        torch.use_deterministic_algorithms(True, warn_only=self.warn_only)

        # Step 6: cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self._applied = True
        return self

    def reset_seeds(self) -> None:
        """Re-apply just the seeds (without touching flags).

        Useful when you want to re-run inference with the exact same
        initial state without calling the full apply() again.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def snapshot(self) -> dict[str, Any]:
        """Return a dict capturing the current determinism state.

        This is useful for logging — you can record this alongside
        inference results to prove environment settings were applied.
        """
        return {
            "seed": self.seed,
            "warn_only": self.warn_only,
            "applied": self._applied,
            "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
            "cublas_was_preset": self._cublas_was_preset,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", ""),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

    def __repr__(self) -> str:
        status = "APPLIED" if self._applied else "NOT APPLIED"
        return f"DeterministicConfig(seed={self.seed}, status={status})"

