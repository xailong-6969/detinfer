"""
detinfer -- Deterministic ML Library

Detect, prevent, and ENFORCE determinism in ML inference and training.

v3: Deterministic agent with token tracing, replay, and diff.

Quick start:
    import detinfer
    detinfer.enforce()  # One line. Everything is now deterministic.

Structure:
    detinfer.inference  -- Core deterministic inference library
    detinfer.agent      -- Deterministic agent with tracing
"""

from __future__ import annotations

import hashlib
from typing import Optional

import torch

# Core inference classes (re-exported from inference subpackage)
from detinfer.inference.config import DeterministicConfig
from detinfer.inference.detector import NonDeterminismDetector
from detinfer.inference.verifier import InferenceVerifier
from detinfer.inference.enforcer import DeterministicEnforcer
from detinfer.inference.canonicalizer import OutputCanonicalizer
from detinfer.inference.guardian import EnvironmentGuardian
from detinfer.inference.engine import DeterministicEngine
from detinfer.inference.utils import hash_tensor, hash_string, get_environment_snapshot

# Wrapper requires transformers -- import lazily to avoid hard dependency
try:
    from detinfer.inference.wrapper import DeterministicLLM
except ImportError:
    DeterministicLLM = None

# Agent requires transformers -- import lazily
try:
    from detinfer.agent.runtime import DeterministicAgent
except ImportError:
    DeterministicAgent = None

__version__ = "0.2.3"

# ---------------------------------------------------------------------------
# Module-level state for the enforce() API
# ---------------------------------------------------------------------------
_config: Optional[DeterministicConfig] = None
_enforced: bool = False


def enforce(seed: int = 42, warn_only: bool = True) -> DeterministicConfig:
    """One-line determinism enforcement for any ML code.

    Call this ONCE at the start of your script. It locks:
      - Python random, NumPy, PyTorch, CUDA seeds
      - cuDNN deterministic mode
      - cuBLAS workspace config
      - torch.use_deterministic_algorithms(True)

    Works for both inference AND training.

    Usage:
        import detinfer
        detinfer.enforce(seed=42)

        # Now ANY PyTorch code is deterministic:
        output = model(input)           # inference
        loss.backward()                 # training
        optimizer.step()                # weight updates

    Args:
        seed: Master random seed (default: 42)
        warn_only: If True, warn instead of error for ops without
                   deterministic GPU implementations (default: True)

    Returns:
        DeterministicConfig instance for inspection
    """
    global _config, _enforced
    _config = DeterministicConfig(seed=seed, warn_only=warn_only)
    _config.apply()
    _enforced = True
    return _config


def status() -> dict:
    """Check current determinism enforcement status.

    Returns:
        Dict with enforcement state, seed, and flag values
    """
    if _config is None or not _enforced:
        return {"enforced": False, "message": "Call detinfer.enforce() first"}
    snapshot = _config.snapshot()
    snapshot["enforced"] = True
    return snapshot


def checkpoint_hash(model: torch.nn.Module) -> str:
    """Compute a canonical SHA-256 hash of model weights.

    Use this during training to verify that two machines
    produce identical weights at the same training step.

    Usage:
        detinfer.enforce(seed=42)
        for step, batch in enumerate(dataloader):
            loss = model(batch).loss
            loss.backward()
            optimizer.step()

            # Verify weights match across machines:
            h = detinfer.checkpoint_hash(model)
            print(f"Step {step}: {h}")

    Args:
        model: Any PyTorch nn.Module

    Returns:
        SHA-256 hex digest of all model parameters
    """
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        # Sort by name to ensure consistent ordering
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


__all__ = [
    # v2 top-level API
    "enforce",
    "status",
    "checkpoint_hash",
    # v3 agent
    "DeterministicAgent",
    # v2 classes
    "DeterministicEngine",
    "DeterministicEnforcer",
    "OutputCanonicalizer",
    "EnvironmentGuardian",
    # v1 API (still available)
    "DeterministicConfig",
    "NonDeterminismDetector",
    "InferenceVerifier",
    "DeterministicLLM",
    # Utilities
    "hash_tensor",
    "hash_string",
    "get_environment_snapshot",
]
