"""
determl — Deterministic ML Inference Library

Detect, prevent, and verify non-determinism in ML inference.
"""

from determl.config import DeterministicConfig
from determl.detector import NonDeterminismDetector
from determl.verifier import InferenceVerifier
from determl.utils import hash_tensor, hash_string, get_environment_snapshot

# Wrapper requires transformers — import lazily to avoid hard dependency
try:
    from determl.wrapper import DeterministicLLM
except ImportError:
    DeterministicLLM = None

__version__ = "0.1.0"

__all__ = [
    "DeterministicConfig",
    "NonDeterminismDetector",
    "InferenceVerifier",
    "DeterministicLLM",
    "hash_tensor",
    "hash_string",
    "get_environment_snapshot",
]
