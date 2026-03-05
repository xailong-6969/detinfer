"""
determl -- Deterministic ML Inference Library

Detect, prevent, and ENFORCE determinism in ML inference.

v2: Real enforcement, not just convenience wrappers.
"""

# Core (always available)
from determl.config import DeterministicConfig
from determl.detector import NonDeterminismDetector
from determl.verifier import InferenceVerifier
from determl.enforcer import DeterministicEnforcer
from determl.canonicalizer import OutputCanonicalizer
from determl.guardian import EnvironmentGuardian
from determl.engine import DeterministicEngine
from determl.utils import hash_tensor, hash_string, get_environment_snapshot

# Wrapper requires transformers -- import lazily to avoid hard dependency
try:
    from determl.wrapper import DeterministicLLM
except ImportError:
    DeterministicLLM = None

__version__ = "0.2.0"

__all__ = [
    # v2 API
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
