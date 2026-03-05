"""
determl.utils — Hashing & Environment Utilities

Provides cryptographic hashing for tensors and strings, plus
a function to snapshot the full compute environment for reproducibility.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from typing import Any

import numpy as np
import torch


def hash_tensor(tensor: torch.Tensor) -> str:
    """Compute SHA-256 hash of a tensor's raw bytes.

    The tensor is moved to CPU (if needed), converted to contiguous
    numpy bytes, then hashed. This produces a deterministic fingerprint
    of the exact numerical values.

    Args:
        tensor: Any PyTorch tensor.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    # Ensure contiguous CPU tensor → numpy → bytes
    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def hash_string(text: str) -> str:
    """Compute SHA-256 hash of a string.

    Args:
        text: Any string.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_environment_snapshot() -> dict[str, Any]:
    """Capture the full compute environment for reproducibility tracking.

    Returns a dict containing:
        - Python version
        - PyTorch version and build config
        - CUDA availability and version
        - cuDNN version
        - GPU names (if available)
        - OS and architecture
        - NumPy version
        - Deterministic algorithm settings

    This should be stored alongside inference results so you can verify
    that the same environment is used when comparing outputs.
    """
    snapshot: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "cudnn_version": None,
        "gpu_devices": [],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }

    if torch.cuda.is_available():
        snapshot["cuda_version"] = torch.version.cuda
        snapshot["cudnn_version"] = torch.backends.cudnn.version()
        snapshot["gpu_devices"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
                "total_memory_mb": round(
                    torch.cuda.get_device_properties(i).total_mem / (1024**2), 1
                ),
            }
            for i in range(torch.cuda.device_count())
        ]

    return snapshot
