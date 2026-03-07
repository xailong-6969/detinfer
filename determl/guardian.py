"""
determl.guardian -- Environment Enforcement

Instead of just capturing a snapshot (what v1 does), the guardian
ENFORCES that environments match before comparing results.

For decentralized AI: if two nodes have incompatible environments,
the guardian refuses to compare their outputs (because the comparison
would be meaningless).
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

from determl.utils import get_environment_snapshot, hash_string


class CompatibilityLevel(str, Enum):
    """How compatible two environments are."""
    STRICT = "STRICT"           # Identical hardware + software -> bit-exact
    COMPATIBLE = "COMPATIBLE"   # Same GPU family + PyTorch major -> canonical match
    INCOMPATIBLE = "INCOMPATIBLE"  # Different GPU family -> unreliable


@dataclass
class EnvironmentFingerprint:
    """Compact fingerprint of the compute environment.

    This is what gets shared between nodes for comparison.
    """
    torch_version: str
    torch_major: str
    cuda_version: str | None
    cudnn_version: int | None
    gpu_name: str | None
    gpu_family: str | None  # e.g., "Ampere", "Hopper", "Ada"
    os_platform: str
    python_version: str
    numpy_version: str
    deterministic_algorithms: bool
    fingerprint_hash: str = ""

    def __post_init__(self) -> None:
        # Auto-compute a hash of this fingerprint
        content = (
            f"{self.torch_version}|{self.cuda_version}|{self.cudnn_version}|"
            f"{self.gpu_name}|{self.os_platform}|{self.deterministic_algorithms}"
        )
        self.fingerprint_hash = hash_string(content)[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "torch_version": self.torch_version,
            "torch_major": self.torch_major,
            "cuda_version": self.cuda_version,
            "cudnn_version": self.cudnn_version,
            "gpu_name": self.gpu_name,
            "gpu_family": self.gpu_family,
            "os_platform": self.os_platform,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "deterministic_algorithms": self.deterministic_algorithms,
            "fingerprint_hash": self.fingerprint_hash,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentFingerprint:
        data.pop("fingerprint_hash", None)
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> EnvironmentFingerprint:
        return cls.from_dict(json.loads(json_str))

    def __str__(self) -> str:
        gpu = self.gpu_name or "CPU only"
        cuda = self.cuda_version or "N/A"
        return (
            f"Environment [{self.fingerprint_hash}]\n"
            f"  PyTorch:  {self.torch_version}\n"
            f"  CUDA:     {cuda}\n"
            f"  GPU:      {gpu} ({self.gpu_family or 'unknown'})\n"
            f"  Python:   {self.python_version}\n"
            f"  Platform: {self.os_platform}\n"
            f"  Deterministic: {self.deterministic_algorithms}"
        )


@dataclass
class ComparisonResult:
    """Result of comparing two environment fingerprints."""
    level: CompatibilityLevel
    local: EnvironmentFingerprint
    remote: EnvironmentFingerprint
    warnings: list[str] = field(default_factory=list)
    details: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f"Compatibility: {self.level.value}"]
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  - {w}")
        if self.details:
            lines.append("Differences:")
            for key, (local_val, remote_val) in self.details.items():
                lines.append(f"  {key}: {local_val} vs {remote_val}")
        return "\n".join(lines)


# GPU architecture family detection
_GPU_FAMILIES: dict[str, str] = {
    "V100": "Volta",
    "T4": "Turing",
    "A100": "Ampere",
    "A10": "Ampere",
    "A30": "Ampere",
    "A40": "Ampere",
    "A6000": "Ampere",
    "L4": "Ada",
    "L40": "Ada",
    "RTX 4090": "Ada",
    "RTX 4080": "Ada",
    "RTX 4070": "Ada",
    "RTX 3090": "Ampere",
    "RTX 3080": "Ampere",
    "H100": "Hopper",
    "H200": "Hopper",
    "B100": "Blackwell",
    "B200": "Blackwell",
}


def _detect_gpu_family(gpu_name: str | None) -> str | None:
    """Detect GPU architecture family from device name."""
    if gpu_name is None:
        return None
    for pattern, family in _GPU_FAMILIES.items():
        if pattern in gpu_name:
            return family
    return "unknown"


class EnvironmentGuardian:
    """Enforces environment consistency for deterministic verification.

    Usage:
        guardian = EnvironmentGuardian()

        # Create a fingerprint of this machine
        local = guardian.create_fingerprint()

        # Compare with a remote node
        result = guardian.compare(local, remote_fingerprint)
        print(result)  # STRICT / COMPATIBLE / INCOMPATIBLE

        # Enforce — raise if incompatible
        guardian.enforce(remote_fingerprint)
    """

    def create_fingerprint(self) -> EnvironmentFingerprint:
        """Capture this machine's environment fingerprint."""
        snapshot = get_environment_snapshot()

        # Extract GPU info
        gpu_name = None
        gpu_devices = snapshot.get("gpu_devices", [])
        if gpu_devices:
            gpu_name = gpu_devices[0].get("name", None)

        torch_version = torch.__version__
        torch_major = torch_version.split(".")[0] + "." + torch_version.split(".")[1]

        return EnvironmentFingerprint(
            torch_version=torch_version,
            torch_major=torch_major,
            cuda_version=snapshot.get("cuda_version"),
            cudnn_version=snapshot.get("cudnn_version"),
            gpu_name=gpu_name,
            gpu_family=_detect_gpu_family(gpu_name),
            os_platform=platform.platform(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            numpy_version=np.__version__,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        )

    def compare(
        self,
        local: EnvironmentFingerprint,
        remote: EnvironmentFingerprint,
    ) -> ComparisonResult:
        """Compare two environment fingerprints.

        Returns:
            ComparisonResult with compatibility level.
        """
        warnings: list[str] = []
        details: dict[str, tuple[Any, Any]] = {}

        # Check deterministic mode
        if not local.deterministic_algorithms:
            warnings.append("Local environment has deterministic algorithms DISABLED")
        if not remote.deterministic_algorithms:
            warnings.append("Remote environment has deterministic algorithms DISABLED")

        # Check PyTorch version
        if local.torch_version != remote.torch_version:
            details["torch_version"] = (local.torch_version, remote.torch_version)

        # Check CUDA
        if local.cuda_version != remote.cuda_version:
            details["cuda_version"] = (local.cuda_version, remote.cuda_version)

        # Check GPU
        if local.gpu_name != remote.gpu_name:
            details["gpu_name"] = (local.gpu_name, remote.gpu_name)

        # Determine compatibility level
        level = self._determine_level(local, remote)

        return ComparisonResult(
            level=level,
            local=local,
            remote=remote,
            warnings=warnings,
            details=details,
        )

    def enforce(
        self,
        required: EnvironmentFingerprint,
        min_level: CompatibilityLevel = CompatibilityLevel.COMPATIBLE,
    ) -> ComparisonResult:
        """Enforce that this machine is compatible with a required environment.

        Args:
            required: The environment fingerprint to match against.
            min_level: Minimum compatibility level required.

        Raises:
            EnvironmentMismatchError: If compatibility is below min_level.
        """
        local = self.create_fingerprint()
        result = self.compare(local, required)

        level_order = [
            CompatibilityLevel.INCOMPATIBLE,
            CompatibilityLevel.COMPATIBLE,
            CompatibilityLevel.STRICT,
        ]

        if level_order.index(result.level) < level_order.index(min_level):
            raise EnvironmentMismatchError(
                f"Environment incompatible. Required: {min_level.value}, "
                f"Got: {result.level.value}.\n{result}"
            )

        return result

    def _determine_level(
        self,
        local: EnvironmentFingerprint,
        remote: EnvironmentFingerprint,
    ) -> CompatibilityLevel:
        """Determine compatibility level between two environments."""
        # STRICT: Everything matches
        if (
            local.torch_version == remote.torch_version
            and local.cuda_version == remote.cuda_version
            and local.cudnn_version == remote.cudnn_version
            and local.gpu_name == remote.gpu_name
        ):
            return CompatibilityLevel.STRICT

        # COMPATIBLE: Same GPU family and PyTorch major version
        if (
            local.torch_major == remote.torch_major
            and local.gpu_family == remote.gpu_family
            and local.gpu_family is not None
        ):
            return CompatibilityLevel.COMPATIBLE

        # Both CPU-only is COMPATIBLE
        if local.gpu_name is None and remote.gpu_name is None:
            if local.torch_major == remote.torch_major:
                return CompatibilityLevel.COMPATIBLE

        return CompatibilityLevel.INCOMPATIBLE


class EnvironmentMismatchError(Exception):
    """Raised when environments are too different for reliable comparison."""
    pass
