"""
determl.detector — Non-Determinism Detector

Scans a PyTorch model and reports which layers/operations are known
to be non-deterministic on GPU. This helps you catch problems BEFORE
running inference, rather than discovering mismatches later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn


class Severity(str, Enum):
    """How serious is the non-determinism risk."""
    WARNING = "WARNING"   # Known non-deterministic forward pass
    INFO = "INFO"         # Non-deterministic backward only, or conditional


# --------------------------------------------------------------------------
# Known non-deterministic PyTorch modules (as of PyTorch 2.x)
# Reference: https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
# --------------------------------------------------------------------------
_NONDETERMINISTIC_MODULES: dict[type, tuple[Severity, str]] = {
    nn.Dropout: (
        Severity.INFO,
        "Dropout is random during training but is a no-op during eval. "
        "Make sure model.eval() is called before inference.",
    ),
    nn.AlphaDropout: (
        Severity.INFO,
        "AlphaDropout is random during training but is a no-op during eval. "
        "Make sure model.eval() is called before inference.",
    ),
    nn.AdaptiveAvgPool2d: (
        Severity.WARNING,
        "AdaptiveAvgPool2d has non-deterministic backward on CUDA. "
        "Forward pass is deterministic.",
    ),
    nn.AdaptiveAvgPool3d: (
        Severity.WARNING,
        "AdaptiveAvgPool3d has non-deterministic backward on CUDA.",
    ),
    nn.AdaptiveMaxPool2d: (
        Severity.WARNING,
        "AdaptiveMaxPool2d has non-deterministic backward on CUDA.",
    ),
    nn.MaxPool3d: (
        Severity.WARNING,
        "MaxPool3d has non-deterministic backward on CUDA.",
    ),
    nn.FractionalMaxPool2d: (
        Severity.WARNING,
        "FractionalMaxPool2d uses random samples and is non-deterministic.",
    ),
    nn.FractionalMaxPool3d: (
        Severity.WARNING,
        "FractionalMaxPool3d uses random samples and is non-deterministic.",
    ),
    nn.RReLU: (
        Severity.INFO,
        "RReLU uses randomness during training. Deterministic in eval mode.",
    ),
    nn.Embedding: (
        Severity.INFO,
        "Embedding with padding_idx has non-deterministic backward on CUDA. "
        "Forward pass is deterministic.",
    ),
}

# String-based detection for ops that appear as substrings in module names
# (catches custom wrappers and transformer attention variants)
_NONDETERMINISTIC_PATTERNS: dict[str, tuple[Severity, str]] = {
    "scaled_dot_product_attention": (
        Severity.WARNING,
        "ScaledDotProductAttention may use non-deterministic Flash Attention "
        "or memory-efficient kernels on CUDA. Consider using "
        "torch.nn.attention.sdpa_kernel() to select a deterministic backend.",
    ),
}


@dataclass
class Finding:
    """A single non-determinism finding in a model scan."""
    layer_name: str
    module_type: str
    severity: Severity
    description: str

    def __str__(self) -> str:
        icon = "[!] " if self.severity == Severity.WARNING else "[i] "
        return f"{icon}[{self.severity.value}] '{self.layer_name}' ({self.module_type}): {self.description}"


@dataclass
class ScanReport:
    """Complete report from scanning a model for non-deterministic ops."""
    model_name: str
    total_modules: int
    findings: list[Finding] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """True if no WARNING-level findings were found."""
        return not any(f.severity == Severity.WARNING for f in self.findings)

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.WARNING]

    @property
    def infos(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.INFO]

    def __str__(self) -> str:
        lines = [f"Scan Report for '{self.model_name}' ({self.total_modules} modules scanned)"]
        lines.append("=" * len(lines[0]))

        if not self.findings:
            lines.append("No non-deterministic operations found.")
            return "\n".join(lines)

        if self.warnings:
            lines.append(f"\nFound {len(self.warnings)} WARNING-level findings:")
            for f in self.warnings:
                lines.append(f"  {f}")

        if self.infos:
            lines.append(f"\nFound {len(self.infos)} INFO-level findings:")
            for f in self.infos:
                lines.append(f"  {f}")

        if self.is_clean:
            lines.append("\nAll findings are INFO-level -- model should be deterministic in eval mode.")
        else:
            lines.append("\nWARNING-level findings detected -- outputs may vary on GPU.")

        return "\n".join(lines)


class NonDeterminismDetector:
    """Scans PyTorch models for potentially non-deterministic operations.

    Usage:
        detector = NonDeterminismDetector()
        report = detector.scan(model)
        print(report)
        if not report.is_clean:
            print("Model has non-deterministic ops!")
    """

    def scan(self, model: nn.Module, model_name: str | None = None) -> ScanReport:
        """Scan a model and return a report of non-deterministic operations.

        Args:
            model: Any PyTorch nn.Module.
            model_name: Optional human-readable name for the report.

        Returns:
            ScanReport with all findings.
        """
        if model_name is None:
            model_name = model.__class__.__name__

        findings: list[Finding] = []
        total = 0

        for name, module in model.named_modules():
            total += 1
            module_type = type(module)
            module_type_name = module_type.__name__

            # Check against known non-deterministic module types
            if module_type in _NONDETERMINISTIC_MODULES:
                severity, desc = _NONDETERMINISTIC_MODULES[module_type]
                findings.append(Finding(
                    layer_name=name or "(root)",
                    module_type=module_type_name,
                    severity=severity,
                    description=desc,
                ))

            # Check string patterns in the module's full class path
            full_type_str = f"{module_type.__module__}.{module_type_name}".lower()
            for pattern, (severity, desc) in _NONDETERMINISTIC_PATTERNS.items():
                if pattern in full_type_str or pattern in name.lower():
                    findings.append(Finding(
                        layer_name=name or "(root)",
                        module_type=module_type_name,
                        severity=severity,
                        description=desc,
                    ))

        return ScanReport(
            model_name=model_name,
            total_modules=total,
            findings=findings,
        )
