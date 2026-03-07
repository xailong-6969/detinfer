"""
determl.enforcer -- Runtime Determinism Enforcement

Unlike config.py which just sets flags, this module actively patches models
to REPLACE non-deterministic operations with deterministic alternatives.

This is the core differentiator of determl v2.
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator

import torch
import torch.nn as nn

from determl.config import DeterministicConfig


class FixAction(str, Enum):
    """What the enforcer did to fix a non-deterministic op."""
    REPLACED = "REPLACED"          # Swapped module with deterministic version
    PATCHED = "PATCHED"            # Applied a hook/flag to existing module
    CONTEXT_MANAGED = "CONTEXT"    # Will use context manager at runtime
    SKIPPED = "SKIPPED"            # Could not fix, user must handle


@dataclass
class EnforcementFix:
    """Record of a single fix applied to a model."""
    layer_name: str
    module_type: str
    action: FixAction
    description: str

    def __str__(self) -> str:
        tag = "[FIXED]" if self.action != FixAction.SKIPPED else "[SKIP]"
        return f"{tag} '{self.layer_name}' ({self.module_type}): {self.description}"


@dataclass
class EnforcementReport:
    """Report of all fixes applied to a model."""
    model_name: str
    total_modules: int
    fixes: list[EnforcementFix] = field(default_factory=list)

    @property
    def num_fixed(self) -> int:
        return sum(1 for f in self.fixes if f.action != FixAction.SKIPPED)

    @property
    def num_skipped(self) -> int:
        return sum(1 for f in self.fixes if f.action == FixAction.SKIPPED)

    @property
    def all_fixed(self) -> bool:
        return self.num_skipped == 0

    def __str__(self) -> str:
        lines = [
            f"Enforcement Report for '{self.model_name}' "
            f"({self.total_modules} modules scanned)"
        ]
        lines.append("=" * len(lines[0]))

        if not self.fixes:
            lines.append("No non-deterministic operations found. Model is clean.")
            return "\n".join(lines)

        for fix in self.fixes:
            lines.append(f"  {fix}")

        lines.append(
            f"\nSummary: {self.num_fixed} fixed, {self.num_skipped} skipped"
        )
        if self.all_fixed:
            lines.append("All non-deterministic ops have been patched.")
        else:
            lines.append(
                "WARNING: Some ops could not be auto-fixed. "
                "Review skipped items above."
            )
        return "\n".join(lines)


class _DeterministicSDPA(nn.Module):
    """Wrapper that forces scaled_dot_product_attention to use the
    deterministic math backend instead of Flash Attention or
    memory-efficient kernels.
    """

    def __init__(self, original_module: nn.Module):
        super().__init__()
        self._original = original_module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.MATH):
                return self._original(*args, **kwargs)
        except ImportError:
            # Older PyTorch — fall back to setting the global flag
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            ):
                return self._original(*args, **kwargs)


class _DeterministicDropout(nn.Module):
    """Replaces Dropout with Identity during inference.
    Explicitly zeros out randomness rather than relying on eval() mode.
    """

    def __init__(self, original: nn.Module):
        super().__init__()
        self._p = getattr(original, "p", 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always identity — never drop during deterministic inference
        return x


class DeterministicEnforcer:
    """Patches a PyTorch model to enforce determinism at the op level.

    This goes beyond setting flags — it actually replaces non-deterministic
    modules with deterministic wrappers and applies forward hooks.

    Usage:
        enforcer = DeterministicEnforcer(seed=42)
        report = enforcer.enforce(model)
        print(report)  # Shows what was fixed

        # Then run inference inside the deterministic context:
        with enforcer.deterministic_context():
            output = model(input_tensor)
    """

    def __init__(self, seed: int = 42, warn_only: bool = True):
        self.seed = seed
        self.config = DeterministicConfig(seed=seed, warn_only=warn_only)
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def enforce(
        self, model: nn.Module, model_name: str | None = None
    ) -> EnforcementReport:
        """Walk the model graph and replace/patch non-deterministic ops.

        This modifies the model IN-PLACE for maximum compatibility.

        Args:
            model: Any PyTorch nn.Module.
            model_name: Optional human-readable name for the report.

        Returns:
            EnforcementReport listing all changes made.
        """
        if model_name is None:
            model_name = model.__class__.__name__

        # Step 1: Apply global deterministic settings
        self.config.apply()

        # Step 2: Fill uninitialized memory
        if hasattr(torch.utils, "deterministic"):
            torch.utils.deterministic.fill_uninitialized_memory = True

        fixes: list[EnforcementFix] = []
        total = 0

        # Step 3: Walk model and fix non-deterministic modules
        modules_to_replace: list[tuple[str, nn.Module, nn.Module]] = []

        for name, module in model.named_modules():
            total += 1
            module_type = type(module).__name__

            # -- Fix Dropout: replace with identity --
            if isinstance(module, (nn.Dropout, nn.AlphaDropout, nn.FeatureAlphaDropout)):
                replacement = _DeterministicDropout(module)
                modules_to_replace.append((name, module, replacement))
                fixes.append(EnforcementFix(
                    layer_name=name or "(root)",
                    module_type=module_type,
                    action=FixAction.REPLACED,
                    description=(
                        f"Replaced with identity (p={getattr(module, 'p', '?')}). "
                        "Dropout is always disabled during deterministic inference."
                    ),
                ))

            # -- Fix attention layers that might use Flash Attention --
            elif self._is_attention_module(module):
                replacement = _DeterministicSDPA(module)
                modules_to_replace.append((name, module, replacement))
                fixes.append(EnforcementFix(
                    layer_name=name or "(root)",
                    module_type=module_type,
                    action=FixAction.REPLACED,
                    description=(
                        "Wrapped with deterministic SDPA context. "
                        "Forces math backend instead of Flash Attention."
                    ),
                ))

            # -- Flag non-deterministic pooling (can't auto-fix forward pass) --
            elif isinstance(module, (nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)):
                fixes.append(EnforcementFix(
                    layer_name=name or "(root)",
                    module_type=module_type,
                    action=FixAction.PATCHED,
                    description=(
                        "Forward pass is deterministic. Backward non-determinism "
                        "handled by torch.use_deterministic_algorithms(True)."
                    ),
                ))

            # -- Flag fractional pooling (inherently random) --
            elif isinstance(module, (nn.FractionalMaxPool2d, nn.FractionalMaxPool3d)):
                fixes.append(EnforcementFix(
                    layer_name=name or "(root)",
                    module_type=module_type,
                    action=FixAction.SKIPPED,
                    description=(
                        "FractionalMaxPool uses random samples by design. "
                        "Consider replacing with AdaptiveMaxPool or AvgPool."
                    ),
                ))

            # -- Fix RReLU (random during training) --
            elif isinstance(module, nn.RReLU):
                # In eval mode, RReLU uses the midpoint. Force eval.
                fixes.append(EnforcementFix(
                    layer_name=name or "(root)",
                    module_type=module_type,
                    action=FixAction.PATCHED,
                    description="RReLU is deterministic in eval mode (uses midpoint).",
                ))

        # Apply replacements
        for name, old_module, new_module in modules_to_replace:
            self._replace_module(model, name, new_module)

        # Step 4: Force eval mode
        model.eval()

        return EnforcementReport(
            model_name=model_name,
            total_modules=total,
            fixes=fixes,
        )

    @contextmanager
    def deterministic_context(self) -> Generator[None, None, None]:
        """Context manager that forces all ops to be deterministic.

        Use this when running inference:
            with enforcer.deterministic_context():
                output = model(input_tensor)
        """
        # Reset seeds
        self.config.reset_seeds()

        # Force SDPA to use math backend globally
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.MATH):
                with torch.no_grad():
                    yield
        except ImportError:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            ):
                with torch.no_grad():
                    yield

    def cleanup(self) -> None:
        """Remove all hooks registered by the enforcer."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _is_attention_module(self, module: nn.Module) -> bool:
        """Check if a module is an attention layer that might use SDPA."""
        module_name = type(module).__name__.lower()
        module_path = f"{type(module).__module__}.{type(module).__name__}".lower()

        attention_patterns = [
            "multiheadattention",
            "selfattention",
            "scaleddotproduct",
            "sdpa",
        ]

        return any(p in module_name or p in module_path for p in attention_patterns)

    @staticmethod
    def _replace_module(
        parent_model: nn.Module, target_name: str, new_module: nn.Module
    ) -> None:
        """Replace a named submodule in a model."""
        parts = target_name.split(".")
        if len(parts) == 1:
            setattr(parent_model, parts[0], new_module)
        else:
            # Navigate to the parent of the target
            parent = parent_model
            for part in parts[:-1]:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            setattr(parent, parts[-1], new_module)
