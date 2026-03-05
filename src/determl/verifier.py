"""
determl.verifier — Inference Verifier

Runs inference multiple times with seed resets, hashes every output,
and compares them to prove bitwise reproducibility.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn

from determl.config import DeterministicConfig
from determl.utils import hash_tensor, get_environment_snapshot


@dataclass
class VerificationResult:
    """Result of a determinism verification run.

    Attributes:
        is_deterministic: True if all runs produced identical output.
        num_runs: How many times inference was executed.
        hashes: SHA-256 hash from each run.
        unique_hashes: Deduplicated set of hashes.
        outputs: Raw output tensors from each run (optional).
        elapsed_seconds: Total wall-clock time for all runs.
        environment: Environment snapshot at time of verification.
        seed: Seed used for verification.
    """
    is_deterministic: bool
    num_runs: int
    hashes: list[str]
    unique_hashes: set[str]
    outputs: list[torch.Tensor] = field(default_factory=list, repr=False)
    elapsed_seconds: float = 0.0
    environment: dict[str, Any] = field(default_factory=dict)
    seed: int = 42

    def __str__(self) -> str:
        if self.is_deterministic:
            status = f"DETERMINISTIC: All {self.num_runs} runs produced identical output"
            hash_line = f"SHA-256: {self.hashes[0][:16]}..."
        else:
            status = f"NON-DETERMINISTIC: {len(self.unique_hashes)} different outputs from {self.num_runs} runs"
            hash_line = "Hashes:\n" + "\n".join(
                f"  Run {i+1}: {h[:16]}..." for i, h in enumerate(self.hashes)
            )

        return (
            f"{status}\n"
            f"{hash_line}\n"
            f"Time: {self.elapsed_seconds:.3f}s | Seed: {self.seed}"
        )


class InferenceVerifier:
    """Verifies that a model produces deterministic inference output.

    Runs inference N times, resetting all seeds before each run,
    then compares SHA-256 hashes of the output tensors.

    Usage:
        verifier = InferenceVerifier(model, tokenizer)
        result = verifier.verify("What is 2+2?", num_runs=5, seed=42)
        print(result)

    For models without a tokenizer (e.g. vision models), use
    verify_with_input() and pass raw tensors directly.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any | None = None,
        device: str | torch.device = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def verify(
        self,
        prompt: str,
        num_runs: int = 5,
        seed: int = 42,
        max_new_tokens: int = 50,
        store_outputs: bool = False,
    ) -> VerificationResult:
        """Verify determinism by running text generation multiple times.

        Requires a tokenizer to be set. The model should support
        .generate() (e.g., HuggingFace CausalLM models).

        Args:
            prompt: Text prompt to generate from.
            num_runs: Number of repeated runs.
            seed: Master seed to reset before each run.
            max_new_tokens: Max tokens to generate per run.
            store_outputs: If True, store raw output tensors in result.

        Returns:
            VerificationResult with pass/fail and per-run hashes.
        """
        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer required for verify(). Use verify_with_input() "
                "for models without a tokenizer."
            )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        def run_fn() -> torch.Tensor:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding — no randomness
            )
            return output_ids

        return self._run_verification(run_fn, num_runs, seed, store_outputs)

    @torch.no_grad()
    def verify_with_input(
        self,
        input_tensor: torch.Tensor,
        num_runs: int = 5,
        seed: int = 42,
        store_outputs: bool = False,
        forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
    ) -> VerificationResult:
        """Verify determinism with a raw input tensor (no tokenizer needed).

        Args:
            input_tensor: Raw input tensor for the model.
            num_runs: Number of repeated runs.
            seed: Master seed to reset before each run.
            store_outputs: If True, store raw output tensors in result.
            forward_fn: Optional custom function (model, input) -> output.
                        Defaults to model(input_tensor).

        Returns:
            VerificationResult.
        """
        input_tensor = input_tensor.to(self.device)

        def run_fn() -> torch.Tensor:
            if forward_fn is not None:
                return forward_fn(self.model, input_tensor)
            return self.model(input_tensor)

        return self._run_verification(run_fn, num_runs, seed, store_outputs)

    def _run_verification(
        self,
        run_fn: Callable[[], torch.Tensor],
        num_runs: int,
        seed: int,
        store_outputs: bool,
    ) -> VerificationResult:
        """Core verification loop: reset seeds → run → hash → compare."""
        config = DeterministicConfig(seed=seed, warn_only=True)
        config.apply()

        hashes: list[str] = []
        outputs: list[torch.Tensor] = []
        start = time.perf_counter()

        for _ in range(num_runs):
            # Reset seeds before EVERY run
            config.reset_seeds()

            output = run_fn()

            # Handle different output types
            if isinstance(output, torch.Tensor):
                tensor_out = output
            elif hasattr(output, "logits"):
                tensor_out = output.logits
            elif isinstance(output, tuple):
                tensor_out = output[0]
            else:
                tensor_out = torch.tensor(output)

            hashes.append(hash_tensor(tensor_out))
            if store_outputs:
                outputs.append(tensor_out.cpu().clone())

        elapsed = time.perf_counter() - start
        unique = set(hashes)

        return VerificationResult(
            is_deterministic=len(unique) == 1,
            num_runs=num_runs,
            hashes=hashes,
            unique_hashes=unique,
            outputs=outputs if store_outputs else [],
            elapsed_seconds=elapsed,
            environment=get_environment_snapshot(),
            seed=seed,
        )
