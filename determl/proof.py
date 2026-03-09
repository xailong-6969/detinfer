"""
determl.proof -- Cross-GPU Verification Proofs

Export deterministic inference proofs from one machine and verify
them on another. This is the core mechanism for proving that determl
produces identical results across different hardware.

Flow:
  1. Machine A: determl export proof.json
     → Runs inference, saves {model, seed, prompt, canonical_hash, env}
  2. Machine B: determl cross-verify proof.json
     → Loads proof, runs same inference, compares canonical hashes
     → MATCH = determinism proven across GPUs
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class InferenceProof:
    """A proof of deterministic inference that can be exported and verified."""

    # What was computed
    model_name: str
    seed: int
    prompt: str
    max_new_tokens: int
    precision: str

    # The proof itself
    canonical_hash: str
    raw_hash: str
    text_output: str

    # Environment snapshot
    gpu_name: str
    cuda_version: Optional[str]
    torch_version: str
    python_version: str
    platform: str

    # Metadata
    timestamp: str = ""
    determl_version: str = "2.0.0"
    
    # Token-level details (Gold standard schema)
    input_tokens_hash: str = ""
    output_tokens_hash: str = ""
    
    # Backend details
    model_dtype: str = "unknown"
    quantization: str = "none"
    transformers_version: str = "unknown"

    def save(self, path: str | Path) -> None:
        """Export proof to a JSON file."""
        path = Path(path)
        data = asdict(self)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "InferenceProof":
        """Load proof from a JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  INFERENCE PROOF",
            "=" * 60,
            f"  Model:          {self.model_name}",
            f"  Seed:           {self.seed}",
            f"  Prompt:         {self.prompt[:60]}{'...' if len(self.prompt) > 60 else ''}",
            f"  Max tokens:     {self.max_new_tokens}",
            f"  Precision:      {self.precision}",
            f"  Dtype:          {self.model_dtype} | Quant: {self.quantization}",
            "-" * 60,
            f"  Canonical hash: {self.canonical_hash}",
            f"  Input tokens:   {self.input_tokens_hash}",
            f"  Output tokens:  {self.output_tokens_hash}",
            f"  Raw hash:       {self.raw_hash}",
            "-" * 60,
            f"  GPU:            {self.gpu_name}",
            f"  CUDA:           {self.cuda_version or 'N/A'}",
            f"  PyTorch:        {self.torch_version} | HF: {self.transformers_version}",
            f"  Timestamp:      {self.timestamp}",
            "=" * 60,
        ]
        return "\n".join(lines)


@dataclass
class CrossVerifyResult:
    """Result of cross-GPU verification."""

    # The original proof
    original_proof: InferenceProof

    # Local results
    local_canonical_hash: str
    local_raw_hash: str
    local_gpu_name: str
    local_text_output: str

    # Comparison
    canonical_match: bool
    raw_match: bool
    text_match: bool
    input_tokens_match: bool
    output_tokens_match: bool

    elapsed_seconds: float = 0.0

    @property
    def verified(self) -> bool:
        """Cross-GPU verification passes if canonical hashes match."""
        return self.canonical_match

    def __str__(self) -> str:
        lines = [
            "",
            "=" * 65,
            "  CROSS-GPU VERIFICATION RESULT",
            "=" * 65,
            "",
            "  Original (remote):",
            f"    GPU:            {self.original_proof.gpu_name}",
            f"    Canonical hash: {self.original_proof.canonical_hash}",
            f"    Raw hash:       {self.original_proof.raw_hash}",
            "",
            "  Local (this machine):",
            f"    GPU:            {self.local_gpu_name}",
            f"    Canonical hash: {self.local_canonical_hash}",
            f"    Raw hash:       {self.local_raw_hash}",
            "",
            "-" * 65,
            f"  Canonical hash match: {'✓ YES' if self.canonical_match else '✗ NO'}",
            f"  Input tokens match:   {'✓ YES' if self.input_tokens_match else '✗ NO (Tokenizer drift!)'}",
            f"  Output tokens match:  {'✓ YES' if self.output_tokens_match else '✗ NO'}",
            f"  Raw hash match:       {'✓ YES' if self.raw_match else '✗ NO'}",
            f"  Text output match:    {'✓ YES' if self.text_match else '✗ NO'}",
            "-" * 65,
        ]

        if self.verified:
            lines.extend([
                "",
                "  ✓ VERIFIED — Deterministic execution confirmed across GPUs!",
                f"    {self.original_proof.gpu_name} and {self.local_gpu_name}",
                f"    produced identical canonical hashes.",
                "",
                "    This proves both machines computed the same result.",
            ])
        else:
            lines.extend([
                "",
                "  ✗ MISMATCH — Canonical hashes differ across GPUs.",
                "",
            ])
            if self.text_match and not self.canonical_match:
                lines.append(
                    "    Note: Text output matched but tensor-level hashes differ."
                )
                lines.append(
                    "    This is expected — the canonicalizer may need tuning"
                )
                lines.append(
                    "    for this GPU combination."
                )
            elif not self.text_match:
                lines.append(
                    "    The models produced different text output."
                )
                lines.append(
                    "    Check that both machines use the same model and seed."
                )

        lines.extend([
            "",
            f"  Time: {self.elapsed_seconds:.1f}s",
            "=" * 65,
        ])
        return "\n".join(lines)


def create_proof(engine, prompt: str, max_new_tokens: int = 256) -> InferenceProof:
    """
    Run deterministic inference and create an exportable proof.

    Args:
        engine: A loaded DeterministicEngine
        prompt: The input prompt
        max_new_tokens: Maximum tokens to generate

    Returns:
        InferenceProof ready to export
    """
    import platform
    import sys
    from datetime import datetime, timezone

    import torch

    # Run inference
    result = engine.run(prompt, max_new_tokens=max_new_tokens)

    # Get environment info
    gpu_name = "CPU"
    cuda_version = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda

    # Extract precision as string (engine stores it as Precision enum)
    precision_val = getattr(engine, "precision", "high")
    if hasattr(precision_val, "value"):
        precision_val = precision_val.value

    try:
        import transformers
        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = "unknown"

    return InferenceProof(
        model_name=getattr(engine, "model_name", "unknown"),
        seed=getattr(engine, "seed", 42),
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        precision=precision_val,
        input_tokens_hash=getattr(result, 'input_tokens_hash', ''),
        output_tokens_hash=getattr(result, 'output_tokens_hash', ''),
        model_dtype=getattr(result, 'model_dtype', 'unknown'),
        quantization=getattr(result, 'quantization', 'none'),
        transformers_version=transformers_version,
        canonical_hash=result.canonical_hash,
        raw_hash=result.raw_hash,
        text_output=result.text or "",
        gpu_name=gpu_name,
        cuda_version=cuda_version,
        torch_version=torch.__version__,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def cross_verify(proof: InferenceProof) -> CrossVerifyResult:
    """
    Verify a proof by re-running the same inference locally.

    Args:
        proof: An InferenceProof loaded from file

    Returns:
        CrossVerifyResult with comparison details
    """
    import torch
    from determl.engine import DeterministicEngine

    start = time.time()

    # Recreate the exact same engine configuration
    engine = DeterministicEngine(
        seed=proof.seed,
        precision=proof.precision,
    )
    engine.load(proof.model_name)

    # Run the same inference
    result = engine.run(proof.prompt, max_new_tokens=proof.max_new_tokens)

    # Get local GPU info
    local_gpu = "CPU"
    if torch.cuda.is_available():
        local_gpu = torch.cuda.get_device_name(0)

    elapsed = time.time() - start

    return CrossVerifyResult(
        original_proof=proof,
        local_canonical_hash=result.canonical_hash,
        local_raw_hash=result.raw_hash,
        local_gpu_name=local_gpu,
        local_text_output=result.text or "",
        canonical_match=(result.canonical_hash == proof.canonical_hash),
        raw_match=(result.raw_hash == proof.raw_hash),
        text_match=((result.text or "").strip() == proof.text_output.strip()),
        input_tokens_match=(getattr(result, 'input_tokens_hash', '') == getattr(proof, 'input_tokens_hash', '')),
        output_tokens_match=(getattr(result, 'output_tokens_hash', '') == getattr(proof, 'output_tokens_hash', '')),
        elapsed_seconds=elapsed,
    )
