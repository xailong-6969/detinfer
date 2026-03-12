"""
detinfer.engine -- DeterministicEngine (High-Level API)

This is the main user-facing class in detinfer v2. It combines:
- DeterministicConfig (seed locking)
- DeterministicEnforcer (op interception)
- OutputCanonicalizer (cross-hardware normalization)
- EnvironmentGuardian (environment enforcement)
- InferenceVerifier (hash verification)

Usage:
    from detinfer import DeterministicEngine

    engine = DeterministicEngine(seed=42)
    engine.load("<hf-model>")

    result = engine.run("Write hello world in Python")
    print(result.text)
    print(result.canonical_hash)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

# Suppress HuggingFace warnings about setting top_k/temperature when do_sample=False
warnings.filterwarnings("ignore", message=".*do_sample.*is set to.*")
warnings.filterwarnings("ignore", message=".*`do_sample` is set to `False`.*")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")

# Also suppress via logging (HuggingFace uses loggers, not warnings)
import logging
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
logging.getLogger("transformers.generation.configuration_utils").setLevel(logging.ERROR)

from detinfer.inference.config import DeterministicConfig
from detinfer.inference.enforcer import DeterministicEnforcer, EnforcementReport
from detinfer.inference.canonicalizer import OutputCanonicalizer, Precision
from detinfer.inference.guardian import EnvironmentGuardian, EnvironmentFingerprint
from detinfer.inference.utils import hash_string, hash_tensor, get_environment_snapshot
from detinfer.inference.verifier import InferenceVerifier, VerificationResult


@dataclass
class DeterministicResult:
    """Result of a deterministic inference run.

    Contains the output text/tensor, canonical hash, raw hash,
    environment fingerprint, and timing information.
    """
    text: str | None = None
    output_tensor: torch.Tensor | None = None
    raw_hash: str = ""
    canonical_hash: str = ""
    input_tokens_hash: str = ""
    output_tokens_hash: str = ""
    model_dtype: str = ""
    quantization: str = ""
    precision: str = ""
    seed: int = 42
    elapsed_seconds: float = 0.0
    environment: EnvironmentFingerprint | None = None

    def to_proof(self) -> dict[str, Any]:
        """Export as a verification proof dict.

        This can be sent to other nodes to prove execution.
        """
        return {
            "canonical_hash": self.canonical_hash,
            "raw_hash": self.raw_hash,
            "input_tokens_hash": self.input_tokens_hash,
            "output_tokens_hash": self.output_tokens_hash,
            "model_dtype": self.model_dtype,
            "quantization": self.quantization,
            "precision": self.precision,
            "seed": self.seed,
            "elapsed_seconds": self.elapsed_seconds,
            "environment": self.environment.to_dict() if self.environment else {},
            "text_hash": hash_string(self.text) if self.text else None,
        }

    def __str__(self) -> str:
        lines = []
        if self.text is not None:
            preview = self.text[:100] + "..." if len(self.text) > 100 else self.text
            lines.append(f"Output: {preview}")
        lines.append(f"Canonical hash: {self.canonical_hash}")
        lines.append(f"Raw hash:       {self.raw_hash}")
        lines.append(f"Precision: {self.precision} | Seed: {self.seed}")
        lines.append(f"Time: {self.elapsed_seconds:.3f}s")
        return "\n".join(lines)


class DeterministicEngine:
    """End-to-end deterministic inference engine.

    The main entry point for detinfer v2. Handles:
    1. Model loading with deterministic config
    2. Automatic op enforcement (replaces non-det ops)
    3. Output canonicalization for cross-hardware consistency
    4. Environment fingerprinting and verification

    Usage with model name:
        engine = DeterministicEngine(seed=42)
        engine.load("<hf-model>")
        result = engine.run("Write hello world")

    Usage with pre-loaded model:
        engine = DeterministicEngine(seed=42)
        engine.load_model(model, tokenizer)
        result = engine.run("Write hello world")

    Usage with raw tensors (no tokenizer):
        engine = DeterministicEngine(seed=42)
        engine.load_model(model)
        result = engine.run_tensor(input_tensor)
    """

    def __init__(
        self,
        seed: int = 42,
        precision: str | Precision = Precision.HIGH,
        device: str | None = None,
        warn_only: bool = True,
    ):
        """Initialize the deterministic engine.

        Args:
            seed: Master random seed for all RNG sources.
            precision: Canonicalization precision ("exact", "high", "medium",
                       "low", "token"). Higher = more sensitive to differences.
            device: Device to use ("cpu", "cuda", "cuda:0").
                    Auto-detected if None.
            warn_only: If True, warn instead of error on non-deterministic ops
                       that can't be fully replaced.
        """
        self.seed = seed
        self.precision = Precision(precision) if isinstance(precision, str) else precision
        self.warn_only = warn_only

        # Core components
        self.config = DeterministicConfig(seed=seed, warn_only=warn_only)
        self.enforcer = DeterministicEnforcer(seed=seed, warn_only=warn_only)
        self.canonicalizer = OutputCanonicalizer(precision=self.precision)
        self.guardian = EnvironmentGuardian()

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Model state (set by load/load_model)
        self.model: nn.Module | None = None
        self.tokenizer: Any | None = None
        self.model_name: str = ""
        self.enforcement_report: EnforcementReport | None = None
        self.fingerprint: EnvironmentFingerprint | None = None
        self._multi_gpu: bool = False  # True when using device_map

    def load(
        self,
        model_name: str,
        torch_dtype: torch.dtype | str = "auto",
        device_map: str | dict | None = None,
        quantize: str | None = None,
    ) -> EnforcementReport:
        """Load a HuggingFace model by name and enforce determinism.

        Args:
            model_name: HuggingFace model ID (e.g., "owner/model-name").
            torch_dtype: Data type for model loading.
            device_map: Device placement strategy for multi-GPU.
                        Use "auto" to split across all available GPUs.
                        Use None for single-device (default).
            quantize: Quantization mode (experimental). Currently supports "int8".
                      INT8 may improve cross-device consistency but does not
                      guarantee bitwise determinism across GPU architectures.

        Returns:
            EnforcementReport showing what ops were fixed.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "The 'transformers' package is required for loading models by name. "
                "Install it with: pip install detinfer[transformers]"
            ) from e

        self.model_name = model_name
        self._multi_gpu = device_map is not None

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model — with or without device_map and quantization
        load_kwargs = {"torch_dtype": torch_dtype}
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        # INT8 quantization (experimental)
        if quantize == "int8":
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "The 'bitsandbytes' package is required for INT8 quantization. "
                    "Install it with: pip install detinfer[quantized]"
                ) from e
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = device_map or "auto"
            self._multi_gpu = True  # INT8 requires device_map

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs
        )

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Move to device (only for single-device mode)
        if not self._multi_gpu:
            self.model.to(self.device)

        # Enforce determinism (patches model in-place)
        self.enforcement_report = self.enforcer.enforce(
            self.model, model_name=model_name
        )

        # Lock seeds on ALL GPUs
        self.config.apply()

        # Capture environment fingerprint
        self.fingerprint = self.guardian.create_fingerprint()

        return self.enforcement_report

    def load_model(
        self,
        model: nn.Module,
        tokenizer: Any | None = None,
        model_name: str | None = None,
    ) -> EnforcementReport:
        """Load a pre-created model and enforce determinism.

        Args:
            model: Any PyTorch nn.Module.
            tokenizer: Optional tokenizer for text generation.
            model_name: Optional name for reports.

        Returns:
            EnforcementReport showing what ops were fixed.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name or model.__class__.__name__
        has_device_map = isinstance(getattr(self.model, "hf_device_map", None), dict)
        is_quantized = bool(
            getattr(self.model, "is_loaded_in_8bit", False)
            or getattr(self.model, "is_loaded_in_4bit", False)
        )
        self._multi_gpu = has_device_map

        if not has_device_map and not is_quantized:
            self.model.to(self.device)

        # Enforce determinism
        self.enforcement_report = self.enforcer.enforce(
            self.model, model_name=self.model_name
        )

        # Capture environment fingerprint
        self.fingerprint = self.guardian.create_fingerprint()

        return self.enforcement_report

    @torch.no_grad()
    def run(self, prompt: str, max_new_tokens: int = 256) -> DeterministicResult:
        """Run deterministic text generation.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum new tokens to generate.

        Returns:
            DeterministicResult with text, hashes, and proof.
        """
        self._check_loaded()

        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer required for text generation. "
                "Use run_tensor() for models without a tokenizer."
            )

        start = time.perf_counter()

        with self.enforcer.deterministic_context():
            # Auto-apply chat template if available (for chat models)
            formatted_prompt = prompt
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                try:
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    pass  # Fall back to raw prompt if template fails

            # Tokenize — send to correct device
            input_device = self._get_input_device()
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(input_device)
            input_tokens_hash = hash_tensor(inputs["input_ids"])

            # Generate with strict greedy decoding (no randomness whatsoever)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,   # No sampling
                num_beams=1,       # No beam search (also non-deterministic)
            )
            output_tokens_hash = hash_tensor(output_ids)

            # Decode new tokens only
            prompt_length = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0, prompt_length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        elapsed = time.perf_counter() - start

        # Canonicalize
        raw_hash = hash_string(text)
        canonical = self.canonicalizer.canonicalize(output_ids.float())

        # Get backend metadata
        model_dtype = str(next(self.model.parameters()).dtype) if self.model else "unknown"
        quant_config = getattr(getattr(self.model, "config", None), "quantization_config", None)
        quantization = "bitsandbytes" if quant_config else "none" # Simplify for now, could expand to awq/gptq

        return DeterministicResult(
            text=text,
            output_tensor=output_ids.cpu(),
            raw_hash=raw_hash,
            canonical_hash=canonical.canonical_hash,
            input_tokens_hash=input_tokens_hash,
            output_tokens_hash=output_tokens_hash,
            model_dtype=model_dtype,
            quantization=quantization,
            precision=self.precision.value,
            seed=self.seed,
            elapsed_seconds=elapsed,
            environment=self.fingerprint,
        )

    @torch.no_grad()
    def run_tensor(self, input_tensor: torch.Tensor) -> DeterministicResult:
        """Run deterministic inference with a raw tensor.

        Args:
            input_tensor: Raw input tensor for the model.

        Returns:
            DeterministicResult with output tensor and hashes.
        """
        self._check_loaded()
        start = time.perf_counter()

        with self.enforcer.deterministic_context():
            input_device = self._get_input_device()
            input_tensor = input_tensor.to(input_device)
            output = self.model(input_tensor)

            # Handle different output types
            if isinstance(output, torch.Tensor):
                tensor_out = output
            elif hasattr(output, "logits"):
                tensor_out = output.logits
            elif isinstance(output, tuple):
                tensor_out = output[0]
            else:
                tensor_out = torch.tensor(output)

        elapsed = time.perf_counter() - start

        raw_hash = hash_tensor(tensor_out)
        canonical = self.canonicalizer.canonicalize(tensor_out)

        return DeterministicResult(
            output_tensor=tensor_out.cpu(),
            raw_hash=raw_hash,
            canonical_hash=canonical.canonical_hash,
            precision=self.precision.value,
            seed=self.seed,
            elapsed_seconds=elapsed,
            environment=self.fingerprint,
        )

    def verify(
        self,
        prompt: str | None = None,
        input_tensor: torch.Tensor | None = None,
        num_runs: int = 5,
    ) -> VerificationResult:
        """Verify determinism by running inference multiple times.

        Provide either prompt (for LLMs) or input_tensor (for any model).
        If neither provided, uses a default test prompt.

        Args:
            prompt: Text prompt (for LLMs with tokenizer).
            input_tensor: Raw tensor input.
            num_runs: Number of repeated runs.

        Returns:
            VerificationResult.
        """
        self._check_loaded()

        verifier = InferenceVerifier(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        if input_tensor is not None:
            return verifier.verify_with_input(
                input_tensor, num_runs=num_runs, seed=self.seed
            )
        elif prompt is not None:
            return verifier.verify(
                prompt, num_runs=num_runs, seed=self.seed
            )
        else:
            # Default test prompt
            default_prompt = "What is 2 + 2? Answer with just the number."
            return verifier.verify(
                default_prompt, num_runs=num_runs, seed=self.seed
            )

    def scan(self) -> EnforcementReport:
        """Return the enforcement report from model loading.

        Shows what non-deterministic ops were found and fixed.
        """
        if self.enforcement_report is None:
            raise RuntimeError("No model loaded. Call load() or load_model() first.")
        return self.enforcement_report

    def get_info(self) -> dict[str, Any]:
        """Return complete engine and environment information."""
        return {
            "model_name": self.model_name,
            "seed": self.seed,
            "precision": self.precision.value,
            "device": self.device,
            "enforcement": str(self.enforcement_report) if self.enforcement_report else None,
            "environment": self.fingerprint.to_dict() if self.fingerprint else None,
        }

    def _check_loaded(self) -> None:
        if self.model is None:
            raise RuntimeError(
                "No model loaded. Call engine.load('model_name') or "
                "engine.load_model(model) first."
            )

    def __repr__(self) -> str:
        model = self.model_name or "no model"
        device_info = "multi-gpu" if self._multi_gpu else self.device
        return (
            f"DeterministicEngine(model='{model}', seed={self.seed}, "
            f"precision='{self.precision.value}', device='{device_info}')"
        )

    def _get_input_device(self) -> str | torch.device:
        """Get the correct device for input tensors.

        For multi-GPU models, inputs go to the first device in the model's
        device map. For single-device, inputs go to self.device.
        """
        if self.model is not None:
            has_device_map = isinstance(getattr(self.model, "hf_device_map", None), dict)
            is_quantized = bool(
                getattr(self.model, "is_loaded_in_8bit", False)
                or getattr(self.model, "is_loaded_in_4bit", False)
            )
            if self._multi_gpu or has_device_map or is_quantized:
                # For dispatched/quantized models, use parameter device directly.
                try:
                    first_param = next(self.model.parameters())
                    return first_param.device
                except StopIteration:
                    return self.device
        return self.device

