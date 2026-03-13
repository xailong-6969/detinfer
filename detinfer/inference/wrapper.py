"""
detinfer.wrapper — Deterministic LLM Wrapper

Wraps any HuggingFace causal language model with deterministic settings:
greedy decoding, seed locking, and batch isolation.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    raise ImportError(
        "The 'transformers' package is required for DeterministicLLM. "
        "Install it with: pip install detinfer[transformers]"
    ) from e

from detinfer.inference.config import DeterministicConfig
from detinfer.inference.utils import hash_string, get_environment_snapshot
from detinfer.inference.verifier import InferenceVerifier


class DeterministicLLM:
    """Wraps a HuggingFace causal LM with deterministic inference.

    Every call to .generate() will:
        1. Reset all random seeds
        2. Use greedy decoding (do_sample=False)
        3. Produce identical output for the same input

    Usage:
        llm = DeterministicLLM("<hf-model>", seed=42)
        output = llm.generate("Write hello world in Python")
        # Same output every time

    For pre-loaded models, pass model and tokenizer directly:
        llm = DeterministicLLM(model=model, tokenizer=tokenizer, seed=42)
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        seed: int = 42,
        device: str | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        warn_only: bool = False,
        torch_dtype: torch.dtype | str = "auto",
    ):
        """Initialize with either a model name (auto-loads) or pre-loaded model.

        Args:
            model_name: HuggingFace model ID or local path. Ignored if
                        model + tokenizer are provided.
            seed: Master random seed.
            device: Device to use ('cpu', 'cuda', 'cuda:0', etc.).
                    Auto-detected if not specified.
            model: Pre-loaded HuggingFace model (optional).
            tokenizer: Pre-loaded tokenizer (optional).
            warn_only: If True, warn rather than error on non-deterministic ops.
            torch_dtype: Data type for model loading (default: "auto").
        """
        self.seed = seed
        self.config = DeterministicConfig(seed=seed, warn_only=warn_only)
        self.config.apply()

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load or use provided model
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            self.model_name = model_name or model.__class__.__name__
        elif model_name is not None:
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
            )
        else:
            raise ValueError(
                "Either model_name or both model + tokenizer must be provided."
            )

        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> str:
        """Generate text deterministically.

        Seeds are reset before every call. Greedy decoding is enforced.
        Any sampling kwargs (do_sample, temperature, top_k, top_p) are
        overridden to ensure determinism.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of new tokens to generate.
            **kwargs: Additional kwargs passed to model.generate().
                      Sampling params are overridden for determinism.

        Returns:
            Generated text string (excluding the prompt).
        """
        # Reset seeds before every generation
        self.config.reset_seeds()

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Force deterministic generation settings
        kwargs.update({
            "do_sample": False,       # Greedy — no randomness
            "max_new_tokens": max_new_tokens,
            "temperature": 1.0,       # No scaling (irrelevant with greedy, but explicit)
            "top_k": 0,               # Disabled
            "top_p": 1.0,             # Disabled
        })

        # Generate
        output_ids = self.model.generate(**inputs, **kwargs)

        # Decode only the NEW tokens (strip the prompt)
        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return text

    def generate_with_hash(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> dict[str, str]:
        """Generate text and return both the output and its SHA-256 hash.

        Useful for verification workflows where you need both the text
        and a deterministic fingerprint.

        Returns:
            Dict with keys: "text", "hash", "prompt_hash"
        """
        text = self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
        return {
            "text": text,
            "hash": hash_string(text),
            "prompt_hash": hash_string(prompt),
        }

    def verify(
        self,
        prompt: str,
        num_runs: int = 5,
        max_new_tokens: int = 50,
    ):
        """Run the built-in verifier on this model.

        Convenience method that creates an InferenceVerifier and runs it.

        Args:
            prompt: Text prompt to test.
            num_runs: Number of repeated runs.
            max_new_tokens: Tokens per run.

        Returns:
            VerificationResult.
        """
        verifier = InferenceVerifier(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        return verifier.verify(
            prompt=prompt,
            num_runs=num_runs,
            seed=self.seed,
            max_new_tokens=max_new_tokens,
        )

    def get_info(self) -> dict[str, Any]:
        """Return model and environment information."""
        return {
            "model_name": self.model_name,
            "seed": self.seed,
            "device": str(self.device),
            "config_snapshot": self.config.snapshot(),
            "environment": get_environment_snapshot(),
        }

    def __repr__(self) -> str:
        return (
            f"DeterministicLLM(model='{self.model_name}', "
            f"seed={self.seed}, device='{self.device}')"
        )

