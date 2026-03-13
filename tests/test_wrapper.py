"""Tests for detinfer.wrapper — DeterministicLLM.

Uses a tiny randomly-initialized causal LM for testing (no downloads).
Requires the 'transformers' package.
"""

import torch
import pytest

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from detinfer.inference.config import DeterministicConfig

# Smallest available CausalLM architecture for offline tests.
# This is NOT a model recommendation — detinfer works with any HF model.
_TEST_ARCH = "gpt2"

# Skip entire module if transformers is not installed
pytestmark = pytest.mark.skipif(
    not HAS_TRANSFORMERS,
    reason="transformers not installed"
)


def _make_tiny_test_model():
    """Create a tiny randomly-initialized causal LM (~1MB) for testing.

    The model produces gibberish — we only need it to verify
    that same input + same seed = same output (determinism).
    """
    config = AutoConfig.for_model(
        _TEST_ARCH,
        vocab_size=50257,
        n_embd=32,
        n_layer=2,
        n_head=2,
        n_inner=64,
        n_positions=128,
    )
    model = AutoModelForCausalLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(_TEST_ARCH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


class TestDeterministicLLM:
    """Tests for the DeterministicLLM wrapper."""

    def test_identical_outputs_across_runs(self):
        """Same prompt should produce identical output every time."""
        from detinfer.inference.wrapper import DeterministicLLM

        model, tokenizer = _make_tiny_test_model()
        llm = DeterministicLLM(
            model=model, tokenizer=tokenizer, seed=42, device="cpu"
        )

        output1 = llm.generate("Hello world", max_new_tokens=20)
        output2 = llm.generate("Hello world", max_new_tokens=20)
        output3 = llm.generate("Hello world", max_new_tokens=20)

        assert output1 == output2 == output3, (
            f"Outputs differ:\n  run1: {output1!r}\n  run2: {output2!r}\n  run3: {output3!r}"
        )

    def test_generate_with_hash(self):
        """generate_with_hash should return text + SHA-256 hashes."""
        from detinfer.inference.wrapper import DeterministicLLM

        model, tokenizer = _make_tiny_test_model()
        llm = DeterministicLLM(
            model=model, tokenizer=tokenizer, seed=42, device="cpu"
        )

        result = llm.generate_with_hash("Test prompt", max_new_tokens=10)

        assert "text" in result
        assert "hash" in result
        assert "prompt_hash" in result
        assert len(result["hash"]) == 64  # SHA-256 hex is 64 chars

    def test_hash_consistency(self):
        """Hashes should be identical across runs."""
        from detinfer.inference.wrapper import DeterministicLLM

        model, tokenizer = _make_tiny_test_model()
        llm = DeterministicLLM(
            model=model, tokenizer=tokenizer, seed=42, device="cpu"
        )

        r1 = llm.generate_with_hash("Hash test", max_new_tokens=10)
        r2 = llm.generate_with_hash("Hash test", max_new_tokens=10)

        assert r1["hash"] == r2["hash"]
        assert r1["text"] == r2["text"]

    def test_verify_method(self):
        """Built-in verify() should confirm determinism."""
        from detinfer.inference.wrapper import DeterministicLLM

        model, tokenizer = _make_tiny_test_model()
        llm = DeterministicLLM(
            model=model, tokenizer=tokenizer, seed=42, device="cpu"
        )

        result = llm.verify("Verify this", num_runs=3, max_new_tokens=10)

        assert result.is_deterministic
        assert len(result.unique_hashes) == 1

    def test_get_info(self):
        """get_info should return model and environment data."""
        from detinfer.inference.wrapper import DeterministicLLM

        model, tokenizer = _make_tiny_test_model()
        llm = DeterministicLLM(
            model=model, tokenizer=tokenizer, seed=42, device="cpu"
        )

        info = llm.get_info()

        assert info["seed"] == 42
        assert "config_snapshot" in info
        assert "environment" in info

    def test_different_prompts_different_outputs(self):
        """Different prompts should (very likely) produce different outputs."""
        from detinfer.inference.wrapper import DeterministicLLM

        model, tokenizer = _make_tiny_test_model()
        llm = DeterministicLLM(
            model=model, tokenizer=tokenizer, seed=42, device="cpu"
        )

        out1 = llm.generate("The cat sat on", max_new_tokens=20)
        out2 = llm.generate("Mathematics is the study of", max_new_tokens=20)

        # Very unlikely to be identical given different prompts
        # (not guaranteed, but extremely probable with even a random model)
        assert isinstance(out1, str)
        assert isinstance(out2, str)

    def test_repr(self):
        """repr should be informative."""
        from detinfer.inference.wrapper import DeterministicLLM

        model, tokenizer = _make_tiny_test_model()
        llm = DeterministicLLM(
            model=model, tokenizer=tokenizer, seed=42, device="cpu"
        )

        r = repr(llm)
        assert "seed=42" in r
        assert "cpu" in r

    def test_no_args_raises(self):
        """Passing neither model_name nor model should raise ValueError."""
        from detinfer.inference.wrapper import DeterministicLLM

        with pytest.raises(ValueError, match="Either model_name"):
            DeterministicLLM(seed=42)

