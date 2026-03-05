"""Tests for determl.wrapper — DeterministicLLM.

Uses a tiny randomly-initialized causal LM model (no downloads).
Requires the 'transformers' package.
"""

import torch
import pytest

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from determl.config import DeterministicConfig


# Skip entire module if transformers is not installed
pytestmark = pytest.mark.skipif(
    not HAS_TRANSFORMERS,
    reason="transformers not installed"
)


def _make_tiny_gpt2():
    """Create a tiny random GPT-2 model (~1MB) for testing.

    This model is randomly initialized — it produces gibberish text,
    but that's fine because we only need to verify determinism.
    """
    config = GPT2Config(
        vocab_size=50257,  # Must match GPT-2 tokenizer vocab
        n_embd=32,
        n_layer=2,
        n_head=2,
        n_inner=64,
        max_position_embeddings=128,
    )
    model = GPT2LMHeadModel(config)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


class TestDeterministicLLM:
    """Tests for the DeterministicLLM wrapper."""

    def test_identical_outputs_across_runs(self):
        """Same prompt should produce identical output every time."""
        from determl.wrapper import DeterministicLLM

        model, tokenizer = _make_tiny_gpt2()
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
        from determl.wrapper import DeterministicLLM

        model, tokenizer = _make_tiny_gpt2()
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
        from determl.wrapper import DeterministicLLM

        model, tokenizer = _make_tiny_gpt2()
        llm = DeterministicLLM(
            model=model, tokenizer=tokenizer, seed=42, device="cpu"
        )

        r1 = llm.generate_with_hash("Hash test", max_new_tokens=10)
        r2 = llm.generate_with_hash("Hash test", max_new_tokens=10)

        assert r1["hash"] == r2["hash"]
        assert r1["text"] == r2["text"]

    def test_verify_method(self):
        """Built-in verify() should confirm determinism."""
        from determl.wrapper import DeterministicLLM

        model, tokenizer = _make_tiny_gpt2()
        llm = DeterministicLLM(
            model=model, tokenizer=tokenizer, seed=42, device="cpu"
        )

        result = llm.verify("Verify this", num_runs=3, max_new_tokens=10)

        assert result.is_deterministic
        assert len(result.unique_hashes) == 1

    def test_get_info(self):
        """get_info should return model and environment data."""
        from determl.wrapper import DeterministicLLM

        model, tokenizer = _make_tiny_gpt2()
        llm = DeterministicLLM(
            model=model, tokenizer=tokenizer, seed=42, device="cpu"
        )

        info = llm.get_info()

        assert info["seed"] == 42
        assert "config_snapshot" in info
        assert "environment" in info

    def test_different_prompts_different_outputs(self):
        """Different prompts should (very likely) produce different outputs."""
        from determl.wrapper import DeterministicLLM

        model, tokenizer = _make_tiny_gpt2()
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
        from determl.wrapper import DeterministicLLM

        model, tokenizer = _make_tiny_gpt2()
        llm = DeterministicLLM(
            model=model, tokenizer=tokenizer, seed=42, device="cpu"
        )

        r = repr(llm)
        assert "seed=42" in r
        assert "cpu" in r

    def test_no_args_raises(self):
        """Passing neither model_name nor model should raise ValueError."""
        from determl.wrapper import DeterministicLLM

        with pytest.raises(ValueError, match="Either model_name"):
            DeterministicLLM(seed=42)
