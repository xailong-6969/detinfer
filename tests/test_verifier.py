"""Tests for determl.verifier — InferenceVerifier.

Uses tiny randomly-initialized models for fast CPU-only testing.
No large model downloads required.
"""

import torch
import torch.nn as nn
import pytest

from determl.verifier import InferenceVerifier


class TinyModel(nn.Module):
    """A minimal deterministic model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.output = nn.Linear(5, 3)

    def forward(self, x):
        return self.output(self.relu(self.linear(x)))


class TestInferenceVerifier:
    """Tests that the verifier correctly identifies deterministic outputs."""

    def test_deterministic_model_passes(self):
        """A simple linear model should produce identical outputs every run."""
        model = TinyModel()
        model.eval()

        verifier = InferenceVerifier(model, device="cpu")
        input_tensor = torch.randn(1, 10)

        result = verifier.verify_with_input(input_tensor, num_runs=5, seed=42)

        assert result.is_deterministic
        assert result.num_runs == 5
        assert len(result.hashes) == 5
        assert len(result.unique_hashes) == 1

    def test_all_hashes_identical(self):
        """Every hash in the result should be the same string."""
        model = TinyModel()
        model.eval()

        verifier = InferenceVerifier(model, device="cpu")
        input_tensor = torch.randn(1, 10)

        result = verifier.verify_with_input(input_tensor, num_runs=3, seed=42)

        assert all(h == result.hashes[0] for h in result.hashes)

    def test_different_seeds_same_input(self):
        """Same model + same input + different seeds should still be deterministic
        (the model itself is deterministic, seeds only affect random ops)."""
        model = TinyModel()
        model.eval()

        input_tensor = torch.randn(1, 10)

        verifier = InferenceVerifier(model, device="cpu")
        result1 = verifier.verify_with_input(input_tensor, num_runs=3, seed=42)
        result2 = verifier.verify_with_input(input_tensor, num_runs=3, seed=99)

        # Both should be deterministic within their own runs
        assert result1.is_deterministic
        assert result2.is_deterministic

    def test_store_outputs(self):
        """When store_outputs=True, output tensors should be captured."""
        model = TinyModel()
        model.eval()

        verifier = InferenceVerifier(model, device="cpu")
        input_tensor = torch.randn(1, 10)

        result = verifier.verify_with_input(
            input_tensor, num_runs=3, seed=42, store_outputs=True
        )

        assert len(result.outputs) == 3
        # All outputs should be identical tensors
        for out in result.outputs[1:]:
            assert torch.equal(result.outputs[0], out)

    def test_elapsed_time_positive(self):
        """Elapsed time should be positive."""
        model = TinyModel()
        model.eval()

        verifier = InferenceVerifier(model, device="cpu")
        input_tensor = torch.randn(1, 10)

        result = verifier.verify_with_input(input_tensor, num_runs=2, seed=42)

        assert result.elapsed_seconds > 0

    def test_environment_captured(self):
        """Environment snapshot should be captured in result."""
        model = TinyModel()
        model.eval()

        verifier = InferenceVerifier(model, device="cpu")
        input_tensor = torch.randn(1, 10)

        result = verifier.verify_with_input(input_tensor, num_runs=2, seed=42)

        assert "torch_version" in result.environment
        assert "python_version" in result.environment

    def test_result_str_deterministic(self):
        """String representation should indicate deterministic for deterministic model."""
        model = TinyModel()
        model.eval()

        verifier = InferenceVerifier(model, device="cpu")
        input_tensor = torch.randn(1, 10)

        result = verifier.verify_with_input(input_tensor, num_runs=3, seed=42)

        result_str = str(result)
        assert "DETERMINISTIC" in result_str

    def test_custom_forward_fn(self):
        """Custom forward function should be used when provided."""
        model = TinyModel()
        model.eval()

        def custom_fn(m, x):
            # Only return logits from first output position
            return m(x)[:, :2]

        verifier = InferenceVerifier(model, device="cpu")
        input_tensor = torch.randn(1, 10)

        result = verifier.verify_with_input(
            input_tensor, num_runs=3, seed=42, forward_fn=custom_fn
        )

        assert result.is_deterministic

    def test_verify_without_tokenizer_raises(self):
        """verify() without tokenizer should raise ValueError."""
        model = TinyModel()
        verifier = InferenceVerifier(model, device="cpu")

        with pytest.raises(ValueError, match="Tokenizer required"):
            verifier.verify("test prompt")
