"""Tests for determl.config — DeterministicConfig seed locking."""

import os
import random

import numpy as np
import torch
import pytest

from determl.config import DeterministicConfig


class TestDeterministicConfig:
    """Test that DeterministicConfig correctly locks all RNG sources."""

    def test_apply_sets_torch_seed(self):
        """After apply(), torch random should be reproducible."""
        config = DeterministicConfig(seed=123)
        config.apply()
        a = torch.rand(5)

        config.apply()
        b = torch.rand(5)

        assert torch.equal(a, b), "Torch random outputs differ after seed reset"

    def test_apply_sets_numpy_seed(self):
        """After apply(), numpy random should be reproducible."""
        config = DeterministicConfig(seed=456)
        config.apply()
        a = np.random.rand(5)

        config.apply()
        b = np.random.rand(5)

        np.testing.assert_array_equal(a, b)

    def test_apply_sets_python_random(self):
        """After apply(), Python random should be reproducible."""
        config = DeterministicConfig(seed=789)
        config.apply()
        a = [random.random() for _ in range(5)]

        config.apply()
        b = [random.random() for _ in range(5)]

        assert a == b

    def test_deterministic_algorithms_enabled(self):
        """After apply(), torch deterministic algorithms should be on."""
        config = DeterministicConfig(seed=42)
        config.apply()
        assert torch.are_deterministic_algorithms_enabled()

    def test_cudnn_flags(self):
        """After apply(), cuDNN flags should be set for determinism."""
        config = DeterministicConfig(seed=42)
        config.apply()
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_cublas_workspace(self):
        """After apply(), CUBLAS_WORKSPACE_CONFIG should be set."""
        config = DeterministicConfig(seed=42)
        config.apply()
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"

    def test_pythonhashseed(self):
        """After apply(), PYTHONHASHSEED should match seed."""
        config = DeterministicConfig(seed=42)
        config.apply()
        assert os.environ.get("PYTHONHASHSEED") == "42"

    def test_snapshot_reflects_state(self):
        """Snapshot should capture the current determinism state."""
        config = DeterministicConfig(seed=42)
        config.apply()
        snap = config.snapshot()

        assert snap["seed"] == 42
        assert snap["applied"] is True
        assert snap["torch_deterministic"] is True
        assert snap["cudnn_deterministic"] is True
        assert snap["cudnn_benchmark"] is False

    def test_reset_seeds(self):
        """reset_seeds() should re-apply seeds without touching flags."""
        config = DeterministicConfig(seed=42)
        config.apply()
        a = torch.rand(3)

        config.reset_seeds()
        b = torch.rand(3)

        assert torch.equal(a, b)

    def test_warn_only_mode(self):
        """warn_only=True should not raise on non-deterministic ops."""
        config = DeterministicConfig(seed=42, warn_only=True)
        config.apply()  # Should not raise
        assert torch.are_deterministic_algorithms_enabled()

    def test_repr(self):
        """repr should show seed and status."""
        config = DeterministicConfig(seed=42)
        assert "NOT APPLIED" in repr(config)
        config.apply()
        assert "APPLIED" in repr(config)

    def test_chaining(self):
        """apply() should return self for chaining."""
        config = DeterministicConfig(seed=42)
        result = config.apply()
        assert result is config
