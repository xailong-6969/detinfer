"""Tests for detinfer.engine -- DeterministicEngine."""

import torch.nn as nn

from detinfer.inference.engine import DeterministicEngine


class TrackToModel(nn.Module):
    """Small model that records whether .to() was called."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.to_called = False

    def to(self, *args, **kwargs):
        self.to_called = True
        return self

    def forward(self, x):
        return self.linear(x)


def _make_engine(monkeypatch) -> DeterministicEngine:
    engine = DeterministicEngine(seed=42, device="cpu")
    monkeypatch.setattr(
        engine.enforcer,
        "enforce",
        lambda model, model_name=None: None,
    )
    monkeypatch.setattr(engine.guardian, "create_fingerprint", lambda: {})
    return engine


def test_load_model_moves_standard_model(monkeypatch):
    """Regular preloaded models should be moved to engine.device."""
    engine = _make_engine(monkeypatch)
    model = TrackToModel()

    engine.load_model(model)

    assert model.to_called is True


def test_load_model_skips_move_for_device_map(monkeypatch):
    """Device-mapped models must not be moved with a blanket .to()."""
    engine = _make_engine(monkeypatch)
    model = TrackToModel()
    model.hf_device_map = {"linear": "cuda:0"}

    engine.load_model(model)

    assert model.to_called is False
    assert engine._multi_gpu is True


def test_load_model_skips_move_for_quantized(monkeypatch):
    """Quantized models are managed by loaders and must not be moved."""
    engine = _make_engine(monkeypatch)
    model = TrackToModel()
    model.is_loaded_in_8bit = True

    engine.load_model(model)

    assert model.to_called is False
    assert engine._multi_gpu is False
    assert str(engine._get_input_device()) == str(model.linear.weight.device)


def test_load_model_applies_deterministic_config(monkeypatch):
    """Preloaded models should still enable deterministic runtime settings."""
    engine = _make_engine(monkeypatch)
    model = TrackToModel()
    applied = {"count": 0}

    def fake_apply():
        applied["count"] += 1

    monkeypatch.setattr(engine.config, "apply", fake_apply)
    engine.load_model(model)

    assert applied["count"] == 1
