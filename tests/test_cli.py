"""Tests for detinfer.cli exit-code behavior."""

import argparse
from types import SimpleNamespace

import pytest

import detinfer.cli as cli_mod
import detinfer.agent.replay as replay_mod
import detinfer.agent.trace as trace_mod
import detinfer.inference.engine as engine_mod


class DummyVerifyResult:
    def __init__(self, is_deterministic: bool):
        self.is_deterministic = is_deterministic

    def __str__(self) -> str:
        return "verify-result"


def test_cmd_verify_exits_nonzero_when_not_deterministic(monkeypatch):
    class FakeEngine:
        def __init__(self, seed=42, precision="high", device=None):
            pass

        def load(self, model_name):
            self.model_name = model_name

        def verify(self, prompt=None, num_runs=5):
            return DummyVerifyResult(is_deterministic=False)

    monkeypatch.setattr(engine_mod, "DeterministicEngine", FakeEngine)

    args = argparse.Namespace(
        model="test-model",
        seed=42,
        precision="high",
        device="cpu",
        prompt="hello",
        runs=3,
    )

    with pytest.raises(SystemExit) as exc:
        cli_mod.cmd_verify(args)

    assert exc.value.code == 1


def test_cmd_verify_returns_success_when_deterministic(monkeypatch):
    class FakeEngine:
        def __init__(self, seed=42, precision="high", device=None):
            pass

        def load(self, model_name):
            self.model_name = model_name

        def verify(self, prompt=None, num_runs=5):
            return DummyVerifyResult(is_deterministic=True)

    monkeypatch.setattr(engine_mod, "DeterministicEngine", FakeEngine)

    args = argparse.Namespace(
        model="test-model",
        seed=42,
        precision="high",
        device="cpu",
        prompt="hello",
        runs=3,
    )

    cli_mod.cmd_verify(args)


def test_cmd_verify_session_exits_nonzero_on_replay_mismatch(monkeypatch, tmp_path):
    class DummySession:
        model = "test-model"
        model_hash = "abc123"
        seed = 42
        generations = [object()]
        session_hash = "hash123"
        schema_version = "1"
        environment = {}

    result = SimpleNamespace(
        passed=False,
        total_turns=1,
        failure_turn=1,
        failure_reason="Output token mismatch",
        failure_step=0,
        expected_token=10,
        observed_token=20,
        details=["detail"],
    )

    monkeypatch.setattr(trace_mod.SessionTrace, "from_json", staticmethod(lambda path: DummySession()))
    monkeypatch.setattr(replay_mod, "replay_session", lambda trace_path, model_name=None, strict=False: result)

    args = argparse.Namespace(
        session_file=str(tmp_path / "session.json"),
        model=None,
        strict=False,
    )

    with pytest.raises(SystemExit) as exc:
        cli_mod.cmd_verify_session(args)

    assert exc.value.code == 1
