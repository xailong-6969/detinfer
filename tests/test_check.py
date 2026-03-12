"""
Tests for detinfer.check -- Regression check module.

Tests comparison logic, mismatch classification, severity,
fail-on/allow policy, and text rendering.
"""

import json

import pytest

from detinfer.check import (
    CheckMismatch,
    CheckReport,
    check_sessions,
    render_check_report,
    mismatch_severity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_session(
    model="gpt2",
    model_hash="abc123",
    seed=42,
    tokenizer_hash="tok_abc",
    output_tokens=None,
    environment=None,
    stop_reason="eos",
) -> dict:
    """Build a minimal session dict for testing."""
    return {
        "schema_version": "1",
        "model": model,
        "model_hash": model_hash,
        "seed": seed,
        "session_hash": "deadbeef",
        "trace_mode": "standard",
        "generation_config": {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": None,
            "top_k": None,
            "max_new_tokens": 256,
        },
        "tokenizer": {
            "name": model,
            "vocab_size": 50257,
            "tokenizer_hash": tokenizer_hash,
            "chat_template_hash": "tpl_abc",
        },
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
        "generations": [
            {
                "turn": 1,
                "prompt_hash": "ph_abc",
                "input_tokens_hash": "ith_abc",
                "output_tokens": output_tokens or [10, 20, 30],
                "output_tokens_hash": "oth_abc",
                "stop_reason": stop_reason,
            }
        ],
        "environment": environment or {
            "python": "3.11",
            "torch": "2.3.0",
            "device": "cpu",
        },
        "quantization": {"mode": None, "backend": None},
        "agent_steps": [],
        "registered_tools": [],
    }


# ---------------------------------------------------------------------------
# Severity tests
# ---------------------------------------------------------------------------

class TestSeverity:
    def test_error_types(self):
        assert mismatch_severity("MODEL_DRIFT") == "error"
        assert mismatch_severity("OUTPUT_DRIFT") == "error"
        assert mismatch_severity("CONFIG_DRIFT") == "error"

    def test_warning_types(self):
        assert mismatch_severity("ENVIRONMENT_DRIFT") == "warning"

    def test_info_types(self):
        assert mismatch_severity("TRACE_DETAIL_DRIFT") == "info"
        assert mismatch_severity("UNKNOWN_TYPE") == "info"


# ---------------------------------------------------------------------------
# Identical sessions
# ---------------------------------------------------------------------------

class TestIdenticalSessions:
    def test_pass(self):
        a = _make_session()
        b = _make_session()
        report = check_sessions(a, b)
        assert report.status == "pass"
        assert report.primary_type is None
        assert len(report.mismatches) == 0
        assert len(report.matched) > 0


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

class TestModelDrift:
    def test_model_name_changed(self):
        a = _make_session(model="gpt2")
        b = _make_session(model="gpt2-large")
        report = check_sessions(a, b)
        assert report.status == "failed"
        assert report.primary_type == "MODEL_DRIFT"

    def test_model_hash_changed(self):
        a = _make_session(model_hash="abc")
        b = _make_session(model_hash="xyz")
        report = check_sessions(a, b)
        assert report.status == "failed"
        types = {m.type for m in report.mismatches}
        assert "MODEL_DRIFT" in types


class TestTokenizerDrift:
    def test_tokenizer_hash_changed(self):
        a = _make_session(tokenizer_hash="tok_v1")
        b = _make_session(tokenizer_hash="tok_v2")
        report = check_sessions(a, b)
        assert report.status == "failed"
        types = {m.type for m in report.mismatches}
        assert "TOKENIZER_DRIFT" in types


class TestConfigDrift:
    def test_seed_changed(self):
        a = _make_session(seed=42)
        b = _make_session(seed=99)
        report = check_sessions(a, b)
        assert report.status == "failed"
        types = {m.type for m in report.mismatches}
        assert "CONFIG_DRIFT" in types


class TestOutputDrift:
    def test_output_tokens_changed(self):
        a = _make_session(output_tokens=[10, 20, 30])
        b = _make_session(output_tokens=[10, 20, 99])
        # Fix hashes to differ
        a["generations"][0]["output_tokens_hash"] = "hash_a"
        b["generations"][0]["output_tokens_hash"] = "hash_b"
        report = check_sessions(a, b)
        assert report.status == "failed"
        output_mismatches = [m for m in report.mismatches if m.type == "OUTPUT_DRIFT"]
        assert len(output_mismatches) > 0
        # Should find first mismatch at index 2
        token_mismatch = [m for m in output_mismatches if m.token_index is not None]
        assert len(token_mismatch) > 0
        assert token_mismatch[0].token_index == 2
        assert token_mismatch[0].expected == 30
        assert token_mismatch[0].observed == 99


class TestStopReasonDrift:
    def test_stop_reason_changed(self):
        a = _make_session(stop_reason="eos")
        b = _make_session(stop_reason="max_new_tokens")
        report = check_sessions(a, b)
        assert report.status == "failed"
        types = {m.type for m in report.mismatches}
        assert "STOP_REASON_DRIFT" in types


class TestEnvironmentDrift:
    def test_env_only_is_warning(self):
        a = _make_session(environment={"python": "3.11", "torch": "2.3.0"})
        b = _make_session(environment={"python": "3.12", "torch": "2.3.0"})
        report = check_sessions(a, b)
        # Environment drift alone should NOT fail (warning level)
        assert report.status == "pass"
        assert report.primary_type == "ENVIRONMENT_DRIFT"
        assert len(report.warnings) > 0


# ---------------------------------------------------------------------------
# Policy: --fail-on / --allow
# ---------------------------------------------------------------------------

class TestPolicy:
    def test_fail_on_specific_type(self):
        a = _make_session(environment={"python": "3.11"})
        b = _make_session(environment={"python": "3.12"})
        # By default: pass (environment is warning)
        report = check_sessions(a, b)
        assert report.status == "pass"

        # With fail-on: should fail
        report = check_sessions(a, b, fail_on={"ENVIRONMENT_DRIFT"})
        assert report.status == "failed"

    def test_allow_ignores_type(self):
        a = _make_session(seed=42)
        b = _make_session(seed=99)
        # By default: fail (config drift is error)
        report = check_sessions(a, b)
        assert report.status == "failed"

        # With allow: should pass
        report = check_sessions(a, b, allow={"CONFIG_DRIFT"})
        assert report.status == "pass"


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

class TestReportRendering:
    def test_pass_report(self):
        report = CheckReport(status="pass", primary_type=None, matched=["model"])
        text = render_check_report(report, "a.json", "b.json")
        assert "PASS" in text
        assert "a.json" in text

    def test_failed_report(self):
        report = CheckReport(
            status="failed",
            primary_type="OUTPUT_DRIFT",
            changed=["output_tokens_hash"],
            mismatches=[CheckMismatch(
                type="OUTPUT_DRIFT", severity="error",
                field="generations[1].output_tokens",
                turn=1, token_index=5, expected=287, observed=318,
            )],
        )
        text = render_check_report(report)
        assert "FAILED" in text
        assert "OUTPUT_DRIFT" in text
        assert "287" in text
        assert "318" in text

    def test_json_output(self):
        report = CheckReport(status="pass", primary_type=None, matched=["model"])
        d = report.to_dict()
        assert d["status"] == "pass"
        assert isinstance(d["matched"], list)
        # Serializable
        json.dumps(d)
