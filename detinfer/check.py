"""
detinfer.check -- Baseline regression checking with mismatch classification.

Compares two exported session traces field-by-field and classifies every
difference into a named drift type (MODEL_DRIFT, TOKENIZER_DRIFT, etc.).

Designed for CI pipelines:
    detinfer check baseline.json candidate.json
    detinfer check baseline.json candidate.json --json
    detinfer check baseline.json candidate.json --fail-on OUTPUT_DRIFT
    detinfer check baseline.json candidate.json --allow ENVIRONMENT_DRIFT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Mismatch types and severity
# ---------------------------------------------------------------------------

ERROR_TYPES = {
    "SCHEMA_MISMATCH",
    "TYPE_MISMATCH",
    "MODEL_DRIFT",
    "TOKENIZER_DRIFT",
    "CONFIG_DRIFT",
    "PROMPT_DRIFT",
    "INPUT_TOKEN_DRIFT",
    "OUTPUT_DRIFT",
    "STOP_REASON_DRIFT",
}

WARNING_TYPES = {
    "ENVIRONMENT_DRIFT",
    "SESSION_HASH_DRIFT",
}

INFO_TYPES = {
    "TRACE_DETAIL_DRIFT",
}


def mismatch_severity(mismatch_type: str) -> str:
    """Return severity level for a mismatch type."""
    if mismatch_type in ERROR_TYPES:
        return "error"
    if mismatch_type in WARNING_TYPES:
        return "warning"
    return "info"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CheckMismatch:
    """A single mismatch between baseline and candidate."""
    type: str
    severity: str
    field: str
    turn: int | None = None
    token_index: int | None = None
    expected: Any = None
    observed: Any = None


@dataclass
class CheckReport:
    """Full regression check report."""
    status: str  # "pass" or "failed"
    primary_type: str | None = None
    matched: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    mismatches: list[CheckMismatch] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "primary_type": self.primary_type,
            "matched": self.matched,
            "changed": self.changed,
            "warnings": self.warnings,
            "mismatches": [
                {
                    "type": m.type,
                    "severity": m.severity,
                    "field": m.field,
                    "turn": m.turn,
                    "token_index": m.token_index,
                    "expected": m.expected,
                    "observed": m.observed,
                }
                for m in self.mismatches
            ],
        }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def check_sessions(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    fail_on: set[str] | None = None,
    allow: set[str] | None = None,
) -> CheckReport:
    """Compare two session traces and classify mismatches.

    Args:
        baseline: Baseline session trace dict (from JSON).
        candidate: Candidate session trace dict (from JSON).
        fail_on: Set of mismatch types that should cause failure.
            If empty/None, all error-severity types cause failure.
        allow: Set of mismatch types to ignore entirely.

    Returns:
        CheckReport with status, matched/changed fields, and mismatches.
    """
    fail_on = {x.upper() for x in (fail_on or set())}
    allow = {x.upper() for x in (allow or set())}

    report = CheckReport(status="pass", primary_type=None)

    # Run comparisons in order of diagnostic priority
    _compare_schema(baseline, candidate, report)
    _compare_trace_type(baseline, candidate, report)
    _compare_identity_and_config(baseline, candidate, report)
    _compare_environment(baseline, candidate, report)
    _compare_messages_and_prompts(baseline, candidate, report)
    _compare_generations(baseline, candidate, report)

    # Session hash is derivative metadata, so report it separately from
    # root-cause execution drift and do not double-count it as OUTPUT_DRIFT.
    _compare_scalar_field(baseline, candidate, report,
                          "session_hash", "SESSION_HASH_DRIFT")

    _finalize_report(report, fail_on=fail_on, allow=allow)
    return report


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _compare_schema(
    baseline: dict, candidate: dict, report: CheckReport,
) -> None:
    _compare_scalar_field(baseline, candidate, report,
                          "schema_version", "SCHEMA_MISMATCH")


def _compare_trace_type(
    baseline: dict, candidate: dict, report: CheckReport,
) -> None:
    b_type = baseline.get("trace_type", "inference")
    c_type = candidate.get("trace_type", "inference")
    if b_type == c_type:
        report.matched.append("trace_type")
    else:
        report.changed.append("trace_type")
        report.mismatches.append(CheckMismatch(
            type="TYPE_MISMATCH",
            severity=mismatch_severity("TYPE_MISMATCH"),
            field="trace_type",
            expected=b_type,
            observed=c_type,
        ))


def _compare_identity_and_config(
    baseline: dict, candidate: dict, report: CheckReport,
) -> None:
    # Scalar identity fields
    for fld, mtype in [
        ("model", "MODEL_DRIFT"),
        ("model_hash", "MODEL_DRIFT"),
        ("seed", "CONFIG_DRIFT"),
    ]:
        _compare_scalar_field(baseline, candidate, report, fld, mtype)

    # Nested dicts
    _compare_nested_dict(
        baseline.get("generation_config", {}),
        candidate.get("generation_config", {}),
        report, prefix="generation_config", mismatch_type="CONFIG_DRIFT",
    )
    _compare_nested_dict(
        baseline.get("tokenizer", {}),
        candidate.get("tokenizer", {}),
        report, prefix="tokenizer", mismatch_type="TOKENIZER_DRIFT",
    )
    _compare_nested_dict(
        baseline.get("quantization", {}),
        candidate.get("quantization", {}),
        report, prefix="quantization", mismatch_type="CONFIG_DRIFT",
    )


def _compare_environment(
    baseline: dict, candidate: dict, report: CheckReport,
) -> None:
    _compare_nested_dict(
        baseline.get("environment", {}),
        candidate.get("environment", {}),
        report, prefix="environment", mismatch_type="ENVIRONMENT_DRIFT",
    )


def _compare_messages_and_prompts(
    baseline: dict, candidate: dict, report: CheckReport,
) -> None:
    if baseline.get("messages") == candidate.get("messages"):
        report.matched.append("messages")
    else:
        report.changed.append("messages")
        report.mismatches.append(CheckMismatch(
            type="PROMPT_DRIFT",
            severity=mismatch_severity("PROMPT_DRIFT"),
            field="messages",
            expected=str(baseline.get("messages", []))[:200],
            observed=str(candidate.get("messages", []))[:200],
        ))

    # Per-turn prompt hashes
    b_gens = baseline.get("generations", [])
    c_gens = candidate.get("generations", [])
    for idx, (bg, cg) in enumerate(zip(b_gens, c_gens), start=1):
        if bg.get("prompt_hash") == cg.get("prompt_hash"):
            report.matched.append(f"generations[{idx}].prompt_hash")
        else:
            report.changed.append(f"generations[{idx}].prompt_hash")
            report.mismatches.append(CheckMismatch(
                type="PROMPT_DRIFT",
                severity=mismatch_severity("PROMPT_DRIFT"),
                field=f"generations[{idx}].prompt_hash",
                turn=idx,
                expected=bg.get("prompt_hash"),
                observed=cg.get("prompt_hash"),
            ))


def _compare_generations(
    baseline: dict, candidate: dict, report: CheckReport,
) -> None:
    b_gens = baseline.get("generations", [])
    c_gens = candidate.get("generations", [])

    if len(b_gens) != len(c_gens):
        report.changed.append("generations.length")
        report.mismatches.append(CheckMismatch(
            type="OUTPUT_DRIFT",
            severity=mismatch_severity("OUTPUT_DRIFT"),
            field="generations.length",
            expected=len(b_gens),
            observed=len(c_gens),
        ))
        return

    for idx, (bg, cg) in enumerate(zip(b_gens, c_gens), start=1):
        _compare_generation_turn(bg, cg, idx, report)


def _compare_generation_turn(
    bg: dict, cg: dict, turn: int, report: CheckReport,
) -> None:
    # Input tokens hash
    if bg.get("input_tokens_hash") == cg.get("input_tokens_hash"):
        report.matched.append(f"turn_{turn}.input_tokens_hash")
    else:
        report.changed.append(f"turn_{turn}.input_tokens_hash")
        report.mismatches.append(CheckMismatch(
            type="INPUT_TOKEN_DRIFT",
            severity=mismatch_severity("INPUT_TOKEN_DRIFT"),
            field=f"generations[{turn}].input_tokens_hash",
            turn=turn,
            expected=bg.get("input_tokens_hash"),
            observed=cg.get("input_tokens_hash"),
        ))

    # Output tokens hash
    if bg.get("output_tokens_hash") == cg.get("output_tokens_hash"):
        report.matched.append(f"turn_{turn}.output_tokens_hash")
    else:
        report.changed.append(f"turn_{turn}.output_tokens_hash")
        first_idx, exp_tok, obs_tok = _first_token_mismatch(
            bg.get("output_tokens", []),
            cg.get("output_tokens", []),
        )
        report.mismatches.append(CheckMismatch(
            type="OUTPUT_DRIFT",
            severity=mismatch_severity("OUTPUT_DRIFT"),
            field=f"generations[{turn}].output_tokens",
            turn=turn,
            token_index=first_idx,
            expected=exp_tok,
            observed=obs_tok,
        ))

    # Stop reason
    if bg.get("stop_reason") == cg.get("stop_reason"):
        report.matched.append(f"turn_{turn}.stop_reason")
    else:
        report.changed.append(f"turn_{turn}.stop_reason")
        report.mismatches.append(CheckMismatch(
            type="STOP_REASON_DRIFT",
            severity=mismatch_severity("STOP_REASON_DRIFT"),
            field=f"generations[{turn}].stop_reason",
            turn=turn,
            expected=bg.get("stop_reason"),
            observed=cg.get("stop_reason"),
        ))

    # Step trace details (info-level only)
    b_steps = bg.get("steps")
    c_steps = cg.get("steps")
    if b_steps is not None and c_steps is not None and b_steps != c_steps:
        report.changed.append(f"turn_{turn}.steps")
        report.mismatches.append(CheckMismatch(
            type="TRACE_DETAIL_DRIFT",
            severity=mismatch_severity("TRACE_DETAIL_DRIFT"),
            field=f"generations[{turn}].steps",
            turn=turn,
        ))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _compare_scalar_field(
    baseline: dict, candidate: dict, report: CheckReport,
    fld: str, mismatch_type: str,
) -> None:
    b = baseline.get(fld)
    c = candidate.get(fld)
    if b == c:
        report.matched.append(fld)
    else:
        report.changed.append(fld)
        report.mismatches.append(CheckMismatch(
            type=mismatch_type,
            severity=mismatch_severity(mismatch_type),
            field=fld,
            expected=b,
            observed=c,
        ))


def _compare_nested_dict(
    baseline: dict, candidate: dict, report: CheckReport,
    prefix: str, mismatch_type: str,
) -> None:
    keys = sorted(set(baseline.keys()) | set(candidate.keys()))
    for key in keys:
        fld = f"{prefix}.{key}"
        b = baseline.get(key)
        c = candidate.get(key)
        if b == c:
            report.matched.append(fld)
        else:
            report.changed.append(fld)
            report.mismatches.append(CheckMismatch(
                type=mismatch_type,
                severity=mismatch_severity(mismatch_type),
                field=fld,
                expected=b,
                observed=c,
            ))


def _first_token_mismatch(
    expected: list[int], observed: list[int],
) -> tuple[int | None, int | None, int | None]:
    """Find the first index where token lists differ."""
    min_len = min(len(expected), len(observed))
    for i in range(min_len):
        if expected[i] != observed[i]:
            return i, expected[i], observed[i]
    if len(expected) != len(observed):
        exp = expected[min_len] if min_len < len(expected) else None
        obs = observed[min_len] if min_len < len(observed) else None
        return min_len, exp, obs
    return None, None, None


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------

def _finalize_report(
    report: CheckReport,
    fail_on: set[str],
    allow: set[str],
) -> None:
    """Determine pass/fail status based on mismatches and policy."""
    effective = [m for m in report.mismatches if m.type not in allow]

    if not effective:
        report.status = "pass"
        report.primary_type = None
        return

    if fail_on:
        failing = [m for m in effective if m.type in fail_on]
    else:
        failing = [m for m in effective if m.severity == "error"]

    if failing:
        report.status = "failed"
        report.primary_type = failing[0].type
    else:
        report.status = "pass"
        report.primary_type = effective[0].type
        report.warnings.append("Only warning/info mismatches found.")


# ---------------------------------------------------------------------------
# Text renderer
# ---------------------------------------------------------------------------

def render_check_report(
    report: CheckReport,
    baseline_path: str = "",
    candidate_path: str = "",
) -> str:
    """Render a human-readable regression report."""
    lines = []
    lines.append("Detinfer Regression Report")
    lines.append("-" * 26)

    if baseline_path:
        lines.append(f"Baseline:  {baseline_path}")
    if candidate_path:
        lines.append(f"Candidate: {candidate_path}")

    lines.append(f"\nStatus: {report.status.upper()}")

    if report.primary_type:
        lines.append(f"Primary type: {report.primary_type}")

    if report.warnings:
        for w in report.warnings:
            lines.append(f"⚠ {w}")

    if report.matched:
        lines.append("\nMatched:")
        shown = report.matched[:12]
        for item in shown:
            lines.append(f"  ✓ {item}")
        if len(report.matched) > 12:
            lines.append(f"  ... and {len(report.matched) - 12} more")

    if report.changed:
        lines.append("\nChanged:")
        for item in report.changed:
            lines.append(f"  ✗ {item}")

    if report.mismatches:
        first = report.mismatches[0]
        lines.append("\nFirst mismatch:")
        lines.append(f"  type:  {first.type}")
        lines.append(f"  field: {first.field}")
        if first.turn is not None:
            lines.append(f"  turn:  {first.turn}")
        if first.token_index is not None:
            lines.append(f"  token_index: {first.token_index}")
        if first.expected is not None:
            lines.append(f"  expected: {first.expected}")
        if first.observed is not None:
            lines.append(f"  observed: {first.observed}")

        if len(report.mismatches) > 1:
            lines.append(f"\n({len(report.mismatches)} total mismatches)")

    return "\n".join(lines)
