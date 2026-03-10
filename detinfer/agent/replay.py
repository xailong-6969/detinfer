"""
detinfer.replay -- Session replay and token-level diff.

Provides tools to:
- Replay a saved session trace and verify determinism turn-by-turn
- Diff two session traces to find the first point of divergence
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from detinfer.agent.trace import SessionTrace


# ---------------------------------------------------------------------------
# Replay result
# ---------------------------------------------------------------------------

@dataclass
class ReplayResult:
    """Result of a session replay verification."""
    passed: bool
    total_turns: int
    verified_turns: int
    failure_turn: int | None = None
    failure_step: int | None = None
    failure_reason: str = ""
    expected_token: int | None = None
    observed_token: int | None = None
    details: list[str] | None = None

    def __str__(self) -> str:
        if self.passed:
            return (
                f"Replay verification: PASSED\n"
                f"  All {self.total_turns} turns verified\n"
                f"  Session is deterministic ✓"
            )
        msg = f"Replay verification: FAILED\n"
        if self.failure_turn is not None:
            msg += f"  Turn {self.failure_turn}: {self.failure_reason}\n"
        if self.failure_step is not None:
            msg += f"  Step {self.failure_step}: "
            msg += f"expected token {self.expected_token}, "
            msg += f"observed token {self.observed_token}\n"
        if self.details:
            for d in self.details:
                msg += f"  {d}\n"
        return msg


# ---------------------------------------------------------------------------
# Diff result
# ---------------------------------------------------------------------------

@dataclass
class DiffResult:
    """Result of comparing two session traces."""
    identical: bool
    total_turns_a: int
    total_turns_b: int
    first_mismatch_turn: int | None = None
    first_mismatch_step: int | None = None
    mismatch_type: str = ""  # "prompt", "input_tokens", "output_tokens", "step", "stop_reason"
    expected: Any = None
    observed: Any = None
    details: list[str] | None = None

    def __str__(self) -> str:
        if self.identical:
            return (
                f"Trace comparison: IDENTICAL\n"
                f"  {self.total_turns_a} turns matched perfectly"
            )
        msg = f"Trace comparison: DIFFERENT\n"
        if self.first_mismatch_turn is not None:
            msg += f"  First mismatch: turn {self.first_mismatch_turn}\n"
            msg += f"  Type: {self.mismatch_type}\n"
        if self.first_mismatch_step is not None:
            msg += f"  Step: {self.first_mismatch_step}\n"
        if self.expected is not None:
            msg += f"  Expected: {self.expected}\n"
            msg += f"  Observed: {self.observed}\n"
        if self.details:
            for d in self.details:
                msg += f"  {d}\n"
        return msg


# ---------------------------------------------------------------------------
# Replay: re-run session and verify
# ---------------------------------------------------------------------------

def replay_session(
    trace_path: str,
    model_name: str | None = None,
    strict: bool = False,
) -> ReplayResult:
    """Replay a saved session and verify determinism.

    Loads the session trace, re-runs all turns with the same config,
    and compares output tokens turn-by-turn.

    Args:
        trace_path: Path to the session JSON file.
        model_name: Override model name (uses trace model if None).
        strict: If True, verify every generation step, not just final tokens.

    Returns:
        ReplayResult with verification outcome.
    """
    from detinfer.agent.runtime import DeterministicAgent

    # Load original trace
    original = SessionTrace.from_json(trace_path)

    # Use model from trace or override
    replay_model = model_name or original.model
    if not replay_model:
        return ReplayResult(
            passed=False, total_turns=0, verified_turns=0,
            failure_reason="No model specified in trace or arguments",
        )

    # Extract user messages
    user_messages = [m["content"] for m in original.messages if m["role"] == "user"]
    total_turns = len(user_messages)

    if total_turns == 0:
        return ReplayResult(passed=True, total_turns=0, verified_turns=0)

    # Create agent with same config
    max_tokens = original.generation_config.get("max_new_tokens", 256)
    agent = DeterministicAgent(
        model_name=replay_model,
        seed=original.seed,
        max_new_tokens=max_tokens,
        trace_mode=original.trace_mode,
        quantize=original.quantization.get("mode"),
    )

    details = []
    verified = 0

    for i, user_msg in enumerate(user_messages):
        turn_num = i + 1

        # Find original generation for this turn
        orig_gen = None
        for g in original.generations:
            if g.turn == turn_num:
                orig_gen = g
                break

        if orig_gen is None:
            return ReplayResult(
                passed=False, total_turns=total_turns, verified_turns=verified,
                failure_turn=turn_num,
                failure_reason=f"No generation trace found for turn {turn_num}",
            )

        # Run the turn
        _ = agent.chat(user_msg)

        # Get replay generation
        replay_gen = agent.session.generations[-1]

        # Check prompt hash
        if orig_gen.prompt_hash and replay_gen.prompt_hash != orig_gen.prompt_hash:
            return ReplayResult(
                passed=False, total_turns=total_turns, verified_turns=verified,
                failure_turn=turn_num,
                failure_reason="Prompt hash mismatch",
                details=[
                    f"Expected prompt hash: {orig_gen.prompt_hash[:16]}...",
                    f"Observed prompt hash: {replay_gen.prompt_hash[:16]}...",
                    "Generation was not attempted",
                ],
            )

        # Check input tokens
        if orig_gen.input_tokens and replay_gen.input_tokens != orig_gen.input_tokens:
            return ReplayResult(
                passed=False, total_turns=total_turns, verified_turns=verified,
                failure_turn=turn_num,
                failure_reason="Input token mismatch",
                details=[
                    f"Expected {len(orig_gen.input_tokens)} input tokens",
                    f"Observed {len(replay_gen.input_tokens)} input tokens",
                ],
            )

        # Check output tokens
        if replay_gen.output_tokens != orig_gen.output_tokens:
            # Find first differing token
            for step_idx in range(min(len(orig_gen.output_tokens), len(replay_gen.output_tokens))):
                if replay_gen.output_tokens[step_idx] != orig_gen.output_tokens[step_idx]:
                    prev_matched = step_idx
                    return ReplayResult(
                        passed=False, total_turns=total_turns, verified_turns=verified,
                        failure_turn=turn_num, failure_step=step_idx,
                        failure_reason="Output token mismatch",
                        expected_token=orig_gen.output_tokens[step_idx],
                        observed_token=replay_gen.output_tokens[step_idx],
                        details=[f"Previous {prev_matched} steps matched exactly"],
                    )

            # Length mismatch
            return ReplayResult(
                passed=False, total_turns=total_turns, verified_turns=verified,
                failure_turn=turn_num,
                failure_reason="Output length mismatch",
                details=[
                    f"Expected {len(orig_gen.output_tokens)} tokens",
                    f"Observed {len(replay_gen.output_tokens)} tokens",
                ],
            )

        # Check stop reason
        if orig_gen.stop_reason and replay_gen.stop_reason != orig_gen.stop_reason:
            return ReplayResult(
                passed=False, total_turns=total_turns, verified_turns=verified,
                failure_turn=turn_num,
                failure_reason=f"Stop reason mismatch: expected '{orig_gen.stop_reason}', got '{replay_gen.stop_reason}'",
            )

        # Strict mode: check every step
        if strict and orig_gen.steps:
            for step_idx, (orig_step, replay_step) in enumerate(
                zip(orig_gen.steps, replay_gen.steps)
            ):
                if replay_step.chosen_token != orig_step.chosen_token:
                    return ReplayResult(
                        passed=False, total_turns=total_turns, verified_turns=verified,
                        failure_turn=turn_num, failure_step=step_idx,
                        failure_reason="Step-level token mismatch",
                        expected_token=orig_step.chosen_token,
                        observed_token=replay_step.chosen_token,
                    )

        verified += 1
        details.append(f"Turn {turn_num}: ✓ ({len(replay_gen.output_tokens)} tokens)")

    return ReplayResult(
        passed=True, total_turns=total_turns, verified_turns=verified,
        details=details,
    )


# ---------------------------------------------------------------------------
# Diff: compare two session traces
# ---------------------------------------------------------------------------

def diff_sessions(path_a: str, path_b: str) -> DiffResult:
    """Compare two session traces token-by-token.

    Reports the first point of divergence between two traces.

    Args:
        path_a: Path to first session JSON.
        path_b: Path to second session JSON.

    Returns:
        DiffResult with comparison outcome.
    """
    trace_a = SessionTrace.from_json(path_a)
    trace_b = SessionTrace.from_json(path_b)

    details = []

    # Check model
    if trace_a.model != trace_b.model:
        details.append(f"Model: {trace_a.model} vs {trace_b.model}")

    # Check seed
    if trace_a.seed != trace_b.seed:
        details.append(f"Seed: {trace_a.seed} vs {trace_b.seed}")

    # Check generation config
    if trace_a.generation_config != trace_b.generation_config:
        details.append("Generation config differs")

    # Compare generations turn by turn
    min_turns = min(len(trace_a.generations), len(trace_b.generations))

    for i in range(min_turns):
        gen_a = trace_a.generations[i]
        gen_b = trace_b.generations[i]
        turn_num = i + 1

        # Prompt hash
        if gen_a.prompt_hash and gen_b.prompt_hash and gen_a.prompt_hash != gen_b.prompt_hash:
            return DiffResult(
                identical=False,
                total_turns_a=len(trace_a.generations),
                total_turns_b=len(trace_b.generations),
                first_mismatch_turn=turn_num,
                mismatch_type="prompt",
                expected=gen_a.prompt_hash[:16] + "...",
                observed=gen_b.prompt_hash[:16] + "...",
                details=details,
            )

        # Input tokens
        if gen_a.input_tokens and gen_b.input_tokens and gen_a.input_tokens != gen_b.input_tokens:
            return DiffResult(
                identical=False,
                total_turns_a=len(trace_a.generations),
                total_turns_b=len(trace_b.generations),
                first_mismatch_turn=turn_num,
                mismatch_type="input_tokens",
                expected=f"{len(gen_a.input_tokens)} tokens",
                observed=f"{len(gen_b.input_tokens)} tokens",
                details=details,
            )

        # Output tokens
        if gen_a.output_tokens != gen_b.output_tokens:
            # Find first mismatch
            for step_idx in range(min(len(gen_a.output_tokens), len(gen_b.output_tokens))):
                if gen_a.output_tokens[step_idx] != gen_b.output_tokens[step_idx]:
                    return DiffResult(
                        identical=False,
                        total_turns_a=len(trace_a.generations),
                        total_turns_b=len(trace_b.generations),
                        first_mismatch_turn=turn_num,
                        first_mismatch_step=step_idx,
                        mismatch_type="output_tokens",
                        expected=gen_a.output_tokens[step_idx],
                        observed=gen_b.output_tokens[step_idx],
                        details=details + [f"Previous {step_idx} steps matched"],
                    )

            return DiffResult(
                identical=False,
                total_turns_a=len(trace_a.generations),
                total_turns_b=len(trace_b.generations),
                first_mismatch_turn=turn_num,
                mismatch_type="output_length",
                expected=len(gen_a.output_tokens),
                observed=len(gen_b.output_tokens),
                details=details,
            )

        # Stop reason
        if gen_a.stop_reason and gen_b.stop_reason and gen_a.stop_reason != gen_b.stop_reason:
            return DiffResult(
                identical=False,
                total_turns_a=len(trace_a.generations),
                total_turns_b=len(trace_b.generations),
                first_mismatch_turn=turn_num,
                mismatch_type="stop_reason",
                expected=gen_a.stop_reason,
                observed=gen_b.stop_reason,
                details=details,
            )

    # Check turn count
    if len(trace_a.generations) != len(trace_b.generations):
        return DiffResult(
            identical=False,
            total_turns_a=len(trace_a.generations),
            total_turns_b=len(trace_b.generations),
            mismatch_type="turn_count",
            expected=len(trace_a.generations),
            observed=len(trace_b.generations),
            details=details + [f"Matched {min_turns} turns before length difference"],
        )

    # Compare agent steps (tool calls, reasoning)
    min_steps = min(len(trace_a.agent_steps), len(trace_b.agent_steps))
    for i in range(min_steps):
        step_a = trace_a.agent_steps[i]
        step_b = trace_b.agent_steps[i]

        if step_a.type != step_b.type:
            return DiffResult(
                identical=False,
                total_turns_a=len(trace_a.generations),
                total_turns_b=len(trace_b.generations),
                first_mismatch_turn=step_a.turn,
                first_mismatch_step=step_a.step,
                mismatch_type="agent_step_type",
                expected=step_a.type,
                observed=step_b.type,
                details=details + [f"Agent step {i+1}: expected {step_a.type}, got {step_b.type}"],
            )

        if step_a.type == "tool_call":
            if step_a.tool != step_b.tool:
                return DiffResult(
                    identical=False,
                    total_turns_a=len(trace_a.generations),
                    total_turns_b=len(trace_b.generations),
                    first_mismatch_turn=step_a.turn,
                    first_mismatch_step=step_a.step,
                    mismatch_type="tool_name",
                    expected=step_a.tool,
                    observed=step_b.tool,
                    details=details + [f"Expected tool: {step_a.tool}, Observed tool: {step_b.tool}"],
                )
            if step_a.arguments != step_b.arguments:
                return DiffResult(
                    identical=False,
                    total_turns_a=len(trace_a.generations),
                    total_turns_b=len(trace_b.generations),
                    first_mismatch_turn=step_a.turn,
                    first_mismatch_step=step_a.step,
                    mismatch_type="tool_arguments",
                    expected=str(step_a.arguments),
                    observed=str(step_b.arguments),
                    details=details + [f"Tool '{step_a.tool}' arguments differ"],
                )

        if step_a.type == "tool_result":
            if step_a.result != step_b.result:
                return DiffResult(
                    identical=False,
                    total_turns_a=len(trace_a.generations),
                    total_turns_b=len(trace_b.generations),
                    first_mismatch_turn=step_a.turn,
                    first_mismatch_step=step_a.step,
                    mismatch_type="tool_result",
                    expected=step_a.result[:100],
                    observed=step_b.result[:100],
                    details=details + [f"Tool '{step_a.tool}' result differs"],
                )

    if len(trace_a.agent_steps) != len(trace_b.agent_steps):
        return DiffResult(
            identical=False,
            total_turns_a=len(trace_a.generations),
            total_turns_b=len(trace_b.generations),
            mismatch_type="agent_step_count",
            expected=len(trace_a.agent_steps),
            observed=len(trace_b.agent_steps),
            details=details + [f"Matched {min_steps} agent steps before count difference"],
        )

    return DiffResult(
        identical=True,
        total_turns_a=len(trace_a.generations),
        total_turns_b=len(trace_b.generations),
        details=details or None,
    )
