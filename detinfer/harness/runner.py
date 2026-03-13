"""
detinfer.harness.runner -- Automated agent harness runner.

Runs task definitions through the DeterministicAgent, records
full session traces, and optionally compares against baseline.

Uses the existing SessionTrace format — same trace, same replay,
same diff/check logic. No separate harness trace format.

Usage:
    from detinfer.harness import HarnessRunner, load_task

    task = load_task("task.json")
    runner = HarnessRunner()
    result = runner.run_task(task)
    print(result)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from detinfer.harness.task_schema import TaskDefinition, ToolDefinition


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """Result of running a single harness task."""
    name: str
    passed: bool
    status: str  # "passed", "failed", "error", "match_failed", "drift"
    output: str = ""
    expected_value: str = ""
    match_mode: str = ""
    trace_path: str = ""
    session_hash: str = ""
    duration_ms: float = 0.0
    turns_executed: int = 0
    error: str = ""
    drift_type: str = ""  # Only set if --against was used

    # Task metadata stored in the trace
    task_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "passed": self.passed,
            "status": self.status,
            "output": self.output[:500],  # Truncate for readability
            "session_hash": self.session_hash,
            "duration_ms": round(self.duration_ms, 2),
            "turns_executed": self.turns_executed,
        }
        if self.trace_path:
            d["trace_path"] = self.trace_path
        if self.expected_value:
            d["expected_value"] = self.expected_value
            d["match_mode"] = self.match_mode
        if self.error:
            d["error"] = self.error
        if self.drift_type:
            d["drift_type"] = self.drift_type
        return d


@dataclass
class SuiteResult:
    """Result of running a task suite."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    results: list[TaskResult] = field(default_factory=list)
    duration_ms: float = 0.0
    manifest_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "duration_ms": round(self.duration_ms, 2),
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# HarnessRunner
# ---------------------------------------------------------------------------

class HarnessRunner:
    """Automated harness runner for deterministic agent tasks.

    Loads task definitions, creates a DeterministicAgent per task,
    runs the agent loop, collects results, and exports traces.

    Reuses the existing SessionTrace format — all replay, diff,
    and check tooling works on harness output.

    Args:
        output_dir: Directory for trace output files.
        against: Optional baseline trace path for comparison.
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        against: str | None = None,
    ):
        self.output_dir = Path(output_dir) if output_dir else None
        self.against = against

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_task(self, task: TaskDefinition) -> TaskResult:
        """Run a single task and return the result.

        Steps:
            1. Create DeterministicAgent with task config
            2. Register mock tools (if any)
            3. Send prompt(s)
            4. Check expected output (if defined)
            5. Export trace
            6. Compare against baseline (if provided)

        Args:
            task: Task definition to run.

        Returns:
            TaskResult with pass/fail, output, trace path, timing.
        """
        start = time.perf_counter()

        try:
            return self._execute_task(task, start)
        except Exception as e:
            return TaskResult(
                name=task.name,
                passed=False,
                status="error",
                error=str(e),
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def _execute_task(self, task: TaskDefinition, start: float) -> TaskResult:
        """Core task execution logic."""
        from detinfer.agent.runtime import DeterministicAgent

        # 1. Create agent
        max_tokens = task.generation_config.get("max_new_tokens", task.max_tokens)
        agent = DeterministicAgent(
            model_name=task.model,
            seed=task.seed,
            max_new_tokens=max_tokens,
            trace_mode=task.trace_mode,
            quantize=task.quantize,
            device=task.device,
            system_prompt=task.system_prompt,
            max_context_tokens=task.max_context_tokens,
        )

        # 2. Register mock tools
        for tool_def in task.tools:
            _mock = tool_def.mock_result
            def _make_tool(result: str):
                return lambda **kwargs: result
            agent.register_tool(tool_def.name, _make_tool(_mock))

        # 3. Run prompts
        responses = []

        # First prompt
        response = agent.chat(task.prompt)
        responses.append(response)

        # Follow-up prompts (for multi-turn tasks)
        for follow_up in task.follow_ups[:task.max_turns - 1]:
            response = agent.chat(follow_up)
            responses.append(response)

        # 4. Compute result
        final_output = responses[-1] if responses else ""
        turns_executed = len(responses)

        # Add task metadata to session trace
        # (stored in agent_steps as a checkpoint)
        from detinfer.agent.trace import AgentStep
        agent._agent_step_counter += 1
        agent.session.add_agent_step(AgentStep(
            step=agent._agent_step_counter,
            type="checkpoint",
            turn=agent._turn_count,
            checkpoint_data={
                "harness_task": task.name,
                "harness_version": 1,
            },
        ))
        session_hash = agent.get_session_hash()

        # 5. Export trace
        trace_path = ""
        if self.output_dir:
            trace_file = self.output_dir / f"{task.name}_trace.json"
            agent.export_session(str(trace_file))
            trace_path = str(trace_file)

        # 6. Check expected output
        passed = True
        status = "passed"
        expected_value = ""
        match_mode = ""

        if task.expected:
            expected_value = task.expected.value
            match_mode = task.expected.match
            if not task.expected.check(final_output):
                passed = False
                status = "match_failed"

        # 7. Compare against baseline
        drift_type = ""
        if self.against and passed:
            drift_type, baseline_passed = self._compare_against(
                agent.session.to_dict(), self.against
            )
            if not baseline_passed:
                passed = False
                status = "drift"

        duration = (time.perf_counter() - start) * 1000

        return TaskResult(
            name=task.name,
            passed=passed,
            status=status,
            output=final_output,
            expected_value=expected_value,
            match_mode=match_mode,
            trace_path=trace_path,
            session_hash=session_hash,
            duration_ms=duration,
            turns_executed=turns_executed,
            drift_type=drift_type,
            task_metadata={"model": task.model, "seed": task.seed},
        )

    def _compare_against(
        self, candidate: dict, baseline_path: str,
    ) -> tuple[str, bool]:
        """Compare candidate trace against a baseline.

        Uses the existing check_sessions logic for consistency.

        Returns:
            (drift_type, passed) tuple.
        """
        from detinfer.check import check_sessions

        # Load baseline
        baseline_p = Path(baseline_path)
        if baseline_p.suffix == ".gz":
            import gzip
            with gzip.open(baseline_p, "rt", encoding="utf-8") as f:
                baseline = json.load(f)
        else:
            with open(baseline_p, "r", encoding="utf-8") as f:
                baseline = json.load(f)

        report = check_sessions(baseline, candidate)
        if report.status == "pass":
            return "", True
        return report.primary_type or "UNKNOWN", False

    def run_suite(
        self,
        tasks: list[TaskDefinition],
        fail_fast: bool = False,
    ) -> SuiteResult:
        """Run a suite of tasks and collect results.

        Args:
            tasks: List of task definitions.
            fail_fast: Stop on first failure.

        Returns:
            SuiteResult with summary stats and per-task results.
        """
        start = time.perf_counter()
        suite = SuiteResult(total=len(tasks))

        for task in tasks:
            result = self.run_task(task)
            suite.results.append(result)

            if result.passed:
                suite.passed += 1
            elif result.status == "error":
                suite.errors += 1
                suite.failed += 1
            else:
                suite.failed += 1

            if fail_fast and not result.passed:
                break

        suite.duration_ms = (time.perf_counter() - start) * 1000

        # Write manifest
        if self.output_dir:
            manifest_path = self.output_dir / "manifest.json"
            manifest = {
                "summary": {
                    "total": suite.total,
                    "passed": suite.passed,
                    "failed": suite.failed,
                    "errors": suite.errors,
                    "duration_ms": round(suite.duration_ms, 2),
                },
                "results": [r.to_dict() for r in suite.results],
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            suite.manifest_path = str(manifest_path)

        return suite


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

def render_task_result(result: TaskResult) -> str:
    """Render a single task result as human-readable text."""
    icon = "✓" if result.passed else "✗"
    line = f"  {icon} {result.name}"
    if result.duration_ms:
        line += f" ({result.duration_ms:.0f}ms)"
    if not result.passed:
        line += f" [{result.status}]"
        if result.error:
            line += f" — {result.error[:80]}"
        if result.drift_type:
            line += f" — {result.drift_type}"
    return line


def render_suite_result(suite: SuiteResult) -> str:
    """Render full suite result as human-readable text."""
    lines = [
        "Agent Harness Report",
        "=" * 20,
        "",
    ]
    for result in suite.results:
        lines.append(render_task_result(result))

    lines.append("")
    lines.append(f"Total: {suite.total}  |  "
                 f"Passed: {suite.passed}  |  "
                 f"Failed: {suite.failed}  |  "
                 f"Errors: {suite.errors}")
    lines.append(f"Duration: {suite.duration_ms:.0f}ms")

    if suite.manifest_path:
        lines.append(f"Manifest: {suite.manifest_path}")

    return "\n".join(lines)
