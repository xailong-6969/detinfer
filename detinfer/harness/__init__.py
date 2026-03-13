"""
detinfer.harness -- Agent Harness for Deterministic Task Execution

Automated harness that loads task definitions, runs the agent loop,
collects structured results, and exports traces.

Usage:
    from detinfer.harness import HarnessRunner, load_task

    task = load_task("task.json")
    runner = HarnessRunner(output_dir="runs/")
    result = runner.run_task(task)
    print(result.status)  # "passed" or "failed"
"""

from detinfer.harness.task_schema import (
    TaskDefinition,
    ToolDefinition,
    ExpectedMatch,
    load_task,
    load_task_suite,
)
from detinfer.harness.runner import (
    HarnessRunner,
    TaskResult,
    SuiteResult,
    render_task_result,
    render_suite_result,
)

__all__ = [
    "TaskDefinition",
    "ToolDefinition",
    "ExpectedMatch",
    "load_task",
    "load_task_suite",
    "HarnessRunner",
    "TaskResult",
    "SuiteResult",
    "render_task_result",
    "render_suite_result",
]
