"""
Tests for detinfer.harness -- Task schema, runner, and suite.

All tests work WITHOUT a real model by mocking the DeterministicAgent.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

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


# ===========================================================================
# Task schema tests
# ===========================================================================


class TestToolDefinition:
    def test_from_dict_full(self):
        d = {"name": "calc", "description": "Math", "mock_result": "42"}
        tool = ToolDefinition.from_dict(d)
        assert tool.name == "calc"
        assert tool.description == "Math"
        assert tool.mock_result == "42"

    def test_from_dict_minimal(self):
        d = {"name": "ping"}
        tool = ToolDefinition.from_dict(d)
        assert tool.name == "ping"
        assert tool.description == ""
        assert tool.mock_result == ""

    def test_to_dict(self):
        tool = ToolDefinition(name="calc", description="Math", mock_result="42")
        d = tool.to_dict()
        assert d["name"] == "calc"
        assert d["description"] == "Math"
        assert d["mock_result"] == "42"

    def test_to_dict_minimal(self):
        tool = ToolDefinition(name="ping")
        d = tool.to_dict()
        assert d == {"name": "ping"}


class TestExpectedMatch:
    def test_exact_match(self):
        m = ExpectedMatch(match="exact", value="hello")
        assert m.check("hello") is True
        assert m.check("  hello  ") is True  # strip
        assert m.check("hello world") is False

    def test_contains_match(self):
        m = ExpectedMatch(match="contains", value="world")
        assert m.check("hello world") is True
        assert m.check("hi there") is False

    def test_regex_match(self):
        m = ExpectedMatch(match="regex", value=r"\d+")
        assert m.check("answer is 42") is True
        assert m.check("no numbers") is False

    def test_from_dict_none(self):
        assert ExpectedMatch.from_dict(None) is None

    def test_from_dict(self):
        m = ExpectedMatch.from_dict({"match": "exact", "value": "test"})
        assert m.match == "exact"
        assert m.value == "test"

    def test_to_dict(self):
        m = ExpectedMatch(match="contains", value="hi")
        d = m.to_dict()
        assert d == {"match": "contains", "value": "hi"}


class TestTaskDefinition:
    def test_from_dict_full(self):
        data = {
            "name": "test_task",
            "model": "test-model",
            "seed": 99,
            "prompt": "Hello",
            "system_prompt": "Be nice",
            "max_turns": 3,
            "max_tokens": 128,
            "trace_mode": "verbose",
            "generation_config": {"max_new_tokens": 128, "do_sample": False},
            "tools": [{"name": "calc", "mock_result": "4"}],
            "expected": {"match": "contains", "value": "4"},
            "follow_ups": ["More?"],
            "description": "Test",
            "tags": ["test"],
        }
        task = TaskDefinition.from_dict(data)
        assert task.name == "test_task"
        assert task.model == "test-model"
        assert task.seed == 99
        assert task.prompt == "Hello"
        assert task.system_prompt == "Be nice"
        assert task.max_turns == 3
        assert task.trace_mode == "verbose"
        assert len(task.tools) == 1
        assert task.tools[0].name == "calc"
        assert task.expected is not None
        assert task.expected.value == "4"
        assert task.follow_ups == ["More?"]
        assert task.description == "Test"
        assert task.tags == ["test"]

    def test_from_dict_defaults(self):
        data = {"name": "minimal", "prompt": "Hi"}
        task = TaskDefinition.from_dict(data)
        assert task.seed == 42
        assert task.max_turns == 1
        assert task.max_tokens == 256
        assert task.trace_mode == "standard"
        assert task.tools == []
        assert task.expected is None

    def test_to_dict_roundtrip(self):
        data = {
            "name": "roundtrip",
            "model": "test-model",
            "seed": 42,
            "prompt": "Hello",
            "max_turns": 1,
            "max_tokens": 256,
            "generation_config": {
                "max_new_tokens": 256,
                "do_sample": False,
                "temperature": 0.0,
            },
        }
        task = TaskDefinition.from_dict(data)
        d = task.to_dict()
        assert d["name"] == "roundtrip"
        assert d["prompt"] == "Hello"
        assert d["generation_config"]["do_sample"] is False

    def test_validate_ok(self):
        task = TaskDefinition(name="ok", prompt="Hi", model="test-model")
        assert task.validate() == []

    def test_validate_missing_name(self):
        task = TaskDefinition(name="", prompt="Hi")
        errors = task.validate()
        assert any("name" in e.lower() for e in errors)

    def test_validate_missing_prompt(self):
        task = TaskDefinition(name="test", prompt="")
        errors = task.validate()
        assert any("prompt" in e.lower() for e in errors)

    def test_validate_bad_match_mode(self):
        task = TaskDefinition(
            name="test", prompt="Hi",
            expected=ExpectedMatch(match="fuzzy", value="x"),
        )
        errors = task.validate()
        assert any("match mode" in e.lower() for e in errors)

    def test_validate_bad_regex(self):
        task = TaskDefinition(
            name="test", prompt="Hi",
            expected=ExpectedMatch(match="regex", value="[invalid"),
        )
        errors = task.validate()
        assert any("regex" in e.lower() for e in errors)

    def test_validate_bad_max_turns(self):
        task = TaskDefinition(name="test", prompt="Hi", max_turns=0)
        errors = task.validate()
        assert any("max_turns" in e.lower() for e in errors)

    def test_validate_missing_model(self):
        task = TaskDefinition(name="test", prompt="Hi", model="")
        errors = task.validate()
        assert any("model" in e.lower() for e in errors)


class TestLoadTask:
    def test_load_valid_task(self, tmp_path):
        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps({
            "name": "test",
            "prompt": "Hello",
            "model": "test-model",
        }))
        task = load_task(str(task_file))
        assert task.name == "test"
        assert task.prompt == "Hello"

    def test_load_auto_name(self, tmp_path):
        task_file = tmp_path / "my_task.json"
        task_file.write_text(json.dumps({"prompt": "Hello", "model": "test-model"}))
        task = load_task(str(task_file))
        assert task.name == "my_task"

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_task("/nonexistent/task.json")

    def test_load_invalid_task(self, tmp_path):
        task_file = tmp_path / "bad.json"
        task_file.write_text(json.dumps({"name": "bad"}))  # no prompt
        with pytest.raises(ValueError, match="prompt"):
            load_task(str(task_file))


class TestLoadTaskSuite:
    def test_load_suite(self, tmp_path):
        for i in range(3):
            (tmp_path / f"task_{i}.json").write_text(json.dumps({
                "name": f"task_{i}",
                "prompt": f"Hello {i}",
                "model": "test-model",
            }))
        tasks = load_task_suite(str(tmp_path))
        assert len(tasks) == 3
        assert tasks[0].name == "task_0"

    def test_load_suite_skips_invalid(self, tmp_path):
        (tmp_path / "good.json").write_text(json.dumps({
            "name": "good", "prompt": "Hi", "model": "test-model",
        }))
        (tmp_path / "bad.json").write_text(json.dumps({"name": "bad"}))
        tasks = load_task_suite(str(tmp_path))
        assert len(tasks) == 1
        assert tasks[0].name == "good"

    def test_load_suite_missing_dir(self):
        with pytest.raises(FileNotFoundError):
            load_task_suite("/nonexistent/dir/")

    def test_load_suite_empty(self, tmp_path):
        tasks = load_task_suite(str(tmp_path))
        assert tasks == []


# ===========================================================================
# Result rendering tests
# ===========================================================================


class TestResultRendering:
    def test_render_task_passed(self):
        r = TaskResult(name="test", passed=True, status="passed", duration_ms=42.0)
        text = render_task_result(r)
        assert "✓" in text
        assert "test" in text

    def test_render_task_failed(self):
        r = TaskResult(name="test", passed=False, status="match_failed", error="bad output")
        text = render_task_result(r)
        assert "✗" in text
        assert "match_failed" in text

    def test_render_task_drift(self):
        r = TaskResult(name="test", passed=False, status="drift", drift_type="OUTPUT_DRIFT")
        text = render_task_result(r)
        assert "OUTPUT_DRIFT" in text

    def test_render_suite(self):
        results = [
            TaskResult(name="a", passed=True, status="passed"),
            TaskResult(name="b", passed=False, status="failed"),
        ]
        suite = SuiteResult(total=2, passed=1, failed=1, results=results)
        text = render_suite_result(suite)
        assert "Passed: 1" in text
        assert "Failed: 1" in text

    def test_task_result_to_dict(self):
        r = TaskResult(name="test", passed=True, status="passed",
                       output="hello", session_hash="abc123")
        d = r.to_dict()
        assert d["name"] == "test"
        assert d["passed"] is True
        assert d["session_hash"] == "abc123"

    def test_suite_result_to_dict(self):
        r = TaskResult(name="test", passed=True, status="passed")
        suite = SuiteResult(total=1, passed=1, failed=0, results=[r])
        d = suite.to_dict()
        assert d["total"] == 1
        assert len(d["results"]) == 1


# ===========================================================================
# Runner tests (mocked — no real model)
# ===========================================================================


def _make_mock_agent():
    """Create a mock DeterministicAgent."""
    agent = MagicMock()
    agent.chat.return_value = "The answer is 4."
    agent.get_session_hash.return_value = "fakehash123"
    agent._turn_count = 0
    agent._agent_step_counter = 0

    # Mock session
    session = MagicMock()
    session.to_dict.return_value = {
        "schema_version": "1",
        "trace_type": "agent",
        "model": "test-model",
        "seed": 42,
        "session_hash": "fakehash123",
        "generation_config": {},
        "tokenizer": {},
        "messages": [],
        "generations": [],
        "environment": {},
        "quantization": {},
        "agent_steps": [],
        "registered_tools": [],
    }
    agent.session = session
    agent.export_session.return_value = "fakehash123"

    return agent


class TestHarnessRunnerMocked:
    """Tests using mocked DeterministicAgent."""

    @patch("detinfer.harness.runner.HarnessRunner._execute_task")
    def test_run_task_passes(self, mock_exec):
        mock_exec.return_value = TaskResult(
            name="test", passed=True, status="passed",
            output="The answer is 4.", session_hash="abc",
        )
        runner = HarnessRunner()
        task = TaskDefinition(name="test", prompt="What is 2+2?")
        result = runner.run_task(task)
        assert result.passed is True
        assert result.status == "passed"

    @patch("detinfer.harness.runner.HarnessRunner._execute_task")
    def test_run_task_error_caught(self, mock_exec):
        mock_exec.side_effect = RuntimeError("model failed")
        runner = HarnessRunner()
        task = TaskDefinition(name="test", prompt="Hello")
        result = runner.run_task(task)
        assert result.passed is False
        assert result.status == "error"
        assert "model failed" in result.error

    def test_run_suite(self):
        runner = HarnessRunner()
        results = [
            TaskResult(name="a", passed=True, status="passed"),
            TaskResult(name="b", passed=False, status="failed"),
            TaskResult(name="c", passed=True, status="passed"),
        ]

        # Mock run_task to return pre-built results
        runner.run_task = MagicMock(side_effect=results)
        tasks = [
            TaskDefinition(name="a", prompt="1"),
            TaskDefinition(name="b", prompt="2"),
            TaskDefinition(name="c", prompt="3"),
        ]
        suite = runner.run_suite(tasks)
        assert suite.total == 3
        assert suite.passed == 2
        assert suite.failed == 1

    def test_run_suite_fail_fast(self):
        runner = HarnessRunner()
        results = [
            TaskResult(name="a", passed=True, status="passed"),
            TaskResult(name="b", passed=False, status="failed"),
        ]
        runner.run_task = MagicMock(side_effect=results)
        tasks = [
            TaskDefinition(name="a", prompt="1"),
            TaskDefinition(name="b", prompt="2"),
            TaskDefinition(name="c", prompt="3"),
        ]
        suite = runner.run_suite(tasks, fail_fast=True)
        assert suite.total == 3
        assert suite.passed == 1
        assert suite.failed == 1
        assert len(suite.results) == 2  # stopped after b

    def test_suite_manifest_output(self, tmp_path):
        runner = HarnessRunner(output_dir=str(tmp_path))
        results = [
            TaskResult(name="a", passed=True, status="passed"),
        ]
        runner.run_task = MagicMock(side_effect=results)
        tasks = [TaskDefinition(name="a", prompt="1")]
        suite = runner.run_suite(tasks)
        assert suite.manifest_path
        manifest = json.loads(Path(suite.manifest_path).read_text())
        assert manifest["summary"]["total"] == 1
        assert manifest["summary"]["passed"] == 1


# ===========================================================================
# Expected match integration tests
# ===========================================================================


class TestExpectedMatchEdgeCases:
    def test_empty_value_contains(self):
        m = ExpectedMatch(match="contains", value="")
        assert m.check("anything") is True  # empty string is in everything

    def test_empty_output_contains(self):
        m = ExpectedMatch(match="contains", value="hello")
        assert m.check("") is False

    def test_regex_case_insensitive(self):
        m = ExpectedMatch(match="regex", value=r"(?i)hello")
        assert m.check("HELLO world") is True

    def test_invalid_match_mode(self):
        m = ExpectedMatch(match="fuzzy", value="x")
        assert m.check("x") is False  # returns False for unknown mode


# ===========================================================================
# Example task loading tests
# ===========================================================================

class TestExampleTasks:
    """Test that the example task files are valid."""

    @pytest.fixture
    def examples_dir(self):
        root = Path(__file__).parent.parent / "examples"
        if not root.exists():
            pytest.skip("examples/ directory not found")
        return root

    def test_all_examples_load(self, examples_dir):
        tasks = load_task_suite(str(examples_dir))
        assert len(tasks) >= 3, f"Expected at least 3 example tasks, got {len(tasks)}"

    def test_all_examples_validate(self, examples_dir):
        tasks = load_task_suite(str(examples_dir))
        for task in tasks:
            errors = task.validate()
            assert errors == [], f"Task '{task.name}' has errors: {errors}"

    def test_basic_math_example(self, examples_dir):
        task = load_task(str(examples_dir / "basic_math.json"))
        assert task.name == "basic_math"
        assert task.expected is not None
        assert task.expected.match == "contains"

    def test_tool_calculator_example(self, examples_dir):
        task = load_task(str(examples_dir / "tool_calculator.json"))
        assert task.name == "tool_calculator"
        assert len(task.tools) == 1
        assert task.tools[0].name == "calculator"
        assert task.tools[0].mock_result == "15"

    def test_multi_turn_example(self, examples_dir):
        task = load_task(str(examples_dir / "multi_turn_math.json"))
        assert task.max_turns == 2
        assert len(task.follow_ups) == 1
