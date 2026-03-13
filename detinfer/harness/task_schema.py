"""
detinfer.harness.task_schema -- Task definition loading and validation.

Defines the schema for agent-run task files and provides loaders
for single tasks and task suites (directories).

Example task.json:
    {
      "name": "basic_math",
      "model": "<hf-model>",
      "seed": 42,
      "prompt": "What is 2+2?",
      "max_turns": 1,
      "generation_config": {"max_new_tokens": 64, "do_sample": false},
      "tools": [{"name": "calculator", "mock_result": "4"}],
      "expected": {"match": "contains", "value": "4"}
    }
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@dataclass
class ToolDefinition:
    """A tool available to the agent during a harness run.

    For deterministic replay, tools use mock_result instead of
    real execution. This ensures the harness produces identical
    traces regardless of external state.

    Args:
        name: Tool name (e.g., "calculator", "send_message").
        description: Human-readable description of the tool.
        mock_result: Fixed result to return when called (deterministic).
    """
    name: str
    description: str = ""
    mock_result: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "ToolDefinition":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            mock_result=data.get("mock_result", ""),
        )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"name": self.name}
        if self.description:
            d["description"] = self.description
        if self.mock_result:
            d["mock_result"] = self.mock_result
        return d


# ---------------------------------------------------------------------------
# Expected match
# ---------------------------------------------------------------------------

@dataclass
class ExpectedMatch:
    """Expected output matcher for a task.

    Supports three match modes:
        exact:    output must equal value exactly (after strip)
        contains: output must contain value as substring
        regex:    output must match the regex pattern

    Args:
        match: Match mode ("exact", "contains", "regex").
        value: String to match against.
    """
    match: str = "contains"   # "exact", "contains", "regex"
    value: str = ""

    def check(self, output: str) -> bool:
        """Check if output matches the expected value."""
        if self.match == "exact":
            return output.strip() == self.value.strip()
        elif self.match == "contains":
            return self.value in output
        elif self.match == "regex":
            return bool(re.search(self.value, output))
        return False

    @classmethod
    def from_dict(cls, data: dict | None) -> "ExpectedMatch | None":
        if data is None:
            return None
        return cls(
            match=data.get("match", "contains"),
            value=data.get("value", ""),
        )

    def to_dict(self) -> dict:
        return {"match": self.match, "value": self.value}


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

@dataclass
class TaskDefinition:
    """A single harness task definition.

    Loaded from a JSON file, defines everything needed to run
    a deterministic agent task: model, prompt, tools, and
    expected output.
    """
    name: str
    model: str = ""
    seed: int = 42
    prompt: str = ""
    system_prompt: str | None = None
    max_turns: int = 1
    max_tokens: int = 256
    max_context_tokens: int | None = None
    trace_mode: str = "standard"
    quantize: str | None = None
    device: str | None = None

    generation_config: dict = field(default_factory=lambda: {
        "max_new_tokens": 256,
        "do_sample": False,
        "temperature": 0.0,
    })

    tools: list[ToolDefinition] = field(default_factory=list)
    expected: ExpectedMatch | None = None

    # Follow-up prompts for multi-turn tasks
    follow_ups: list[str] = field(default_factory=list)

    # Metadata
    description: str = ""
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "TaskDefinition":
        """Load a TaskDefinition from a dict (parsed JSON)."""
        tools = [ToolDefinition.from_dict(t) for t in data.get("tools", [])]
        expected = ExpectedMatch.from_dict(data.get("expected"))

        gen_config = data.get("generation_config", {})
        max_tokens = data.get("max_tokens", 256)
        if "max_new_tokens" not in gen_config:
            gen_config["max_new_tokens"] = max_tokens
        gen_config.setdefault("do_sample", False)
        gen_config.setdefault("temperature", 0.0)

        return cls(
            name=data.get("name", "unnamed"),
            model=data.get("model", ""),
            seed=data.get("seed", 42),
            prompt=data.get("prompt", ""),
            system_prompt=data.get("system_prompt"),
            max_turns=data.get("max_turns", 1),
            max_tokens=max_tokens,
            max_context_tokens=data.get("max_context_tokens"),
            trace_mode=data.get("trace_mode", "standard"),
            quantize=data.get("quantize"),
            device=data.get("device"),
            generation_config=gen_config,
            tools=tools,
            expected=expected,
            follow_ups=data.get("follow_ups", []),
            description=data.get("description", ""),
            tags=data.get("tags", []),
        )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "name": self.name,
            "model": self.model,
            "seed": self.seed,
            "prompt": self.prompt,
            "max_turns": self.max_turns,
            "max_tokens": self.max_tokens,
            "trace_mode": self.trace_mode,
            "generation_config": self.generation_config,
        }
        if self.system_prompt:
            d["system_prompt"] = self.system_prompt
        if self.max_context_tokens is not None:
            d["max_context_tokens"] = self.max_context_tokens
        if self.quantize:
            d["quantize"] = self.quantize
        if self.device:
            d["device"] = self.device
        if self.tools:
            d["tools"] = [t.to_dict() for t in self.tools]
        if self.expected:
            d["expected"] = self.expected.to_dict()
        if self.follow_ups:
            d["follow_ups"] = self.follow_ups
        if self.description:
            d["description"] = self.description
        if self.tags:
            d["tags"] = self.tags
        return d

    def validate(self) -> list[str]:
        """Validate the task definition. Returns list of errors."""
        errors = []
        if not self.name:
            errors.append("Task name is required")
        if not self.prompt:
            errors.append("Task prompt is required")
        if not self.model:
            errors.append("Task model is required")
        if self.max_turns < 1:
            errors.append("max_turns must be >= 1")
        if self.max_tokens < 1:
            errors.append("max_tokens must be >= 1")
        if self.expected and self.expected.match not in ("exact", "contains", "regex"):
            errors.append(f"Invalid match mode: {self.expected.match}")
        if self.expected and self.expected.match == "regex":
            try:
                re.compile(self.expected.value)
            except re.error as e:
                errors.append(f"Invalid regex pattern: {e}")
        return errors


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_task(path: str | Path) -> TaskDefinition:
    """Load a single task definition from a JSON file.

    Args:
        path: Path to a task.json file.

    Returns:
        Parsed TaskDefinition.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the task is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Use filename as task name if not specified
    if "name" not in data:
        data["name"] = path.stem

    task = TaskDefinition.from_dict(data)
    errors = task.validate()
    if errors:
        raise ValueError(f"Invalid task '{task.name}': {'; '.join(errors)}")
    return task


def load_task_suite(dir_path: str | Path) -> list[TaskDefinition]:
    """Load all task JSON files from a directory.

    Loads all .json files from the directory, sorted by name.
    Skips files that fail validation (logs warning).

    Args:
        dir_path: Path to a directory of task JSON files.

    Returns:
        List of valid TaskDefinitions.

    Raises:
        FileNotFoundError: If the directory doesn't exist.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Task directory not found: {dir_path}")

    tasks = []
    for json_file in sorted(dir_path.glob("*.json")):
        try:
            task = load_task(json_file)
            tasks.append(task)
        except (ValueError, json.JSONDecodeError) as e:
            import sys
            print(f"⚠ Skipping {json_file.name}: {e}", file=sys.stderr)

    return tasks
