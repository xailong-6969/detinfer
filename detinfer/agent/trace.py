"""
detinfer.trace -- Token-level trace recording, hashing, and export/import.

The foundation module for deterministic session tracking.
Records every generation step, computes canonical session hashes,
and provides JSON serialization for replay and diff.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class TraceMode(str, Enum):
    """Trace detail levels.

    minimal  = proof only (CI, benchmarks, sharing)
    standard = replay/debugging (default for local use)
    verbose  = deep diagnosis (top-k tokens, full env)

    IMPORTANT: The canonical session hash is computed from the same
    core fields regardless of trace mode.  Debug extras (rendered_prompt,
    input_tokens list, per-step top-k, extended env) are NEVER part of
    the canonical hash.
    """
    MINIMAL = "minimal"
    STANDARD = "standard"
    VERBOSE = "verbose"


# ---------------------------------------------------------------------------
# Generation step (one token)
# ---------------------------------------------------------------------------

@dataclass
class GenerationStep:
    """A single generation step in the trace.

    Minimal mode: only step + chosen_token.
    Standard mode: + is_ambiguous flag.
    Verbose mode: + top_tokens, top_scores.
    """
    step: int
    chosen_token: int
    top_tokens: list[int] | None = None
    top_scores: list[float] | None = None
    is_ambiguous: bool = False  # True when top-2 logits are within epsilon

    def to_dict(self, mode: TraceMode = TraceMode.STANDARD, verbose: bool = False) -> dict:
        """Serialize step. verbose=True is legacy compat for mode=VERBOSE."""
        use_verbose = (mode == TraceMode.VERBOSE) or verbose
        d = {"step": self.step, "chosen_token": self.chosen_token}
        if self.is_ambiguous:
            d["is_ambiguous"] = True
        if use_verbose and self.top_tokens is not None:
            d["top_tokens"] = self.top_tokens
            if self.top_scores is not None:
                d["top_scores"] = self.top_scores
        return d


# ---------------------------------------------------------------------------
# Agent step (tool calls, reasoning steps)
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    """A single agent step in the workflow trace.

    Records tool calls, tool results, LLM generations,
    and checkpoints. Only serializable logical state is stored.
    No runtime objects (tensors, model refs, etc.).
    """
    step: int
    type: str  # "llm_generation", "tool_call", "tool_result", "checkpoint"
    turn: int = 0

    # For tool_call / tool_result
    tool: str = ""
    arguments: dict = field(default_factory=dict)
    result: str = ""

    # For llm_generation (references GenerationTrace by turn)
    generation_turn: int | None = None

    # For checkpoint
    checkpoint_data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"step": self.step, "type": self.type, "turn": self.turn}
        if self.type == "tool_call":
            d["tool"] = self.tool
            d["arguments"] = self.arguments
        elif self.type == "tool_result":
            d["tool"] = self.tool
            d["result"] = self.result
        elif self.type == "llm_generation":
            if self.generation_turn is not None:
                d["generation_turn"] = self.generation_turn
        elif self.type == "checkpoint":
            d["checkpoint_data"] = self.checkpoint_data
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "AgentStep":
        return cls(
            step=data["step"],
            type=data["type"],
            turn=data.get("turn", 0),
            tool=data.get("tool", ""),
            arguments=data.get("arguments", {}),
            result=data.get("result", ""),
            generation_turn=data.get("generation_turn"),
            checkpoint_data=data.get("checkpoint_data", {}),
        )


# ---------------------------------------------------------------------------
# Generation trace (one turn of generation)
# ---------------------------------------------------------------------------

@dataclass
class GenerationTrace:
    """Trace of a single generation turn.

    Records the rendered prompt, input/output tokens, generation steps,
    and stop reason. Supports both minimal and verbose trace modes.
    """
    turn: int
    rendered_prompt: str = ""
    prompt_hash: str = ""
    input_tokens: list[int] = field(default_factory=list)
    input_tokens_hash: str = ""
    output_tokens: list[int] = field(default_factory=list)
    output_tokens_hash: str = ""
    stop_reason: str = ""  # "eos" or "max_new_tokens"
    steps: list[GenerationStep] = field(default_factory=list)

    def add_step(
        self,
        step: int,
        chosen_token: int,
        top_tokens: list[int] | None = None,
        top_scores: list[float] | None = None,
        is_ambiguous: bool = False,
    ) -> None:
        """Record a generation step."""
        self.steps.append(GenerationStep(
            step=step,
            chosen_token=chosen_token,
            top_tokens=top_tokens,
            top_scores=top_scores,
            is_ambiguous=is_ambiguous,
        ))

    def finalize(self, eos_token_id: int | None = None) -> None:
        """Compute hashes and determine stop reason."""
        self.prompt_hash = _hash_string(self.rendered_prompt)
        self.input_tokens_hash = _hash_token_list(self.input_tokens)
        self.output_tokens_hash = _hash_token_list(self.output_tokens)

        if not self.stop_reason:
            if self.output_tokens and eos_token_id is not None and \
               self.output_tokens[-1] == eos_token_id:
                self.stop_reason = "eos"
            else:
                self.stop_reason = "max_new_tokens"

    def to_dict(self, mode: TraceMode = TraceMode.STANDARD, verbose: bool = False) -> dict:
        """Serialize generation trace according to trace mode.

        minimal:  prompt_hash, input_tokens_hash, output_tokens/hash, stop_reason
        standard: + rendered_prompt, input_tokens, steps (chosen_token only)
        verbose:  + steps with top_tokens/top_scores
        """
        effective = TraceMode.VERBOSE if verbose else mode

        # Core fields present in ALL modes (used for canonical hash)
        d: dict[str, Any] = {
            "turn": self.turn,
            "prompt_hash": self.prompt_hash,
            "input_tokens_hash": self.input_tokens_hash,
            "output_tokens": self.output_tokens,
            "output_tokens_hash": self.output_tokens_hash,
            "stop_reason": self.stop_reason,
        }

        # Standard adds rendered_prompt, input_tokens, and step trace
        if effective in (TraceMode.STANDARD, TraceMode.VERBOSE):
            d["rendered_prompt"] = self.rendered_prompt
            d["input_tokens"] = self.input_tokens
            d["steps"] = [s.to_dict(mode=effective) for s in self.steps]

        # Verbose adds top-k (already handled inside step.to_dict)
        # Nothing extra needed here — step.to_dict handles it

        return d


# ---------------------------------------------------------------------------
# Session trace (full multi-turn session)
# ---------------------------------------------------------------------------

@dataclass
class SessionTrace:
    """Full deterministic session trace.

    Contains everything needed to replay and verify a multi-turn
    conversation: model info, generation config, tokenizer fingerprint,
    messages, generation traces, and environment.
    """
    # Schema
    schema_version: str = "1"

    # Trace type: "inference" or "agent"
    trace_type: str = "inference"

    # Model
    model: str = ""
    model_hash: str = ""
    seed: int = 42

    # Generation config
    generation_config: dict = field(default_factory=lambda: {
        "do_sample": False,
        "temperature": 0.0,
        "top_p": None,
        "top_k": None,
        "max_new_tokens": 256,
    })

    # Tokenizer
    tokenizer_info: dict = field(default_factory=lambda: {
        "name": "",
        "vocab_size": 0,
        "tokenizer_hash": "",
        "chat_template_hash": "",
    })

    # Trace mode
    trace_mode: TraceMode = TraceMode.STANDARD

    # Messages (semantic)
    messages: list[dict] = field(default_factory=list)

    # Generations (execution)
    generations: list[GenerationTrace] = field(default_factory=list)

    # Environment
    environment: dict = field(default_factory=dict)

    # Session hash (computed)
    session_hash: str = ""

    # Quantization
    quantization: dict = field(default_factory=lambda: {
        "mode": None,
        "backend": None,
    })

    # Agent workflow steps (tool calls, reasoning)
    agent_steps: list[AgentStep] = field(default_factory=list)

    # Registered tools (names only, for verification)
    registered_tools: list[str] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content})

    def add_generation(self, trace: GenerationTrace) -> None:
        """Add a generation trace."""
        self.generations.append(trace)

    def add_agent_step(self, agent_step: AgentStep) -> None:
        """Add an agent workflow step (tool call, generation, etc.)."""
        self.agent_steps.append(agent_step)

    def compute_session_hash(self) -> str:
        """Compute deterministic session hash from CANONICAL fields only.

        The canonical hash is INDEPENDENT of trace mode. minimal,
        standard, and verbose traces of the same run produce the
        same session_hash.  This is a critical design invariant.

        Canonical fields:
            schema_version, model, model_hash, seed,
            generation_config, tokenizer fingerprint,
            messages, per-generation (turn, prompt_hash,
            input_tokens_hash, output_tokens, output_tokens_hash,
            stop_reason), quantization, agent_steps,
            registered_tools.

        NOT included: trace_mode, rendered_prompt, input_tokens,
        steps, top_tokens, top_scores, environment.
        """
        d = self._canonical_dict()
        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
        self.session_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return self.session_hash

    def _canonical_dict(self) -> dict:
        """Core execution fields only — deterministic across trace modes."""
        return {
            "schema_version": self.schema_version,
            "model": self.model,
            "model_hash": self.model_hash,
            "seed": self.seed,
            "generation_config": self.generation_config,
            "tokenizer": self.tokenizer_info,
            "messages": self.messages,
            "generations": [
                {
                    "turn": g.turn,
                    "prompt_hash": g.prompt_hash,
                    "input_tokens_hash": g.input_tokens_hash,
                    "output_tokens": g.output_tokens,
                    "output_tokens_hash": g.output_tokens_hash,
                    "stop_reason": g.stop_reason,
                }
                for g in self.generations
            ],
            "quantization": self.quantization,
            "agent_steps": [s.to_dict() for s in self.agent_steps],
            "registered_tools": self.registered_tools,
        }

    def to_dict(self) -> dict:
        """Convert full session to dict for JSON export.

        Output varies by trace_mode:
          minimal:  canonical fields + session_hash + basic env
          standard: + rendered_prompt, input_tokens, steps
          verbose:  + top_tokens, top_scores, extended env
        """
        mode = self.trace_mode if isinstance(self.trace_mode, TraceMode) else TraceMode(self.trace_mode)
        return {
            "schema_version": self.schema_version,
            "trace_type": self.trace_type,
            "trace_mode": mode.value,
            "model": self.model,
            "model_hash": self.model_hash,
            "seed": self.seed,
            "session_hash": self.session_hash,
            "generation_config": self.generation_config,
            "tokenizer": self.tokenizer_info,
            "messages": self.messages,
            "generations": [g.to_dict(mode=mode) for g in self.generations],
            "environment": self.environment,
            "quantization": self.quantization,
            "agent_steps": [s.to_dict() for s in self.agent_steps],
            "registered_tools": self.registered_tools,
        }

    def export_json(self, path: str) -> None:
        """Export session trace to JSON file.

        Supports gzip: pass a path ending in .gz for compressed output.
        """
        self.compute_session_hash()
        data = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        if path.endswith(".gz"):
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(data)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)

    @classmethod
    def from_json(cls, path: str) -> "SessionTrace":
        """Load session trace from JSON file (supports .gz)."""
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionTrace":
        """Reconstruct SessionTrace from dict."""
        session = cls(
            schema_version=data.get("schema_version", "1"),
            trace_type=data.get("trace_type", "inference"),
            model=data.get("model", ""),
            model_hash=data.get("model_hash", ""),
            seed=data.get("seed", 42),
            generation_config=data.get("generation_config", {}),
            tokenizer_info=data.get("tokenizer", {}),
            trace_mode=TraceMode(data.get("trace_mode", "standard")),
            messages=data.get("messages", []),
            environment=data.get("environment", {}),
            session_hash=data.get("session_hash", ""),
            quantization=data.get("quantization", {"mode": None, "backend": None}),
            registered_tools=data.get("registered_tools", []),
        )

        for gen_data in data.get("generations", []):
            trace = GenerationTrace(
                turn=gen_data["turn"],
                rendered_prompt=gen_data.get("rendered_prompt", ""),
                prompt_hash=gen_data.get("prompt_hash", ""),
                input_tokens=gen_data.get("input_tokens", []),
                input_tokens_hash=gen_data.get("input_tokens_hash", ""),
                output_tokens=gen_data.get("output_tokens", []),
                output_tokens_hash=gen_data.get("output_tokens_hash", ""),
                stop_reason=gen_data.get("stop_reason", ""),
            )
            for step_data in gen_data.get("steps", []):
                trace.steps.append(GenerationStep(
                    step=step_data["step"],
                    chosen_token=step_data["chosen_token"],
                    top_tokens=step_data.get("top_tokens"),
                    top_scores=step_data.get("top_scores"),
                ))
            session.generations.append(trace)

        for step_data in data.get("agent_steps", []):
            session.agent_steps.append(AgentStep.from_dict(step_data))

        return session


# ---------------------------------------------------------------------------
# Environment fingerprint builder (minimal, deterministic)
# ---------------------------------------------------------------------------

def build_environment() -> dict:
    """Build a minimal, deterministic environment fingerprint.

    Avoids timestamps, hostnames, paths — only includes
    version info relevant to reproducibility.
    """
    import torch
    env = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": torch.__version__.split("+")[0],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "detinfer": _get_detinfer_version(),
    }

    # Transformers version (optional)
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except ImportError:
        pass

    # CUDA version
    if torch.cuda.is_available():
        env["cuda"] = torch.version.cuda or "unknown"
        env["gpu"] = torch.cuda.get_device_name(0)

    return env


# ---------------------------------------------------------------------------
# Hash utilities
# ---------------------------------------------------------------------------

def _hash_string(s: str) -> str:
    """SHA-256 hash of a string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _hash_token_list(tokens: list[int]) -> str:
    """SHA-256 hash of a token ID list."""
    # Convert to canonical string representation
    canonical = ",".join(str(t) for t in tokens)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_model_hash(model) -> str:
    """Compute a lightweight model hash from config JSON.

    Uses the model config (architecture, vocab size, etc.)
    rather than full weights for speed.
    """
    try:
        config_json = json.dumps(
            model.config.to_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return "unknown"


def compute_tokenizer_hash(tokenizer) -> str:
    """Compute a hash of the tokenizer vocabulary."""
    try:
        vocab = json.dumps(
            sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]),
            separators=(",", ":"),
        )
        return hashlib.sha256(vocab.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return "unknown"


def compute_chat_template_hash(tokenizer) -> str:
    """Hash the chat template string if present."""
    template = getattr(tokenizer, "chat_template", None)
    if template:
        return hashlib.sha256(str(template).encode("utf-8")).hexdigest()[:16]
    return "none"


def _get_detinfer_version() -> str:
    """Get detinfer version without circular imports."""
    try:
        import detinfer
        return getattr(detinfer, "__version__", "unknown")
    except ImportError:
        return "unknown"
