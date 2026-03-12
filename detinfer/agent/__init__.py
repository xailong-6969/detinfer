"""
detinfer.agent -- Deterministic Agent System

Multi-turn deterministic agent with token-level tracing,
session replay, and diff verification.
"""

from detinfer.agent.runtime import DeterministicAgent, deterministic_argmax
from detinfer.agent.trace import (
    TraceMode,
    GenerationStep,
    GenerationTrace,
    AgentStep,
    SessionTrace,
    build_environment,
    compute_model_hash,
    compute_tokenizer_hash,
    compute_chat_template_hash,
)
from detinfer.agent.replay import (
    replay_session,
    diff_sessions,
    ReplayResult,
    DiffResult,
)

__all__ = [
    "DeterministicAgent",
    "deterministic_argmax",
    "TraceMode",
    "GenerationStep",
    "GenerationTrace",
    "AgentStep",
    "SessionTrace",
    "build_environment",
    "compute_model_hash",
    "compute_tokenizer_hash",
    "compute_chat_template_hash",
    "replay_session",
    "diff_sessions",
    "ReplayResult",
    "DiffResult",
]

