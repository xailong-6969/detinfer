"""
Tests for detinfer v0.3.0 agent, trace, and replay modules.

These tests verify the core data structures and logic without
requiring a GPU or transformers (except where noted).
"""

import json
import os
import tempfile

import pytest

from detinfer.agent.trace import (
    GenerationStep,
    GenerationTrace,
    SessionTrace,
    TraceMode,
    _hash_string,
    _hash_token_list,
)


# ---------------------------------------------------------------------------
# GenerationStep tests
# ---------------------------------------------------------------------------

class TestGenerationStep:
    def test_minimal_step(self):
        step = GenerationStep(step=0, chosen_token=42)
        d = step.to_dict(verbose=False)
        assert d == {"step": 0, "chosen_token": 42}

    def test_verbose_step(self):
        step = GenerationStep(
            step=0, chosen_token=42,
            top_tokens=[42, 100, 200],
            top_scores=[9.5, 7.2, 3.1],
        )
        d = step.to_dict(verbose=True)
        assert d["top_tokens"] == [42, 100, 200]
        assert d["top_scores"] == [9.5, 7.2, 3.1]

    def test_verbose_without_data(self):
        step = GenerationStep(step=0, chosen_token=42)
        d = step.to_dict(verbose=True)
        assert "top_tokens" not in d


# ---------------------------------------------------------------------------
# GenerationTrace tests
# ---------------------------------------------------------------------------

class TestGenerationTrace:
    def test_add_steps(self):
        trace = GenerationTrace(turn=1)
        trace.add_step(step=0, chosen_token=10)
        trace.add_step(step=1, chosen_token=20)
        assert len(trace.steps) == 2
        assert trace.steps[0].chosen_token == 10
        assert trace.steps[1].chosen_token == 20

    def test_finalize_eos(self):
        trace = GenerationTrace(turn=1)
        trace.rendered_prompt = "Hello"
        trace.input_tokens = [1, 2, 3]
        trace.output_tokens = [10, 20, 50256]  # 50256 = eos
        trace.finalize(eos_token_id=50256)
        assert trace.stop_reason == "eos"
        assert trace.prompt_hash != ""
        assert trace.input_tokens_hash != ""
        assert trace.output_tokens_hash != ""

    def test_finalize_max_tokens(self):
        trace = GenerationTrace(turn=1)
        trace.rendered_prompt = "Hello"
        trace.input_tokens = [1, 2, 3]
        trace.output_tokens = [10, 20, 30]
        trace.finalize(eos_token_id=50256)
        assert trace.stop_reason == "max_new_tokens"

    def test_to_dict_minimal(self):
        trace = GenerationTrace(turn=1)
        trace.add_step(step=0, chosen_token=42)
        d = trace.to_dict(mode=TraceMode.MINIMAL)
        assert "rendered_prompt" not in d
        assert "steps" not in d  # minimal mode excludes steps
        assert "input_tokens" not in d
        assert d["turn"] == 1

    def test_to_dict_standard(self):
        trace = GenerationTrace(turn=1, rendered_prompt="Hello")
        trace.add_step(step=0, chosen_token=42)
        d = trace.to_dict(mode=TraceMode.STANDARD)
        assert d["rendered_prompt"] == "Hello"
        assert len(d["steps"]) == 1
        assert "top_tokens" not in d["steps"][0]

    def test_to_dict_verbose(self):
        trace = GenerationTrace(turn=1, rendered_prompt="Hello")
        d = trace.to_dict(verbose=True)
        assert d["rendered_prompt"] == "Hello"


# ---------------------------------------------------------------------------
# SessionTrace tests
# ---------------------------------------------------------------------------

class TestSessionTrace:
    def test_add_message(self):
        session = SessionTrace(model="test-model", seed=42)
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there")
        assert len(session.messages) == 2
        assert session.messages[0]["role"] == "user"

    def test_session_hash_deterministic(self):
        s1 = SessionTrace(model="test-model", seed=42)
        s1.add_message("user", "Hello")
        h1 = s1.compute_session_hash()

        s2 = SessionTrace(model="test-model", seed=42)
        s2.add_message("user", "Hello")
        h2 = s2.compute_session_hash()

        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_session_hash_changes_with_content(self):
        s1 = SessionTrace(model="test-model", seed=42)
        s1.add_message("user", "Hello")
        h1 = s1.compute_session_hash()

        s2 = SessionTrace(model="test-model", seed=42)
        s2.add_message("user", "Goodbye")
        h2 = s2.compute_session_hash()

        assert h1 != h2

    def test_export_import_roundtrip(self):
        session = SessionTrace(model="test-model", seed=42, trace_mode=TraceMode.STANDARD)
        session.add_message("user", "What is 2+2?")
        session.add_message("assistant", "4")

        gen = GenerationTrace(turn=1, rendered_prompt="What is 2+2?")
        gen.input_tokens = [2061, 318, 362, 10, 17]
        gen.output_tokens = [19]
        gen.add_step(step=0, chosen_token=19)
        gen.finalize(eos_token_id=50256)
        session.add_generation(gen)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            session.export_json(path)

            # Reload
            loaded = SessionTrace.from_json(path)
            assert loaded.model == "test-model"
            assert loaded.seed == 42
            assert len(loaded.messages) == 2
            assert len(loaded.generations) == 1
            assert loaded.generations[0].output_tokens == [19]
            assert loaded.generations[0].steps[0].chosen_token == 19
            assert loaded.session_hash != ""

            # Re-export and verify stable hash
            loaded.compute_session_hash()
            assert loaded.session_hash == session.session_hash
        finally:
            os.unlink(path)

    def test_step_ambiguity_roundtrip(self):
        session = SessionTrace(model="test-model", seed=42, trace_mode=TraceMode.STANDARD)
        session.add_message("user", "Hello")

        gen = GenerationTrace(turn=1, rendered_prompt="Hello")
        gen.add_step(step=0, chosen_token=42, is_ambiguous=True)
        gen.output_tokens = [42]
        gen.finalize(eos_token_id=50256)
        session.add_generation(gen)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            session.export_json(path)
            loaded = SessionTrace.from_json(path)
            assert loaded.generations[0].steps[0].is_ambiguous is True
        finally:
            os.unlink(path)

    def test_canonical_hash_independent_of_trace_mode(self):
        """Critical invariant: same run, different trace modes, same hash."""
        def make_session(mode: TraceMode) -> SessionTrace:
            s = SessionTrace(model="test-model", seed=42, trace_mode=mode)
            s.add_message("user", "test")
            gen = GenerationTrace(turn=1, rendered_prompt="test")
            gen.input_tokens = [1, 2, 3]
            gen.output_tokens = [10, 20]
            gen.add_step(step=0, chosen_token=10, top_tokens=[10, 5], top_scores=[9.0, 3.0])
            gen.add_step(step=1, chosen_token=20)
            gen.finalize(eos_token_id=50256)
            s.add_generation(gen)
            return s

        s_min = make_session(TraceMode.MINIMAL)
        s_std = make_session(TraceMode.STANDARD)
        s_verb = make_session(TraceMode.VERBOSE)

        h_min = s_min.compute_session_hash()
        h_std = s_std.compute_session_hash()
        h_verb = s_verb.compute_session_hash()

        assert h_min == h_std == h_verb


# ---------------------------------------------------------------------------
# Hash utility tests
# ---------------------------------------------------------------------------

class TestHashing:
    def test_hash_string_deterministic(self):
        h1 = _hash_string("hello")
        h2 = _hash_string("hello")
        assert h1 == h2

    def test_hash_string_different(self):
        h1 = _hash_string("hello")
        h2 = _hash_string("world")
        assert h1 != h2

    def test_hash_token_list_deterministic(self):
        h1 = _hash_token_list([1, 2, 3])
        h2 = _hash_token_list([1, 2, 3])
        assert h1 == h2

    def test_hash_token_list_different(self):
        h1 = _hash_token_list([1, 2, 3])
        h2 = _hash_token_list([3, 2, 1])
        assert h1 != h2


# ---------------------------------------------------------------------------
# Diff tests (using SessionTrace directly)
# ---------------------------------------------------------------------------

class TestDiff:
    def _make_session(self, tokens: list[int]) -> str:
        session = SessionTrace(model="test-model", seed=42)
        session.add_message("user", "test")
        session.add_message("assistant", "response")
        gen = GenerationTrace(turn=1)
        gen.output_tokens = tokens
        gen.finalize(eos_token_id=50256)
        session.add_generation(gen)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name
        session.export_json(path)
        return path

    def test_identical_sessions(self):
        from detinfer.agent.replay import diff_sessions
        p1 = self._make_session([10, 20, 30])
        p2 = self._make_session([10, 20, 30])
        try:
            result = diff_sessions(p1, p2)
            assert result.identical
        finally:
            os.unlink(p1)
            os.unlink(p2)

    def test_different_sessions(self):
        from detinfer.agent.replay import diff_sessions
        p1 = self._make_session([10, 20, 30])
        p2 = self._make_session([10, 20, 99])
        try:
            result = diff_sessions(p1, p2)
            assert not result.identical
            assert result.first_mismatch_turn == 1
            assert result.first_mismatch_step == 2
            assert result.expected == 30
            assert result.observed == 99
        finally:
            os.unlink(p1)
            os.unlink(p2)


class TestReplay:
    def test_replay_preserves_tool_messages(self, monkeypatch):
        from detinfer.agent.replay import replay_session
        import detinfer.agent.runtime as runtime_mod

        class FakeSession:
            def __init__(self):
                self.generations = []
                self.messages = []

            def add_message(self, role: str, content: str) -> None:
                self.messages.append({"role": role, "content": content})

        class FakeAgent:
            def __init__(
                self,
                model_name: str,
                seed: int = 42,
                max_new_tokens: int = 256,
                trace_mode: str = "standard",
                quantize: str | None = None,
                system_prompt: str | None = None,
            ):
                self.model_name = model_name
                self.seed = seed
                self.max_new_tokens = max_new_tokens
                self.trace_mode = trace_mode
                self.quantize = quantize
                self.session = FakeSession()
                self._conversation_history = []
                self._turn_count = 0

                if system_prompt:
                    self._conversation_history.append(
                        {"role": "system", "content": system_prompt}
                    )
                    self.session.add_message("system", system_prompt)

            def chat(self, message: str) -> str:
                self._turn_count += 1
                self._conversation_history.append({"role": "user", "content": message})
                self.session.add_message("user", message)

                gen = GenerationTrace(turn=self._turn_count)
                if self._turn_count == 1:
                    gen.output_tokens = [10, 20]
                else:
                    has_tool = any(
                        m.get("role") == "tool"
                        and m.get("content") == "[Tool: calculator] Result: 4"
                        for m in self._conversation_history
                    )
                    gen.output_tokens = [30] if has_tool else [99]
                gen.stop_reason = "max_new_tokens"
                self.session.generations.append(gen)

                self._conversation_history.append({"role": "assistant", "content": "ok"})
                self.session.add_message("assistant", "ok")
                return "ok"

        session = SessionTrace(trace_type="agent", model="test-model", seed=42)
        session.add_message("system", "You are helpful.")
        session.add_message("user", "first")
        session.add_message("assistant", "a1")
        session.add_message("tool", "[Tool: calculator] Result: 4")
        session.add_message("user", "second")
        session.add_message("assistant", "a2")

        g1 = GenerationTrace(turn=1)
        g1.output_tokens = [10, 20]
        g1.stop_reason = "max_new_tokens"
        session.add_generation(g1)

        g2 = GenerationTrace(turn=2)
        g2.output_tokens = [30]
        g2.stop_reason = "max_new_tokens"
        session.add_generation(g2)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            session.export_json(path)
            monkeypatch.setattr(runtime_mod, "DeterministicAgent", FakeAgent)
            result = replay_session(path)
            assert result.passed
            assert result.verified_turns == 2
        finally:
            os.unlink(path)

    def test_replay_strict_requires_step_data(self, monkeypatch):
        from detinfer.agent.replay import replay_session
        import detinfer.agent.runtime as runtime_mod

        class FakeSession:
            def __init__(self):
                self.generations = []
                self.messages = []

            def add_message(self, role: str, content: str) -> None:
                self.messages.append({"role": role, "content": content})

        class FakeAgent:
            def __init__(
                self,
                model_name: str,
                seed: int = 42,
                max_new_tokens: int = 256,
                trace_mode: str = "standard",
                quantize: str | None = None,
                system_prompt: str | None = None,
            ):
                self.model_name = model_name
                self.seed = seed
                self.max_new_tokens = max_new_tokens
                self.trace_mode = trace_mode
                self.quantize = quantize
                self.session = FakeSession()
                self._turn_count = 0

                if system_prompt:
                    self.session.add_message("system", system_prompt)

            def chat(self, message: str) -> str:
                self._turn_count += 1
                self.session.add_message("user", message)
                gen = GenerationTrace(turn=self._turn_count)
                gen.output_tokens = [10]
                gen.stop_reason = "max_new_tokens"
                self.session.generations.append(gen)
                self.session.add_message("assistant", "ok")
                return "ok"

        session = SessionTrace(
            trace_type="agent",
            model="test-model",
            seed=42,
            trace_mode=TraceMode.MINIMAL,
        )
        session.add_message("user", "hello")
        session.add_message("assistant", "ok")

        gen = GenerationTrace(turn=1)
        gen.output_tokens = [10]
        gen.stop_reason = "max_new_tokens"
        session.add_generation(gen)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            session.export_json(path)
            monkeypatch.setattr(runtime_mod, "DeterministicAgent", FakeAgent)
            result = replay_session(path, strict=True)
            assert not result.passed
            assert result.failure_turn == 1
            assert result.failure_reason == "Strict mode requires per-step trace data"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TruncationPolicy tests
# ---------------------------------------------------------------------------

class TestTruncationPolicy:
    def test_policy_disabled_by_default(self):
        from detinfer.agent.runtime import TruncationPolicy
        policy = TruncationPolicy()
        assert not policy.enabled
        assert policy.max_context_tokens is None

    def test_policy_enabled(self):
        from detinfer.agent.runtime import TruncationPolicy
        policy = TruncationPolicy(max_context_tokens=512)
        assert policy.enabled
        assert policy.max_context_tokens == 512

    def test_policy_preserve_pairs_default(self):
        from detinfer.agent.runtime import TruncationPolicy
        policy = TruncationPolicy(max_context_tokens=100)
        assert policy.preserve_pairs is True


# ---------------------------------------------------------------------------
# AgentStep tool call recording tests
# ---------------------------------------------------------------------------

class TestAgentStepRecording:
    def test_tool_call_step(self):
        from detinfer.agent.trace import AgentStep
        step = AgentStep(
            step=1, type="tool_call", turn=3,
            tool="calculator", arguments={"expr": "2+2"},
        )
        d = step.to_dict()
        assert d["type"] == "tool_call"
        assert d["tool"] == "calculator"
        assert d["arguments"] == {"expr": "2+2"}

    def test_tool_result_step(self):
        from detinfer.agent.trace import AgentStep
        step = AgentStep(
            step=2, type="tool_result", turn=3,
            tool="calculator", result="4",
        )
        d = step.to_dict()
        assert d["type"] == "tool_result"
        assert d["result"] == "4"

    def test_checkpoint_step(self):
        from detinfer.agent.trace import AgentStep
        step = AgentStep(
            step=3, type="checkpoint", turn=3,
            checkpoint_data={"event": "truncation", "dropped": 2},
        )
        d = step.to_dict()
        assert d["type"] == "checkpoint"
        assert d["checkpoint_data"]["event"] == "truncation"

    def test_roundtrip(self):
        from detinfer.agent.trace import AgentStep
        original = AgentStep(
            step=5, type="tool_call", turn=2,
            tool="search", arguments={"q": "hello"},
        )
        d = original.to_dict()
        restored = AgentStep.from_dict(d)
        assert restored.step == 5
        assert restored.tool == "search"
        assert restored.arguments == {"q": "hello"}


# ---------------------------------------------------------------------------
# Session save / resume tests
# ---------------------------------------------------------------------------

class TestSessionSaveResume:
    def test_save_state_roundtrip(self):
        """Test that save_state/load_state preserves conversation history."""
        session = SessionTrace(model="test-model", seed=42, trace_mode=TraceMode.STANDARD)
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there")

        gen = GenerationTrace(turn=1)
        gen.output_tokens = [10, 20]
        gen.finalize(eos_token_id=50256)
        session.add_generation(gen)

        # Simulate save_state structure
        state = {
            "version": 1,
            "agent_config": {
                "model_name": "test-model",
                "seed": 42,
                "max_new_tokens": 256,
                "trace_mode": "standard",
                "quantize": None,
                "system_prompt": None,
                "max_context_tokens": None,
            },
            "conversation_history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "turn_count": 1,
            "agent_step_counter": 1,
            "session_trace": session.to_dict(),
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name
            json.dump(state, f, indent=2)

        try:
            # Load and verify
            loaded_state = json.loads(open(path).read())
            assert loaded_state["turn_count"] == 1
            assert len(loaded_state["conversation_history"]) == 2
            assert loaded_state["agent_config"]["model_name"] == "test-model"

            # Verify session trace roundtrip
            loaded_session = SessionTrace.from_dict(loaded_state["session_trace"])
            assert loaded_session.model == "test-model"
            assert len(loaded_session.generations) == 1
        finally:
            os.unlink(path)

    def test_gzip_session_roundtrip(self):
        """Test gzip export/import roundtrip."""
        session = SessionTrace(model="test-model", seed=42)
        session.add_message("user", "test")
        gen = GenerationTrace(turn=1)
        gen.output_tokens = [5, 10]
        gen.finalize(eos_token_id=50256)
        session.add_generation(gen)

        with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as f:
            path = f.name

        try:
            session.export_json(path)
            loaded = SessionTrace.from_json(path)
            assert loaded.model == "test-model"
            assert loaded.generations[0].output_tokens == [5, 10]
            assert loaded.session_hash == session.session_hash
        finally:
            os.unlink(path)


    def test_save_state_computes_session_hash(self, monkeypatch):
        """save_state() should persist a usable session hash in the saved trace."""
        import detinfer.agent.runtime as runtime_mod

        class DummyConfig:
            def to_dict(self):
                return {"arch": "dummy"}

        class DummyModel:
            config = DummyConfig()

        class DummyTokenizer:
            chat_template = None

            def __len__(self):
                return 2

            def get_vocab(self):
                return {"a": 0, "b": 1}

        def fake_load(self, model_name, **kwargs):
            self.model = DummyModel()
            self.tokenizer = DummyTokenizer()
            self.model_name = model_name
            return None

        monkeypatch.setattr(runtime_mod.DeterministicEngine, "load", fake_load)
        agent = runtime_mod.DeterministicAgent("test-model", seed=42)
        agent.session.add_message("user", "hello")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            agent.save_state(path)
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            assert state["session_trace"]["session_hash"] != ""
        finally:
            os.unlink(path)

    def test_load_state_rejects_config_mismatch(self, monkeypatch):
        """Resuming with a different deterministic config should fail explicitly."""
        import detinfer.agent.runtime as runtime_mod

        class DummyConfig:
            def to_dict(self):
                return {"arch": "dummy"}

        class DummyModel:
            config = DummyConfig()

        class DummyTokenizer:
            chat_template = None

            def __len__(self):
                return 2

            def get_vocab(self):
                return {"a": 0, "b": 1}

        def fake_load(self, model_name, **kwargs):
            self.model = DummyModel()
            self.tokenizer = DummyTokenizer()
            self.model_name = model_name
            return None

        monkeypatch.setattr(runtime_mod.DeterministicEngine, "load", fake_load)
        agent = runtime_mod.DeterministicAgent("test-model", seed=42, max_new_tokens=64)

        session = SessionTrace(model="test-model", seed=99, trace_mode=TraceMode.STANDARD)
        session.compute_session_hash()
        state = {
            "version": 1,
            "agent_config": {
                "model_name": "test-model",
                "seed": 99,
                "max_new_tokens": 64,
                "trace_mode": "standard",
                "quantize": None,
                "system_prompt": None,
                "max_context_tokens": None,
            },
            "conversation_history": [],
            "turn_count": 0,
            "agent_step_counter": 0,
            "session_trace": session.to_dict(),
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
            json.dump(state, f, indent=2)

        try:
            with pytest.raises(ValueError, match="seed"):
                agent.load_state(path)
        finally:
            os.unlink(path)
