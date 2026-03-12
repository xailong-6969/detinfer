"""
detinfer.agent -- Deterministic Agent

Multi-turn deterministic agent with full token tracing,
prompt snapshotting, and session export.

Usage:
    from detinfer import DeterministicAgent

    agent = DeterministicAgent("<model>", seed=42)  # any HuggingFace model
    agent.chat("What is 2+2?")
    agent.chat("Explain more")
    agent.export_session("session.json")
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import torch

from detinfer.inference.engine import DeterministicEngine
from detinfer.agent.trace import (
    AgentStep,
    GenerationTrace,
    SessionTrace,
    TraceMode,
    build_environment,
    compute_model_hash,
    compute_tokenizer_hash,
    compute_chat_template_hash,
    _hash_string,
    _hash_token_list,
)


def deterministic_argmax(
    logits: torch.Tensor, epsilon: float = 1e-6
) -> tuple[int, bool]:
    """Deterministic argmax with stable tie-breaking and ambiguity detection.

    When multiple tokens share the maximum logit value,
    selects the smallest token ID. Also detects near-ties
    (top-2 within epsilon) and flags them as numerically ambiguous.

    Args:
        logits: 1D tensor of logit scores for each token.
        epsilon: Threshold for near-tie detection. If the top-2
            logits differ by less than this, the step is marked
            as ambiguous.

    Returns:
        Tuple of (token_id, is_ambiguous). is_ambiguous is True
        when the top-2 logits are within epsilon of each other.
    """
    max_val = torch.max(logits)
    candidates = torch.where(logits == max_val)[0]
    token_id = int(candidates.min().item())

    # Detect near-ties: is the gap between #1 and #2 within epsilon?
    is_ambiguous = False
    if logits.shape[0] > 1:
        top2 = torch.topk(logits, min(2, logits.shape[0]))
        if top2.values.shape[0] >= 2:
            gap = abs(top2.values[0].item() - top2.values[1].item())
            is_ambiguous = gap < epsilon

    return token_id, is_ambiguous


@dataclass
class TruncationPolicy:
    """Deterministic truncation policy for long conversations.

    When the rendered prompt exceeds max_context_tokens, this policy
    defines exactly which messages to drop.  The algorithm is fully
    deterministic: same history + same policy = same truncation.

    Rules (applied in order):
        1. System prompt is ALWAYS kept (never truncated).
        2. The latest user message is ALWAYS kept.
        3. Oldest non-system, non-latest-user turns are dropped first.
        4. Truncation events are recorded in the session trace.

    Args:
        max_context_tokens: Max tokens for the rendered prompt
            (default: None = no truncation).
        preserve_pairs: If True, drop user+assistant pairs
            together to avoid orphaned messages.
    """
    max_context_tokens: int | None = None
    preserve_pairs: bool = True

    @property
    def enabled(self) -> bool:
        return self.max_context_tokens is not None


class DeterministicAgent:
    """Multi-turn deterministic agent with full tracing.

    Wraps DeterministicEngine to provide:
    - Multi-turn conversation with history tracking
    - Token-level trace recording per turn
    - Prompt snapshotting (rendered_prompt + hash per turn)
    - Tool registration and call tracing
    - Deterministic truncation policy for long conversations
    - Re-seeding before each generation
    - Session export with full trace for replay/verification
    - Session save/resume via checkpoint files

    Usage:
        agent = DeterministicAgent("<model>", seed=42)  # any HuggingFace model
        response = agent.chat("Hello!")
        print(response)
        agent.export_session("session.json")
    """

    def __init__(
        self,
        model_name: str,
        seed: int = 42,
        max_new_tokens: int = 256,
        trace_mode: str | TraceMode = "standard",
        quantize: str | None = None,
        device: str | None = None,
        system_prompt: str | None = None,
        max_context_tokens: int | None = None,
    ):
        """Initialize the deterministic agent.

        Args:
            model_name: HuggingFace model ID.
            seed: Master random seed.
            max_new_tokens: Max tokens per generation.
            trace_mode: Trace detail level ("minimal", "standard", "verbose").
            quantize: Quantization mode (None or "int8", experimental).
            device: Device to use (auto-detected if None).
            system_prompt: Optional system prompt (e.g., "You are a math tutor").
            max_context_tokens: Max prompt tokens before truncation (None = no limit).
        """
        self.model_name = model_name
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.trace_mode = TraceMode(trace_mode) if isinstance(trace_mode, str) else trace_mode
        self.quantize = quantize
        self.system_prompt = system_prompt
        self.truncation = TruncationPolicy(max_context_tokens=max_context_tokens)

        # Initialize engine
        self.engine = DeterministicEngine(seed=seed, device=device)

        # Load model
        load_kwargs = {}
        if quantize == "int8":
            load_kwargs["quantize"] = "int8"
        self.engine.load(model_name, **load_kwargs)

        # Build session trace
        self.session = SessionTrace(
            trace_type="agent",
            model=model_name,
            model_hash=compute_model_hash(self.engine.model),
            seed=seed,
            trace_mode=trace_mode,
            generation_config={
                "do_sample": False,
                "temperature": 0.0,
                "top_p": None,
                "top_k": None,
                "max_new_tokens": max_new_tokens,
            },
            tokenizer_info={
                "name": model_name,
                "vocab_size": len(self.engine.tokenizer),
                "tokenizer_hash": compute_tokenizer_hash(self.engine.tokenizer),
                "chat_template_hash": compute_chat_template_hash(self.engine.tokenizer),
            },
            environment=build_environment(),
            quantization={
                "mode": quantize,
                "backend": "bitsandbytes" if quantize else None,
            },
        )

        self._turn_count = 0
        self._agent_step_counter = 0
        self._conversation_history: list[dict] = []
        self._tools: dict[str, Any] = {}  # name -> callable

        # Add system prompt to conversation history if provided
        if system_prompt:
            self._conversation_history.append({"role": "system", "content": system_prompt})
            self.session.add_message("system", system_prompt)

    def chat(self, message: str) -> str:
        """Send a message and get a deterministic response.

        Uses manual token-by-token generation with deterministic
        argmax (smallest-token-ID tie-breaking) for strict
        reproducibility. Records full token trace.

        Args:
            message: User message.

        Returns:
            Assistant response text.
        """
        self._turn_count += 1

        # Add user message
        self._conversation_history.append({"role": "user", "content": message})
        self.session.add_message("user", message)

        # Re-seed for strict reproducibility
        self.engine.config.apply()

        # Truncate if needed, then render prompt
        self._truncate_history()
        rendered_prompt = self._render_prompt()

        # Tokenize
        input_device = self.engine._get_input_device()
        inputs = self.engine.tokenizer(
            rendered_prompt, return_tensors="pt"
        ).to(input_device)
        input_token_ids = inputs["input_ids"][0].tolist()

        # Create generation trace for this turn
        gen_trace = GenerationTrace(
            turn=self._turn_count,
            rendered_prompt=rendered_prompt,
            input_tokens=input_token_ids,
        )

        tokenizer = self.engine.tokenizer
        eos_id = tokenizer.eos_token_id
        generated_ids = []
        current_ids = inputs["input_ids"]
        verbose = self.trace_mode == TraceMode.VERBOSE

        # Token-by-token generation with deterministic argmax + KV cache
        past_key_values = None
        with torch.no_grad(), self.engine.enforcer.deterministic_context():
            for step in range(self.max_new_tokens):
                outputs = self.engine.model(
                    current_ids, past_key_values=past_key_values,
                    use_cache=True,
                )
                next_logits = outputs.logits[0, -1, :]
                past_key_values = outputs.past_key_values

                # Deterministic argmax with stable tie-breaking
                token_id, is_ambiguous = deterministic_argmax(next_logits)

                # Capture top-k for verbose trace
                top_tokens = None
                top_scores = None
                if verbose:
                    k = min(10, next_logits.shape[0])
                    top_vals, top_ids = torch.topk(next_logits, k)
                    # Stable sort: by (-score, token_id) for cross-hardware consistency
                    pairs = list(zip(top_vals.tolist(), top_ids.tolist()))
                    pairs.sort(key=lambda p: (-p[0], p[1]))
                    top_tokens = [p[1] for p in pairs]
                    top_scores = [round(p[0], 6) for p in pairs]

                gen_trace.add_step(
                    step=step,
                    chosen_token=token_id,
                    top_tokens=top_tokens,
                    top_scores=top_scores,
                    is_ambiguous=is_ambiguous,
                )
                generated_ids.append(token_id)

                # Stop on EOS
                if token_id == eos_id:
                    break

                # Next iteration: only feed the new token (KV cache has the rest)
                current_ids = torch.tensor([[token_id]], device=input_device)

        # Set output tokens and finalize
        gen_trace.output_tokens = generated_ids
        gen_trace.finalize(eos_token_id=eos_id)

        # Add to session
        self.session.add_generation(gen_trace)

        # Record agent step for this LLM generation
        self._agent_step_counter += 1
        self.session.add_agent_step(AgentStep(
            step=self._agent_step_counter,
            type="llm_generation",
            turn=self._turn_count,
            generation_turn=self._turn_count,
        ))

        # Decode response
        response = tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        # Add assistant message
        self._conversation_history.append({"role": "assistant", "content": response})
        self.session.add_message("assistant", response)

        return response

    def chat_stream(self, message: str):
        """Send a message and yield tokens as they are generated.

        Same deterministic guarantees as chat(), but yields decoded
        text chunks in real-time for streaming display.

        Args:
            message: User message.

        Yields:
            str: Decoded text chunk for each generated token.

        After iteration completes, the full response is recorded in
        the session trace (identical to what chat() would produce).
        """
        self._turn_count += 1

        # Add user message
        self._conversation_history.append({"role": "user", "content": message})
        self.session.add_message("user", message)

        # Re-seed for strict reproducibility
        self.engine.config.apply()

        # Truncate if needed, then render
        self._truncate_history()
        rendered_prompt = self._render_prompt()
        input_device = self.engine._get_input_device()
        inputs = self.engine.tokenizer(
            rendered_prompt, return_tensors="pt"
        ).to(input_device)
        input_token_ids = inputs["input_ids"][0].tolist()

        # Create generation trace
        gen_trace = GenerationTrace(
            turn=self._turn_count,
            rendered_prompt=rendered_prompt,
            input_tokens=input_token_ids,
        )

        tokenizer = self.engine.tokenizer
        eos_id = tokenizer.eos_token_id
        generated_ids = []
        current_ids = inputs["input_ids"]
        verbose = self.trace_mode == TraceMode.VERBOSE

        # Token-by-token generation with streaming + KV cache
        past_key_values = None
        with torch.no_grad(), self.engine.enforcer.deterministic_context():
            for step in range(self.max_new_tokens):
                outputs = self.engine.model(
                    current_ids, past_key_values=past_key_values,
                    use_cache=True,
                )
                next_logits = outputs.logits[0, -1, :]
                past_key_values = outputs.past_key_values

                # Deterministic argmax with tie-breaking
                token_id, is_ambiguous = deterministic_argmax(next_logits)

                # Capture top-k for verbose trace
                top_tokens = None
                top_scores = None
                if verbose:
                    k = min(10, next_logits.shape[0])
                    top_vals, top_ids = torch.topk(next_logits, k)
                    # Stable sort: by (-score, token_id)
                    pairs = list(zip(top_vals.tolist(), top_ids.tolist()))
                    pairs.sort(key=lambda p: (-p[0], p[1]))
                    top_tokens = [p[1] for p in pairs]
                    top_scores = [round(p[0], 6) for p in pairs]

                # Record trace step
                gen_trace.add_step(
                    step=step,
                    chosen_token=token_id,
                    top_tokens=top_tokens,
                    top_scores=top_scores,
                    is_ambiguous=is_ambiguous,
                )
                generated_ids.append(token_id)

                # Decode just this token and yield
                chunk = tokenizer.decode(
                    [token_id], skip_special_tokens=True
                )
                if chunk:
                    yield chunk

                # Stop on EOS
                if token_id == eos_id:
                    break

                # Next iteration: only feed the new token (KV cache has the rest)
                current_ids = torch.tensor([[token_id]], device=input_device)

        # Finalize trace
        gen_trace.output_tokens = generated_ids
        gen_trace.finalize(eos_token_id=eos_id)
        self.session.add_generation(gen_trace)

        # Record agent step for this LLM generation (matches chat() behavior)
        self._agent_step_counter += 1
        self.session.add_agent_step(AgentStep(
            step=self._agent_step_counter,
            type="llm_generation",
            turn=self._turn_count,
            generation_turn=self._turn_count,
        ))

        # Decode full response
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        self._conversation_history.append({"role": "assistant", "content": response})
        self.session.add_message("assistant", response)

    def _truncate_history(self) -> None:
        """Apply deterministic truncation to conversation history.

        Algorithm (fully deterministic):
            1. If max_context_tokens is None, do nothing.
            2. Render the current history into a prompt string.
            3. If token count <= budget, do nothing.
            4. Otherwise, drop oldest non-protected messages one at a time
               until within budget. Protected = system prompt + latest user message.
            5. Record truncation event as an AgentStep in the trace.
        """
        if not self.truncation.enabled:
            return

        tokenizer = self.engine.tokenizer
        budget = self.truncation.max_context_tokens

        # Quick check: render current history and count tokens
        rendered = self._render_prompt()
        token_count = len(tokenizer.encode(rendered))
        if token_count <= budget:
            return

        # Build drop candidates: indices of messages we CAN drop
        # Never drop: system prompt (index 0 if present) or last user message
        last_user_idx = None
        for i in range(len(self._conversation_history) - 1, -1, -1):
            if self._conversation_history[i]["role"] == "user":
                last_user_idx = i
                break

        droppable = []
        for i, msg in enumerate(self._conversation_history):
            if msg["role"] == "system":
                continue  # protect system prompt
            if i == last_user_idx:
                continue  # protect latest user message
            droppable.append(i)

        # Drop oldest first (smallest index first = most deterministic)
        dropped_count = 0
        original_len = len(self._conversation_history)

        if self.truncation.preserve_pairs:
            # Drop user+assistant pairs together
            # Build pairs: [(user_idx, assistant_idx), ...]
            pairs = []
            i = 0
            while i < len(droppable) - 1:
                idx_a = droppable[i]
                idx_b = droppable[i + 1]
                msg_a = self._conversation_history[idx_a]
                msg_b = self._conversation_history[idx_b]
                if msg_a["role"] == "user" and msg_b["role"] == "assistant":
                    pairs.append((idx_a, idx_b))
                    i += 2
                else:
                    pairs.append((idx_a,))
                    i += 1
            if i < len(droppable):
                pairs.append((droppable[i],))

            # Drop pairs oldest-first
            indices_to_drop = set()
            for pair in pairs:
                indices_to_drop.update(pair)
                # Rebuild history without dropped indices
                remaining = [
                    msg for j, msg in enumerate(self._conversation_history)
                    if j not in indices_to_drop
                ]
                test_prompt = self._render_prompt_from(remaining)
                if len(tokenizer.encode(test_prompt)) <= budget:
                    break

            dropped_count = len(indices_to_drop)
            self._conversation_history = [
                msg for j, msg in enumerate(self._conversation_history)
                if j not in indices_to_drop
            ]
        else:
            # Drop individual messages oldest-first
            indices_to_drop = set()
            for idx in droppable:
                indices_to_drop.add(idx)
                remaining = [
                    msg for j, msg in enumerate(self._conversation_history)
                    if j not in indices_to_drop
                ]
                test_prompt = self._render_prompt_from(remaining)
                if len(tokenizer.encode(test_prompt)) <= budget:
                    break

            dropped_count = len(indices_to_drop)
            self._conversation_history = [
                msg for j, msg in enumerate(self._conversation_history)
                if j not in indices_to_drop
            ]

        # Record truncation event in trace
        if dropped_count > 0:
            self._agent_step_counter += 1
            self.session.add_agent_step(AgentStep(
                step=self._agent_step_counter,
                type="checkpoint",
                turn=self._turn_count,
                checkpoint_data={
                    "event": "truncation",
                    "messages_before": original_len,
                    "messages_after": len(self._conversation_history),
                    "messages_dropped": dropped_count,
                    "max_context_tokens": budget,
                },
            ))

    def _render_prompt_from(self, messages: list[dict]) -> str:
        """Render a prompt from an arbitrary message list."""
        tokenizer = self.engine.tokenizer
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                pass
        parts = []
        for msg in messages:
            role = msg["role"].capitalize()
            if role == "System":
                parts.append(f"System: {msg['content']}")
            else:
                parts.append(f"{role}: {msg['content']}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _render_prompt(self) -> str:
        """Render the full conversation into a prompt string.

        Uses chat template if available, otherwise raw concatenation.
        System prompt is included at the start of the conversation.
        """
        return self._render_prompt_from(self._conversation_history)

    def export_session(self, path: str) -> str:
        """Export the full session trace to JSON.

        Args:
            path: Output file path.

        Returns:
            Session hash.
        """
        self.session.export_json(path)
        return self.session.session_hash

    def get_session_hash(self) -> str:
        """Compute and return the current session hash."""
        return self.session.compute_session_hash()

    def save_state(self, path: str) -> None:
        """Save the full agent state for later resume.

        Exports conversation history, session trace, turn counter,
        and agent config into a single JSON.  Load with load_state().

        Args:
            path: Output file path (.json or .json.gz).
        """
        state = {
            "version": 1,
            "agent_config": {
                "model_name": self.model_name,
                "seed": self.seed,
                "max_new_tokens": self.max_new_tokens,
                "trace_mode": self.trace_mode.value,
                "quantize": self.quantize,
                "system_prompt": self.system_prompt,
                "max_context_tokens": self.truncation.max_context_tokens,
            },
            "conversation_history": self._conversation_history,
            "turn_count": self._turn_count,
            "agent_step_counter": self._agent_step_counter,
            "session_trace": self.session.to_dict(),
        }
        data = json.dumps(state, indent=2, ensure_ascii=False)
        if path.endswith(".gz"):
            import gzip
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(data)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)

    def load_state(self, path: str) -> None:
        """Resume agent from a saved state file.

        Restores conversation history, session trace, and counters.
        The model must already be loaded (same model_name).

        Args:
            path: Path to a state file created by save_state().

        Raises:
            ValueError: If the state was saved with a different model.
        """
        if path.endswith(".gz"):
            import gzip
            with gzip.open(path, "rt", encoding="utf-8") as f:
                state = json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)

        config = state["agent_config"]
        if config["model_name"] != self.model_name:
            raise ValueError(
                f"State was saved with model '{config['model_name']}' "
                f"but this agent is loaded with '{self.model_name}'"
            )

        self._conversation_history = state["conversation_history"]
        self._turn_count = state["turn_count"]
        self._agent_step_counter = state["agent_step_counter"]
        self.session = SessionTrace.from_dict(state["session_trace"])

    @property
    def history(self) -> list[dict]:
        """Return the conversation history."""
        return list(self._conversation_history)

    @property
    def turn_count(self) -> int:
        """Return the number of completed turns."""
        return self._turn_count

    # ------------------------------------------------------------------
    # Tool registration and calling
    # ------------------------------------------------------------------

    def register_tool(self, name: str, fn: Any) -> None:
        """Register a tool for deterministic agent workflows.

        Only the tool name, arguments, and result are stored in the
        trace. The callable itself is never serialized.

        Args:
            name: Tool name (e.g., 'calculator', 'web_search').
            fn: Callable that accepts keyword arguments and returns a string.
        """
        self._tools[name] = fn
        if name not in self.session.registered_tools:
            self.session.registered_tools.append(name)

    def call_tool(self, name: str, arguments: dict | None = None) -> str:
        """Call a registered tool and record the call in the agent trace.

        Records both the tool_call (name + arguments) and the
        tool_result (output) as separate agent steps. Only stores
        serializable data — no runtime objects.

        Args:
            name: Name of a previously registered tool.
            arguments: Keyword arguments to pass to the tool.

        Returns:
            Tool result as a string.

        Raises:
            KeyError: If the tool is not registered.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered. "
                           f"Available: {list(self._tools.keys())}")

        args = arguments or {}

        # Record tool_call step
        self._agent_step_counter += 1
        self.session.add_agent_step(AgentStep(
            step=self._agent_step_counter,
            type="tool_call",
            turn=self._turn_count,
            tool=name,
            arguments=args,
        ))

        # Execute tool
        result = str(self._tools[name](**args))

        # Record tool_result step
        self._agent_step_counter += 1
        self.session.add_agent_step(AgentStep(
            step=self._agent_step_counter,
            type="tool_result",
            turn=self._turn_count,
            tool=name,
            result=result,
        ))

        # Add tool result to conversation for context
        self._conversation_history.append({
            "role": "user",
            "content": f"[Tool: {name}] Result: {result}",
        })
        self.session.add_message("tool", f"[{name}] {result}")

        return result

    def checkpoint(self, data: dict | None = None) -> None:
        """Record a checkpoint in the agent workflow trace.

        Useful for marking key decision points during replay.

        Args:
            data: Optional serializable checkpoint data.
        """
        self._agent_step_counter += 1
        self.session.add_agent_step(AgentStep(
            step=self._agent_step_counter,
            type="checkpoint",
            turn=self._turn_count,
            checkpoint_data=data or {},
        ))
