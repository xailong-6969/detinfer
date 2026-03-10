"""
detinfer.agent -- Deterministic Chat Agent

Multi-turn deterministic chat with full token tracing,
prompt snapshotting, and session export.

Usage:
    from detinfer import DeterministicAgent

    agent = DeterministicAgent("gpt2", seed=42)
    agent.chat("What is 2+2?")
    agent.chat("Explain more")
    agent.export_session("session.json")
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import torch

from detinfer.inference.engine import DeterministicEngine
from detinfer.agent.trace import (
    GenerationTrace,
    SessionTrace,
    build_environment,
    compute_model_hash,
    compute_tokenizer_hash,
    compute_chat_template_hash,
    _hash_string,
    _hash_token_list,
)


def deterministic_argmax(logits: torch.Tensor) -> int:
    """Deterministic argmax with stable tie-breaking.

    When multiple tokens share the maximum logit value,
    selects the smallest token ID. This guarantees identical
    results across hardware.

    Args:
        logits: 1D tensor of logit scores for each token.

    Returns:
        Token ID of the selected token.
    """
    max_val = torch.max(logits)
    candidates = torch.where(logits == max_val)[0]
    return int(candidates.min().item())


class DeterministicAgent:
    """Multi-turn deterministic chat agent with full tracing.

    Wraps DeterministicEngine to provide:
    - Multi-turn conversation with history tracking
    - Token-level trace recording per turn
    - Prompt snapshotting (rendered_prompt + hash per turn)
    - Re-seeding before each generation
    - Session export with full trace for replay/verification

    Usage:
        agent = DeterministicAgent("gpt2", seed=42)
        response = agent.chat("Hello!")
        print(response)
        agent.export_session("session.json")
    """

    def __init__(
        self,
        model_name: str,
        seed: int = 42,
        max_new_tokens: int = 256,
        trace_mode: str = "minimal",
        quantize: str | None = None,
        device: str | None = None,
        system_prompt: str | None = None,
    ):
        """Initialize the deterministic agent.

        Args:
            model_name: HuggingFace model ID.
            seed: Master random seed.
            max_new_tokens: Max tokens per generation.
            trace_mode: "minimal" (default) or "topk" for verbose trace.
            quantize: Quantization mode (None or "int8", experimental).
            device: Device to use (auto-detected if None).
            system_prompt: Optional system prompt (e.g., "You are a math tutor").
        """
        self.model_name = model_name
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.trace_mode = trace_mode
        self.quantize = quantize
        self.system_prompt = system_prompt

        # Initialize engine
        self.engine = DeterministicEngine(seed=seed, device=device)

        # Load model
        load_kwargs = {}
        if quantize == "int8":
            load_kwargs["quantize"] = "int8"
        self.engine.load(model_name, **load_kwargs)

        # Build session trace
        self.session = SessionTrace(
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
        self._conversation_history: list[dict] = []

        # Add system prompt to conversation history if provided
        if system_prompt:
            self._conversation_history.append({"role": "system", "content": system_prompt})
            self.session.add_message("system", system_prompt)

    def chat(self, message: str) -> str:
        """Send a message and get a deterministic response.

        Re-seeds RNG before each generation for strict reproducibility.
        Records full token trace for the generation.

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

        # Render prompt using chat template or plain history
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

        # Generate with tracing
        with torch.no_grad(), self.engine.enforcer.deterministic_context():
            output_ids = self.engine.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )

        # Extract generated tokens
        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_length:]
        output_token_ids = generated_ids.tolist()

        # Record steps (one per generated token)
        for i, token_id in enumerate(output_token_ids):
            gen_trace.add_step(step=i, chosen_token=token_id)

        # Set output tokens and finalize
        gen_trace.output_tokens = output_token_ids
        eos_id = self.engine.tokenizer.eos_token_id
        gen_trace.finalize(eos_token_id=eos_id)

        # Add to session
        self.session.add_generation(gen_trace)

        # Decode response
        response = self.engine.tokenizer.decode(
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

        # Render prompt and tokenize
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

        # Token-by-token generation with streaming
        with torch.no_grad(), self.engine.enforcer.deterministic_context():
            for step in range(self.max_new_tokens):
                outputs = self.engine.model(current_ids)
                next_logits = outputs.logits[0, -1, :]

                # Deterministic argmax with tie-breaking
                next_token = deterministic_argmax(next_logits)
                token_id = next_token.item()

                # Record trace step
                gen_trace.add_step(step=step, chosen_token=token_id)
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

                # Append token for next iteration
                next_token_tensor = next_token.unsqueeze(0).unsqueeze(0).to(input_device)
                current_ids = torch.cat([current_ids, next_token_tensor], dim=1)

        # Finalize trace
        gen_trace.output_tokens = generated_ids
        gen_trace.finalize(eos_token_id=eos_id)
        self.session.add_generation(gen_trace)

        # Decode full response
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        self._conversation_history.append({"role": "assistant", "content": response})
        self.session.add_message("assistant", response)

    def _render_prompt(self) -> str:
        """Render the full conversation into a prompt string.

        Uses chat template if available, otherwise raw concatenation.
        System prompt is included at the start of the conversation.
        """
        tokenizer = self.engine.tokenizer

        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            try:
                return tokenizer.apply_chat_template(
                    self._conversation_history,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Fallback: simple concatenation
        parts = []
        for msg in self._conversation_history:
            role = msg["role"].capitalize()
            if role == "System":
                parts.append(f"System: {msg['content']}")
            else:
                parts.append(f"{role}: {msg['content']}")
        parts.append("Assistant:")
        return "\n".join(parts)

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

    @property
    def history(self) -> list[dict]:
        """Return the conversation history."""
        return list(self._conversation_history)

    @property
    def turn_count(self) -> int:
        """Return the number of completed turns."""
        return self._turn_count
