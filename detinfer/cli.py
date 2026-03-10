"""
detinfer.cli -- Command-Line Interface

Provides the `detinfer` command with subcommands:
  detinfer run <model>              -- Interactive deterministic inference
  detinfer chat <model>             -- Deterministic multi-turn chat agent
  detinfer replay <session.json>    -- Replay and verify a saved session
  detinfer diff <a.json> <b.json>   -- Token-level comparison of two sessions
  detinfer scan <model>             -- Scan for non-deterministic ops
  detinfer verify <model>           -- Verify determinism (auto-prompt)
  detinfer compare <model>          -- Before/after detinfer comparison
  detinfer export <model>           -- Export inference proof to JSON
  detinfer cross-verify <proof.json> -- Verify proof on this machine
  detinfer info                     -- Show environment info
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time

import torch


def cmd_info(args: argparse.Namespace) -> None:
    """Show environment information."""
    from detinfer.inference.guardian import EnvironmentGuardian

    guardian = EnvironmentGuardian()
    fingerprint = guardian.create_fingerprint()
    print(fingerprint)


def cmd_scan(args: argparse.Namespace) -> None:
    """Scan a model for non-deterministic ops."""
    from detinfer.inference.engine import DeterministicEngine

    print(f"Loading model: {args.model}...")
    engine = DeterministicEngine(
        seed=args.seed,
        device=args.device,
    )
    report = engine.load(args.model)
    print(f"\n{report}")


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify a model produces deterministic output."""
    from detinfer.engine import DeterministicEngine

    print(f"Loading model: {args.model}...")
    engine = DeterministicEngine(
        seed=args.seed,
        precision=args.precision,
        device=args.device,
    )
    engine.load(args.model)

    prompt = args.prompt or "What is 2 + 2? Answer with just the number."
    print(f"Verifying with prompt: {prompt!r}")
    print(f"Running {args.runs} times...\n")

    result = engine.verify(prompt=prompt, num_runs=args.runs)
    print(result)


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run the auto-scaling determinism benchmark."""
    from detinfer.inference.engine import DeterministicEngine
    from detinfer.inference.benchmark import BenchmarkConfig, run_benchmark, estimate_param_count

    print(f"Loading model: {args.model}...")
    engine = DeterministicEngine(
        seed=args.seed,
        precision=args.precision,
        device=args.device,
    )
    engine.load(args.model)

    # Auto-detect model size for scaling
    param_b = estimate_param_count(engine.model) if engine.model else None
    config = BenchmarkConfig.from_depth(args.depth, param_b)

    print(f"Benchmark depth: {config.depth}")
    print(f"Prompts: {config.num_prompts} | Runs per prompt: {config.runs_per_prompt} | Total: {config.total_runs}")
    print("\nRunning benchmark...\n")

    result = run_benchmark(engine, config, max_new_tokens=args.max_tokens)
    print(result)


def cmd_export(args: argparse.Namespace) -> None:
    """Export an inference proof to a JSON file."""
    from detinfer.inference.engine import DeterministicEngine
    from detinfer.inference.proof import create_proof

    print(f"Loading model: {args.model}...")
    engine = DeterministicEngine(
        seed=args.seed,
        precision=args.precision,
        device=args.device,
    )
    engine.load(args.model)

    prompt = args.prompt or "What is 2 + 2? Answer with just the number."
    print(f"Running inference with prompt: {prompt!r}")

    proof = create_proof(engine, prompt, max_new_tokens=args.max_tokens)
    proof.save(args.output)

    print(f"\n{proof}")
    print(f"\nProof saved to: {args.output}")
    print(f"\nCopy this file to another machine and run:")
    print(f"  detinfer cross-verify {args.output}")


def cmd_cross_verify(args: argparse.Namespace) -> None:
    """Verify an inference proof on this machine."""
    from detinfer.inference.proof import InferenceProof, cross_verify

    print(f"Loading proof from: {args.proof_file}")
    proof = InferenceProof.load(args.proof_file)
    print(f"\nOriginal proof:")
    print(proof)

    print(f"\nRe-running inference locally...")
    result = cross_verify(proof)
    print(result)


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare model output with and without detinfer enforcement."""
    import random
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompt = args.prompt or "What is 2 + 2? Answer with just the number."
    num_runs = args.runs
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Model:  {args.model}")
    print(f"Prompt: {prompt!r}")
    print(f"Device: {device}")
    print(f"Runs:   {num_runs}")

    # ── Phase 1: WITHOUT detinfer ──
    print("\n" + "=" * 60)
    print("  WITHOUT detinfer (raw PyTorch, no enforcement)")
    print("=" * 60)

    print("Loading model (raw)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()

    raw_hashes = []
    raw_texts = []
    for i in range(num_runs):
        # Do NOT lock seeds — let it be natural
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )
        output_bytes = output.cpu().numpy().tobytes()
        h = hashlib.sha256(output_bytes).hexdigest()
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        raw_hashes.append(h)
        raw_texts.append(text)
        match = "" if i == 0 else (" ✓ same" if h == raw_hashes[0] else " ✗ DIFFERENT")
        print(f"  Run {i+1}: {h}{match}")

    unique_raw = len(set(raw_hashes))
    if unique_raw == 1:
        print(f"\n  Result: All {num_runs} hashes match")
        print(f"  (Greedy decoding on {device.upper()} happens to be stable,")
        print(f"   but internal float values may still drift across different hardware)")
    else:
        print(f"\n  Result: NON-DETERMINISTIC — {unique_raw} different hashes!")

    # Cleanup raw model
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Phase 2: WITH detinfer ──
    print("\n" + "=" * 60)
    print("  WITH detinfer (enforcement ON)")
    print("=" * 60)

    from detinfer.inference.engine import DeterministicEngine

    engine = DeterministicEngine(
        seed=args.seed,
        precision=args.precision,
        device=device,
    )
    report = engine.load(args.model)
    print(f"\n{report}\n")

    determl_hashes = []
    for i in range(num_runs):
        result = engine.run(prompt, max_new_tokens=50)
        h = result.canonical_hash
        determl_hashes.append(h)
        match = "" if i == 0 else (" ✓ same" if h == determl_hashes[0] else " ✗ DIFFERENT")
        print(f"  Run {i+1}: {h}{match}")

    unique_determl = len(set(determl_hashes))
    if unique_determl == 1:
        print(f"\n  Result: DETERMINISTIC — All {num_runs} hashes identical!")
    else:
        print(f"\n  Result: {unique_determl} different hashes")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Without detinfer: {unique_raw} unique hash(es) across {num_runs} runs")
    print(f"  With detinfer:    {unique_determl} unique hash(es) across {num_runs} runs")
    if unique_determl == 1:
        print(f"\n  ✓ detinfer canonical hash: {determl_hashes[0]}")
        print(f"  ✓ Seed: {args.seed}")
        print(f"  ✓ This hash will be identical on any machine with the same model.")
    print()


def cmd_run(args: argparse.Namespace) -> None:
    """Interactive deterministic inference."""
    from detinfer.inference.engine import DeterministicEngine

    print(f"Loading model: {args.model}...")
    engine = DeterministicEngine(
        seed=args.seed,
        precision=args.precision,
        device=args.device,
    )
    report = engine.load(args.model)

    print(f"\n{report}")
    print(f"\nEngine: {engine}")
    print("\nReady! Type your prompt (Ctrl+C to exit):\n")

    try:
        while True:
            prompt = input("> ").strip()
            if not prompt:
                continue

            result = engine.run(prompt, max_new_tokens=args.max_tokens)
            print(f"\n{result}\n")

    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")


def cmd_chat(args: argparse.Namespace) -> None:
    """Deterministic multi-turn chat agent."""
    from detinfer.agent.runtime import DeterministicAgent

    print(f"Loading model: {args.model}...")
    agent = DeterministicAgent(
        model_name=args.model,
        seed=args.seed,
        max_new_tokens=args.max_tokens,
        trace_mode="topk" if args.verbose_trace else "minimal",
        quantize=args.quantize,
        device=args.device,
        system_prompt=args.system,
    )

    print(f"Model loaded. Seed: {args.seed}")
    if args.system:
        print(f"System: {args.system}")
    print(f"Deterministic chat ready.\n")

    # Non-interactive mode
    if args.prompt:
        if args.stream:
            import sys
            print(f"User:      {args.prompt}")
            print(f"Assistant: ", end="", flush=True)
            for chunk in agent.chat_stream(args.prompt):
                print(chunk, end="", flush=True)
            print()
        else:
            response = agent.chat(args.prompt)
            print(f"User:      {args.prompt}")
            print(f"Assistant: {response}")
        print(f"\nSession hash: {agent.get_session_hash()}")
        if args.export:
            session_hash = agent.export_session(args.export)
            print(f"Session exported to: {args.export}")
        return

    # Interactive mode
    print("Type your messages (Ctrl+C to exit):\n")
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if args.stream:
                import sys
                print("Assistant: ", end="", flush=True)
                for chunk in agent.chat_stream(user_input):
                    print(chunk, end="", flush=True)
                print("\n")
            else:
                response = agent.chat(user_input)
                print(f"Assistant: {response}\n")

    except (KeyboardInterrupt, EOFError):
        print(f"\n\nSession: {agent.turn_count} turns")
        print(f"Session hash: {agent.get_session_hash()}")

        if args.export:
            session_hash = agent.export_session(args.export)
            print(f"Session exported to: {args.export}")

        print("Goodbye!")


def cmd_replay(args: argparse.Namespace) -> None:
    """Replay and verify a saved session."""
    from detinfer.agent.replay import replay_session

    print(f"Loading session: {args.session_file}")
    print(f"Strict mode: {'ON' if args.strict else 'OFF'}")
    print(f"Replaying...\n")

    result = replay_session(
        trace_path=args.session_file,
        model_name=args.model,
        strict=args.strict,
    )
    print(result)


def cmd_diff(args: argparse.Namespace) -> None:
    """Token-level comparison of two session traces."""
    from detinfer.agent.replay import diff_sessions

    print(f"Comparing:")
    print(f"  A: {args.file_a}")
    print(f"  B: {args.file_b}")
    print()

    result = diff_sessions(args.file_a, args.file_b)
    print(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="detinfer",
        description="Deterministic ML Inference Tool",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- detinfer info --
    subparsers.add_parser("info", help="Show environment information")

    # -- detinfer scan <model> --
    scan_parser = subparsers.add_parser("scan", help="Scan model for non-deterministic ops")
    scan_parser.add_argument("model", help="HuggingFace model name (e.g., Qwen/Qwen2.5-Coder-0.5B-Instruct)")
    scan_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    scan_parser.add_argument("--device", default=None, help="Device (cpu/cuda, default: auto)")

    # -- detinfer verify <model> --
    verify_parser = subparsers.add_parser("verify", help="Verify model determinism")
    verify_parser.add_argument("model", help="HuggingFace model name")
    verify_parser.add_argument("--prompt", default=None, help="Test prompt (default: auto)")
    verify_parser.add_argument("--runs", type=int, default=5, help="Number of runs (default: 5)")
    verify_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    verify_parser.add_argument("--precision", default="high", help="Canonical precision (default: high)")
    verify_parser.add_argument("--device", default=None, help="Device (default: auto)")

    # -- detinfer compare <model> --
    compare_parser = subparsers.add_parser("compare", help="Before/after detinfer comparison")
    compare_parser.add_argument("model", help="HuggingFace model name")
    compare_parser.add_argument("--prompt", default=None, help="Test prompt (default: auto)")
    compare_parser.add_argument("--runs", type=int, default=5, help="Number of runs (default: 5)")
    compare_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    compare_parser.add_argument("--precision", default="high", help="Canonical precision (default: high)")
    compare_parser.add_argument("--device", default=None, help="Device (default: auto)")

    # -- detinfer benchmark <model> --
    bench_parser = subparsers.add_parser("benchmark", help="Auto-scaling determinism benchmark")
    bench_parser.add_argument("model", help="HuggingFace model name")
    bench_parser.add_argument("--depth", default="auto", choices=["auto", "light", "standard", "deep"], help="Benchmark depth (default: auto)")
    bench_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    bench_parser.add_argument("--precision", default="high", help="Canonical precision (default: high)")
    bench_parser.add_argument("--device", default=None, help="Device (default: auto)")
    bench_parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens per prompt (default: 256)")

    # -- detinfer export <model> --
    export_parser = subparsers.add_parser("export", help="Export inference proof to JSON")
    export_parser.add_argument("model", help="HuggingFace model name")
    export_parser.add_argument("--output", "-o", default="proof.json", help="Output file (default: proof.json)")
    export_parser.add_argument("--prompt", default=None, help="Test prompt (default: auto)")
    export_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    export_parser.add_argument("--precision", default="high", help="Canonical precision (default: high)")
    export_parser.add_argument("--device", default=None, help="Device (default: auto)")
    export_parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens (default: 256)")

    # -- detinfer cross-verify <proof.json> --
    xverify_parser = subparsers.add_parser("cross-verify", help="Verify proof on this machine")
    xverify_parser.add_argument("proof_file", help="Path to proof JSON file")

    # -- detinfer run <model> --
    run_parser = subparsers.add_parser("run", help="Interactive deterministic inference")
    run_parser.add_argument("model", help="HuggingFace model name")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    run_parser.add_argument("--precision", default="high", help="Canonical precision (default: high)")
    run_parser.add_argument("--device", default=None, help="Device (default: auto)")
    run_parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens (default: 256)")

    # -- detinfer chat <model> --
    chat_parser = subparsers.add_parser("chat", help="Deterministic multi-turn chat agent")
    chat_parser.add_argument("model", help="HuggingFace model name")
    chat_parser.add_argument("--prompt", default=None, help="Non-interactive: single prompt (for CI/scripts)")
    chat_parser.add_argument("--export", default=None, help="Export session trace to JSON file")
    chat_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    chat_parser.add_argument("--device", default=None, help="Device (default: auto)")
    chat_parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens per turn (default: 256)")
    chat_parser.add_argument("--system", default=None, help="System prompt (e.g., 'You are a math tutor')")
    chat_parser.add_argument("--quantize", default=None, choices=["int8"], help="Quantization mode (experimental)")
    chat_parser.add_argument("--verbose-trace", action="store_true", help="Record top-k tokens per step")
    chat_parser.add_argument("--stream", action="store_true", help="Stream tokens as they are generated")

    # -- detinfer replay <session.json> --
    replay_parser = subparsers.add_parser("replay", help="Replay and verify a saved session")
    replay_parser.add_argument("session_file", help="Path to session JSON file")
    replay_parser.add_argument("--model", default=None, help="Override model (uses trace model if not set)")
    replay_parser.add_argument("--strict", action="store_true", help="Verify every generation step")

    # -- detinfer diff <a.json> <b.json> --
    diff_parser = subparsers.add_parser("diff", help="Token-level comparison of two sessions")
    diff_parser.add_argument("file_a", help="First session JSON")
    diff_parser.add_argument("file_b", help="Second session JSON")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "info": cmd_info,
        "scan": cmd_scan,
        "verify": cmd_verify,
        "compare": cmd_compare,
        "benchmark": cmd_benchmark,
        "export": cmd_export,
        "cross-verify": cmd_cross_verify,
        "run": cmd_run,
        "chat": cmd_chat,
        "replay": cmd_replay,
        "diff": cmd_diff,
    }

    handlers[args.command](args)


if __name__ == "__main__":
    main()

