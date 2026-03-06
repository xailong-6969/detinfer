"""
determl.cli -- Command-Line Interface

Provides the `determl` command with subcommands:
  determl run <model>              -- Interactive deterministic inference
  determl scan <model>             -- Scan for non-deterministic ops
  determl verify <model>           -- Verify determinism (auto-prompt)
  determl compare <model>          -- Before/after determl comparison
  determl info                     -- Show environment info
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time

import torch


def cmd_info(args: argparse.Namespace) -> None:
    """Show environment information."""
    from determl.guardian import EnvironmentGuardian

    guardian = EnvironmentGuardian()
    fingerprint = guardian.create_fingerprint()
    print(fingerprint)


def cmd_scan(args: argparse.Namespace) -> None:
    """Scan a model for non-deterministic ops."""
    from determl.engine import DeterministicEngine

    print(f"Loading model: {args.model}...")
    engine = DeterministicEngine(
        seed=args.seed,
        device=args.device,
    )
    report = engine.load(args.model)
    print(f"\n{report}")


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify a model produces deterministic output."""
    from determl.engine import DeterministicEngine

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


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare model output with and without determl enforcement."""
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

    # ── Phase 1: WITHOUT determl ──
    print("\n" + "=" * 60)
    print("  WITHOUT determl (raw PyTorch, no enforcement)")
    print("=" * 60)

    print("Loading model (raw)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
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
        print("  (Greedy decoding on this GPU happens to be stable,")
        print("   but internal float values may still drift across different GPUs)")
    else:
        print(f"\n  Result: NON-DETERMINISTIC — {unique_raw} different hashes!")

    # Cleanup raw model
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Phase 2: WITH determl ──
    print("\n" + "=" * 60)
    print("  WITH determl (enforcement ON)")
    print("=" * 60)

    from determl.engine import DeterministicEngine

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
    print(f"  Without determl: {unique_raw} unique hash(es) across {num_runs} runs")
    print(f"  With determl:    {unique_determl} unique hash(es) across {num_runs} runs")
    if unique_determl == 1:
        print(f"\n  ✓ determl canonical hash: {determl_hashes[0]}")
        print(f"  ✓ Seed: {args.seed}")
        print(f"  ✓ This hash will be identical on any machine with the same model.")
    print()


def cmd_run(args: argparse.Namespace) -> None:
    """Interactive deterministic inference."""
    from determl.engine import DeterministicEngine

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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="determl",
        description="Deterministic ML Inference Tool",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- determl info --
    subparsers.add_parser("info", help="Show environment information")

    # -- determl scan <model> --
    scan_parser = subparsers.add_parser("scan", help="Scan model for non-deterministic ops")
    scan_parser.add_argument("model", help="HuggingFace model name (e.g., Qwen/Qwen2.5-Coder-0.5B-Instruct)")
    scan_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    scan_parser.add_argument("--device", default=None, help="Device (cpu/cuda, default: auto)")

    # -- determl verify <model> --
    verify_parser = subparsers.add_parser("verify", help="Verify model determinism")
    verify_parser.add_argument("model", help="HuggingFace model name")
    verify_parser.add_argument("--prompt", default=None, help="Test prompt (default: auto)")
    verify_parser.add_argument("--runs", type=int, default=5, help="Number of runs (default: 5)")
    verify_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    verify_parser.add_argument("--precision", default="high", help="Canonical precision (default: high)")
    verify_parser.add_argument("--device", default=None, help="Device (default: auto)")

    # -- determl compare <model> --
    compare_parser = subparsers.add_parser("compare", help="Before/after determl comparison")
    compare_parser.add_argument("model", help="HuggingFace model name")
    compare_parser.add_argument("--prompt", default=None, help="Test prompt (default: auto)")
    compare_parser.add_argument("--runs", type=int, default=5, help="Number of runs (default: 5)")
    compare_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    compare_parser.add_argument("--precision", default="high", help="Canonical precision (default: high)")
    compare_parser.add_argument("--device", default=None, help="Device (default: auto)")

    # -- determl run <model> --
    run_parser = subparsers.add_parser("run", help="Interactive deterministic inference")
    run_parser.add_argument("model", help="HuggingFace model name")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    run_parser.add_argument("--precision", default="high", help="Canonical precision (default: high)")
    run_parser.add_argument("--device", default=None, help="Device (default: auto)")
    run_parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens (default: 256)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "info": cmd_info,
        "scan": cmd_scan,
        "verify": cmd_verify,
        "compare": cmd_compare,
        "run": cmd_run,
    }

    handlers[args.command](args)


if __name__ == "__main__":
    main()
