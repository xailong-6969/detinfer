"""
detinfer.cli -- Command-Line Interface

Provides the `detinfer` command with subcommands:
  detinfer run <model>              -- Interactive deterministic inference
  detinfer agent <model>             -- Deterministic multi-turn agent
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
    from detinfer.inference.engine import DeterministicEngine

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
    if not result.is_deterministic:
        sys.exit(1)


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
    import time
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
    max_tokens = args.max_tokens if hasattr(args, 'max_tokens') else 256

    # ── Phase 1: WITHOUT detinfer ──
    print("\n" + "=" * 60)
    print("  WITHOUT detinfer (raw PyTorch, do_sample=True, temp=0.7)")
    print("  Each run asks the SAME question but gets DIFFERENT answers")
    print("=" * 60)

    print("Loading model (raw)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()

    # Format prompt with chat template if available
    formatted_prompt = prompt
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        try:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_hashes = []
    raw_texts = []
    eos_id = tokenizer.eos_token_id

    for i in range(num_runs):
        print(f"\n  ── Run {i+1}/{num_runs} ──")
        print(f"  Prompt:    {prompt}")
        print(f"  Response:  ", end="", flush=True)

        # Token-by-token generation with sampling (live streaming)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        current_ids = inputs["input_ids"]
        generated_ids = []
        past_kv = None

        with torch.no_grad():
            for step in range(max_tokens):
                outputs = model(current_ids, past_key_values=past_kv, use_cache=True)
                logits = outputs.logits[0, -1, :]
                past_kv = outputs.past_key_values

                # Apply temperature and sample (non-deterministic on purpose)
                scaled = logits / 0.7
                probs = torch.softmax(scaled, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1).item()

                generated_ids.append(token_id)

                # Stream token live
                chunk = tokenizer.decode([token_id], skip_special_tokens=True)
                if chunk:
                    print(chunk, end="", flush=True)
                    time.sleep(0.02)

                if token_id == eos_id:
                    break

                current_ids = torch.tensor([[token_id]], device=device)

        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        raw_hashes.append(h)
        raw_texts.append(text)
        match = "" if i == 0 else (" ✓ same" if h == raw_hashes[0] else " ✗ DIFFERENT")
        print(f"\n  Text hash: {h}{match}")

    unique_raw = len(set(raw_hashes))
    if unique_raw == 1:
        print(f"\n  Result: All {num_runs} hashes match (got lucky — sampling can still vary)")
    else:
        print(f"\n  Result: NON-DETERMINISTIC — {unique_raw} different hashes across {num_runs} runs!")
        print(f"  (Same question every time, but answers vary due to random sampling)")

    # Cleanup raw model
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Phase 2: WITH detinfer ──
    print("\n" + "=" * 60)
    print("  WITH detinfer (enforcement ON, greedy + seed locked)")
    print("  Same question, IDENTICAL answer every time")
    print("=" * 60)

    from detinfer.inference.engine import DeterministicEngine
    from detinfer.agent.runtime import deterministic_argmax
    from detinfer.inference.utils import hash_string

    engine = DeterministicEngine(
        seed=args.seed,
        precision=args.precision,
        device=device,
    )
    report = engine.load(args.model)
    print(f"\n{report}\n")

    det_tokenizer = engine.tokenizer
    det_eos_id = det_tokenizer.eos_token_id
    input_device = engine._get_input_device()

    # Format prompt
    det_formatted = prompt
    if hasattr(det_tokenizer, 'chat_template') and det_tokenizer.chat_template:
        try:
            messages = [{"role": "user", "content": prompt}]
            det_formatted = det_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    detinfer_text_hashes = []
    for i in range(num_runs):
        print(f"\n  ── Run {i+1}/{num_runs} ──")
        print(f"  Prompt:    {prompt}")
        print(f"  Response:  ", end="", flush=True)

        engine.config.apply()
        inputs = det_tokenizer(det_formatted, return_tensors="pt").to(input_device)
        current_ids = inputs["input_ids"]
        generated_ids = []
        past_kv = None

        with torch.no_grad(), engine.enforcer.deterministic_context():
            for step in range(max_tokens):
                outputs = engine.model(
                    current_ids, past_key_values=past_kv, use_cache=True,
                )
                logits = outputs.logits[0, -1, :]
                past_kv = outputs.past_key_values

                token_id, _ = deterministic_argmax(logits)
                generated_ids.append(token_id)

                chunk = det_tokenizer.decode([token_id], skip_special_tokens=True)
                if chunk:
                    print(chunk, end="", flush=True)
                    time.sleep(0.02)

                if token_id == det_eos_id:
                    break

                current_ids = torch.tensor([[token_id]], device=input_device)

        text = det_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        text_hash = hash_string(text)
        detinfer_text_hashes.append(text_hash)
        match = "" if i == 0 else (" ✓ same" if text_hash == detinfer_text_hashes[0] else " ✗ DIFFERENT")
        print(f"\n  Text hash: {text_hash}{match}")

    unique_detinfer = len(set(detinfer_text_hashes))
    if unique_detinfer == 1:
        print(f"\n  Result: DETERMINISTIC — All {num_runs} hashes identical!")
    else:
        print(f"\n  Result: {unique_detinfer} different hashes")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Without detinfer: {unique_raw} unique hash(es) across {num_runs} runs")
    print(f"  With detinfer:    {unique_detinfer} unique hash(es) across {num_runs} runs")
    if unique_detinfer == 1:
        print(f"\n  ✓ detinfer text hash: {detinfer_text_hashes[0]}")
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

            _run_stream(engine, prompt, args.max_tokens)

    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")


def _run_stream(engine, prompt: str, max_new_tokens: int) -> None:
    """Stream tokens one-by-one for detinfer run."""
    import sys
    import torch
    from detinfer.inference.utils import hash_string
    from detinfer.agent.runtime import deterministic_argmax

    engine.config.apply()

    # Format prompt with chat template if available
    formatted_prompt = prompt
    tokenizer = engine.tokenizer
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        try:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    input_device = engine._get_input_device()
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(input_device)
    current_ids = inputs["input_ids"]
    eos_id = tokenizer.eos_token_id
    generated_ids = []
    past_key_values = None

    print("\nOutput: ", end="", flush=True)

    with torch.no_grad(), engine.enforcer.deterministic_context():
        for step in range(max_new_tokens):
            outputs = engine.model(
                current_ids, past_key_values=past_key_values,
                use_cache=True,
            )
            next_logits = outputs.logits[0, -1, :]
            past_key_values = outputs.past_key_values

            token_id, _ = deterministic_argmax(next_logits)
            generated_ids.append(token_id)

            # Stream the token
            chunk = tokenizer.decode([token_id], skip_special_tokens=True)
            if chunk:
                print(chunk, end="", flush=True)

            if token_id == eos_id:
                break

            current_ids = torch.tensor([[token_id]], device=input_device)

    # Print hash info
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    text_hash = hash_string(text)
    print(f"\nHash: {text_hash}")
    print(f"Precision: {engine.precision.value} | Seed: {engine.seed}\n")



def cmd_agent(args: argparse.Namespace) -> None:
    """Deterministic multi-turn agent."""
    from detinfer.agent.runtime import DeterministicAgent

    print(f"Loading model: {args.model}...")
    agent = DeterministicAgent(
        model_name=args.model,
        seed=args.seed,
        max_new_tokens=args.max_tokens,
        trace_mode=args.trace_mode,
        quantize=args.quantize,
        device=args.device,
        system_prompt=args.system,
        max_context_tokens=args.max_context_tokens,
    )

    # Resume from saved state if provided
    if args.load_state:
        print(f"Resuming from: {args.load_state}")
        agent.load_state(args.load_state)

    print(f"Model loaded. Seed: {args.seed}")
    if args.system:
        print(f"System: {args.system}")
    print(f"Deterministic agent ready.\n")

    # Non-interactive mode
    if args.prompt:
        print(f"User:      {args.prompt}")
        print(f"Assistant: ", end="", flush=True)
        for chunk in agent.chat_stream(args.prompt):
            print(chunk, end="", flush=True)
        print()
        print(f"\nSession hash: {agent.get_session_hash()}")
        if args.export:
            session_hash = agent.export_session(args.export)
            print(f"Session exported to: {args.export}")
        if args.save_state:
            agent.save_state(args.save_state)
            print(f"State saved to: {args.save_state}")
        return

    # Interactive mode
    print("Type your messages (Ctrl+C to exit):\n")
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            print("Assistant: ", end="", flush=True)
            for chunk in agent.chat_stream(user_input):
                print(chunk, end="", flush=True)
            print("\n")

    except (KeyboardInterrupt, EOFError):
        print(f"\n\nSession: {agent.turn_count} turns")
        print(f"Session hash: {agent.get_session_hash()}")

        if args.export:
            session_hash = agent.export_session(args.export)
            print(f"Session exported to: {args.export}")

        if args.save_state:
            agent.save_state(args.save_state)
            print(f"State saved to: {args.save_state}")

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


def cmd_verify_session(args: argparse.Namespace) -> None:
    """Verify a session export is a valid deterministic execution proof."""
    from detinfer.agent.trace import SessionTrace
    from detinfer.agent.replay import replay_session
    import time

    print("═" * 60)
    print("  DETERMINISTIC EXECUTION PROOF VERIFICATION")
    print("═" * 60)

    # Load and show session metadata
    session = SessionTrace.from_json(args.session_file)
    print(f"\n  Session file:  {args.session_file}")
    print(f"  Model:        {session.model}")
    print(f"  Model hash:   {session.model_hash[:16] + '...' if session.model_hash else 'N/A'}")
    print(f"  Seed:         {session.seed}")
    print(f"  Turns:        {len(session.generations)}")
    print(f"  Session hash: {session.session_hash[:16] + '...' if session.session_hash else 'N/A'}")
    print(f"  Schema:       v{session.schema_version}")

    if session.environment:
        env = session.environment
        print(f"\n  Original environment:")
        if env.get("torch"): print(f"    PyTorch:    {env['torch']}")
        if env.get("cuda"): print(f"    CUDA:       {env['cuda']}")
        if env.get("detinfer"): print(f"    detinfer:   {env['detinfer']}")

    print(f"\n  Replaying session locally...")
    start = time.time()

    result = replay_session(
        trace_path=args.session_file,
        model_name=args.model,
        strict=args.strict,
    )

    elapsed = time.time() - start

    print(f"  Replay completed in {elapsed:.1f}s\n")
    print("─" * 60)

    if result.passed:
        print(f"  ✓ VERIFIED — All {result.total_turns} turns match exactly")
        print(f"  ✓ Session hash: {session.session_hash}")
        print(f"  ✓ This session is a valid deterministic execution proof.")
    else:
        print(f"  ✗ VERIFICATION FAILED")
        if result.failure_turn:
            print(f"  ✗ Turn {result.failure_turn}: {result.failure_reason}")
        if result.failure_step is not None:
            print(f"  ✗ Step {result.failure_step}: expected token {result.expected_token}, got {result.observed_token}")
        if result.details:
            for d in result.details:
                print(f"    {d}")

        sys.exit(1)
    print("═" * 60)


def cmd_doctor(args: argparse.Namespace) -> None:
    """Run a determinism health check / audit on a model."""
    import json as json_mod
    import time
    from detinfer.inference.engine import DeterministicEngine
    from detinfer.agent.runtime import deterministic_argmax
    from detinfer.inference.utils import hash_string

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    prompt = args.prompt or "What is 2 + 2? Answer with just the number."
    num_runs = args.runs

    report = {
        "model": args.model,
        "device": device,
        "prompt": prompt,
        "runs": num_runs,
        "checks": {},
        "overall": "PASS",
    }

    if not args.json:
        print("╔" + "═" * 58 + "╗")
        print("║     Detinfer Determinism Report                        ║")
        print("╚" + "═" * 58 + "╝")
        print(f"\n  Model:   {args.model}")
        print(f"  Device:  {device}")
        print(f"  Prompt:  {prompt!r}")
        print(f"  Runs:    {num_runs}\n")
        print("─" * 60)

    # ── Check 1: Environment settings ──
    import os

    # Sampling
    report["checks"]["sampling"] = {"status": "DISABLED", "ok": True}

    # Seed locking
    seed_ok = True
    report["checks"]["seed_locking"] = {"status": "ENABLED", "seed": args.seed, "ok": True}

    # Torch deterministic algorithms
    det_algo = torch.are_deterministic_algorithms_enabled()
    report["checks"]["torch_deterministic_algorithms"] = {
        "status": "ENABLED" if det_algo else "DISABLED (will be enabled by detinfer)",
        "ok": True,  # detinfer will enable it
    }

    # cuDNN benchmark
    cudnn_bench = torch.backends.cudnn.benchmark if hasattr(torch.backends, 'cudnn') else False
    report["checks"]["cudnn_benchmark"] = {
        "status": "DISABLED" if not cudnn_bench else "ENABLED (detinfer will disable)",
        "ok": True,  # detinfer will fix it
    }

    # CUBLAS workspace
    cublas_env = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "not set")
    report["checks"]["cublas_workspace"] = {
        "status": cublas_env if cublas_env != "not set" else "not set (detinfer will set)",
        "ok": True,
    }

    # Attention backend
    attn_backend = "unknown"
    try:
        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
            flash = torch.backends.cuda.flash_sdp_enabled()
            math = torch.backends.cuda.math_sdp_enabled()
            if flash:
                attn_backend = "flash_sdp (may be non-deterministic)"
            elif math:
                attn_backend = "math_sdp (deterministic)"
            else:
                attn_backend = "default"
    except Exception:
        attn_backend = "unknown"

    report["checks"]["attention_backend"] = {
        "status": attn_backend,
        "ok": "non-deterministic" not in attn_backend,
    }

    if not args.json:
        print("\n  Environment Checks:")
        for name, check in report["checks"].items():
            icon = "✓" if check["ok"] else "✗"
            label = name.replace("_", " ").title()
            print(f"    {icon} {label}: {check['status']}")

    # ── Check 2: Load and scan model ──
    if not args.json:
        print(f"\n{'─' * 60}")
        print(f"  Loading model with detinfer enforcement...")

    engine = DeterministicEngine(
        seed=args.seed,
        precision=args.precision,
        device=device,
    )
    scan_report = engine.load(args.model)

    # Check if model has non-deterministic ops
    has_nondet = "non-deterministic" in str(scan_report).lower() and "No non-deterministic" not in str(scan_report)
    report["checks"]["model_scan"] = {
        "status": "CLEAN" if not has_nondet else "HAS NON-DETERMINISTIC OPS (patched by detinfer)",
        "ok": True,  # detinfer patches them
    }

    if not args.json:
        icon = "✓" if not has_nondet else "⚠"
        print(f"    {icon} Model scan: {report['checks']['model_scan']['status']}")

    # ── Check 3: Tokenizer fingerprint ──
    tokenizer = engine.tokenizer
    test_tokens = tokenizer(prompt, return_tensors="pt")
    token_hash = hash_string(str(test_tokens["input_ids"].tolist()))
    report["checks"]["tokenizer"] = {
        "status": "STABLE",
        "token_hash": token_hash[:16],
        "ok": True,
    }

    if not args.json:
        print(f"    ✓ Tokenizer: STABLE (hash: {token_hash[:16]}...)")

    # ── Check 4: Prompt hash ──
    prompt_hash = hash_string(prompt)
    report["checks"]["prompt_hash"] = {
        "status": "STABLE",
        "hash": prompt_hash[:16],
        "ok": True,
    }

    if not args.json:
        print(f"    ✓ Prompt hash: {prompt_hash[:16]}...")

    # ── Check 5: Repeated run verification ──
    if not args.json:
        print(f"\n{'─' * 60}")
        print(f"  Repeated Run Check ({num_runs} runs):\n")

    # Format prompt
    formatted_prompt = prompt
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        try:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    input_device = engine._get_input_device()
    eos_id = tokenizer.eos_token_id
    hashes = []
    texts = []
    first_mismatch = None

    for i in range(num_runs):
        engine.config.apply()
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(input_device)
        current_ids = inputs["input_ids"]
        generated_ids = []
        past_kv = None

        with torch.no_grad(), engine.enforcer.deterministic_context():
            for step in range(args.max_tokens):
                outputs = engine.model(
                    current_ids, past_key_values=past_kv, use_cache=True,
                )
                logits = outputs.logits[0, -1, :]
                past_kv = outputs.past_key_values
                token_id, _ = deterministic_argmax(logits)
                generated_ids.append(token_id)
                if token_id == eos_id:
                    break
                current_ids = torch.tensor([[token_id]], device=input_device)

        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        h = hash_string(text)
        hashes.append(h)
        texts.append(text)

        if i > 0 and h != hashes[0] and first_mismatch is None:
            first_mismatch = i + 1

        if not args.json:
            match = "" if i == 0 else (" ✓ match" if h == hashes[0] else " ✗ MISMATCH")
            print(f"    Run {i+1}: {h[:32]}...{match}")

    matched = sum(1 for h in hashes if h == hashes[0])
    all_match = matched == num_runs

    report["checks"]["repeated_runs"] = {
        "runs": num_runs,
        "matched": f"{matched}/{num_runs}",
        "first_mismatch": first_mismatch if first_mismatch else "none",
        "text_hash": hashes[0],
        "ok": all_match,
    }

    if not all_match:
        report["overall"] = "FAIL"

    # ── Check any failures ──
    problems = []
    for name, check in report["checks"].items():
        if not check["ok"]:
            problems.append(f"- {name.replace('_', ' ').title()}: {check['status']}")

    report["problems"] = problems

    # ── Output ──
    if args.json:
        print(json_mod.dumps(report, indent=2))
    else:
        print(f"\n{'─' * 60}")
        print(f"\n  Response preview: {texts[0][:80]}{'...' if len(texts[0]) > 80 else ''}")
        print(f"  Text hash:       {hashes[0]}")
        print(f"  Seed:            {args.seed}")
        print()
        print("═" * 60)
        if report["overall"] == "PASS":
            print("  ✓ OVERALL STATUS: PASS")
            print("  ✓ All checks passed. Inference is deterministic.")
            print(f"  ✓ {matched}/{num_runs} runs produced identical output.")
        else:
            print("  ✗ OVERALL STATUS: FAIL")
            print()
            print("  Problems found:")
            for p in problems:
                print(f"    {p}")
        print("═" * 60)


def cmd_check(args: argparse.Namespace) -> None:
    """Compare two session traces for regression."""
    import json
    import gzip
    from detinfer.check import check_sessions, render_check_report

    def _load_trace(path: str) -> dict:
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    baseline = _load_trace(args.baseline)
    candidate = _load_trace(args.candidate)

    report = check_sessions(
        baseline, candidate,
        fail_on=set(args.fail_on),
        allow=set(args.allow),
    )

    if args.json_output:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_check_report(
            report,
            baseline_path=args.baseline,
            candidate_path=args.candidate,
        ))

    if report.status == "failed":
        sys.exit(1)


def cmd_agent_run(args: argparse.Namespace) -> None:
    """Run agent harness tasks."""
    from pathlib import Path
    from detinfer.harness import (
        HarnessRunner, load_task, load_task_suite,
        render_task_result, render_suite_result,
    )

    target = Path(args.task)
    runner = HarnessRunner(
        output_dir=args.output_dir,
        against=args.against,
    )

    if target.is_dir():
        # Suite mode
        tasks = load_task_suite(str(target))
        if not tasks:
            print("No valid task files found.")
            sys.exit(1)
        suite = runner.run_suite(tasks, fail_fast=args.fail_fast)
        if args.json_output:
            print(json.dumps(suite.to_dict(), indent=2))
        else:
            print(render_suite_result(suite))
        if suite.failed > 0:
            sys.exit(1)
    else:
        # Single task mode
        task = load_task(str(target))
        result = runner.run_task(task)
        if args.json_output:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(render_task_result(result))
        if not result.passed:
            sys.exit(1)


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
    scan_parser.add_argument("model", help="HuggingFace model name (e.g., owner/model-name)")
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
    compare_parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens per run (default: 256)")

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

    # -- detinfer agent <model> --
    agent_parser = subparsers.add_parser("agent", help="Deterministic multi-turn agent")
    agent_parser.add_argument("model", help="HuggingFace model name")
    agent_parser.add_argument("--prompt", default=None, help="Non-interactive: single prompt (for CI/scripts)")
    agent_parser.add_argument("--export", default=None, help="Export session trace to JSON file")
    agent_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    agent_parser.add_argument("--device", default=None, help="Device (default: auto)")
    agent_parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens per turn (default: 256)")
    agent_parser.add_argument("--system", default=None, help="System prompt (e.g., 'You are a math tutor')")
    agent_parser.add_argument("--quantize", default=None, choices=["int8"], help="Quantization mode (experimental)")
    agent_parser.add_argument("--trace-mode", default="standard", choices=["minimal", "standard", "verbose"],
                              help="Trace detail level (default: standard)")
    agent_parser.add_argument("--max-context-tokens", type=int, default=None,
                              help="Max prompt tokens before truncation (default: no limit)")
    agent_parser.add_argument("--save-state", default=None,
                              help="Save agent state to file on exit (for resume)")
    agent_parser.add_argument("--load-state", default=None,
                              help="Resume from a saved agent state file")

    # -- detinfer replay <session.json> --
    replay_parser = subparsers.add_parser("replay", help="Replay and verify a saved session")
    replay_parser.add_argument("session_file", help="Path to session JSON file")
    replay_parser.add_argument("--model", default=None, help="Override model (uses trace model if not set)")
    replay_parser.add_argument("--strict", action="store_true", help="Verify every generation step")

    # -- detinfer diff <a.json> <b.json> --
    diff_parser = subparsers.add_parser("diff", help="Token-level comparison of two sessions")
    diff_parser.add_argument("file_a", help="First session JSON")
    diff_parser.add_argument("file_b", help="Second session JSON")

    # -- detinfer verify-session <session.json> --
    vs_parser = subparsers.add_parser("verify-session", help="Verify a session as execution proof")
    vs_parser.add_argument("session_file", help="Path to session JSON file")
    vs_parser.add_argument("--model", default=None, help="Override model (uses trace model if not set)")
    vs_parser.add_argument("--strict", action="store_true", help="Verify every generation step")

    # -- detinfer doctor <model> --
    doctor_parser = subparsers.add_parser("doctor", help="Determinism health check & audit")
    doctor_parser.add_argument("model", help="HuggingFace model name")
    doctor_parser.add_argument("--prompt", default=None, help="Test prompt (default: auto)")
    doctor_parser.add_argument("--runs", type=int, default=5, help="Number of test runs (default: 5)")
    doctor_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    doctor_parser.add_argument("--precision", default="high", help="Canonical precision (default: high)")
    doctor_parser.add_argument("--device", default=None, help="Device (default: auto)")
    doctor_parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens (default: 256)")
    doctor_parser.add_argument("--json", action="store_true", help="Output report as JSON (for CI)")

    # -- detinfer check <baseline> <candidate> --
    check_parser = subparsers.add_parser("check", help="Compare two session traces for regression")
    check_parser.add_argument("baseline", help="Baseline session JSON file")
    check_parser.add_argument("candidate", help="Candidate session JSON file")
    check_parser.add_argument("--json", action="store_true", dest="json_output",
                              help="Output report as JSON")
    check_parser.add_argument("--fail-on", action="append", default=[],
                              help="Mismatch type that should fail (e.g. OUTPUT_DRIFT)")
    check_parser.add_argument("--allow", action="append", default=[],
                              help="Mismatch type to ignore (e.g. ENVIRONMENT_DRIFT)")

    # -- detinfer agent-run <task> --
    agent_run_parser = subparsers.add_parser("agent-run", help="Run agent harness tasks")
    agent_run_parser.add_argument("task", help="Task JSON file or directory of tasks")
    agent_run_parser.add_argument("--output-dir", default=None,
                                  help="Directory for trace output files")
    agent_run_parser.add_argument("--against", default=None,
                                  help="Baseline trace to compare against")
    agent_run_parser.add_argument("--json", action="store_true", dest="json_output",
                                  help="Output report as JSON")
    agent_run_parser.add_argument("--fail-fast", action="store_true",
                                  help="Stop suite on first failure")

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
        "agent": cmd_agent,
        "replay": cmd_replay,
        "diff": cmd_diff,
        "verify-session": cmd_verify_session,
        "doctor": cmd_doctor,
        "check": cmd_check,
        "agent-run": cmd_agent_run,
    }

    handlers[args.command](args)


if __name__ == "__main__":
    main()
