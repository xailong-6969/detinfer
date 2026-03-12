# detinfer -- Deterministic Inference & Replay Toolkit

**Deterministic runtime controls, session tracing, and replay verification for LLM workflows.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-163%20passed-brightgreen.svg)]()
[![PyPI](https://img.shields.io/badge/pypi-v0.3.0-blue.svg)](https://pypi.org/project/detinfer/)

> **New to detinfer?** Start with the [Simple Guide](docs/GUIDE.md) -- explains everything in plain language.

---

## Table of Contents

1. [What is detinfer](#what-is-detinfer)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Part 1: Inference](#part-1-inference)
5. [Part 2: Agent](#part-2-agent)
6. [Part 3: Agent Harness](#part-3-agent-harness)
7. [How It Works](#how-it-works)
8. [CLI Reference](#cli-reference)
9. [API Reference](#api-reference)
10. [GitHub Action](#github-action)
11. [Compatibility](#compatibility)
12. [Running Tests](#running-tests)

---

## What is detinfer

Most reproducibility guides stop at setting RNG seeds. In practice, LLM outputs can still drift because of decoding settings, backend behavior, prompt rendering, tokenizer differences, and attention implementations.

detinfer tackles this by combining:

- **Runtime enforcement** -- locks all random seeds, disables non-deterministic CUDA ops
- **Session tracing** -- records every token generated, with hashes for verification
- **Replay and diff** -- re-run or compare any two sessions token by token
- **Regression checking** -- classify exactly what changed between two runs

### What detinfer does NOT do

- Guarantee determinism for all PyTorch workloads universally
- Guarantee bitwise-identical outputs across different GPU models
- Replace careful environment control (tokenizer, prompt format, backend)

---

## Installation

### Step 1: Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows
```

### Step 2: Install detinfer

Basic install (core only):

```bash
pip install detinfer
```

With HuggingFace model support (recommended):

```bash
pip install "detinfer[transformers]"
```

With INT8 quantization (experimental):

```bash
pip install "detinfer[quantized]"
```

### From source

```bash
git clone https://github.com/xailong-6969/detinfer.git
cd detinfer
python3 -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

pip install -e "."                # Basic install
pip install -e ".[transformers]"  # With HuggingFace support
pip install -e ".[dev]"           # For development and testing
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- NVIDIA GPU (recommended) or CPU

---

## Quick Start

### One-line determinism

```python
import detinfer
detinfer.enforce(seed=42)

# Everything after this line runs with detinfer's deterministic controls enabled
output = model(input)  # Same input = same output, every time
```

### Run from CLI

```bash
# Interactive deterministic inference
detinfer run <hf-model>

# Verify determinism (runs 5 times, checks all match)
detinfer verify <hf-model>

# Multi-turn deterministic agent
detinfer agent <hf-model>
```

Replace `<hf-model>` with a supported HuggingFace model name. See [Compatibility](#compatibility).

---

## Part 1: Inference

The inference module enforces deterministic runtime settings and provides tools to verify, benchmark, and export deterministic proofs.

### 1.1 Runtime Enforcement

```python
import detinfer

detinfer.enforce(seed=42)
```

This single call:

- Locks Python, NumPy, PyTorch, and CUDA random seeds
- Disables cuDNN autotuning
- Enables `torch.use_deterministic_algorithms(True)`
- Configures cuBLAS workspace

### 1.2 DeterministicEngine

```python
from detinfer import DeterministicEngine

engine = DeterministicEngine(seed=42)
engine.load("<hf-model>")

result = engine.run("Write hello world in Python")
print(result.text)            # The generated text
print(result.canonical_hash)  # SHA-256 hash for verification
```

### 1.3 Verify Determinism

Run the same prompt multiple times and verify all outputs match:

```python
result = engine.verify(num_runs=5)
print(result)
# DETERMINISTIC: All 5 runs produced identical output
```

CLI:

```bash
detinfer verify <hf-model>
detinfer verify <hf-model> --runs 10
```

### 1.4 Benchmark

Tests determinism across 8 categories of prompts (auto-scales by model size):

```bash
detinfer benchmark <hf-model>
```

| Tier | Category | What it tests |
|------|----------|--------------|
| 1 | Sanity | Basic questions |
| 2 | Long output | 200+ token generations |
| 3 | Uncertain | Creative prompts |
| 4 | Complex code | Merge sort, LRU cache |
| 5 | Reasoning | Logic puzzles |
| 6 | Deep context | Long code + passage analysis |
| 7 | Adversarial | Designed to break determinism |
| 8 | Edge cases | Unicode, empty, special chars |

### 1.5 Scan for Non-Deterministic Ops

```bash
detinfer scan <hf-model>        # Detects Dropout, Flash Attention, etc.
```

### 1.6 Before vs After Comparison

```bash
detinfer compare <hf-model>     # Runs without detinfer, then with detinfer
```

### 1.7 Cross-GPU Verification

```bash
# Machine A: export proof
detinfer export <hf-model> -o proof.json

# Transfer proof.json to Machine B

# Machine B: verify
detinfer cross-verify proof.json
```

### 1.8 Training Verification

```python
import detinfer

detinfer.enforce(seed=42)

for step, batch in enumerate(dataloader):
    loss = model(batch).loss
    loss.backward()
    optimizer.step()

    h = detinfer.checkpoint_hash(model)
    print(f"Step {step}: {h}")
```

### 1.9 Health Check

```bash
detinfer doctor <hf-model>          # Full determinism audit
detinfer doctor <hf-model> --json   # JSON for CI pipelines
```

---

## Part 2: Agent

The agent module provides multi-turn deterministic conversations with full token tracing, replay, and diff.

### 2.1 Deterministic Agent

```python
from detinfer import DeterministicAgent

agent = DeterministicAgent("<hf-model>", seed=42)
response = agent.chat("What is 2+2?")
print(response)

# With system prompt
agent = DeterministicAgent(
    "<hf-model>",
    seed=42,
    system_prompt="You are a math tutor"
)
response = agent.chat("What is calculus?")
```

Both `chat()` and `chat_stream()` use manual token-by-token generation with deterministic argmax (smallest-token-ID tie-breaking).

### 2.2 Streaming

```python
for chunk in agent.chat_stream("Explain gravity"):
    print(chunk, end="", flush=True)
```

### 2.3 Tool Calls

Register tools and call them -- every call is recorded in the trace:

```python
agent.register_tool("calculator", lambda expression: str(eval(expression)))
agent.register_tool("lookup", lambda query: f"Result for: {query}")

response = agent.chat("What is 2+2?")
result = agent.call_tool("calculator", {"expression": "2+2"})
response = agent.chat(f"The answer is {result}. Explain why.")

# Checkpoint key decision points
agent.checkpoint({"decision": "used_calculator", "confidence": "high"})
```

### 2.4 Export Session

```python
agent.export_session("session.json")
```

The exported trace contains:

```json
{
  "schema_version": "1",
  "trace_type": "agent",
  "trace_mode": "standard",
  "model": "<hf-model>",
  "seed": 42,
  "session_hash": "a1b2c3...",
  "messages": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."}
  ],
  "generations": [
    {
      "turn": 1,
      "prompt_hash": "abc...",
      "input_tokens": [464, 318],
      "output_tokens": [17, 10],
      "output_tokens_hash": "def...",
      "stop_reason": "eos",
      "steps": [
        {"step": 0, "chosen_token": 17},
        {"step": 1, "chosen_token": 10}
      ]
    }
  ]
}
```

With `--trace-mode verbose`, each step also includes `top_tokens` and `top_scores` (top-10 candidates).

### 2.5 Trace Modes

Three levels of trace detail -- all produce the same canonical session hash:

| Mode | What is captured | Use case |
|------|-----------------|----------|
| `minimal` | Hashes + output tokens only | CI, benchmarks |
| `standard` | + rendered prompt, input tokens, step trace | Replay, debugging |
| `verbose` | + top-k tokens and scores per step | Deep diagnosis |

```bash
detinfer agent <hf-model> --trace-mode minimal
detinfer agent <hf-model> --trace-mode verbose
```

### 2.6 Replay

Re-run a saved session and verify it produces the same output:

```bash
detinfer replay session.json
detinfer replay session.json --strict    # Step-by-step verification
```

### 2.7 Diff

Compare two sessions and find the first divergence:

```bash
detinfer diff run_a.json run_b.json
```

### 2.8 Verify Session

Verify a session trace as a deterministic execution proof:

```bash
detinfer verify-session session.json
```

```
  ======================================================
    DETERMINISTIC EXECUTION PROOF VERIFICATION
  ======================================================

    Model:        <hf-model>
    Seed:         42
    Turns:        3

    [PASS] VERIFIED -- All 3 turns match exactly
    [PASS] This session is a valid deterministic execution proof.
  ======================================================
```

### 2.9 Regression Check

Compare two session traces and classify exactly what changed:

```bash
detinfer check baseline.json candidate.json
```

```
Detinfer Regression Report
--------------------------
Status: FAILED
Primary type: TOKENIZER_DRIFT

Matched:
  [OK] model
  [OK] seed
  [OK] generation_config

Changed:
  [FAIL] tokenizer.tokenizer_hash
```

Mismatch types:

| Type | Severity | Meaning |
|------|----------|---------|
| `SCHEMA_MISMATCH` | error | Schema version changed |
| `TYPE_MISMATCH` | error | Comparing inference vs agent trace |
| `MODEL_DRIFT` | error | Model name or weights changed |
| `TOKENIZER_DRIFT` | error | Tokenizer or chat template changed |
| `CONFIG_DRIFT` | error | Seed, temperature, or gen config changed |
| `PROMPT_DRIFT` | error | Messages or rendered prompt changed |
| `INPUT_TOKEN_DRIFT` | error | Same text tokenized differently |
| `OUTPUT_DRIFT` | error | Model produced different tokens |
| `STOP_REASON_DRIFT` | error | EOS vs max_tokens changed |
| `ENVIRONMENT_DRIFT` | warning | Torch/Python version changed |
| `TRACE_DETAIL_DRIFT` | info | Verbose-only fields differ |

CI flags:

```bash
detinfer check a.json b.json --json                    # JSON output for CI
detinfer check a.json b.json --fail-on OUTPUT_DRIFT    # Fail on specific type
detinfer check a.json b.json --allow ENVIRONMENT_DRIFT # Ignore env changes
```

### 2.10 Deterministic Truncation

When conversations exceed the context window:

- System prompt is **never** dropped
- Latest user message is **never** dropped
- Oldest turns are dropped first (user+assistant pairs together)
- Every truncation event is recorded in the trace

```python
agent = DeterministicAgent(
    "<hf-model>", seed=42,
    max_context_tokens=2048
)
```

### 2.11 Save and Resume

```python
# Save
agent.save_state("agent_state.json")

# Resume (same model must be loaded)
agent.load_state("agent_state.json")
response = agent.chat("Continue where we left off")
```

CLI:

```bash
detinfer agent <hf-model> --save-state state.json
detinfer agent <hf-model> --load-state state.json
```

---

## Part 3: Agent Harness

The harness automates agent testing. Define tasks as JSON, run them, compare against baselines.

### 3.1 Define a Task

Create a JSON file:

```json
{
  "name": "basic_math",
  "model": "<hf-model>",
  "seed": 42,
  "prompt": "What is 2+2?",
  "max_turns": 1,
  "generation_config": {
    "max_new_tokens": 64,
    "do_sample": false
  },
  "expected": {
    "match": "contains",
    "value": "4"
  }
}
```

### 3.2 Run a Task

```bash
detinfer agent-run task.json
```

### 3.3 Run a Suite

Put multiple JSON files in a directory:

```bash
detinfer agent-run examples/
```

Output:

```
Agent Harness Report
====================

  [PASS] basic_math (42ms)
  [PASS] python_function (85ms)
  [PASS] explain_gravity (71ms)

Total: 3  |  Passed: 3  |  Failed: 0  |  Errors: 0
Duration: 198ms
```

### 3.4 Compare Against Baseline

```bash
detinfer agent-run task.json --against baseline.json
```

### 3.5 Export Traces

```bash
detinfer agent-run examples/ --output-dir runs/
```

The harness uses the same `SessionTrace` format -- all replay, diff, and check tools work on harness output.

### 3.6 Expected Match Modes

| Mode | Behavior |
|------|----------|
| `exact` | Output must equal value exactly |
| `contains` | Output must contain value as substring |
| `regex` | Output must match regex pattern |

### 3.7 Task Schema Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `name` | yes | -- | Task name |
| `model` | yes | -- | HuggingFace model name |
| `prompt` | yes | -- | Initial prompt |
| `seed` | no | 42 | Random seed |
| `max_turns` | no | 1 | Max conversation turns |
| `system_prompt` | no | -- | System prompt |
| `follow_ups` | no | [] | Follow-up prompts for multi-turn |
| `generation_config` | no | greedy | Generation configuration |
| `tools` | no | [] | Mock tools to register |
| `expected` | no | -- | Expected output matcher |
| `tags` | no | [] | Tags for filtering |

### 3.8 Example Tasks

The repo includes example tasks in `examples/`:

| File | Type | Tests |
|------|------|-------|
| `basic_math.json` | Smoke | Simple arithmetic |
| `python_function.json` | Medium | Code generation |
| `explain_gravity.json` | Medium | Explanation + system prompt |
| `tool_calculator.json` | Tool | Mock tool usage |
| `long_reasoning.json` | Stress | Long generation + regex match |
| `multi_turn_math.json` | Stress | Multi-turn with follow-ups |

---

## How It Works

detinfer applies deterministic runtime settings for 7 sources of non-determinism:

| Source | Problem | detinfer fix |
|--------|---------|-------------|
| Random seeds | Separate RNGs in Python, NumPy, PyTorch, CUDA | Locks all seeds in one call |
| CUDA atomics | Non-deterministic `atomicAdd` operations | Enables `torch.use_deterministic_algorithms(True)` |
| Flash Attention | Non-deterministic `scaled_dot_product_attention` | Replaces with deterministic math backend |
| cuDNN tuning | Auto-selects different algorithms per run | Disables benchmark mode |
| cuBLAS workspace | Matrix multiply results vary with workspace | Sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| Float ordering | Different GPUs produce different float results | Canonicalizes outputs before hashing |
| LLM decoding | Temperature and sampling add randomness | Uses deterministic argmax with stable tie-breaking |

---

## CLI Reference

### All Commands

```bash
# INFERENCE
detinfer run <hf-model>                              # Interactive deterministic inference
detinfer run <hf-model> --seed 42 --max-tokens 512   # Custom seed and token limit
detinfer verify <hf-model>                           # Run 5 times, compare hashes
detinfer verify <hf-model> --runs 10                 # Custom number of runs
detinfer scan <hf-model>                             # Scan for non-deterministic ops
detinfer compare <hf-model>                          # Before vs after comparison
detinfer benchmark <hf-model>                        # Full benchmark
detinfer export <hf-model> -o proof.json             # Export proof
detinfer cross-verify proof.json                     # Verify proof from another machine
detinfer doctor <hf-model>                           # Determinism health check
detinfer doctor <hf-model> --json                    # JSON report for CI

# AGENT
detinfer agent <hf-model>                            # Multi-turn deterministic agent
detinfer agent <hf-model> --prompt "What is 2+2?"    # Non-interactive
detinfer agent <hf-model> --system "You are a tutor" # System prompt
detinfer agent <hf-model> --export session.json      # Export session trace
detinfer agent <hf-model> --trace-mode verbose       # Record top-k tokens
detinfer agent <hf-model> --save-state state.json    # Save state on exit
detinfer agent <hf-model> --load-state state.json    # Resume from state
detinfer replay session.json                         # Replay saved session
detinfer replay session.json --strict                # Step-by-step verify
detinfer verify-session session.json                 # Verify as proof
detinfer diff run_a.json run_b.json                  # Token-level diff

# REGRESSION CHECK
detinfer check baseline.json candidate.json          # Compare traces
detinfer check a.json b.json --json                  # JSON output
detinfer check a.json b.json --fail-on OUTPUT_DRIFT  # Fail on specific type
detinfer check a.json b.json --allow ENVIRONMENT_DRIFT

# AGENT HARNESS
detinfer agent-run task.json                         # Run single task
detinfer agent-run examples/                         # Run task suite
detinfer agent-run task.json --against baseline.json # Compare against baseline
detinfer agent-run examples/ --output-dir runs/      # Export traces
detinfer agent-run examples/ --json                  # JSON output
detinfer agent-run examples/ --fail-fast             # Stop on first failure

# INFO
detinfer info                                        # GPU and environment details
```

### Flag Reference

#### `detinfer agent`

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | -- | Non-interactive mode |
| `--system` | -- | System prompt |
| `--seed` | 42 | Random seed |
| `--max-tokens` | 256 | Max tokens per turn |
| `--device` | auto | Device (cpu, cuda, auto) |
| `--export` | -- | Export session trace to JSON |
| `--quantize` | -- | Quantization mode (`int8`) |
| `--trace-mode` | standard | Trace detail level |
| `--max-context-tokens` | -- | Max prompt tokens before truncation |
| `--save-state` | -- | Save agent state on exit |
| `--load-state` | -- | Resume from saved state |

#### `detinfer check`

| Flag | Default | Description |
|------|---------|-------------|
| `--json` | off | JSON output |
| `--fail-on` | -- | Mismatch type that causes failure |
| `--allow` | -- | Mismatch type to ignore |

#### `detinfer agent-run`

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | -- | Trace output directory |
| `--against` | -- | Baseline trace to compare |
| `--json` | off | JSON output |
| `--fail-fast` | off | Stop on first failure |

---

## API Reference

### Top-Level

```python
import detinfer

detinfer.enforce(seed=42)          # Apply deterministic runtime settings
detinfer.status()                   # Check enforcement state
detinfer.checkpoint_hash(model)     # Hash model weights (for training)
```

### DeterministicEngine

| Method | Description |
|--------|-------------|
| `DeterministicEngine(seed, precision, device)` | Create engine |
| `.load(model_name)` | Load model, apply deterministic settings |
| `.load(model_name, quantize="int8")` | Load with INT8 quantization |
| `.run(prompt, max_new_tokens)` | Run inference |
| `.verify(prompt, num_runs)` | Run N times, compare hashes |
| `.scan()` | Show enforcement report |

### DeterministicAgent

| Method | Description |
|--------|-------------|
| `DeterministicAgent(model, seed, system_prompt)` | Create agent |
| `.chat(message)` | Send message, get response |
| `.chat_stream(message)` | Stream tokens |
| `.register_tool(name, fn)` | Register a callable tool |
| `.call_tool(name, args)` | Call tool, record in trace |
| `.checkpoint(data)` | Record checkpoint event |
| `.export_session(path)` | Export full token trace |
| `.get_session_hash()` | Get canonical session hash |
| `.save_state(path)` | Save agent state |
| `.load_state(path)` | Resume from saved state |

### Regression Check

| Function | Description |
|----------|-------------|
| `check_sessions(baseline, candidate)` | Compare two trace dicts |
| `render_check_report(report)` | Human-readable report |

### Agent Harness

| Function | Description |
|----------|-------------|
| `HarnessRunner(output_dir, against)` | Create harness runner |
| `runner.run_task(task)` | Run single task |
| `runner.run_suite(tasks)` | Run task suite |
| `load_task(path)` | Load task from JSON |
| `load_task_suite(dir_path)` | Load all tasks from directory |

### Replay and Diff

| Function | Description |
|----------|-------------|
| `replay_session(trace_path)` | Re-run session, verify tokens |
| `diff_sessions(path_a, path_b)` | Find first mismatch |

---

## GitHub Action

Add determinism verification to CI:

```yaml
# .github/workflows/determinism.yml
name: Verify Determinism
on: [push]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: xailong-6969/detinfer@main
        with:
          command: verify-session
          session-file: baseline.json
          strict: true
```

---

## Compatibility

### Supported

| Feature | Status | Notes |
|---------|--------|-------|
| PyTorch eager mode | Supported | Default execution mode |
| Greedy decoding | Supported | Enforced via deterministic argmax |
| fp32 / fp16 inference | Supported | Deterministic on supported backends |
| CPU inference | Supported | Fully deterministic |
| NVIDIA GPU (single) | Supported | T4, V100, A100, RTX 3070/4090, etc. |
| HuggingFace CausalLM | Supported | Any CausalLM on HuggingFace Hub |

### Partially Supported

| Feature | Notes |
|---------|-------|
| bf16 inference | Hardware-dependent rounding may differ |
| Multi-GPU (`device_map="auto"`) | May have split-order edge cases |
| Flash Attention | Auto-replaced with MATH backend |

### Experimental

| Feature | Notes |
|---------|-------|
| INT8 (bitsandbytes) | May improve consistency |

### Not Supported

| Feature | Reason |
|---------|--------|
| GPTQ/AWQ quantization | Kernel-specific rounding |
| `torch.compile` | Graph autotuning |
| Beam search | Tie-breaking is implementation-specific |
| vLLM / paged attention | KV cache paging |
| AMD GPUs (ROCm) | Untested |
| Apple Silicon (MPS) | Untested |

---

## Architecture

```
detinfer/
  __init__.py       # Top-level API: enforce(), status(), checkpoint_hash()
  cli.py            # CLI entry point (16 commands)
  check.py          # Regression check: compare two traces, classify drift

  inference/        # Deterministic inference library
    config.py       # Seed locking + deterministic flags
    enforcer.py     # Runtime op patching (Dropout, Flash Attention)
    canonicalizer.py # Cross-hardware output normalization
    guardian.py     # Environment fingerprinting
    engine.py       # High-level DeterministicEngine
    benchmark.py    # Auto-scaling benchmark suite
    proof.py        # Cross-GPU proof export/verify
    detector.py     # Static model scanning
    verifier.py     # Hash-based verification
    wrapper.py      # Simple HuggingFace wrapper
    utils.py        # Hashing + env snapshots

  agent/            # Deterministic agent system
    runtime.py      # DeterministicAgent with truncation + save/resume
    trace.py        # Token-level trace recording + session schema
    replay.py       # Session replay verification + diff

  harness/          # Agent harness for automated testing
    task_schema.py  # Task definition loading + validation
    runner.py       # HarnessRunner with baseline comparison

examples/           # Example harness task files
docs/               # Documentation
  GUIDE.md          # Beginner-friendly guide
  determinism-spec.md  # Technical determinism specification
```

---

## Running Tests

```bash
pip install "detinfer[dev]"
pytest tests/ -v
```

163 tests covering all modules.

---

## License

MIT -- see [LICENSE](LICENSE).
