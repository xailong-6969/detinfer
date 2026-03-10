# detinfer — Deterministic ML Library

**Enforce determinism in ML inference and training. One line of code.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-88%20passed-brightgreen.svg)]()
[![PyPI](https://img.shields.io/badge/pypi-v0.3.0-blue.svg)](https://pypi.org/project/detinfer/)

---

## Why?

ML models give **different outputs every time you run them** — even with the exact same input. This happens because of GPU non-determinism (Flash Attention, CUDA atomics, cuDNN auto-tuning, floating-point ordering).

This breaks everything that requires **verifiable computation**: decentralized AI networks, reproducible research, CI/CD for ML, audit trails.

**detinfer Fixes this.** One import, one function call.

```python
import detinfer
detinfer.enforce()  # Everything is now deterministic.
```

---

## Installation

### Via PyPI (easiest)

```bash
# Create and activate a virtual environment (required on Debian/Ubuntu)
python3 -m venv venv
source venv/bin/activate

# With HuggingFace model support (recommended)
pip install "detinfer[transformers]"

# Core only (if you already have your own model)
pip install detinfer
```

Then use the CLI directly:
```bash
# Replace <model> with any HuggingFace model, e.g. gpt2, Qwen/Qwen2.5-0.5B-Instruct, etc.

# Inference
detinfer run <model>            # Interactive inference — type prompts, get deterministic output
detinfer verify <model>         # Verify determinism — runs 5 times, compares hashes
detinfer benchmark <model>      # Full stress test with 36 prompts across 8 categories
detinfer compare <model>        # Side-by-side: without detinfer vs with detinfer

# Agent (NEW in v0.3.0)
detinfer chat <model>           # Multi-turn deterministic chat agent
detinfer chat <model> --prompt "What is 2+2?"  # Non-interactive (for CI/scripts)
detinfer replay session.json    # Replay and verify a saved session
detinfer diff run_a.json run_b.json  # Token-level comparison of two runs

# Proofs
detinfer export <model> -o proof.json   # Export proof for cross-GPU verification
detinfer cross-verify proof.json        # Verify a proof from another machine
detinfer verify-session session.json    # Verify session as execution proof

# Info
detinfer info                   # Show your GPU and environment details
```

### Via Git (for the interactive menu)

```bash
git clone https://github.com/xailong-6969/detinfer.git
cd detinfer
bash run_detinfer.sh
```

The script automatically:
1. Creates a virtual environment
2. Installs detinfer + all dependencies
3. Detects your GPU/CPU
4. Launches the interactive menu

### Manual Install from Source

```bash
# Clone
git clone https://github.com/xailong-6969/detinfer.git
cd detinfer

# Create and activate a virtual environment (required on Debian/Ubuntu)
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Recommended — includes HuggingFace model support (load any model by name)
pip install -e ".[transformers]"

# For contributors — includes pytest for running the test suite
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- Any NVIDIA GPU (recommended) or CPU

---

## Getting Started

### If you installed via `run_detinfer.sh`

The interactive menu is already open — select an option and follow the prompts. The script handles everything for you.

### If you installed via `pip install`

*(Note: On modern Debian/Ubuntu systems, remember to activate your virtual environment before running the tool, e.g., `source venv/bin/activate`)*

**From Python (in your own code):**

```python
import detinfer

# One line — locks all randomness globally
detinfer.enforce(seed=42)

# Now any PyTorch code is deterministic:
output = model(input)           # inference
loss.backward()                 # training
optimizer.step()                # weight updates
```

---

## Usage

### 1. One-Line Enforcement (for any ML code)

```python
import detinfer

# Lock all sources of randomness globally
detinfer.enforce(seed=42)

# Now ANY PyTorch code is deterministic:
output = model(input)        # inference — deterministic
loss.backward()              # training — deterministic
optimizer.step()             # weight updates — deterministic
```

### 2. DeterministicEngine (for LLM inference)

```python
from detinfer import DeterministicEngine

# Load any HuggingFace model + auto-fix all non-deterministic ops
engine = DeterministicEngine(seed=42)
engine.load("<model>")  # e.g., "Qwen/Qwen2.5-0.5B-Instruct", "meta-llama/Llama-3-8B", etc.

# Run inference — same output every time, on any GPU
result = engine.run("Write hello world in Python")
print(result.text)            # The generated text
print(result.canonical_hash)  # SHA-256 hash — identical across GPUs
```

**For large models across multiple GPUs:**

```python
# Models too large for a single GPU — automatically splits across all available GPUs
engine = DeterministicEngine(seed=42)
engine.load("<model>", device_map="auto")

# Everything else works the same
result = engine.run("Write hello world in Python")
```

### 3. Deterministic Chat Agent (NEW in v0.3.0)

```python
from detinfer import DeterministicAgent

# Multi-turn deterministic chat
agent = DeterministicAgent("gpt2", seed=42)
response = agent.chat("What is 2+2?")
print(response)  # Always the same answer

response = agent.chat("Explain more")
print(response)  # Always the same follow-up

# Export full session trace (token-by-token proof)
agent.export_session("session.json")

# Replay on another machine to verify
# detinfer replay session.json
```

### 4. Verify Determinism

```python
# Run 5 times automatically, compare all hashes
result = engine.verify(num_runs=5)
print(result)
# DETERMINISTIC: All 5 runs produced identical output
# SHA-256: 799519fee8d50aca...
```

### 5. Training Verification

```python
import detinfer

detinfer.enforce(seed=42)

for step, batch in enumerate(dataloader):
    loss = model(batch).loss
    loss.backward()
    optimizer.step()

    # Hash model weights — identical across machines at same step
    h = detinfer.checkpoint_hash(model)
    print(f"Step {step}: {h}")
```

### 6. Cross-GPU Proof Verification

Prove that two different GPUs produce the same output — see the [Cross-GPU Verification Guide](#cross-gpu-verification-guide) below for the full step-by-step walkthrough.

---

## CLI

detinfer includes a full command-line interface:

```bash
# Replace <model> with any HuggingFace model name, e.g.:
#   Qwen/Qwen2.5-0.5B-Instruct
#   meta-llama/Llama-3-8B
#   mistralai/Mistral-7B-v0.1
#   gpt2

# Interactive deterministic inference
detinfer run <model>

# Deterministic multi-turn chat agent (NEW in v0.3.0)
detinfer chat <model>
detinfer chat <model> --prompt "Hello"  # Non-interactive mode
detinfer chat <model> --export session.json
detinfer chat <model> --quantize int8   # Experimental INT8 mode

# Replay and verify a saved chat session
detinfer replay session.json
detinfer replay session.json --strict   # Step-by-step verification

# Token-level comparison of two sessions
detinfer diff run_a.json run_b.json

# Verify a session export as deterministic execution proof
detinfer verify-session session.json
detinfer verify-session session.json --strict

# Scan model for non-deterministic ops (Dropout, Flash Attention, etc.)
detinfer scan <model>

# Verify determinism (run 5 times, compare hashes)
detinfer verify <model>

# Before vs after detinfer comparison
detinfer compare <model>

# Full benchmark (auto-scales based on model size)
detinfer benchmark <model>

# Export inference proof to JSON
detinfer export <model> -o proof.json

# Verify a proof from another machine
detinfer cross-verify proof.json

# Show environment information
detinfer info
```

### Interactive Menu (run_determl.sh)

```
>> What would you like to do?
   1) run          - Interactive deterministic inference
   2) scan         - Scan model for non-deterministic ops
   3) verify       - Verify model produces deterministic output
   4) compare      - Before vs after detinfer comparison
   5) benchmark    - Full determinism benchmark (auto-scales)
   6) export       - Export inference proof (for cross-GPU verify)
   7) cross-verify - Verify a proof from another machine
   8) info         - Show environment information
   9) exit         - Exit detinfer
```

---

## Features

### Auto-Scaling Benchmark

Tests determinism across 8 categories of prompts:

| Tier | Category | What it tests |
|------|----------|--------------|
| 1 | Sanity | Basic questions (baseline check) |
| 2 | Long output | 200+ token generations (many CUDA ops) |
| 3 | Uncertain | Creative prompts (model has low confidence) |
| 4 | Complex code | Merge sort, LRU cache (deep computation) |
| 5 | Reasoning | Logic puzzles, step-by-step (deep attention) |
| 6 | Deep context | Long code + passage analysis |
| 7 | Adversarial | FizzBuzz, pangrams (designed to break determinism) |
| 8 | Edge cases | Unicode, empty, special characters |

Auto-scales based on model size:
- Small models (<3B): 20 prompts × 5 runs = 100 total
- Medium models (3-13B): 10 prompts × 3 runs = 30 total
- Large models (13B+): 5 prompts × 2 runs = 10 total

### Before vs After Comparison

```bash
detinfer compare <model>
```

Runs the model first **without** detinfer (raw PyTorch) then **with** detinfer, showing hash differences side by side.

### Cross-GPU Verification Guide

This proves that two different machines (with different GPUs) produce the exact same output.

**How it works:**

```
Machine A                          Machine B
─────────                          ─────────
Runs inference                     
  → gets hash abc123               
  → saves to proof.json            
                                   
        ──── transfer proof.json ────→
                                   
                                   Reads proof.json (Machine A's hash)
                                   Runs the SAME inference locally
                                     → gets hash abc123
                                   Compares: abc123 == abc123? ✓ MATCH
```

Only **one** `proof.json` is created. Machine B does not create its own — it re-runs the inference and compares its result with Machine A's hash automatically.

**What you need:**
- Two machines with GPUs (e.g., a cloud GPU instance + another server, or any two machines)
- Both machines must have detinfer installed
- Both machines must use the same model

**Step 1: Install detinfer on both machines**

```bash
# Run this on both Machine A and Machine B:
git clone https://github.com/xailong-6969/detinfer.git
cd detinfer
pip install -e ".[transformers]"
```

**Step 2: Export a proof on Machine A**

```bash
detinfer export <model> -o proof.json

# This will:
#   1. Load the model
#   2. Run inference with a test prompt
#   3. Save the canonical hash, environment info, and output to proof.json
```

**Step 3: Transfer proof.json to Machine B**

Copy `proof.json` from Machine A to Machine B however you prefer — `scp`, file upload, copy-paste, etc.

**Step 4: Verify on Machine B**

```bash
cd detinfer
detinfer cross-verify proof.json
```

**What you'll see:**

```
  CROSS-GPU VERIFICATION RESULT
  =================================================================

  Original (remote):
    GPU:            NVIDIA GeForce RTX 3070
    Canonical hash: 799519fee8d50aca...

  Local (this machine):
    GPU:            Tesla T4
    Canonical hash: 799519fee8d50aca...

  Canonical hash match: ✓ YES

  ✓ VERIFIED — Deterministic execution confirmed across GPUs!
```

If both canonical hashes match → the library works. Same model + same input = same output, regardless of hardware.

---

## How It Works

detinfer addresses 7 sources of non-determinism:

| Source | Problem | detinfer Fix |
|--------|---------|-------------|
| Random seeds | Python, NumPy, PyTorch, CUDA each have separate RNGs | Locks ALL seeds in one call |
| CUDA atomics | `scatter_add`, `index_add` use non-deterministic `atomicAdd` | Forces `torch.use_deterministic_algorithms(True)` |
| Flash Attention | `scaled_dot_product_attention` is non-deterministic | Replaces with deterministic math backend |
| cuDNN tuning | Auto-selects different algorithms per run | Disables benchmark mode |
| cuBLAS workspace | Matrix multiplications vary with workspace config | Sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| Float ordering | Different GPUs produce different float results | Canonicalizes outputs before hashing |
| LLM sampling | `temperature`, `top_k`, `top_p` add randomness | Forces greedy decoding |

---

## Architecture

```
detinfer/
  __init__.py       # Top-level API: enforce(), status(), checkpoint_hash()
  cli.py            # CLI entry point (12 commands)

  inference/        # Deterministic inference library
    config.py       # Seed locking + deterministic flags
    enforcer.py     # Runtime op patching (Dropout, Flash Attention)
    canonicalizer.py # Cross-hardware output normalization
    guardian.py     # Environment fingerprinting + compatibility
    engine.py       # High-level DeterministicEngine for LLMs
    benchmark.py    # Auto-scaling benchmark suite (8 tiers, 36 prompts)
    proof.py        # Cross-GPU proof export/import/verify
    detector.py     # Static model scanning
    verifier.py     # Hash-based verification
    wrapper.py      # Simple HuggingFace wrapper
    utils.py        # Hashing + env snapshots

  agent/            # Deterministic agent system (NEW in v0.3.0)
    runtime.py      # DeterministicAgent — multi-turn chat
    trace.py        # Token-level trace recording + session schema
    replay.py       # Session replay verification + diff
```

---

## API Reference

### Top-Level API

```python
import detinfer

detinfer.enforce(seed=42)              # Lock all randomness
detinfer.status()                       # Check enforcement state
detinfer.checkpoint_hash(model)         # Hash model weights (for training)
```

### DeterministicEngine

| Method | Description |
|--------|-------------|
| `DeterministicEngine(seed, precision, device)` | Create engine |
| `.load(model_name)` | Load HuggingFace model, auto-fix ops |
| `.run(prompt, max_new_tokens)` | Deterministic inference |
| `.verify(prompt, num_runs)` | Run N times, compare hashes |
| `.scan()` | Show enforcement report |

### Proof System

| Function | Description |
|----------|-------------|
| `create_proof(engine, prompt)` | Run inference, create exportable proof |
| `cross_verify(proof)` | Re-run locally, compare with proof |
| `InferenceProof.save(path)` | Export proof to JSON |
| `InferenceProof.load(path)` | Load proof from JSON |

### Benchmark

| Function | Description |
|----------|-------------|
| `run_benchmark(engine, config)` | Run full benchmark suite |
| `BenchmarkConfig.from_depth(depth, param_b)` | Auto-scale by model size |

### DeterministicAgent (NEW in v0.3.0)

| Method | Description |
|--------|-------------|
| `DeterministicAgent(model, seed)` | Create agent |
| `.chat(message)` | Send message, get deterministic response |
| `.export_session(path)` | Export full token trace to JSON |
| `.get_session_hash()` | Get canonical session hash |

### Replay & Diff

| Function | Description |
|----------|-------------|
| `replay_session(trace_path)` | Re-run session, verify token-by-token |
| `diff_sessions(path_a, path_b)` | Compare two traces, find first mismatch |

---

## GitHub Action

Other teams can add detinfer to their CI to auto-verify model outputs haven't changed:

```yaml
# .github/workflows/determinism.yml
name: Verify Determinism
on: [push]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Verify a saved session is still reproducible
      - uses: xailong-6969/detinfer@v2-enforcement
        with:
          command: verify-session
          session-file: baseline.json
          strict: true

      # Or generate + export a new session for comparison
      - uses: xailong-6969/detinfer@v2-enforcement
        with:
          command: chat
          model: gpt2
          prompt: "What is 2+2?"
          export: output.json
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Supported Matrix

| Feature | Status | Notes |
|---|---|---|
| PyTorch eager mode | **Fully supported** | Default, fully deterministic |
| Greedy decoding | **Fully supported** | Enforced automatically (`do_sample=False`, `num_beams=1`) |
| fp32 inference | **Fully supported** | Full determinism |
| fp16 inference | **Fully supported** | Deterministic on supported backends |
| CPU inference | **Fully supported** | Fully deterministic |
| NVIDIA GPU (single) | **Fully supported** | T4, V100, A100, RTX 3070/4090, etc. |
| HuggingFace CausalLM | **Fully supported** | GPT-2, Qwen, TinyLlama, LLaMA, etc. |
| Cross-GPU canonical hashes | **Fully supported** | Via `detinfer export` + `detinfer cross-verify` |
| bf16 inference | Partial | Hardware-dependent rounding; canonical hash may differ across GPU generations |
| Multi-GPU (`device_map="auto"`) | Partial | Inference works; very large models may have split-order edge cases |
| Flash Attention | Partial | Auto-replaced with MATH backend; may reduce throughput |
| Quantized models (INT8 via bitsandbytes) | **Experimental** | May improve cross-device consistency; not guaranteed bitwise identical |
| Quantized models (GPTQ/AWQ) | **Not supported** | Kernel-specific rounding breaks cross-machine proof |
| `torch.compile` | **Not supported** | Graph autotuning selects different kernels across runs |
| Beam search (`num_beams > 1`) | **Not supported** | Tie-breaking is implementation-specific |
| Speculative decoding | **Not supported** | Draft model adds nondeterminism |
| vLLM / paged attention | **Not supported** | KV cache paging is non-deterministic |
| Tensor / pipeline parallelism | **Not supported** | Reduction order across devices not controlled |
| AMD GPUs (ROCm) | Untested | |
| Apple Silicon (MPS) | Untested | |

> For the full technical specification of what "deterministic" means in detinfer — including exact hash definitions, proof format, and operating modes — see [docs/determinism-spec.md](docs/determinism-spec.md).

---

## License

MIT — see [LICENSE](LICENSE).


