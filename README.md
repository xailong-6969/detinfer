# determl — Deterministic ML Library

**Enforce determinism in ML inference and training. One line of code.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-70%20passed-brightgreen.svg)]()

---

## Why?

ML models give **different outputs every time you run them** — even with the exact same input. This happens because of GPU non-determinism (Flash Attention, CUDA atomics, cuDNN auto-tuning, floating-point ordering).

This breaks everything that requires **verifiable computation**: decentralized AI networks, reproducible research, CI/CD for ML, audit trails.

**determl fixes this.** One import, one function call.

```python
import determl
determl.enforce()  # Everything is now deterministic.
```

---

## Installation

### Quick Start (recommended)

```bash
git clone -b v2-enforcement https://github.com/xailong-6969/determl.git
cd determl
bash run_determl.sh
```

The script automatically:
1. Creates a virtual environment
2. Installs determl + all dependencies
3. Detects your GPU
4. Launches the interactive menu

### Manual Install

```bash
# Clone
git clone -b v2-enforcement https://github.com/xailong-6969/determl.git
cd determl

# Recommended — includes HuggingFace model support (load any model by name)
pip install -e ".[transformers]"

# Minimal — only core enforcement (use if you have your own model, no HuggingFace)
pip install -e .

# For contributors — includes pytest for running the test suite
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- Any NVIDIA GPU (recommended) or CPU

---

## Getting Started

### If you installed via `run_determl.sh`

The interactive menu is already open — select an option and follow the prompts. The script handles everything for you.

### If you installed manually via `pip install`

After installation, you can use determl in two ways:

**From the terminal (CLI):**

```bash
# Replace <model> with any HuggingFace model, e.g. gpt2, Qwen/Qwen2.5-0.5B-Instruct, etc.

determl run <model>            # Interactive inference — type prompts, get deterministic output
determl verify <model>         # Verify determinism — runs 5 times, compares hashes
determl benchmark <model>      # Full stress test with 36 prompts across 8 categories
determl compare <model>        # Side-by-side: without determl vs with determl
determl info                   # Show your GPU and environment details
```

**From Python (in your own code):**

```python
import determl

# One line — locks all randomness globally
determl.enforce(seed=42)

# Now any PyTorch code is deterministic:
output = model(input)           # inference
loss.backward()                 # training
optimizer.step()                # weight updates
```

---

## Usage

### 1. One-Line Enforcement (for any ML code)

```python
import determl

# Lock all sources of randomness globally
determl.enforce(seed=42)

# Now ANY PyTorch code is deterministic:
output = model(input)        # inference — deterministic
loss.backward()              # training — deterministic
optimizer.step()             # weight updates — deterministic
```

### 2. DeterministicEngine (for LLM inference)

```python
from determl import DeterministicEngine

# Load any HuggingFace model + auto-fix all non-deterministic ops
engine = DeterministicEngine(seed=42)
engine.load("<model>")  # e.g., "Qwen/Qwen2.5-0.5B-Instruct", "meta-llama/Llama-3-8B", etc.

# Run inference — same output every time, on any GPU
result = engine.run("Write hello world in Python")
print(result.text)            # The generated text
print(result.canonical_hash)  # SHA-256 hash — identical across GPUs
```

### 3. Verify Determinism

```python
# Run 5 times automatically, compare all hashes
result = engine.verify(num_runs=5)
print(result)
# DETERMINISTIC: All 5 runs produced identical output
# SHA-256: 799519fee8d50aca...
```

### 4. Training Verification

```python
import determl

determl.enforce(seed=42)

for step, batch in enumerate(dataloader):
    loss = model(batch).loss
    loss.backward()
    optimizer.step()

    # Hash model weights — identical across machines at same step
    h = determl.checkpoint_hash(model)
    print(f"Step {step}: {h}")
```

### 5. Cross-GPU Proof Verification

```python
from determl.proof import create_proof, cross_verify, InferenceProof

# Machine A: export proof
proof = create_proof(engine, "What is 2+2?")
proof.save("proof.json")

# Machine B: verify it
proof = InferenceProof.load("proof.json")
result = cross_verify(proof)
print(result)
# ✓ VERIFIED — RTX 3070 and T4 produced identical canonical hashes
```

---

## CLI

determl includes a full command-line interface:

```bash
# Replace <model> with any HuggingFace model name, e.g.:
#   Qwen/Qwen2.5-0.5B-Instruct
#   meta-llama/Llama-3-8B
#   mistralai/Mistral-7B-v0.1
#   gpt2

# Interactive deterministic inference
determl run <model>

# Scan model for non-deterministic ops (Dropout, Flash Attention, etc.)
determl scan <model>

# Verify determinism (run 5 times, compare hashes)
determl verify <model>

# Before vs after determl comparison
determl compare <model>

# Full benchmark (auto-scales based on model size)
determl benchmark <model>

# Export inference proof to JSON
determl export <model> -o proof.json

# Verify a proof from another machine
determl cross-verify proof.json

# Show environment information
determl info
```

### Interactive Menu (run_determl.sh)

```
>> What would you like to do?
   1) run          - Interactive deterministic inference
   2) scan         - Scan model for non-deterministic ops
   3) verify       - Verify model produces deterministic output
   4) compare      - Before vs after determl comparison
   5) benchmark    - Full determinism benchmark (auto-scales)
   6) export       - Export inference proof (for cross-GPU verify)
   7) cross-verify - Verify a proof from another machine
   8) info         - Show environment information
   9) exit         - Exit determl
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
determl compare Qwen/Qwen2.5-Coder-0.5B-Instruct
```

Runs the model first **without** determl (raw PyTorch) then **with** determl, showing hash differences side by side.

### Cross-GPU Verification

Export a proof on one GPU, verify on another:

```bash
# Machine A (e.g., Vast.ai RTX 3070)
determl export Qwen/Qwen2.5-Coder-0.5B-Instruct -o proof.json

# Copy proof.json to Machine B

# Machine B (e.g., Colab T4)
determl cross-verify proof.json
# → VERIFIED: canonical hashes match across GPUs
```

---

## How It Works

determl addresses 7 sources of non-determinism:

| Source | Problem | determl Fix |
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
determl/
  __init__.py       # Top-level API: enforce(), status(), checkpoint_hash()
  config.py         # Seed locking + deterministic flags
  enforcer.py       # Runtime op patching (Dropout, Flash Attention)
  canonicalizer.py  # Cross-hardware output normalization
  guardian.py       # Environment fingerprinting + compatibility
  engine.py         # High-level DeterministicEngine for LLMs
  benchmark.py      # Auto-scaling benchmark suite (8 tiers, 36 prompts)
  proof.py          # Cross-GPU proof export/import/verify
  detector.py       # Static model scanning
  verifier.py       # Hash-based verification
  wrapper.py        # Simple HuggingFace wrapper
  cli.py            # CLI entry point (9 commands)
  utils.py          # Hashing + env snapshots
```

---

## API Reference

### Top-Level API

```python
import determl

determl.enforce(seed=42)              # Lock all randomness
determl.status()                       # Check enforcement state
determl.checkpoint_hash(model)         # Hash model weights (for training)
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

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

All 70 tests use tiny randomly-initialized models — no large downloads, CPU only, runs in ~60 seconds.

---

## License

MIT — see [LICENSE](LICENSE).
