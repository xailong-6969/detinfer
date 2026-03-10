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

**detinfer fixes this.** One import, one function call.

```python
import detinfer
detinfer.enforce()  # Everything is now deterministic.
```

---

## Installation

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

Update to latest version:

```bash
pip install --upgrade detinfer
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- Any NVIDIA GPU (recommended) or CPU

### Quick Reference — All CLI Commands

```bash
# Replace <model> with any HuggingFace model, e.g. gpt2, Qwen/Qwen2.5-0.5B-Instruct

# ── Inference ──
detinfer run <model>                                # Interactive deterministic inference
detinfer run <model> --seed 42 --max-tokens 512     # Custom seed and token limit

# ── Chat Agent ──
detinfer chat <model>                               # Multi-turn deterministic chat
detinfer chat <model> --prompt "What is 2+2?"       # Non-interactive (single question)
detinfer chat <model> --stream                      # Stream tokens in real-time
detinfer chat <model> --system "You are a tutor"    # Set system prompt
detinfer chat <model> --export session.json         # Export session trace
detinfer chat <model> --quantize int8               # Experimental INT8 mode
detinfer chat <model> --verbose-trace               # Record top-k tokens per step

# ── Verify & Replay ──
detinfer verify <model>                             # Run 5 times, compare hashes
detinfer replay session.json                        # Replay a saved session
detinfer replay session.json --strict               # Step-by-step verification
detinfer verify-session session.json                # Verify session as execution proof
detinfer verify-session session.json --strict       # Strict proof verification
detinfer diff run_a.json run_b.json                 # Token-level comparison of two runs

# ── Analysis ──
detinfer scan <model>                               # Scan for non-deterministic ops
detinfer compare <model>                            # Before vs after detinfer comparison
detinfer benchmark <model>                          # Full benchmark (auto-scales)

# ── Cross-GPU Proofs ──
detinfer export <model> -o proof.json               # Export proof for cross-GPU verification
detinfer cross-verify proof.json                    # Verify proof from another machine

# ── Info ──
detinfer info                                       # Show GPU and environment details
```

## Getting Started

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
engine = DeterministicEngine(seed=42)
engine.load("<model>", device_map="auto")

result = engine.run("Write hello world in Python")
```

### 3. Deterministic Chat Agent

```python
from detinfer import DeterministicAgent

# Multi-turn deterministic chat
agent = DeterministicAgent("gpt2", seed=42)
response = agent.chat("What is 2+2?")
print(response)  # Always the same answer

response = agent.chat("Explain more")
print(response)  # Always the same follow-up

# With system prompt
agent = DeterministicAgent("Qwen/Qwen2.5-0.5B-Instruct", seed=42, system_prompt="You are a math tutor")
response = agent.chat("What is calculus?")

# Streaming output (tokens appear one by one)
for chunk in agent.chat_stream("Explain gravity"):
    print(chunk, end="", flush=True)

# Export full session trace (token-by-token proof)
agent.export_session("session.json")
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

Prove that two different GPUs produce the same output:

```bash
# Machine A: export proof
detinfer export <model> -o proof.json

# Transfer proof.json to Machine B (scp, upload, etc.)

# Machine B: verify
detinfer cross-verify proof.json
```

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

---

## CLI Reference

### `detinfer run` — Interactive inference

```bash
detinfer run <model>
detinfer run <model> --seed 42 --max-tokens 256
```

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | 42 | Random seed |
| `--max-tokens` | 256 | Max tokens to generate |
| `--device` | auto | Device (cpu, cuda, auto) |

### `detinfer chat` — Deterministic chat agent

```bash
detinfer chat <model>
detinfer chat <model> --prompt "What is 2+2?"
detinfer chat <model> --stream
detinfer chat <model> --system "You are a math tutor"
detinfer chat <model> --export session.json
detinfer chat <model> --quantize int8
```

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | — | Non-interactive mode (single question) |
| `--stream` | off | Stream tokens as they are generated |
| `--system` | — | System prompt (e.g., "You are a math tutor") |
| `--seed` | 42 | Random seed |
| `--max-tokens` | 256 | Max tokens per turn |
| `--device` | auto | Device (cpu, cuda, auto) |
| `--export` | — | Export session trace to JSON |
| `--quantize` | — | Quantization mode (`int8`, experimental) |
| `--verbose-trace` | off | Record top-k tokens per step |

### `detinfer verify` — Verify determinism

```bash
detinfer verify <model>
detinfer verify <model> --runs 10
```

| Flag | Default | Description |
|------|---------|-------------|
| `--runs` | 5 | Number of runs to compare |
| `--seed` | 42 | Random seed |

### `detinfer replay` — Replay a saved session

```bash
detinfer replay session.json
detinfer replay session.json --strict
```

| Flag | Default | Description |
|------|---------|-------------|
| `--strict` | off | Verify every generation step, not just final tokens |
| `--model` | — | Override model (uses trace model if not set) |

### `detinfer verify-session` — Verify session as execution proof

```bash
detinfer verify-session session.json
detinfer verify-session session.json --strict
```

Shows model info, session hash, environment, re-runs all turns, and reports if it's a valid deterministic execution proof.

| Flag | Default | Description |
|------|---------|-------------|
| `--strict` | off | Verify every generation step |
| `--model` | — | Override model |

### `detinfer diff` — Compare two sessions

```bash
detinfer diff run_a.json run_b.json
```

Token-level comparison of two session traces. Shows first mismatch.

### `detinfer scan` — Scan for non-deterministic ops

```bash
detinfer scan <model>
```

Detects Dropout, Flash Attention, and other non-deterministic operations in the model.

### `detinfer compare` — Before vs after comparison

```bash
detinfer compare <model>
```

Runs the model first **without** detinfer (raw PyTorch) then **with** detinfer, showing hash differences side by side.

### `detinfer benchmark` — Full determinism benchmark

```bash
detinfer benchmark <model>
```

Tests determinism across 8 categories of prompts (auto-scales based on model size):

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

### `detinfer export` / `detinfer cross-verify` — Cross-GPU proofs

```bash
detinfer export <model> -o proof.json
detinfer cross-verify proof.json
```

### `detinfer info` — Environment information

```bash
detinfer info
```

Shows GPU, PyTorch version, CUDA version, and determinism flags.

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
  cli.py            # CLI entry point (14 commands)

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

  agent/            # Deterministic agent system
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
| `.load(model_name, quantize="int8")` | Load with INT8 quantization (experimental) |
| `.run(prompt, max_new_tokens)` | Deterministic inference |
| `.verify(prompt, num_runs)` | Run N times, compare hashes |
| `.scan()` | Show enforcement report |

### DeterministicAgent

| Method | Description |
|--------|-------------|
| `DeterministicAgent(model, seed, system_prompt)` | Create agent |
| `.chat(message)` | Send message, get deterministic response |
| `.chat_stream(message)` | Stream tokens as they are generated |
| `.export_session(path)` | Export full token trace to JSON |
| `.get_session_hash()` | Get canonical session hash |

### Proof System

| Function | Description |
|----------|-------------|
| `create_proof(engine, prompt)` | Run inference, create exportable proof |
| `cross_verify(proof)` | Re-run locally, compare with proof |
| `InferenceProof.save(path)` | Export proof to JSON |
| `InferenceProof.load(path)` | Load proof from JSON |

### Replay & Diff

| Function | Description |
|----------|-------------|
| `replay_session(trace_path)` | Re-run session, verify token-by-token |
| `diff_sessions(path_a, path_b)` | Compare two traces, find first mismatch |

---

## GitHub Action

Add determinism verification to your CI pipeline:

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

      # Or generate + export a new session
      - uses: xailong-6969/detinfer@v2-enforcement
        with:
          command: chat
          model: gpt2
          prompt: "What is 2+2?"
          export: output.json
```

---

## Supported Matrix

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

> For the full technical specification — see [docs/determinism-spec.md](docs/determinism-spec.md).

---

## Running Tests

```bash
pip install "detinfer[dev]"
pytest tests/ -v
```

---

## License

MIT — see [LICENSE](LICENSE).
