# determl — Deterministic ML Inference Library

**Detect, prevent, and verify non-determinism in ML inference.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## The Problem

Machine learning models can give **different outputs every time you run them** — even with the exact same input. This happens because of:

- **Random seeds** — PyTorch, NumPy, and Python each have their own random number generators
- **GPU non-determinism** — CUDA operations like `scatter_add` and Flash Attention use non-deterministic algorithms by default for speed
- **cuDNN auto-tuning** — picks different algorithms on different runs
- **Floating point arithmetic** — the order of addition can change results across hardware

This is a problem when you need to **prove** that a model was executed correctly — for example, in decentralized AI networks where multiple nodes must agree on the output.

## Quick Start

### Installation

```bash
# Basic (torch + numpy only)
pip install -e .

# With HuggingFace wrapper support
pip install -e ".[transformers]"

# With dev tools (pytest)
pip install -e ".[dev]"
```

### 1. Lock Down Randomness (30 seconds)

```python
from determl import DeterministicConfig

config = DeterministicConfig(seed=42)
config.apply()
# Done! PyTorch, NumPy, Python random, CUDA, cuDNN — all locked.
```

### 2. Scan a Model for Problems

```python
from determl import NonDeterminismDetector

detector = NonDeterminismDetector()
report = detector.scan(your_model)
print(report)
# ⚠️  Found 2 potentially non-deterministic operations:
#   - 'attention': ScaledDotProductAttention (Flash Attention)
#   - 'pool': AdaptiveAvgPool2d (non-deterministic backward on CUDA)
```

### 3. Verify Determinism

```python
from determl import InferenceVerifier

verifier = InferenceVerifier(model, tokenizer)
result = verifier.verify("What is 2+2?", num_runs=5, seed=42)
print(result)
# ✅ DETERMINISTIC: All 5 runs produced identical output
# SHA-256: a1b2c3d4e5f6...
```

### 4. Wrap an LLM (Full Solution)

```python
from determl import DeterministicLLM

llm = DeterministicLLM("Qwen/Qwen2.5-Coder-0.5B-Instruct", seed=42)
output = llm.generate("Write hello world in Python")
# Same output every single time — guaranteed.
```

---

## API Reference

### `DeterministicConfig`

| Method | Description |
|--------|-------------|
| `DeterministicConfig(seed=42, warn_only=False)` | Create config |
| `.apply()` | Lock all RNG sources + set deterministic flags |
| `.reset_seeds()` | Re-apply just the seeds (for repeated runs) |
| `.snapshot()` | Return dict of current determinism state |

### `NonDeterminismDetector`

| Method | Description |
|--------|-------------|
| `NonDeterminismDetector()` | Create detector |
| `.scan(model, model_name=None)` | Scan model → `ScanReport` |

**`ScanReport`** properties: `.is_clean`, `.warnings`, `.infos`, `.findings`

### `InferenceVerifier`

| Method | Description |
|--------|-------------|
| `InferenceVerifier(model, tokenizer=None, device="cpu")` | Create verifier |
| `.verify(prompt, num_runs=5, seed=42)` | Verify text generation |
| `.verify_with_input(tensor, num_runs=5, seed=42)` | Verify with raw tensor |

**`VerificationResult`** properties: `.is_deterministic`, `.hashes`, `.unique_hashes`, `.elapsed_seconds`, `.environment`

### `DeterministicLLM`

| Method | Description |
|--------|-------------|
| `DeterministicLLM(model_name, seed=42)` | Load model by name |
| `DeterministicLLM(model=m, tokenizer=t, seed=42)` | Use pre-loaded model |
| `.generate(prompt, max_new_tokens=256)` | Generate text deterministically |
| `.generate_with_hash(prompt)` | Generate + SHA-256 hash |
| `.verify(prompt, num_runs=5)` | Built-in verification |
| `.get_info()` | Model + environment info |

### Utilities

| Function | Description |
|----------|-------------|
| `hash_tensor(tensor)` | SHA-256 of tensor bytes |
| `hash_string(text)` | SHA-256 of UTF-8 string |
| `get_environment_snapshot()` | Full compute environment dict |

---

## Deep Dive: Sources of Non-Determinism

### 1. Random Seeds
Multiple RNG systems (Python `random`, NumPy, PyTorch CPU, PyTorch CUDA) must ALL be seeded. Missing even one breaks determinism.

### 2. CUDA Operations
Some CUDA kernels use `atomicAdd` which is inherently non-deterministic. `torch.use_deterministic_algorithms(True)` forces deterministic alternatives (slower but reproducible).

### 3. cuDNN Auto-Tuning
`torch.backends.cudnn.benchmark = True` (default) lets cuDNN pick the fastest algorithm, which may vary between runs. We disable this.

### 4. cuBLAS Workspace
Matrix multiplications on GPU can produce different results depending on workspace size. Setting `CUBLAS_WORKSPACE_CONFIG=:4096:8` fixes this.

### 5. Sampling in LLMs
`do_sample=True` (temperature, top-k, top-p) introduces randomness by design. Greedy decoding (`do_sample=False`) eliminates this entirely.

### 6. Hardware Differences
Different GPU architectures (A100 vs V100) can produce different floating-point results. `get_environment_snapshot()` captures hardware info so you can verify environments match.

---

## Glossary

| Term | Meaning |
|------|---------|
| **Deterministic** | Same input → same output, every time |
| **Greedy decoding** | Always pick the highest-probability token (no randomness) |
| **SHA-256** | Cryptographic hash function — produces a unique 64-character fingerprint |
| **cuDNN** | NVIDIA's library for deep learning primitives (convolutions, etc.) |
| **cuBLAS** | NVIDIA's library for linear algebra (matrix multiply, etc.) |
| **Seed** | Starting value for a pseudo-random number generator |
| **Bitwise identical** | Exact same bytes — not just "close enough" |

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

All tests use tiny randomly-initialized models — no large downloads, CPU only.

---

## License

MIT — see [LICENSE](LICENSE).
