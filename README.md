# detinfer

Deterministic runtime controls, tracing, replay verification, and drift checks for supported LLM workflows.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-v0.3.0-blue.svg)](https://pypi.org/project/detinfer/)

Start here if you want plain-language docs: [docs/GUIDE.md](docs/GUIDE.md)  
Detailed semantics and guarantees: [docs/determinism-spec.md](docs/determinism-spec.md)

---

## Table of Contents

1. [What detinfer does](#what-detinfer-does)
2. [Installation (PyPI first)](#installation-pypi-first)
3. [Install from source (git clone)](#install-from-source-git-clone)
4. [Quick start](#quick-start)
5. [Inference commands](#inference-commands)
6. [Agent commands](#agent-commands)
7. [Agent harness commands](#agent-harness-commands)
8. [Core API](#core-api)
9. [Compatibility and limits](#compatibility-and-limits)
10. [Run tests](#run-tests)

---

## What detinfer does

detinfer focuses on deterministic LLM inference and replayability by combining:

- Runtime enforcement (seeds, deterministic torch/cuda settings, backend controls)
- Deterministic decoding for agent mode (argmax with stable tie-breaks)
- Session traces (messages, token traces, hashes)
- Replay, diff, and regression classification (`check`)

What detinfer does not do:

- It is not a full training framework
- It does not guarantee universal cross-hardware bitwise identity for every workload
- It does not replace model/tokenizer/prompt/environment discipline

Training note:

- You can use `detinfer.enforce()` and `detinfer.checkpoint_hash(model)` in training loops to improve reproducibility checks.
- This is a utility layer, not a full training orchestration stack.

---

## Installation (PyPI first)

### 1) Create a Python virtual environment

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

```

### 2) Install detinfer

```bash
# Core
pip install detinfer

# Recommended for HuggingFace model workflows
pip install "detinfer[transformers]"

# Optional experimental int8 path
pip install "detinfer[quantized]"
```

Requirements:

- Python 3.10+
- PyTorch 2.0+
- NVIDIA GPU recommended (CPU works too)

---

## Install from source (git clone)

Use this if you want to develop locally or modify the codebase.

### Clone repository

```bash
git clone https://github.com/xailong-6969/detinfer.git
cd detinfer
```

Then set up env + editable install:

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1


# Editable install
pip install -e "."
pip install -e ".[transformers]"
pip install -e ".[dev]"
```

---

## Quick start

### One-line enforcement

```python
import detinfer
detinfer.enforce(seed=42)
```

### Quick CLI flow

```bash
# Deterministic interactive inference
detinfer run <hf-model>

# Repeat verification (same prompt, N runs)
detinfer verify <hf-model> --runs 5

# Deterministic multi-turn agent
detinfer agent <hf-model>
```

Replace `<hf-model>` with your HuggingFace model id.

---

## Inference commands

These commands are for single prompt / direct inference validation.

| Command | Purpose |
|---|---|
| `detinfer run <hf-model>` | Interactive deterministic inference |
| `detinfer verify <hf-model> --runs N` | Repeat-run determinism check |
| `detinfer compare <hf-model>` | Raw sampling vs detinfer side-by-side |
| `detinfer scan <hf-model>` | Scan and report non-deterministic ops |
| `detinfer doctor <hf-model>` | Health/audit report |
| `detinfer benchmark <hf-model>` | Prompt-suite determinism benchmark |
| `detinfer export <hf-model> -o proof.json` | Export inference proof |
| `detinfer cross-verify proof.json` | Re-run and verify exported proof |
| `detinfer info` | Print environment fingerprint info |

Example:

```bash
detinfer verify meta-llama/Llama-3.2-1B-Instruct --runs 5 --seed 42
detinfer doctor meta-llama/Llama-3.2-1B-Instruct --json
```

---

## Agent commands

These commands are for multi-turn deterministic conversation traces.

| Command | Purpose |
|---|---|
| `detinfer agent <hf-model>` | Interactive deterministic agent |
| `detinfer agent <hf-model> --prompt "..."` | Non-interactive single prompt |
| `detinfer agent <hf-model> --export session.json` | Export session trace |
| `detinfer agent <hf-model> --trace-mode minimal|standard|verbose` | Control trace detail |
| `detinfer agent <hf-model> --save-state state.json` | Save resumable state |
| `detinfer agent <hf-model> --load-state state.json` | Resume state |
| `detinfer replay session.json [--strict]` | Replay and verify session |
| `detinfer verify-session session.json [--strict]` | Verify session as execution proof |
| `detinfer diff run_a.json run_b.json` | First divergence finder |
| `detinfer check baseline.json candidate.json` | Drift classification |

Example:

```bash
detinfer agent <hf-model> --prompt "What is 2+2?" --export run.json
detinfer replay run.json --strict
detinfer check baseline.json run.json --json
```

---

## Agent harness commands

Harness runs deterministic task files (`.json`) for repeatable agent testing.

| Command | Purpose |
|---|---|
| `detinfer agent-run task.json` | Run one task |
| `detinfer agent-run examples/` | Run all tasks in a directory |
| `detinfer agent-run examples/ --output-dir runs/` | Export traces + manifest |
| `detinfer agent-run task.json --against baseline.json` | Compare against baseline |
| `detinfer agent-run examples/ --json` | JSON output (CI friendly) |
| `detinfer agent-run examples/ --fail-fast` | Stop at first failure |

---

## Core API

```python
import detinfer
from detinfer import DeterministicEngine, DeterministicAgent

detinfer.enforce(seed=42)
status = detinfer.status()

engine = DeterministicEngine(seed=42)
engine.load("<hf-model>")
result = engine.run("Write hello world in Python")
print(result.text)
print(result.canonical_hash)

agent = DeterministicAgent("<hf-model>", seed=42)
reply = agent.chat("What is 2+2?")
agent.export_session("session.json")
```

---

## Compatibility and limits

Supported well:

- PyTorch eager mode
- Greedy decoding (`do_sample=False`, `num_beams=1`)
- CPU inference
- Single-GPU CUDA inference
- HuggingFace CausalLM workflows

Partial / best effort:

- `bf16` (hardware dependent)
- Multi-GPU `device_map="auto"`
- INT8 bitsandbytes flows (experimental)

Not supported or out of scope:

- GPTQ / AWQ / GGUF quantized kernels
- `torch.compile` graph autotuning paths
- Beam search determinism guarantees
- vLLM / paged-attention runtimes
- ROCm/MPS paths are not validated in this repo

See [docs/determinism-spec.md](docs/determinism-spec.md) for exact assumptions and proof semantics.

---

## Run tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT, see [LICENSE](LICENSE).






