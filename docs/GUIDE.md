# detinfer Simple Guide

This guide explains detinfer in plain language and gives practical command flows.

If you need formal definitions and proof semantics, read [determinism-spec.md](determinism-spec.md).

---

## What problem detinfer solves

Without deterministic controls, the same model and prompt can produce different outputs across runs.

detinfer helps you make runs repeatable by combining:

- deterministic runtime configuration
- deterministic decoding in agent mode
- trace export for replay and auditing
- diff/check tools to diagnose drift

---

## Install (recommended order)

### 1) Create and activate a virtual environment first

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install from PyPI

```bash
pip install "detinfer[transformers]"
```

Optional:

```bash
pip install "detinfer[quantized]"
```

### 3) Optional source install (for development)

```bash
git clone https://github.com/xailong-6969/detinfer.git
cd detinfer

pip install -e ".[dev]"
```

---

## Core workflow

### A) Inference workflow

Use this when you want deterministic inference for prompts.

```bash
# Interactive deterministic generation
detinfer run <hf-model>

# Verify determinism over repeated runs
detinfer verify <hf-model> --runs 5

# Compare raw sampling vs detinfer
detinfer compare <hf-model>

# Full health audit
detinfer doctor <hf-model>
```

### B) Agent workflow

Use this when you need multi-turn deterministic traces and replay.

```bash
# Run deterministic multi-turn agent
detinfer agent <hf-model>

# One-shot prompt and export trace
detinfer agent <hf-model> --prompt "What is 2+2?" --export session.json

# Replay exported session
detinfer replay session.json --strict

# Verify session as proof
detinfer verify-session session.json --strict

# Compare two sessions
detinfer diff run_a.json run_b.json

# Classify drift
detinfer check baseline.json candidate.json --json
```

### C) Agent harness workflow

Use this when you want task-based deterministic testing.

```bash
# Run one task
detinfer agent-run task.json

# Run a suite
detinfer agent-run examples/ --output-dir runs/

# Compare against baseline
detinfer agent-run task.json --against baseline.json
```

---

## Minimal Python usage

```python
import detinfer
from detinfer import DeterministicEngine, DeterministicAgent

# Runtime enforcement
detinfer.enforce(seed=42)

# Inference engine
engine = DeterministicEngine(seed=42)
engine.load("<hf-model>")
result = engine.run("Write hello world in Python")
print(result.text)
print(result.canonical_hash)

# Agent
agent = DeterministicAgent("<hf-model>", seed=42)
reply = agent.chat("What is 2+2?")
agent.export_session("session.json")
```

---

## Trace modes (agent)

`--trace-mode` controls how much generation detail is saved:

- `minimal`: hashes and output tokens only
- `standard`: prompts, input tokens, and step records
- `verbose`: standard + top candidate tokens/scores

Important:

- `replay --strict` requires per-step trace records.
- For strict replay, export with `trace_mode=standard` or `trace_mode=verbose`.

---

## Training note

detinfer is not a full training framework.

You can still use:

- `detinfer.enforce(seed=...)` for deterministic runtime setup
- `detinfer.checkpoint_hash(model)` to compare model states across runs/machines

---

## FAQ

### Does detinfer support any HuggingFace CausalLM model?
Yes, for supported runtime settings and deterministic decoding constraints.

### Does it guarantee cross-GPU identical text in all cases?
No. Cross-hardware behavior is best-effort and depends on model/config/backend constraints.

### Can I use quantized models?
INT8 bitsandbytes is experimental. GPTQ/AWQ/GGUF are out of scope in this repo.

### Is this only for inference?
Primary focus is deterministic inference and agent replay. Training support is utility-level, not framework-level.




