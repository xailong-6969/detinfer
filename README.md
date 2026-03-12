# detinfer — Deterministic Inference & Replay Toolkit

**Deterministic runtime controls, session tracing, and replay verification for supported LLM workflows.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-115%20passed-brightgreen.svg)]()
[![PyPI](https://img.shields.io/badge/pypi-v0.2.3-blue.svg)](https://pypi.org/project/detinfer/)

---

## Why?

Most reproducibility guides stop at setting RNG seeds. In practice, LLM runs can still drift because of decoding settings, backend behavior, prompt rendering, tokenizer differences, and attention/runtime choices.

detinfer focuses on supported workflows where deterministic settings, token tracing, replay, and diffing can be used together to verify and debug LLM outputs.

```python
import detinfer
detinfer.enforce()  # Apply deterministic runtime settings
```

---

## What detinfer is

- A **deterministic runtime and verification toolkit** for supported LLM inference paths
- A **replay/debugging tool** for agent sessions and token-level generation traces
- A **reproducibility aid** for benchmarking, CI, and regression testing

## What detinfer is not

- A guarantee of universal determinism for all PyTorch workloads
- A guarantee of bitwise-identical outputs across all hardware and library combinations
- A replacement for careful control of tokenizer, prompt formatting, backend, and environment

---

## Note on Determinism

detinfer enforces deterministic execution by disabling stochastic sampling (temperature/top-p) and enabling deterministic CUDA execution modes. This ensures that identical inputs produce identical outputs when run under the same environment.

The determinism stack includes:
- Disabling sampling (`do_sample=False`, greedy decoding)
- Locking RNG seeds (PyTorch, Python, NumPy, CUDA)
- Forcing deterministic CUDA ops (`torch.use_deterministic_algorithms`)
- Disabling cuDNN autotuning
- Canonicalizing floating-point outputs
- Hashing results for verification

Cross-device determinism is best-effort and depends on the underlying hardware and kernels used by the model.

---

## Installation

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

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

# ══════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════

detinfer run <model>                                # Interactive deterministic inference
detinfer run <model> --seed 42 --max-tokens 512     # Custom seed and token limit
detinfer verify <model>                             # Run 5 times, compare hashes
detinfer verify <model> --runs 10                   # Custom number of runs
detinfer scan <model>                               # Scan for non-deterministic ops
detinfer compare <model>                            # Before vs after detinfer comparison
detinfer benchmark <model>                          # Full benchmark (auto-scales by model size)
detinfer export <model> -o proof.json               # Export proof for cross-GPU verification
detinfer cross-verify proof.json                    # Verify proof from another machine
detinfer doctor <model>                             # Determinism health check & audit
detinfer doctor <model> --json                      # JSON report for CI pipelines

# ══════════════════════════════════════
# AGENT
# ══════════════════════════════════════

detinfer agent <model>                              # Multi-turn deterministic agent
detinfer agent <model> --prompt "What is 2+2?"      # Non-interactive (single question)
detinfer agent <model> --system "You are a tutor"   # Set system prompt
detinfer agent <model> --export session.json        # Export session trace
detinfer agent <model> --quantize int8              # Experimental INT8 mode
detinfer agent <model> --trace-mode verbose         # Record top-k tokens per step
detinfer agent <model> --max-context-tokens 2048    # Deterministic context truncation
detinfer agent <model> --save-state state.json      # Save agent state on exit
detinfer agent <model> --load-state state.json      # Resume from saved state
detinfer replay session.json                        # Replay a saved agent session
detinfer replay session.json --strict               # Step-by-step verification
detinfer verify-session session.json                # Verify session as execution proof
detinfer diff run_a.json run_b.json                 # Token-level comparison of two runs

# ══════════════════════════════════════
# REGRESSION CHECK
# ══════════════════════════════════════

detinfer check baseline.json candidate.json         # Compare traces for regression
detinfer check baseline.json candidate.json --json  # JSON output for CI
detinfer check a.json b.json --fail-on OUTPUT_DRIFT # Fail on specific drift type
detinfer check a.json b.json --allow ENVIRONMENT_DRIFT  # Ignore env differences

# ══════════════════════════════════════
# INFO
# ══════════════════════════════════════

detinfer info                                       # Show GPU and environment details
```

---

## Inference

### Runtime Enforcement

```python
import detinfer

# Apply deterministic runtime settings for supported workflows
detinfer.enforce(seed=42)
```

This locks RNG seeds, disables cuDNN benchmarking, enables `torch.use_deterministic_algorithms`, and sets cuBLAS workspace config.

### DeterministicEngine

```python
from detinfer import DeterministicEngine

# Load any HuggingFace model + apply deterministic runtime settings
engine = DeterministicEngine(seed=42)
engine.load("<model>")  # e.g., "Qwen/Qwen2.5-0.5B-Instruct", "gpt2"

# Run inference under deterministic settings
result = engine.run("Write hello world in Python")
print(result.text)            # The generated text
print(result.canonical_hash)  # SHA-256 hash for verification
```

### Verify Determinism

```python
result = engine.verify(num_runs=5)
print(result)
# DETERMINISTIC: All 5 runs produced identical output
```

```bash
detinfer verify <model>         # CLI version
detinfer verify <model> --runs 10
```

### Benchmark

Tests determinism across 8 categories of prompts (auto-scales based on model size):

```bash
detinfer benchmark <model>
```

| Tier | Category | What it tests |
|------|----------|--------------|
| 1 | Sanity | Basic questions (baseline check) |
| 2 | Long output | 200+ token generations |
| 3 | Uncertain | Creative prompts (low confidence) |
| 4 | Complex code | Merge sort, LRU cache |
| 5 | Reasoning | Logic puzzles, step-by-step |
| 6 | Deep context | Long code + passage analysis |
| 7 | Adversarial | Designed to break determinism |
| 8 | Edge cases | Unicode, empty, special characters |

### Scan for Non-Deterministic Ops

```bash
detinfer scan <model>        # Detects Dropout, Flash Attention, etc.
```

### Before vs After Comparison

```bash
detinfer compare <model>     # Runs without detinfer, then with detinfer
```

### Cross-GPU Verification

```bash
# Machine A: export proof
detinfer export <model> -o proof.json

# Transfer proof.json to Machine B

# Machine B: verify
detinfer cross-verify proof.json
```

### Training Verification

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

---

## Agent

### Deterministic Agent

Deterministic agent sessions with replayable execution traces.

Both `chat()` and `chat_stream()` use manual token-by-token generation with deterministic argmax (smallest-token-ID tie-breaking).

```python
from detinfer import DeterministicAgent

# Multi-turn deterministic agent (works with any HuggingFace model)
agent = DeterministicAgent("Qwen/Qwen2.5-0.5B-Instruct", seed=42)
response = agent.chat("What is 2+2?")
print(response)

# With system prompt
agent = DeterministicAgent(
    "Qwen/Qwen2.5-0.5B-Instruct",
    seed=42,
    system_prompt="You are a math tutor"
)
response = agent.chat("What is calculus?")

# Streaming output (tokens appear one by one)
for chunk in agent.chat_stream("Explain gravity"):
    print(chunk, end="", flush=True)

# Export full session trace
agent.export_session("session.json")
```

### Agent Step Replay (Tool Call Tracing)

Register tools and call them — every call is recorded in the trace for replay and diffing.

```python
from detinfer import DeterministicAgent

agent = DeterministicAgent("TinyLlama/TinyLlama-1.1B-Chat-v1.0", seed=42)

# Register tools (only name + callable, never serialized)
agent.register_tool("calculator", lambda expression: str(eval(expression)))
agent.register_tool("lookup", lambda query: f"Result for: {query}")

# Use tools — recorded as agent steps
response = agent.chat("What is 2+2?")
result = agent.call_tool("calculator", {"expression": "2+2"})
response = agent.chat(f"The answer is {result}. Explain why.")

# Checkpoint key decision points
agent.checkpoint({"decision": "used_calculator", "confidence": "high"})

agent.export_session("agent_session.json")
```

The exported trace includes `agent_steps`:

```json
{
  "agent_steps": [
    {"step": 1, "type": "llm_generation", "turn": 1, "generation_turn": 1},
    {"step": 2, "type": "tool_call", "turn": 1, "tool": "calculator", "arguments": {"expression": "2+2"}},
    {"step": 3, "type": "tool_result", "turn": 1, "tool": "calculator", "result": "4"},
    {"step": 4, "type": "llm_generation", "turn": 2, "generation_turn": 2},
    {"step": 5, "type": "checkpoint", "turn": 2, "checkpoint_data": {"decision": "used_calculator"}}
  ],
  "registered_tools": ["calculator", "lookup"]
}
```

`detinfer diff` now detects tool call divergence:

```
Trace comparison: DIFFERENT
  First mismatch: turn 2
  Type: tool_name
  Expected: calculator
  Observed: web_search
```

### Session Export & Trace

Exported sessions contain:

```json
{
  "schema_version": "1",
  "model": "gpt2",
  "seed": 42,
  "messages": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."}
  ],
  "generations": [
    {
      "input_tokens": [464, 318],
      "output_tokens": [17, 10],
      "steps": [
        {"step": 0, "chosen_token": 17},
        {"step": 1, "chosen_token": 10}
      ]
    }
  ]
}
```

With `--trace-mode verbose`, each step also includes `top_tokens` and `top_scores` (top-10 candidates).

### Trace Modes

Three levels of trace detail — all produce the same canonical session hash:

| Mode | What's captured | Use case |
|------|----------------|----------|
| `minimal` | Hashes + output tokens only | CI, benchmarks |
| `standard` | + rendered prompt, input tokens, step trace | Replay, debugging |
| `verbose` | + top-k tokens and scores per step | Deep diagnosis |

```bash
detinfer agent <model> --trace-mode minimal    # Lightweight
detinfer agent <model> --trace-mode verbose    # Full detail
```

### Deterministic Truncation

When conversations exceed the context window, detinfer uses a deterministic truncation policy:

- System prompt is **never** dropped
- Latest user message is **never** dropped
- Oldest turns are dropped first (user+assistant pairs together)
- Every truncation event is recorded in the trace

```python
agent = DeterministicAgent(
    "gpt2", seed=42,
    max_context_tokens=2048  # Truncate when prompt exceeds 2048 tokens
)
```

Same history + same policy = same truncation = same output. This prevents hidden non-determinism from framework-specific truncation.

### Session Save & Resume

Save agent state mid-conversation and resume later:

```python
# Save
agent.save_state("agent_state.json")

# Resume (same model must be loaded)
agent.load_state("agent_state.json")
response = agent.chat("Continue where we left off")
```

CLI:
```bash
detinfer agent <model> --save-state state.json    # Save on exit
detinfer agent <model> --load-state state.json    # Resume
```

### Trace Type Separation

Inference and agent traces are clearly labeled in the JSON:

```json
{"trace_type": "inference", ...}   // from detinfer export
{"trace_type": "agent", ...}       // from detinfer agent --export
```

`detinfer check` will catch if you accidentally compare an inference trace against an agent trace (`TYPE_MISMATCH`).

### Replay & Verification

`detinfer replay` re-runs a saved session and verifies prompt hashes, input tokens, output tokens, and stop conditions.

```bash
detinfer replay run.json
detinfer replay run.json --strict
```

If a divergence occurs, the first mismatch is reported.

### Session Diffing

`detinfer diff` compares two saved sessions and reports the first divergence point at the token/step level.

```bash
detinfer diff run_a.json run_b.json
```

### Session Proof Verification

```bash
detinfer verify-session session.json
```

```
  ══════════════════════════════════════════════════════════════
    DETERMINISTIC EXECUTION PROOF VERIFICATION
  ══════════════════════════════════════════════════════════════

    Model:        gpt2
    Seed:         42
    Turns:        3

    ✓ VERIFIED — All 3 turns match exactly
    ✓ This session is a valid deterministic execution proof.
  ══════════════════════════════════════════════════════════════
```

### Regression Check

Compare two session traces and classify exactly what changed:

```bash
detinfer check baseline.json candidate.json
```

```
Detinfer Regression Report
--------------------------
Baseline:  baseline.json
Candidate: candidate.json

Status: FAILED
Primary type: TOKENIZER_DRIFT

Matched:
  ✓ model
  ✓ seed
  ✓ generation_config

Changed:
  ✗ tokenizer.tokenizer_hash

First mismatch:
  type:  TOKENIZER_DRIFT
  field: tokenizer.tokenizer_hash
```

Mismatch types:

| Type | Severity | Meaning |
|------|----------|---------|
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

---

## CLI Reference

### `detinfer agent`

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | — | Non-interactive mode (single question) |
| `--system` | — | System prompt (e.g., "You are a math tutor") |
| `--seed` | 42 | Random seed |
| `--max-tokens` | 256 | Max tokens per turn |
| `--device` | auto | Device (cpu, cuda, auto) |
| `--export` | — | Export session trace to JSON |
| `--quantize` | — | Quantization mode (`int8`, experimental) |
| `--trace-mode` | standard | Trace detail: `minimal`, `standard`, `verbose` |
| `--max-context-tokens` | — | Max prompt tokens before truncation |
| `--save-state` | — | Save agent state to file on exit |
| `--load-state` | — | Resume from a saved agent state file |

### `detinfer check`

| Flag | Default | Description |
|------|---------|-------------|
| `--json` | off | Output report as JSON |
| `--fail-on` | — | Mismatch type that should fail (repeatable) |
| `--allow` | — | Mismatch type to ignore (repeatable) |

### `detinfer verify-session`

| Flag | Default | Description |
|------|---------|-------------|
| `--strict` | off | Verify every generation step |
| `--model` | — | Override model |

### `detinfer replay`

| Flag | Default | Description |
|------|---------|-------------|
| `--strict` | off | Step-by-step verification |
| `--model` | — | Override model |

### `detinfer verify`

| Flag | Default | Description |
|------|---------|-------------|
| `--runs` | 5 | Number of runs to compare |
| `--seed` | 42 | Random seed |

---

## How It Works

detinfer applies deterministic runtime settings for 7 sources of non-determinism:

| Source | Problem | detinfer Setting |
|--------|---------|-----------------|
| Random seeds | Separate RNGs in Python, NumPy, PyTorch, CUDA | Locks all seeds in one call |
| CUDA atomics | `scatter_add`, `index_add` use non-deterministic `atomicAdd` | Enables `torch.use_deterministic_algorithms(True)` |
| Flash Attention | `scaled_dot_product_attention` is non-deterministic | Replaces with deterministic math backend |
| cuDNN tuning | Auto-selects different algorithms per run | Disables benchmark mode |
| cuBLAS workspace | Matrix multiplications vary with workspace config | Sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| Float ordering | Different GPUs may produce different float results | Canonicalizes outputs before hashing |
| LLM decoding | `temperature`, `top_k`, `top_p` add randomness | Uses deterministic argmax with stable tie-breaking |

---

## Architecture

```
detinfer/
  __init__.py       # Top-level API: enforce(), status(), checkpoint_hash()
  cli.py            # CLI entry point (15 commands)
  check.py          # Regression check: compare two traces, classify drift

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
    runtime.py      # DeterministicAgent — multi-turn with truncation + save/resume
    trace.py        # Token-level trace recording + session schema + trace modes
    replay.py       # Session replay verification + diff
```

---

## API Reference

### Top-Level API

```python
import detinfer

detinfer.enforce(seed=42)              # Apply deterministic runtime settings
detinfer.status()                       # Check enforcement state
detinfer.checkpoint_hash(model)         # Hash model weights (for training)
```

### DeterministicEngine

| Method | Description |
|--------|-------------|
| `DeterministicEngine(seed, precision, device)` | Create engine |
| `.load(model_name)` | Load HuggingFace model, apply deterministic settings |
| `.load(model_name, quantize="int8")` | Load with INT8 quantization (experimental) |
| `.run(prompt, max_new_tokens)` | Run inference under deterministic settings |
| `.verify(prompt, num_runs)` | Run N times, compare hashes |
| `.scan()` | Show enforcement report |

### DeterministicAgent

| Method | Description |
|--------|-------------|
| `DeterministicAgent(model, seed, system_prompt, max_context_tokens)` | Create agent |
| `.chat(message)` | Send message, get response (deterministic argmax) |
| `.chat_stream(message)` | Stream tokens as generated |
| `.register_tool(name, fn)` | Register a callable tool |
| `.call_tool(name, args)` | Call tool and record in trace |
| `.checkpoint(data)` | Record a checkpoint event |
| `.export_session(path)` | Export full token trace to JSON |
| `.get_session_hash()` | Get canonical session hash |
| `.save_state(path)` | Save full agent state for resume |
| `.load_state(path)` | Resume agent from saved state |

### Regression Check

| Function | Description |
|----------|-------------|
| `check_sessions(baseline, candidate)` | Compare two trace dicts, classify mismatches |
| `render_check_report(report)` | Human-readable regression report |

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

      - uses: xailong-6969/detinfer@main
        with:
          command: verify-session
          session-file: baseline.json
          strict: true
```

---

## Support Status

### Supported

- Single-process HuggingFace causal LM inference
- Deterministic agent session export and replay
- Token-level diffing for saved sessions
- Greedy decoding paths under controlled runtime settings

### Experimental

- INT8 quantized mode (may improve consistency, not guaranteed bitwise identical)
- Cross-device consistency checks
- Streaming/verbose trace paths

### Not Guaranteed

- Universal determinism for arbitrary PyTorch code
- Bitwise-identical results across all GPU architectures
- Distributed training or asynchronous multi-node systems
- External API or tool call determinism

### Detailed Compatibility

| Feature | Status | Notes |
|---|---|---|
| PyTorch eager mode | **Supported** | Default execution mode |
| Greedy decoding | **Supported** | Enforced via deterministic argmax |
| fp32 / fp16 inference | **Supported** | Deterministic on supported backends |
| CPU inference | **Supported** | Fully deterministic |
| NVIDIA GPU (single) | **Supported** | T4, V100, A100, RTX 3070/4090, etc. |
| HuggingFace CausalLM | **Supported** | GPT-2, Qwen, TinyLlama, LLaMA, etc. |
| bf16 inference | Partial | Hardware-dependent rounding may differ |
| Multi-GPU (`device_map="auto"`) | Partial | May have split-order edge cases |
| Flash Attention | Partial | Auto-replaced with MATH backend |
| INT8 (bitsandbytes) | **Experimental** | May improve consistency |
| GPTQ/AWQ quantization | **Not supported** | Kernel-specific rounding |
| `torch.compile` | **Not supported** | Graph autotuning |
| Beam search | **Not supported** | Tie-breaking is implementation-specific |
| vLLM / paged attention | **Not supported** | KV cache paging |
| AMD GPUs (ROCm) | Untested | |
| Apple Silicon (MPS) | Untested | |

---

## Running Tests

```bash
pip install "detinfer[dev]"
pytest tests/ -v
```

---

## License

MIT — see [LICENSE](LICENSE).
