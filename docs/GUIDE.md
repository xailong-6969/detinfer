# detinfer — The Simple Guide

Everything you need to understand detinfer, explained simply.

---

## What Problem Does detinfer Solve?

When you run an AI model (like ChatGPT, LLaMA, or Qwen) and ask it the same question twice, **you might get different answers each time**. This is called **non-determinism**.

### Why does this happen?

| Cause | What happens |
|-------|-------------|
| **Random number generators** | Different random seeds = different token sampling |
| **GPU floating-point math** | GPUs process numbers in slightly different orders each run |
| **cuDNN autotuning** | NVIDIA auto-selects different algorithms each run |
| **Flash Attention** | Uses non-deterministic CUDA operations internally |
| **Temperature / sampling** | `temperature > 0` adds randomness by design |

### Why is this a problem?

- **Testing**: You can't write tests if the output changes every time
- **Debugging**: "It worked yesterday" — but you can't reproduce the issue
- **Research**: Your paper claims X, but nobody can replicate your results
- **Production**: Your CI pipeline gives different results on different runs
- **Auditing**: You need proof that a specific model produced a specific output

**detinfer fixes all of this.**

---

## How Does detinfer Work?

detinfer does 3 things:

### 1. Locks Down Randomness

```python
import detinfer
detinfer.enforce(seed=42)
```

This single line:
- Sets Python's `random` seed
- Sets NumPy's seed
- Sets PyTorch's seed (CPU + GPU)
- Sets CUDA's seed
- Disables cuDNN autotuning
- Forces PyTorch to use deterministic algorithms
- Configures cuBLAS workspace

**Result**: Every source of randomness is locked. Same input → same output.

### 2. Records Everything

When you use detinfer's agent, it records a **session trace** — a complete log of:
- What you said (prompts)
- What the model said (responses)
- Every token the model generated
- The hash of each input/output

This trace is saved as a JSON file. Think of it like a **black box recorder** for AI conversations.

### 3. Lets You Verify and Compare

With the traces, you can:
- **Replay** a conversation to verify it produces the same output
- **Diff** two conversations to see exactly where they diverge
- **Check** if anything changed between two runs (model drift, tokenizer change, etc.)

---

## The 5-Minute Tutorial

### Step 1: Install

```bash
pip install "detinfer[transformers]"
```

### Step 2: Make Any Model Deterministic

The simplest way — one line:

```python
import detinfer
detinfer.enforce()  # That's it. Everything is deterministic now.
```

### Step 3: Use the Deterministic Agent

```python
from detinfer import DeterministicAgent

# Replace <hf-model> with any HuggingFace model
agent = DeterministicAgent("<hf-model>", seed=42)

# Chat with the model
response = agent.chat("What is 2+2?")
print(response)  # Always the same answer

# Chat again — it remembers the conversation
response = agent.chat("Explain why")
print(response)  # Always the same follow-up

# Save the full conversation trace
agent.export_session("my_session.json")
```

### Step 4: Verify It's Actually Deterministic

```bash
# Run the same model 5 times, check all outputs match
detinfer verify <hf-model>
```

Output:
```
✓ DETERMINISTIC: All 5 runs produced identical output
  Hash: a1b2c3d4...
```

### Step 5: Compare Two Runs

```bash
# If something changed, find out exactly what
detinfer diff run_a.json run_b.json
```

Output:
```
First divergence at turn 2, token 47:
  Expected: token 1234 ("the")
  Got:      token 5678 ("a")
```

---

## Core Concepts Explained Simply

### What is a "Session Trace"?

A session trace is a JSON file that records everything about a conversation:

```
┌─────────────────────────────────┐
│         Session Trace           │
│                                 │
│  Model: <hf-model>              │
│  Seed:  42                      │
│  Hash:  a1b2c3...               │
│                                 │
│  Messages:                      │
│    User: "What is 2+2?"         │
│    Bot:  "2+2 equals 4."        │
│                                 │
│  Generations:                   │
│    Turn 1:                      │
│      Input tokens:  [464, 318]  │
│      Output tokens: [17, 10]    │
│      Stop reason:   eos         │
│      Steps:                     │
│        Step 0: chose token 17   │
│        Step 1: chose token 10   │
└─────────────────────────────────┘
```

### What is "Deterministic Argmax"?

When a model generates text, at each step it scores every possible next word. Normally you'd pick the highest-scoring word. But what if two words have the **exact same score**?

Different hardware might break the tie differently. detinfer solves this by always picking the **smallest token ID** when there's a tie:

```
Scores:  "the" (ID 100) = 0.95,  "a" (ID 50) = 0.95
Normal:  Could pick either → non-deterministic
detinfer: Always picks ID 50 ("a") → deterministic
```

### What are "Trace Modes"?

How much detail to record:

| Mode | What you get | When to use |
|------|-------------|-------------|
| `minimal` | Just hashes and output tokens | CI pipelines, automated testing |
| `standard` | + full prompts, input tokens, step trace | Normal debugging and replay |
| `verbose` | + top-10 candidate tokens at every step | Deep diagnosis — "why did it pick THIS word?" |

All three modes produce the **same hash** — so you can compare a minimal trace against a verbose trace.

### What is "Regression Checking"?

When you update your model, tokenizer, or environment, things might change. `detinfer check` compares two traces and tells you **exactly what changed**:

```
detinfer check old_run.json new_run.json
```

It classifies changes into types:

| Change | What it means | How serious |
|--------|--------------|-------------|
| `MODEL_DRIFT` | Model weights changed | Critical -- the model is different |
| `TOKENIZER_DRIFT` | Tokenizer version changed | Critical -- same text, different tokens |
| `CONFIG_DRIFT` | Seed or generation config changed | Critical -- settings are different |
| `PROMPT_DRIFT` | The prompt changed | Critical -- different input |
| `OUTPUT_DRIFT` | Model produced different tokens | Critical -- core output changed |
| `ENVIRONMENT_DRIFT` | PyTorch/Python version changed | Warning -- usually harmless |
| `TRACE_DETAIL_DRIFT` | Verbose fields differ | Info -- cosmetic only |

### What is the "Agent Harness"?

The harness lets you define tests for your AI as simple JSON files:

```json
{
  "name": "math_test",
  "model": "<hf-model>",
  "seed": 42,
  "prompt": "What is 2+2?",
  "expected": {
    "match": "contains",
    "value": "4"
  }
}
```

Then run them:

```bash
detinfer agent-run math_test.json
```

Think of it as **unit tests for AI models**.

---

## Common Workflows

### Workflow 1: "I want my model to give the same answer every time"

```python
import detinfer
from detinfer import DeterministicAgent

agent = DeterministicAgent("<hf-model>", seed=42)

# This will ALWAYS produce the same output
response = agent.chat("Explain quantum computing")
print(response)
```

### Workflow 2: "I want to prove my model produced a specific output"

```bash
# Generate output and save proof
detinfer agent <hf-model> --prompt "What is gravity?" --export proof.json

# Send proof.json to someone else
# They can verify it on their machine:
detinfer verify-session proof.json
```

### Workflow 3: "I updated my model and want to check what changed"

```bash
# Before update: save a baseline
detinfer agent <hf-model> --prompt "Test question" --export baseline.json

# After update: run the same thing
detinfer agent <hf-model> --prompt "Test question" --export new_run.json

# Compare
detinfer check baseline.json new_run.json
```

### Workflow 4: "I want automated testing in CI"

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

### Workflow 5: "I want to test my model across different scenarios"

Create task files in `examples/`:

```bash
# Run all tests
detinfer agent-run examples/ --output-dir results/

# Output:
#   ✓ basic_math (42ms)
#   ✓ code_generation (85ms)
#   ✗ edge_case_unicode [match_failed]
#
#   Total: 3  |  Passed: 2  |  Failed: 1
```

---

## CLI Cheat Sheet

| What you want | Command |
|--------------|---------|
| Make a model deterministic | `detinfer run <hf-model>` |
| Verify determinism works | `detinfer verify <hf-model>` |
| Chat with deterministic model | `detinfer agent <hf-model>` |
| Save a conversation | `detinfer agent <hf-model> --export session.json` |
| Replay a conversation | `detinfer replay session.json` |
| Compare two runs | `detinfer diff a.json b.json` |
| Check what changed | `detinfer check old.json new.json` |
| Run test suite | `detinfer agent-run examples/` |
| Scan for issues | `detinfer scan <hf-model>` |
| Health check | `detinfer doctor <hf-model>` |
| System info | `detinfer info` |

---

## FAQ

### Q: Does detinfer work with any HuggingFace model?
**A:** Yes. Any CausalLM model from HuggingFace is supported. You just pass the model name (e.g., `owner/model-name`) and detinfer handles the rest.

### Q: Does it guarantee the same output on different GPUs?
**A:** Best-effort. On the **same hardware**, outputs are bitwise identical. Across different GPUs (e.g., A100 vs RTX 4090), the **canonical hash** provides cross-hardware comparison after floating-point normalization.

### Q: Does it slow down inference?
**A:** Slightly. Disabling cuDNN autotuning and Flash Attention adds ~5-15% overhead on GPU. On CPU, there's no measurable difference.

### Q: Can I use it with training, not just inference?
**A:** Yes. `detinfer.enforce()` locks all seeds and deterministic flags for both training and inference. Use `detinfer.checkpoint_hash(model)` to verify weights match across machines.

### Q: What about quantized models (GPTQ, AWQ)?
**A:** Experimental. INT8 via bitsandbytes may improve consistency but is not guaranteed bitwise identical across hardware. GPTQ/AWQ are not supported.

### Q: Does it work with vLLM or TGI?
**A:** No. These use paged attention and custom CUDA kernels that bypass PyTorch's determinism controls.

---

## Glossary

| Term | Simple explanation |
|------|-------------------|
| **Deterministic** | Same input always gives the same output |
| **Non-deterministic** | Same input might give different outputs each time |
| **Seed** | A number that controls randomness — same seed = same random choices |
| **Session trace** | A recording of an entire AI conversation, including every token |
| **Token** | The smallest unit of text a model works with (roughly a word or syllable) |
| **Hash** | A fingerprint of data — if the data changes, the hash changes |
| **Canonical hash** | A hardware-independent hash that allows cross-GPU comparison |
| **Drift** | When something changes between two runs (model, tokenizer, output, etc.) |
| **Replay** | Re-running a saved session to verify it produces the same result |
| **Argmax** | Picking the highest-scoring next token — the simplest decoding strategy |
| **cuDNN** | NVIDIA's library for neural network operations |
| **Flash Attention** | A fast attention algorithm that is unfortunately non-deterministic |
