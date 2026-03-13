# detinfer Determinism Specification

This document defines what detinfer guarantees, what it records, and what assumptions must hold.

---

## 1) Scope

detinfer is a deterministic inference and replay toolkit for supported LLM workflows.

Primary goals:

- deterministic repeated runs on the same setup
- reproducible session traces
- replay/diff/check tools to diagnose drift

Non-goals:

- trustless cryptographic proof of execution
- universal determinism across all kernels, backends, and model runtimes
- full training framework orchestration

---

## 2) Determinism definition

In detinfer, deterministic means:

> Given the same model weights, tokenizer, prompt bytes, seed, generation configuration,
> and supported backend, generated output tokens are expected to be identical across runs
> on the same hardware/software setup.

Cross-hardware comparison is best-effort. Canonical hashes are useful comparison artifacts,
but they are not a blanket guarantee for every model/runtime combination.

---

## 3) What is hashed

detinfer emits multiple SHA-256 hashes for inference/session artifacts.

| Field | Source | Purpose |
|---|---|---|
| `input_tokens_hash` | Tokenized prompt tensor | Detect tokenizer/template drift |
| `output_tokens_hash` | Generated token tensor | Detect output token drift |
| `raw_hash` | Decoded text output | Human-readable quick check |
| `canonical_hash` | Canonicalized output tensor | Stable comparison field in proofs |

Notes:

- For current CausalLM flows, token-level hashes are the strongest identity signal.
- Text equality is useful but weaker than token-hash equality.

---

## 4) Required assumptions

Deterministic replay/checking requires all of the following to be consistent:

- same model weights/checkpoint
- same tokenizer + chat template behavior
- same prompt bytes
- same seed
- same generation config (`do_sample=False`, greedy path)
- compatible runtime/backend (torch/cuda/model loading path)

---

## 5) Support matrix

### Supported

- PyTorch eager mode
- Greedy decoding (`do_sample=False`, `num_beams=1`)
- CPU inference
- Single-GPU CUDA inference
- HuggingFace CausalLM workflows

### Partial / best effort

- `bf16` inference (hardware-sensitive)
- Multi-GPU `device_map="auto"`
- INT8 bitsandbytes flows (experimental)

### Not supported / out of scope

- GPTQ/AWQ/GGUF quantized kernels
- `torch.compile` deterministic guarantees
- Beam search deterministic guarantees
- vLLM and paged-attention runtimes
- Unvalidated ROCm/MPS paths in this repo

---

## 6) Proof format (`detinfer export`)

`proof.json` includes runtime/config/hash fields such as:

- model id, seed, prompt, max tokens, precision
- `canonical_hash`, `raw_hash`, `input_tokens_hash`, `output_tokens_hash`
- dtype/quantization metadata
- environment metadata (GPU/CUDA/torch/python/transformers)

Proof verification (`cross-verify`) re-runs the same request locally and compares fields.

---

## 7) Session traces and replay

Agent session traces contain:

- semantic messages
- per-turn generation traces
- token hashes
- optional per-step details depending on trace mode

Trace modes:

- `minimal`: hash-focused trace, no detailed step records
- `standard`: includes step records
- `verbose`: includes step records + top candidates

Important replay rule:

- `replay --strict` requires per-step records.
- For strict replay, export traces with `trace_mode=standard` or `trace_mode=verbose`.

---

## 8) Mismatch interpretation

Common mismatch classes in `detinfer check`:

- `MODEL_DRIFT`: model/weights changed
- `TOKENIZER_DRIFT`: tokenizer or template changed
- `CONFIG_DRIFT`: seed or generation config changed
- `PROMPT_DRIFT`: input messages/prompt changed
- `INPUT_TOKEN_DRIFT`: tokenization changed
- `OUTPUT_DRIFT`: generated tokens changed
- `STOP_REASON_DRIFT`: stop condition changed
- `ENVIRONMENT_DRIFT`: runtime version drift
- `TRACE_DETAIL_DRIFT`: detail-only trace differences

Use `--fail-on` and `--allow` in CI to tune policy.

---

## 9) Practical guidance

For strongest reproducibility:

- pin model + tokenizer revisions
- pin torch/transformers versions
- keep seeds and generation config fixed
- use deterministic decode paths
- export and archive traces/proofs for comparisons

