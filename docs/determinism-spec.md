# detinfer Determinism Specification

This document defines exactly what "deterministic" means in detinfer,
what is hashed and why, what assumptions are required, and what is excluded.

---

## 1. Definition of Determinism

In detinfer, **deterministic** means:

> Given the same model weights, tokenizer, prompt bytes, seed, dtype, generation
> config, and supported backend — the output token sequence is **bitwise identical**
> across runs on the **same hardware**.

With canonicalization enabled, detinfer additionally guarantees:

> The **canonical hash** is identical across **different hardware** (e.g. A100 vs RTX 4090)
> when all supported configuration requirements are met.

### What deterministic is NOT

- **Not** trustless verification — the `proof.json` is a replay proof, not a cryptographic
  proof of execution. Another party can re-run the same inference to check, but cannot
  verify without running the model themselves.
- **Not** semantic stability — two outputs may be byte-different but semantically equivalent.
  detinfer enforces byte/token determinism, not fuzzy similarity.
- **Not** guaranteed across unsupported backends — see the Supported Matrix below.

---

## 2. What is Hashed

Each inference run produces the following hashes (all SHA-256):

| Field | What is hashed | Purpose |
|---|---|---|
| `input_tokens_hash` | Raw input token ID tensor | Catches tokenizer/chat-template drift |
| `output_tokens_hash` | Complete output token ID tensor (including prompt) | Token-level proof |
| `raw_hash` | Decoded text string | Quick sanity check |
| `canonical_hash` | Output token IDs after floating-point canonicalization | Cross-hardware comparison |

### Why token IDs, not text?

Decoded text can differ even when token IDs are identical, due to:
- `skip_special_tokens` behaviour differences
- Unicode normalization in the tokenizer decoder
- Whitespace handling across transformers versions

detinfer hashes **token IDs** (integers) which are hardware-independent and unambiguous.

---

## 3. Required Assumptions

For detinfer's determinism guarantee to hold, ALL of the following must be true:

| Assumption | Why it matters |
|---|---|
| Same model weights (identical checkpoint) | Different checkpoints → different outputs |
| Same tokenizer version | Different versions may tokenize identically-typed strings differently |
| Same prompt bytes (exact, not just visually identical) | Hidden Unicode characters or whitespace differences change tokens |
| Same seed (default: 42) | Seeds control all RNG sources |
| Same dtype (fp32 / fp16 / bf16) | Different dtypes produce different floating-point rounding |
| Same generation config | `do_sample=False`, `num_beams=1` enforced automatically |
| Supported backend (see below) | Unsupported backends may have kernel-level nondeterminism |

---

## 4. Supported / Unsupported Matrix

### Fully Supported

| Feature | Notes |
|---|---|
| PyTorch eager mode | Default, fully deterministic |
| Greedy decoding (`do_sample=False`, `num_beams=1`) | Enforced automatically |
| fp32 inference | Full determinism on any hardware |
| fp16 inference | Deterministic on supported backends |
| CPU inference | Fully deterministic |
| CUDA (single GPU) | Deterministic with enforced flags |
| HuggingFace CausalLM models | Any CausalLM model on HuggingFace Hub |
| Cross-GPU canonical comparison | Via canonicalized token-level hashes |

### Partially Supported (best-effort)

| Feature | Limitation |
|---|---|
| bf16 inference | Hardware-dependent rounding; canonical hash may differ across GPU generations |
| Multi-GPU (`device_map="auto"`) | Supported for inference; tensor split order may vary |
| Flash Attention | Replaced with deterministic MATH backend automatically; may reduce throughput |

### Unsupported / Experimental

| Feature | Reason |
|---|---|
| Quantized inference (GPTQ, AWQ, bitsandbytes, GGUF) | Kernel-specific integer rounding is hardware-dependent — proof not reliable cross-machine |
| `torch.compile` | Graph autotuning selects different kernels across runs |
| Beam search (`num_beams > 1`) | Tie-breaking is implementation-specific |
| Speculative decoding | Draft model introduces additional nondeterminism |
| vLLM / paged attention | KV cache layout and paging is non-deterministic |
| Tensor parallelism / pipeline parallelism | Reduction order across devices is not controlled |
| Triton autotuning | Block size and tile selection is hardware-dependent |

---

## 5. Proof Format

The `proof.json` exported by `detinfer export` contains the following fields:

```json
{
  "model_name": "HuggingFace model ID",
  "seed": 42,
  "prompt": "Input prompt string",
  "max_new_tokens": 256,
  "precision": "high",

  "canonical_hash": "SHA-256 of canonicalized token IDs",
  "input_tokens_hash": "SHA-256 of tokenized input token IDs",
  "output_tokens_hash": "SHA-256 of complete generated token IDs",
  "raw_hash": "SHA-256 of decoded text",

  "model_dtype": "torch.float16",
  "quantization": "none",

  "gpu_name": "NVIDIA A100-SXM4-80GB",
  "cuda_version": "12.1",
  "torch_version": "2.2.0",
  "transformers_version": "4.39.0",
  "python_version": "3.11.0",
  "platform": "Linux...",
  "timestamp": "2026-03-09T10:00:00+00:00",
  "detinfer_version": "0.3.0"
}
```

### What the canonical hash proves

The `canonical_hash` proves that:
1. The same model was run
2. The same prompt was tokenized to the same tokens
3. The same token sequence was generated
4. After canonicalization (floating-point rounding to eliminate hardware noise), the result matches

### What the canonical hash does NOT prove

- That the computation was performed on a specific machine
- That the operator did not tamper with results (not a cryptographic proof)
- That quantized models are equivalent across machines

---

## 6. Operating Modes

detinfer provides three effective operating modes depending on your use case:

### Strict Mode (default for `detinfer verify` / `detinfer export`)
- All seeds locked
- Flash Attention replaced with MATH backend
- Greedy decoding enforced
- Full token-level hashing
- Use when: cross-hardware verification, audit trails, decentralized AI

### Compatible Mode (default for `detinfer run`)
- Seeds locked
- Greedy decoding enforced
- Chat template auto-applied
- Use when: interactive inference, development

### Audit Mode (`detinfer cross-verify`)
- Does not force settings — loads proof and re-runs with proof's configuration
- Detects and reports mismatch sources (tokenizer drift, dtype mismatch, quantization)
- Use when: diagnosing why two machines disagree

---

## 7. Mismatch Diagnosis

When `detinfer cross-verify` reports a mismatch, it checks these fields in order:

1. **Input tokens mismatch** → tokenizer or chat template differs between machines
2. **Quantization mismatch** → proof used quantized model, local run did not (or vice versa)
3. **dtype mismatch** → fp16 vs bf16 between machines
4. **Output tokens mismatch, canonical match** → rounding within tolerance, canonical hash is valid
5. **All hashes mismatch** → fundamental non-determinism; check CUDA version and model weights

---

## 8. Terminology

| Term | Definition |
|---|---|
| **Deterministic** | Bitwise identical output on the same hardware setup |
| **Canonical hash** | Hardware-normalized hash for cross-machine comparison |
| **Raw hash** | Hash of decoded text; not suitable for cross-machine comparison |
| **Token hash** | Hash of integer token ID sequence; the strongest form of output identity |
| **Proof** | A JSON record containing hashes and environment metadata for replay verification |
| **Cross-GPU verification** | Re-running the same inference on different hardware and comparing canonical hashes |

