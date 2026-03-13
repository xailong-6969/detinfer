"""Tests for detinfer.proof -- proof replay helpers."""

import detinfer.inference.engine as engine_mod
from detinfer.inference.proof import InferenceProof, cross_verify


def test_cross_verify_restores_quantized_engine_load(monkeypatch):
    captured = {}

    class FakeResult:
        canonical_hash = "canon"
        raw_hash = "raw"
        text = "hello"
        input_tokens_hash = "input"
        output_tokens_hash = "output"

    class FakeEngine:
        def __init__(self, seed=42, precision="high"):
            captured["seed"] = seed
            captured["precision"] = precision

        def load(self, model_name, **kwargs):
            captured["model_name"] = model_name
            captured["load_kwargs"] = kwargs

        def run(self, prompt, max_new_tokens=256):
            captured["prompt"] = prompt
            captured["max_new_tokens"] = max_new_tokens
            return FakeResult()

    monkeypatch.setattr(engine_mod, "DeterministicEngine", FakeEngine)

    proof = InferenceProof(
        model_name="test-model",
        seed=7,
        prompt="Hello",
        max_new_tokens=12,
        precision="high",
        canonical_hash="canon",
        raw_hash="raw",
        text_output="hello",
        gpu_name="GPU",
        cuda_version=None,
        torch_version="2.0.0",
        python_version="3.12",
        platform="test",
        input_tokens_hash="input",
        output_tokens_hash="output",
        quantization="bitsandbytes",
    )

    result = cross_verify(proof)

    assert captured["seed"] == 7
    assert captured["precision"] == "high"
    assert captured["model_name"] == "test-model"
    assert captured["load_kwargs"] == {"quantize": "int8"}
    assert captured["prompt"] == "Hello"
    assert captured["max_new_tokens"] == 12
    assert result.verified
