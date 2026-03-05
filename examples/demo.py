"""
determl Quick Demo
==================

This script demonstrates the core features of determl:
1. Locking down randomness with DeterministicConfig
2. Scanning a model for non-deterministic operations
3. Verifying inference determinism with hashing

No large model downloads required — uses a tiny random model.
"""

import torch
import torch.nn as nn

from determl import (
    DeterministicConfig,
    NonDeterminismDetector,
    InferenceVerifier,
    hash_tensor,
    get_environment_snapshot,
)


# ── 1. A tiny demo model ────────────────────────────────────────────────
class TinyTransformer(nn.Module):
    """A minimal transformer-like model for demonstration."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 32)
        self.transformer = nn.TransformerEncoderLayer(d_model=32, nhead=2, batch_first=True)
        self.output_head = nn.Linear(32, 100)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output_head(x)


def main():
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    print("=" * 60)
    print("  determl — Deterministic ML Inference Demo")
    print("=" * 60)

    # ── Step 1: Lock down randomness ──────────────────────────────
    print("\nStep 1: Applying DeterministicConfig...")
    config = DeterministicConfig(seed=42, warn_only=True)
    config.apply()
    print(f"   {config}")
    print(f"   Snapshot: {config.snapshot()}")

    # ── Step 2: Create and scan a model ───────────────────────────
    print("\nStep 2: Scanning model for non-deterministic ops...")
    model = TinyTransformer()
    model.eval()

    detector = NonDeterminismDetector()
    report = detector.scan(model, model_name="TinyTransformer")
    print(f"\n{report}")

    # ── Step 3: Verify determinism ────────────────────────────────
    print("\nStep 3: Verifying inference determinism...")
    verifier = InferenceVerifier(model, device="cpu")

    # Create a fixed input (sequence of token IDs)
    input_ids = torch.randint(0, 100, (1, 10))

    result = verifier.verify_with_input(
        input_ids, num_runs=5, seed=42, store_outputs=True
    )
    print(f"\n{result}")

    # ── Step 4: Show hashing ──────────────────────────────────────
    print("\nStep 4: Tensor hashing demo...")
    t1 = torch.tensor([1.0, 2.0, 3.0])
    t2 = torch.tensor([1.0, 2.0, 3.0])
    t3 = torch.tensor([1.0, 2.0, 3.1])  # Slightly different

    h1 = hash_tensor(t1)
    h2 = hash_tensor(t2)
    h3 = hash_tensor(t3)

    print(f"   hash([1,2,3]):   {h1[:32]}...")
    print(f"   hash([1,2,3]):   {h2[:32]}...")
    print(f"   hash([1,2,3.1]): {h3[:32]}...")
    print(f"   Same? {h1 == h2}  |  Different? {h1 != h3}")

    # ── Step 5: Environment snapshot ──────────────────────────────
    print("\nStep 5: Environment snapshot...")
    env = get_environment_snapshot()
    for key, value in env.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 60)
    print("  Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
