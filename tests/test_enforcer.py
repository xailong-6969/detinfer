"""Tests for determl.enforcer -- DeterministicEnforcer."""

import torch
import torch.nn as nn
import pytest

from determl.enforcer import (
    DeterministicEnforcer,
    EnforcementReport,
    FixAction,
)


class ModelWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(20, 5)

    def forward(self, x):
        return self.output(self.dropout(self.linear(x)))


class ModelWithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=2, batch_first=True)
        self.linear = nn.Linear(32, 10)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        return self.linear(attn_out)


class CleanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class TestDeterministicEnforcer:
    """Tests for the enforcer module."""

    def setup_method(self):
        self.enforcer = DeterministicEnforcer(seed=42, warn_only=True)

    def test_clean_model_no_fixes(self):
        """Clean model should have no fixes applied."""
        model = CleanModel()
        report = self.enforcer.enforce(model)
        assert report.num_fixed == 0
        assert report.all_fixed

    def test_dropout_replaced(self):
        """Dropout should be replaced with identity."""
        model = ModelWithDropout()
        report = self.enforcer.enforce(model)

        dropout_fixes = [f for f in report.fixes if "Dropout" in f.module_type]
        assert len(dropout_fixes) >= 1
        assert dropout_fixes[0].action == FixAction.REPLACED

        # Verify the dropout is actually replaced — should be identity now
        x = torch.randn(1, 10)
        model.train()  # Even in train mode, replaced dropout is identity
        out1 = model(x)
        out2 = model(x)
        # Without enforcement, dropout in train mode would give different results
        # With enforcement, dropout is replaced with identity
        assert torch.equal(out1, out2)

    def test_attention_wrapped(self):
        """MultiheadAttention should be wrapped with deterministic SDPA."""
        model = ModelWithAttention()
        report = self.enforcer.enforce(model)

        attn_fixes = [f for f in report.fixes if "Attention" in f.module_type]
        assert len(attn_fixes) >= 1
        assert attn_fixes[0].action == FixAction.REPLACED

    def test_model_set_to_eval(self):
        """After enforcement, model should be in eval mode."""
        model = CleanModel()
        model.train()
        self.enforcer.enforce(model)
        assert not model.training

    def test_deterministic_context(self):
        """deterministic_context should reset seeds and run without error."""
        model = CleanModel()
        model.eval()
        self.enforcer.enforce(model)

        x = torch.randn(1, 10)

        with self.enforcer.deterministic_context():
            out1 = model(x)

        with self.enforcer.deterministic_context():
            out2 = model(x)

        assert torch.equal(out1, out2)

    def test_report_str(self):
        """Report string should be informative."""
        model = ModelWithDropout()
        report = self.enforcer.enforce(model)
        report_str = str(report)
        assert "FIXED" in report_str or "No non-deterministic" in report_str

    def test_enforce_idempotent(self):
        """Enforcing twice should not break anything."""
        model = ModelWithDropout()
        report1 = self.enforcer.enforce(model)
        report2 = self.enforcer.enforce(model)
        # Should not crash
        x = torch.randn(1, 10)
        model(x)  # Should work fine
