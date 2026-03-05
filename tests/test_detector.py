"""Tests for determl.detector — NonDeterminismDetector."""

import torch
import torch.nn as nn
import pytest

from determl.detector import NonDeterminismDetector, Severity


class CleanModel(nn.Module):
    """A model with only deterministic operations."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class ModelWithDropout(nn.Module):
    """A model with Dropout (non-deterministic in train mode)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(20, 5)

    def forward(self, x):
        return self.output(self.dropout(self.linear(x)))


class ModelWithNonDetOps(nn.Module):
    """A model with known non-deterministic GPU operations."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.frac_pool = nn.FractionalMaxPool2d(3, output_size=2)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TestNonDeterminismDetector:
    """Tests for the model scanner."""

    def setup_method(self):
        self.detector = NonDeterminismDetector()

    def test_clean_model_passes(self):
        """A model with only Linear + ReLU should have no warnings."""
        model = CleanModel()
        report = self.detector.scan(model)
        assert report.is_clean
        assert len(report.warnings) == 0

    def test_dropout_detected_as_info(self):
        """Dropout should be flagged as INFO (safe in eval mode)."""
        model = ModelWithDropout()
        report = self.detector.scan(model)
        assert len(report.infos) >= 1
        dropout_findings = [f for f in report.findings if "Dropout" in f.module_type]
        assert len(dropout_findings) >= 1
        assert all(f.severity == Severity.INFO for f in dropout_findings)

    def test_adaptive_pool_detected_as_warning(self):
        """AdaptiveAvgPool2d should be flagged as WARNING."""
        model = ModelWithNonDetOps()
        report = self.detector.scan(model)
        pool_findings = [f for f in report.findings if "AdaptiveAvgPool2d" in f.module_type]
        assert len(pool_findings) >= 1
        assert pool_findings[0].severity == Severity.WARNING

    def test_fractional_pool_detected(self):
        """FractionalMaxPool2d should be flagged as WARNING."""
        model = ModelWithNonDetOps()
        report = self.detector.scan(model)
        frac_findings = [f for f in report.findings if "FractionalMaxPool2d" in f.module_type]
        assert len(frac_findings) >= 1
        assert frac_findings[0].severity == Severity.WARNING

    def test_report_not_clean_with_warnings(self):
        """is_clean should be False when WARNING findings exist."""
        model = ModelWithNonDetOps()
        report = self.detector.scan(model)
        assert not report.is_clean

    def test_report_counts_all_modules(self):
        """total_modules should count all named_modules."""
        model = CleanModel()
        report = self.detector.scan(model)
        expected = sum(1 for _ in model.named_modules())
        assert report.total_modules == expected

    def test_custom_model_name(self):
        """Model name should be customizable."""
        model = CleanModel()
        report = self.detector.scan(model, model_name="MyCustomModel")
        assert report.model_name == "MyCustomModel"

    def test_default_model_name(self):
        """Default model name should be the class name."""
        model = CleanModel()
        report = self.detector.scan(model)
        assert report.model_name == "CleanModel"

    def test_report_str_clean(self):
        """String representation of clean report should indicate no issues."""
        model = CleanModel()
        report = self.detector.scan(model)
        report_str = str(report)
        assert "No non-deterministic" in report_str

    def test_report_str_with_findings(self):
        """String representation with findings should contain warnings."""
        model = ModelWithNonDetOps()
        report = self.detector.scan(model)
        report_str = str(report)
        assert "WARNING" in report_str
