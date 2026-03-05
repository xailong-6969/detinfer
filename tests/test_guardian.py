"""Tests for determl.guardian -- EnvironmentGuardian."""

import pytest
import torch

from determl.guardian import (
    EnvironmentGuardian,
    EnvironmentFingerprint,
    CompatibilityLevel,
    EnvironmentMismatchError,
)


class TestEnvironmentFingerprint:
    """Tests for EnvironmentFingerprint."""

    def test_create_fingerprint(self):
        """Should create a valid fingerprint of the current environment."""
        guardian = EnvironmentGuardian()
        fp = guardian.create_fingerprint()
        assert fp.torch_version == torch.__version__
        assert fp.python_version is not None
        assert fp.fingerprint_hash != ""

    def test_fingerprint_hash_deterministic(self):
        """Same environment should produce same fingerprint hash."""
        guardian = EnvironmentGuardian()
        fp1 = guardian.create_fingerprint()
        fp2 = guardian.create_fingerprint()
        assert fp1.fingerprint_hash == fp2.fingerprint_hash

    def test_to_dict_round_trip(self):
        """Should survive to_dict -> from_dict round trip."""
        guardian = EnvironmentGuardian()
        fp1 = guardian.create_fingerprint()
        d = fp1.to_dict()
        fp2 = EnvironmentFingerprint.from_dict(d)
        assert fp2.torch_version == fp1.torch_version
        assert fp2.python_version == fp1.python_version

    def test_to_json_round_trip(self):
        """Should survive to_json -> from_json round trip."""
        guardian = EnvironmentGuardian()
        fp1 = guardian.create_fingerprint()
        json_str = fp1.to_json()
        fp2 = EnvironmentFingerprint.from_json(json_str)
        assert fp2.torch_version == fp1.torch_version

    def test_str_representation(self):
        """String representation should be informative."""
        guardian = EnvironmentGuardian()
        fp = guardian.create_fingerprint()
        s = str(fp)
        assert "PyTorch" in s
        assert "Python" in s


class TestEnvironmentGuardian:
    """Tests for EnvironmentGuardian."""

    def test_self_comparison_strict(self):
        """Comparing environment with itself should be STRICT."""
        guardian = EnvironmentGuardian()
        fp = guardian.create_fingerprint()
        result = guardian.compare(fp, fp)
        assert result.level == CompatibilityLevel.STRICT

    def test_different_torch_version_compatible(self):
        """Same major, different minor should be COMPATIBLE (CPU)."""
        guardian = EnvironmentGuardian()
        fp1 = guardian.create_fingerprint()
        fp2 = guardian.create_fingerprint()

        # Simulate different minor version but same major
        fp2.torch_version = fp1.torch_major + ".999"
        fp2.cudnn_version = fp1.cudnn_version  # Keep same

        result = guardian.compare(fp1, fp2)
        # Should be COMPATIBLE since same major version, CPU
        assert result.level in (CompatibilityLevel.COMPATIBLE, CompatibilityLevel.STRICT)

    def test_completely_different_incompatible(self):
        """Completely different environments should be INCOMPATIBLE."""
        guardian = EnvironmentGuardian()
        fp1 = guardian.create_fingerprint()
        fp2 = EnvironmentFingerprint(
            torch_version="1.0.0",
            torch_major="1.0",
            cuda_version="10.0",
            cudnn_version=7000,
            gpu_name="Tesla V100",
            gpu_family="Volta",
            os_platform="Linux",
            python_version="3.8.0",
            numpy_version="1.20.0",
            deterministic_algorithms=False,
        )
        result = guardian.compare(fp1, fp2)
        assert result.level == CompatibilityLevel.INCOMPATIBLE

    def test_enforce_passes_on_self(self):
        """Enforcing against own fingerprint should pass."""
        guardian = EnvironmentGuardian()
        fp = guardian.create_fingerprint()
        result = guardian.enforce(fp, min_level=CompatibilityLevel.STRICT)
        assert result.level == CompatibilityLevel.STRICT

    def test_enforce_raises_on_incompatible(self):
        """Enforcing against incompatible env should raise."""
        guardian = EnvironmentGuardian()
        remote = EnvironmentFingerprint(
            torch_version="1.0.0",
            torch_major="1.0",
            cuda_version="10.0",
            cudnn_version=7000,
            gpu_name="Tesla V100",
            gpu_family="Volta",
            os_platform="Linux",
            python_version="3.8.0",
            numpy_version="1.20.0",
            deterministic_algorithms=False,
        )
        with pytest.raises(EnvironmentMismatchError):
            guardian.enforce(remote, min_level=CompatibilityLevel.STRICT)

    def test_warnings_for_non_deterministic(self):
        """Should warn if either env has deterministic algorithms disabled."""
        guardian = EnvironmentGuardian()
        fp1 = guardian.create_fingerprint()
        fp2 = guardian.create_fingerprint()
        fp2.deterministic_algorithms = False

        result = guardian.compare(fp1, fp2)
        assert any("DISABLED" in w for w in result.warnings)

    def test_comparison_result_str(self):
        """Comparison result string should be informative."""
        guardian = EnvironmentGuardian()
        fp = guardian.create_fingerprint()
        result = guardian.compare(fp, fp)
        s = str(result)
        assert "STRICT" in s
