"""Tests for calibration module — scoring rules, temperature scaling, artifacts."""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from abba.engine.calibration import (
    CalibrationArtifact,
    accuracy,
    apply_temperature,
    brier_score,
    expected_calibration_error,
    find_temperature,
    log_loss,
    reliability_bins,
)


# ---------------------------------------------------------------------------
# Scoring rules
# ---------------------------------------------------------------------------

class TestLogLoss:
    def test_perfect_predictions(self):
        """Perfect predictions should have very low log loss."""
        y_true = [True, False, True, False]
        y_pred = [0.99, 0.01, 0.99, 0.01]
        ll = log_loss(y_true, y_pred)
        assert ll < 0.02

    def test_coin_flip_baseline(self):
        """Predicting 0.5 always should give ~ln(2) = 0.693."""
        y_true = [True, False] * 50
        y_pred = [0.5] * 100
        ll = log_loss(y_true, y_pred)
        assert abs(ll - math.log(2)) < 0.01

    def test_worse_than_coin_flip(self):
        """Inverted predictions should have log loss > ln(2)."""
        y_true = [True, True, True, False, False, False]
        y_pred = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]
        ll = log_loss(y_true, y_pred)
        assert ll > math.log(2)

    def test_symmetric(self):
        """Log loss should be the same for mirrored predictions."""
        y_true = [True, False]
        y_pred = [0.8, 0.2]
        ll1 = log_loss(y_true, y_pred)
        ll2 = log_loss([False, True], [0.2, 0.8])
        assert abs(ll1 - ll2) < 1e-10

    def test_clipping_prevents_infinity(self):
        """Predictions of exactly 0 or 1 should not cause infinity."""
        y_true = [True, False]
        y_pred = [0.0, 1.0]  # worst possible
        ll = log_loss(y_true, y_pred)
        assert math.isfinite(ll)


class TestBrierScore:
    def test_perfect(self):
        y_true = [True, False, True]
        y_pred = [1.0, 0.0, 1.0]
        assert brier_score(y_true, y_pred) == 0.0

    def test_coin_flip(self):
        y_true = [True, False] * 50
        y_pred = [0.5] * 100
        assert abs(brier_score(y_true, y_pred) - 0.25) < 0.001

    def test_range(self):
        """Brier score should always be between 0 and 1."""
        rng = np.random.default_rng(42)
        y_true = rng.choice([True, False], size=100).tolist()
        y_pred = rng.uniform(0, 1, 100).tolist()
        bs = brier_score(y_true, y_pred)
        assert 0 <= bs <= 1


class TestAccuracy:
    def test_perfect(self):
        y_true = [True, False, True]
        y_pred = [0.9, 0.1, 0.8]
        assert accuracy(y_true, y_pred) == 1.0

    def test_all_wrong(self):
        y_true = [True, True]
        y_pred = [0.3, 0.2]
        assert accuracy(y_true, y_pred) == 0.0

    def test_coin_flip_approximate(self):
        """With random predictions, accuracy should be ~50%."""
        rng = np.random.default_rng(42)
        y_true = rng.choice([True, False], size=10000).tolist()
        y_pred = rng.uniform(0, 1, 10000).tolist()
        acc = accuracy(y_true, y_pred)
        assert 0.45 < acc < 0.55


# ---------------------------------------------------------------------------
# Calibration diagnostics
# ---------------------------------------------------------------------------

class TestReliabilityBins:
    def test_well_calibrated(self):
        """A well-calibrated model should have low per-bin error."""
        rng = np.random.default_rng(42)
        n = 1000
        y_pred = rng.uniform(0.2, 0.8, n)
        y_true = rng.random(n) < y_pred  # perfectly calibrated by construction

        bins = reliability_bins(y_true.tolist(), y_pred.tolist(), n_bins=5)
        for b in bins:
            assert b["calibration_error"] < 0.05, f"Bin {b['bin_low']}-{b['bin_high']} error too high"

    def test_empty_bins_excluded(self):
        """Bins with no predictions should not appear."""
        y_true = [True, True]
        y_pred = [0.55, 0.65]  # only in middle bins
        bins = reliability_bins(y_true, y_pred, n_bins=10)
        assert all(b["count"] > 0 for b in bins)


class TestECE:
    def test_perfectly_calibrated(self):
        """ECE should be near 0 for a perfectly calibrated model."""
        rng = np.random.default_rng(42)
        n = 2000
        y_pred = rng.uniform(0.1, 0.9, n)
        y_true = rng.random(n) < y_pred
        ece = expected_calibration_error(y_true.tolist(), y_pred.tolist())
        assert ece < 0.03

    def test_overconfident_model(self):
        """An overconfident model should have high ECE."""
        # Always predicts 0.9 but actual rate is 55%
        y_true = [True] * 55 + [False] * 45
        y_pred = [0.9] * 100
        ece = expected_calibration_error(y_true, y_pred)
        assert ece > 0.20


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

class TestTemperatureScaling:
    def test_overconfident_model_gets_t_above_1(self):
        """An overconfident model should get T > 1 to soften predictions."""
        rng = np.random.default_rng(42)
        n = 500
        # True probs are near 0.5, but model predicts extreme values
        true_probs = rng.uniform(0.4, 0.6, n)
        y_true = (rng.random(n) < true_probs).tolist()
        y_pred = np.where(np.array(true_probs) > 0.5, 0.85, 0.15).tolist()

        T = find_temperature(y_true, y_pred)
        assert T > 1.0, f"Expected T > 1 for overconfident model, got {T}"

    def test_underconfident_model_gets_t_below_1(self):
        """An underconfident model should get T < 1 to sharpen predictions."""
        rng = np.random.default_rng(42)
        n = 500
        # True probs are spread, but model predicts near 0.5
        true_probs = rng.uniform(0.2, 0.8, n)
        y_true = (rng.random(n) < true_probs).tolist()
        y_pred = (0.5 + (np.array(true_probs) - 0.5) * 0.2).tolist()  # compressed

        T = find_temperature(y_true, y_pred)
        assert T < 1.0, f"Expected T < 1 for underconfident model, got {T}"

    def test_well_calibrated_model_gets_t_near_1(self):
        """A well-calibrated model should get T ≈ 1."""
        rng = np.random.default_rng(42)
        n = 500
        y_pred_raw = rng.uniform(0.3, 0.7, n)
        y_true = (rng.random(n) < y_pred_raw).tolist()

        T = find_temperature(y_true, y_pred_raw.tolist())
        assert 0.7 < T < 1.3, f"Expected T ≈ 1 for calibrated model, got {T}"

    def test_apply_temperature_identity_at_1(self):
        """T=1 should not change predictions."""
        y_pred = [0.3, 0.5, 0.7]
        result = apply_temperature(y_pred, 1.0)
        np.testing.assert_allclose(result, y_pred, atol=1e-6)

    def test_apply_temperature_softens_above_1(self):
        """T > 1 should push predictions toward 0.5."""
        y_pred = [0.2, 0.8]
        result = apply_temperature(y_pred, 2.0)
        assert result[0] > 0.2  # pushed toward 0.5
        assert result[1] < 0.8  # pushed toward 0.5


# ---------------------------------------------------------------------------
# Calibration artifact
# ---------------------------------------------------------------------------

class TestCalibrationArtifact:
    def test_roundtrip_save_load(self, tmp_path):
        """Artifact should survive save/load cycle."""
        artifact = CalibrationArtifact(
            log_loss=0.65,
            brier_score=0.22,
            accuracy=0.58,
            sample_size=100,
            temperature=1.15,
            ece=0.045,
            date_range="2025-10-01 to 2026-03-01",
            beats_coin_flip=True,
        )
        path = tmp_path / "calibration.json"
        artifact.save(path)
        loaded = CalibrationArtifact.load(path)

        assert loaded.log_loss == 0.65
        assert loaded.temperature == 1.15
        assert loaded.beats_coin_flip is True
        assert loaded.sample_size == 100

    def test_calibration_error_property(self):
        """calibration_error should return ECE."""
        artifact = CalibrationArtifact(ece=0.042)
        assert artifact.calibration_error == 0.042

    def test_accuracy_history_format(self):
        """accuracy_history should match confidence.py's expected format."""
        artifact = CalibrationArtifact(
            log_loss=0.65, brier_score=0.22, accuracy=0.58,
            sample_size=100, date_range="2025-10 to 2026-03",
        )
        hist = artifact.accuracy_history
        assert "log_loss" in hist
        assert "brier_score" in hist
        assert "calibration_status" in hist
        assert hist["calibration_status"] == "empirically_validated"

    def test_insufficient_data_status(self):
        """Under 50 samples should report insufficient_data."""
        artifact = CalibrationArtifact(sample_size=30)
        assert artifact.accuracy_history["calibration_status"] == "insufficient_data"

    def test_to_dict_complete(self):
        """to_dict should include all fields."""
        artifact = CalibrationArtifact()
        d = artifact.to_dict()
        assert "log_loss" in d
        assert "calibration_status" in d
        assert "temperature" in d
        assert "baselines" in d


# ---------------------------------------------------------------------------
# Acceptance gate
# ---------------------------------------------------------------------------

class TestAcceptanceGate:
    """Model changes must pass the acceptance gate before shipping."""

    def test_good_model_passes(self):
        artifact = CalibrationArtifact(
            log_loss=0.6844, brier_score=0.2457, ece=0.042,
            sample_size=1037, beats_coin_flip=True, beats_home_bias=True,
        )
        result = artifact.check_acceptance()
        assert result["passed"], f"Good model should pass: {result['failures']}"

    def test_coin_flip_model_fails(self):
        artifact = CalibrationArtifact(
            log_loss=0.6931, brier_score=0.25, ece=0.01,
            sample_size=1000, beats_coin_flip=False, beats_home_bias=False,
        )
        result = artifact.check_acceptance()
        assert not result["passed"]
        assert "log_loss" in result["failures"]
        assert "beats_coin_flip" in result["failures"]

    def test_small_sample_fails(self):
        artifact = CalibrationArtifact(
            log_loss=0.65, brier_score=0.22, ece=0.03,
            sample_size=50, beats_coin_flip=True, beats_home_bias=True,
        )
        result = artifact.check_acceptance()
        assert not result["passed"]
        assert "sample_size" in result["failures"]

    def test_regression_log_loss_fails(self):
        """A model that regresses on log loss should fail."""
        artifact = CalibrationArtifact(
            log_loss=0.6920, brier_score=0.2480, ece=0.05,
            sample_size=500, beats_coin_flip=True, beats_home_bias=True,
        )
        result = artifact.check_acceptance()
        assert not result["passed"]
        assert "log_loss" in result["failures"]

    def test_custom_thresholds(self):
        """Thresholds should be overridable."""
        artifact = CalibrationArtifact(
            log_loss=0.6920, brier_score=0.2480, ece=0.05,
            sample_size=500, beats_coin_flip=True, beats_home_bias=True,
        )
        # Relax log loss threshold
        result = artifact.check_acceptance(max_log_loss=0.6950)
        assert result["passed"]

    def test_current_artifact_passes_gate(self):
        """The shipped calibration artifact must pass the acceptance gate.

        This is the CI canary — if this fails, something regressed.
        """
        path = Path(__file__).parent.parent / "src" / "abba" / "data" / "calibration.json"
        if not path.exists():
            pytest.skip("No calibration artifact yet")
        artifact = CalibrationArtifact.load(path)
        result = artifact.check_acceptance()
        assert result["passed"], (
            f"Shipped calibration artifact FAILS acceptance gate.\n"
            f"Failures: {result['failures']}\n"
            f"Checks: {json.dumps(result['checks'], indent=2, default=str)}"
        )
