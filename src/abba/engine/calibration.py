"""Calibration metrics and temperature scaling for probability predictions.

Provides:
- Proper scoring rules: log loss, Brier score
- Calibration diagnostics: ECE, reliability bins
- Temperature scaling: post-hoc recalibration of model probabilities
- Calibration artifact: serializable results from a backtest run

All computations use numpy only. No sklearn or scipy required.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Scoring rules
# ---------------------------------------------------------------------------

def log_loss(y_true: list[bool] | np.ndarray, y_pred: list[float] | np.ndarray,
             eps: float = 1e-7) -> float:
    """Binary log loss (negative log-likelihood). Lower is better.

    Proper scoring rule: minimized when predicted probabilities equal
    true probabilities. Random baseline (0.5) = ln(2) ≈ 0.6931.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.clip(np.asarray(y_pred, dtype=np.float64), eps, 1 - eps)
    return -float(np.mean(y_t * np.log(y_p) + (1 - y_t) * np.log(1 - y_p)))


def brier_score(y_true: list[bool] | np.ndarray, y_pred: list[float] | np.ndarray) -> float:
    """Brier score (mean squared error of probabilities). Lower is better.

    Range: [0, 1]. Random baseline (0.5) = 0.25. Perfect = 0.0.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_t - y_p) ** 2))


def accuracy(y_true: list[bool] | np.ndarray, y_pred: list[float] | np.ndarray) -> float:
    """Directional accuracy: fraction of games where the favored side won."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    picks = y_p > 0.5
    return float(np.mean(picks == y_t))


# ---------------------------------------------------------------------------
# Calibration diagnostics
# ---------------------------------------------------------------------------

def reliability_bins(
    y_true: list[bool] | np.ndarray,
    y_pred: list[float] | np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Compute reliability diagram bins.

    Groups predictions into equal-width bins and compares the mean predicted
    probability to the observed frequency. Well-calibrated models have
    predicted_mean ≈ observed_frequency in each bin.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    bins = []
    edges = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        mask = (y_p >= edges[i]) & (y_p < edges[i + 1])
        if i == n_bins - 1:  # include right edge in last bin
            mask = (y_p >= edges[i]) & (y_p <= edges[i + 1])

        count = int(np.sum(mask))
        if count == 0:
            continue

        predicted_mean = float(np.mean(y_p[mask]))
        observed_freq = float(np.mean(y_t[mask]))

        bins.append({
            "bin_low": round(float(edges[i]), 2),
            "bin_high": round(float(edges[i + 1]), 2),
            "count": count,
            "predicted_mean": round(predicted_mean, 4),
            "observed_frequency": round(observed_freq, 4),
            "calibration_error": round(abs(predicted_mean - observed_freq), 4),
        })

    return bins


def expected_calibration_error(
    y_true: list[bool] | np.ndarray,
    y_pred: list[float] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Weighted average of per-bin |predicted - observed|, weighted by bin count.
    Lower is better. 0 = perfectly calibrated. >0.05 = poorly calibrated.
    """
    bins = reliability_bins(y_true, y_pred, n_bins=n_bins)
    if not bins:
        return 1.0

    total = sum(b["count"] for b in bins)
    ece = sum(b["calibration_error"] * b["count"] for b in bins) / total
    return round(ece, 4)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def _logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Inverse sigmoid: logit(p) = log(p / (1-p))."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid function, numerically stable."""
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z)),
    )


def find_temperature(
    y_true: list[bool] | np.ndarray,
    y_pred: list[float] | np.ndarray,
    n_iter: int = 200,
) -> float:
    """Find optimal temperature T that minimizes log loss on calibration set.

    Temperature scaling: calibrated_p = sigmoid(logit(p) / T)
    - T > 1: softens probabilities toward 0.5 (overconfident model)
    - T < 1: sharpens probabilities away from 0.5 (underconfident model)
    - T = 1: no change

    Uses grid search + golden-section refinement for robustness.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    logits = _logit(y_p)

    def _nll(T: float) -> float:
        scaled = _sigmoid(logits / T)
        eps = 1e-7
        scaled = np.clip(scaled, eps, 1 - eps)
        return -float(np.mean(y_t * np.log(scaled) + (1 - y_t) * np.log(1 - scaled)))

    # Coarse grid search over [0.1, 10.0]
    candidates = np.linspace(0.1, 10.0, 100)
    losses = [_nll(t) for t in candidates]
    best_idx = int(np.argmin(losses))

    # Refine with golden-section search around the best
    lo = candidates[max(best_idx - 1, 0)]
    hi = candidates[min(best_idx + 1, len(candidates) - 1)]
    gr = (np.sqrt(5) + 1) / 2

    for _ in range(50):
        if hi - lo < 1e-4:
            break
        c1 = hi - (hi - lo) / gr
        c2 = lo + (hi - lo) / gr
        if _nll(c1) < _nll(c2):
            hi = c2
        else:
            lo = c1

    T = round((lo + hi) / 2, 4)
    return max(0.1, min(10.0, T))


def apply_temperature(y_pred: list[float] | np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to predicted probabilities."""
    y_p = np.asarray(y_pred, dtype=np.float64)
    logits = _logit(y_p)
    return _sigmoid(logits / temperature)


# ---------------------------------------------------------------------------
# Calibration artifact
# ---------------------------------------------------------------------------

@dataclass
class CalibrationArtifact:
    """Serializable calibration results from a backtest run.

    This replaces _DEFAULT_ACCURACY_HISTORY in confidence.py with
    empirically measured values.
    """
    log_loss: float = 0.6931  # ln(2), coin flip baseline
    brier_score: float = 0.25  # coin flip baseline
    accuracy: float = 0.50
    sample_size: int = 0
    temperature: float = 1.0
    ece: float = 1.0
    reliability_bins: list[dict[str, Any]] = field(default_factory=list)
    date_range: str = ""
    model_version: str = "2.0.0"
    baselines: dict[str, float] = field(default_factory=dict)

    # Comparison vs baselines
    beats_coin_flip: bool = False
    beats_home_bias: bool = False
    beats_market: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "accuracy": self.accuracy,
            "sample_size": self.sample_size,
            "temperature": self.temperature,
            "ece": self.ece,
            "reliability_bins": self.reliability_bins,
            "date_range": self.date_range,
            "model_version": self.model_version,
            "baselines": self.baselines,
            "beats_coin_flip": self.beats_coin_flip,
            "beats_home_bias": self.beats_home_bias,
            "beats_market": self.beats_market,
            "calibration_status": "empirically_validated" if self.sample_size >= 50 else "insufficient_data",
        }

    def save(self, path: str | Path) -> None:
        """Write calibration artifact to JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> CalibrationArtifact:
        """Load calibration artifact from JSON."""
        data = json.loads(Path(path).read_text())
        return cls(
            log_loss=data.get("log_loss", 0.6931),
            brier_score=data.get("brier_score", 0.25),
            accuracy=data.get("accuracy", 0.50),
            sample_size=data.get("sample_size", 0),
            temperature=data.get("temperature", 1.0),
            ece=data.get("ece", 1.0),
            reliability_bins=data.get("reliability_bins", []),
            date_range=data.get("date_range", ""),
            model_version=data.get("model_version", "2.0.0"),
            baselines=data.get("baselines", {}),
            beats_coin_flip=data.get("beats_coin_flip", False),
            beats_home_bias=data.get("beats_home_bias", False),
            beats_market=data.get("beats_market", False),
        )

    @property
    def calibration_error(self) -> float:
        """The empirically measured calibration error for confidence intervals.

        Replaces the hardcoded _BASE_CALIBRATION_ERROR = 0.08 in confidence.py.
        """
        return self.ece

    @property
    def accuracy_history(self) -> dict[str, Any]:
        """Format compatible with confidence.py's _DEFAULT_ACCURACY_HISTORY.

        Drop-in replacement for the fabricated baselines.
        """
        status = "empirically_validated" if self.sample_size >= 50 else "insufficient_data"
        return {
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "accuracy": self.accuracy,
            "sample_size": self.sample_size,
            "date_range": self.date_range,
            "calibration_status": status,
        }

    def check_acceptance(
        self,
        max_log_loss: float = 0.6900,
        max_brier: float = 0.2490,
        max_ece: float = 0.060,
        min_sample: int = 200,
        must_beat_coin_flip: bool = True,
        must_beat_home_bias: bool = True,
    ) -> dict[str, Any]:
        """Validate whether this model passes the acceptance gate.

        Thresholds are derived from the current best backtest (Mar 2026):
          log_loss=0.6844, brier=0.2457, ECE=0.0419, N=1037

        A model change must not regress beyond these bounds. The defaults
        include headroom (~0.005 log loss) so minor fluctuations don't
        block deployment.

        Returns dict with 'passed' bool and per-check results.
        """
        checks = {
            "log_loss": {
                "value": self.log_loss,
                "threshold": max_log_loss,
                "passed": self.log_loss <= max_log_loss,
                "direction": "lower is better",
            },
            "brier_score": {
                "value": self.brier_score,
                "threshold": max_brier,
                "passed": self.brier_score <= max_brier,
                "direction": "lower is better",
            },
            "ece": {
                "value": self.ece,
                "threshold": max_ece,
                "passed": self.ece <= max_ece,
                "direction": "lower is better",
            },
            "sample_size": {
                "value": self.sample_size,
                "threshold": min_sample,
                "passed": self.sample_size >= min_sample,
                "direction": "higher is better",
            },
            "beats_coin_flip": {
                "value": self.beats_coin_flip,
                "threshold": must_beat_coin_flip,
                "passed": self.beats_coin_flip or not must_beat_coin_flip,
            },
            "beats_home_bias": {
                "value": self.beats_home_bias,
                "threshold": must_beat_home_bias,
                "passed": self.beats_home_bias or not must_beat_home_bias,
            },
        }

        all_passed = all(c["passed"] for c in checks.values())
        failures = [name for name, c in checks.items() if not c["passed"]]

        return {
            "passed": all_passed,
            "checks": checks,
            "failures": failures,
            "summary": "ACCEPTED" if all_passed else f"REJECTED: {', '.join(failures)}",
        }
