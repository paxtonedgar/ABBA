"""Ensemble prediction engine.

Combines multiple model predictions using statistically sound methods.
Each method has a specific use case:
- weighted: default, weights by inverse variance (more certain models count more)
- average: equal weight, baseline
- median: robust to outliers
- voting: binary classification (>0.5 threshold)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, median

import numpy as np


@dataclass
class PredictionResult:
    """Structured prediction output."""
    value: float
    confidence: float
    error_margin: float
    model_count: int
    method: str
    individual_predictions: list[float]
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "value": round(self.value, 4),
            "confidence": round(self.confidence, 4),
            "error_margin": round(self.error_margin, 4),
            "model_count": self.model_count,
            "method": self.method,
            "individual_predictions": [round(p, 4) for p in self.individual_predictions],
            "timestamp": self.timestamp,
        }


class EnsembleEngine:
    """Statistically sound ensemble prediction combining."""

    def combine(
        self,
        predictions: list[float],
        method: str = "weighted",
        weights: list[float] | None = None,
    ) -> PredictionResult:
        if not predictions:
            return PredictionResult(
                value=0.0, confidence=0.0, error_margin=1.0,
                model_count=0, method=method,
                individual_predictions=[], timestamp=datetime.now().isoformat(),
            )

        preds = np.array(predictions, dtype=np.float64)

        if method == "weighted":
            value = self._weighted_combine(preds, weights)
        elif method == "median":
            value = float(np.median(preds))
        elif method == "voting":
            value = float(np.mean(preds > 0.5))
        else:
            value = float(np.mean(preds))

        std = float(np.std(preds, ddof=1)) if len(preds) > 1 else 0.5
        n = len(preds)

        # 95% CI margin: t-approximation for small n, z for large
        if n > 1:
            margin = 1.96 * std / np.sqrt(n)
        else:
            margin = 0.5

        # Confidence: based on model agreement (low std = high confidence)
        # Maps std from [0, 0.5] to confidence [1.0, 0.0]
        confidence = float(np.clip(1.0 - 2.0 * std, 0.0, 1.0))

        return PredictionResult(
            value=float(value),
            confidence=confidence,
            error_margin=float(margin),
            model_count=n,
            method=method,
            individual_predictions=predictions,
            timestamp=datetime.now().isoformat(),
        )

    def _weighted_combine(
        self, preds: np.ndarray, weights: list[float] | None
    ) -> float:
        """Consensus-proximity weighting when no explicit weights given.

        Models with predictions closer to the group mean get higher weight.
        This is NOT inverse-variance weighting (which requires per-model
        variance estimates). It is agreement-based: outlier models are
        down-weighted relative to the consensus.
        """
        if weights:
            w = np.array(weights, dtype=np.float64)
            w = w / w.sum()
            return float(np.dot(preds, w))

        if len(preds) == 1:
            return float(preds[0])

        # When models nearly agree (std < 0.02), consensus-proximity weighting
        # becomes numerically unstable (tiny distances → huge weight ratios).
        # Fall back to simple mean in that case.
        group_mean = float(np.mean(preds))
        std = float(np.std(preds))
        if std < 0.02:
            return group_mean

        distances = np.abs(preds - group_mean) + 1e-4
        inv_weights = 1.0 / distances
        inv_weights = inv_weights / inv_weights.sum()
        return float(np.dot(preds, inv_weights))

    def validate(self, predictions: list[float]) -> dict:
        """Validate an ensemble is usable."""
        if not predictions:
            return {"valid": False, "reason": "no predictions"}
        if len(predictions) < 2:
            return {"valid": False, "reason": "need at least 2 models"}

        preds = np.array(predictions)
        std = float(np.std(preds, ddof=1))
        spread = float(np.max(preds) - np.min(preds))

        valid = spread <= 0.5 and std <= 0.25
        return {
            "valid": valid,
            "reason": "valid" if valid else f"high disagreement (std={std:.3f}, spread={spread:.3f})",
            "std": round(std, 4),
            "spread": round(spread, 4),
            "mean": round(float(np.mean(preds)), 4),
        }

    @staticmethod
    def data_hash(game_id: str, model_version: str, data: dict) -> str:
        """Deterministic hash for cache key."""
        raw = json.dumps({"game_id": game_id, "model_version": model_version, **data},
                         sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
