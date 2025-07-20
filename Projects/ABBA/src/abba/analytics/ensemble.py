"""Ensemble prediction management for ABBA."""

from statistics import mean, median
from typing import Any

import numpy as np

from ..core.logging import get_logger

logger = get_logger(__name__)


class EnsembleManager:
    """Manages ensemble predictions and model combination."""

    def __init__(self):
        """Initialize the ensemble manager."""
        self.combination_methods = {
            "average": self._average_combination,
            "weighted": self._weighted_combination,
            "median": self._median_combination,
            "voting": self._voting_combination,
        }

    async def combine_predictions(
        self, predictions: list[float], method: str = "weighted"
    ) -> float:
        """Combine multiple model predictions.

        Args:
            predictions: List of model predictions
            method: Combination method ('average', 'weighted', 'median', 'voting')

        Returns:
            Combined prediction
        """
        try:
            if not predictions:
                logger.warning("No predictions to combine")
                return 0.0

            if method not in self.combination_methods:
                logger.warning(f"Unknown combination method: {method}, using weighted")
                method = "weighted"

            combination_func = self.combination_methods[method]
            result = combination_func(predictions)

            logger.debug(
                f"Combined {len(predictions)} predictions using {method}: {result:.4f}"
            )

            return float(result)

        except Exception as e:
            logger.error(f"Error combining predictions: {e}")
            return float(mean(predictions)) if predictions else 0.0

    def _average_combination(self, predictions: list[float]) -> float:
        """Combine predictions using simple average.

        Args:
            predictions: List of predictions

        Returns:
            Average prediction
        """
        return mean(predictions)

    def _weighted_combination(self, predictions: list[float]) -> float:
        """Combine predictions using weighted average.

        Args:
            predictions: List of predictions

        Returns:
            Weighted average prediction
        """
        if not predictions:
            return 0.0

        # Simple weighting based on prediction confidence
        # In practice, this could be based on model performance
        weights = [
            1.0 + abs(pred - 0.5) for pred in predictions
        ]  # Higher weight for more confident predictions
        total_weight = sum(weights)

        if total_weight == 0:
            return mean(predictions)

        weighted_sum = sum(
            pred * weight for pred, weight in zip(predictions, weights, strict=False)
        )
        return weighted_sum / total_weight

    def _median_combination(self, predictions: list[float]) -> float:
        """Combine predictions using median.

        Args:
            predictions: List of predictions

        Returns:
            Median prediction
        """
        return median(predictions)

    def _voting_combination(self, predictions: list[float]) -> float:
        """Combine predictions using voting (majority for binary classification).

        Args:
            predictions: List of predictions

        Returns:
            Voting result
        """
        if not predictions:
            return 0.0

        # Convert to binary predictions (threshold at 0.5)
        binary_predictions = [1 if pred > 0.5 else 0 for pred in predictions]

        # Return majority vote as probability
        positive_votes = sum(binary_predictions)
        return positive_votes / len(binary_predictions)

    async def calculate_error_bars(self, predictions: list[float]) -> dict[str, float]:
        """Calculate error bars and confidence intervals.

        Args:
            predictions: List of model predictions

        Returns:
            Dictionary with confidence and margin
        """
        try:
            if not predictions:
                return {"confidence": 0.0, "margin": 0.0}

            # Calculate basic statistics
            _mean_pred = mean(predictions)
            std_pred = np.std(predictions)

            # Calculate confidence based on agreement
            agreement = 1.0 - std_pred  # Higher agreement = higher confidence
            confidence = np.clip(agreement, 0.0, 1.0)

            # Calculate error margin (95% confidence interval)
            margin = 1.96 * std_pred / np.sqrt(len(predictions))  # 95% CI

            return {"confidence": float(confidence), "margin": float(margin)}

        except Exception as e:
            logger.error(f"Error calculating error bars: {e}")
            return {"confidence": 0.5, "margin": 0.1}

    async def validate_ensemble(self, predictions: list[float]) -> dict[str, Any]:
        """Validate ensemble predictions.

        Args:
            predictions: List of model predictions

        Returns:
            Validation metrics
        """
        try:
            if not predictions:
                return {
                    "valid": False,
                    "reason": "No predictions provided",
                    "metrics": {},
                }

            # Check for extreme values
            min_pred = min(predictions)
            max_pred = max(predictions)
            range_pred = max_pred - min_pred

            # Check for agreement
            std_pred = np.std(predictions)
            agreement = 1.0 - std_pred

            # Determine if ensemble is valid
            valid = (
                len(predictions) >= 2
                and range_pred <= 0.5  # Predictions should be reasonably close
                and agreement >= 0.3  # Some level of agreement required
            )

            metrics = {
                "prediction_count": len(predictions),
                "min_prediction": min_pred,
                "max_prediction": max_pred,
                "prediction_range": range_pred,
                "standard_deviation": std_pred,
                "agreement": agreement,
                "mean_prediction": mean(predictions),
            }

            return {
                "valid": valid,
                "reason": (
                    "Valid ensemble" if valid else "Low agreement or extreme values"
                ),
                "metrics": metrics,
            }

        except Exception as e:
            logger.error(f"Error validating ensemble: {e}")
            return {"valid": False, "reason": f"Validation error: {e}", "metrics": {}}
