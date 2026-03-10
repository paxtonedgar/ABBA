"""Biometrics processing for ABBA."""

from typing import Any

import numpy as np

from ..core.logging import get_logger
from .models import BiometricData

logger = get_logger(__name__)


class BiometricsProcessor:
    """Processes biometric data for analytics."""

    def __init__(self):
        """Initialize the biometrics processor."""
        self.fatigue_thresholds = {
            "heart_rate": {"low": 60, "high": 100},
            "movement": {"low": 0.5, "high": 2.0},
            "recovery": {"low": 0.3, "high": 0.8},
        }

    async def process(self, player_data: dict[str, Any]) -> BiometricData | None:
        """Process raw biometric data.

        Args:
            player_data: Raw biometric data

        Returns:
            Processed biometric data or None
        """
        try:
            if not player_data:
                logger.warning("No player data provided")
                return None

            # Process heart rate data
            hr_data = player_data.get("heart_rate", [])
            heart_rate = await self._process_heart_rate(hr_data)

            # Process fatigue metrics
            fatigue_metrics = player_data.get("fatigue_metrics", {})
            fatigue_level = await self._calculate_fatigue(fatigue_metrics)

            # Process movement data
            movement_data = player_data.get("movement", {})
            movement_metrics = await self._process_movement(movement_data)

            # Calculate recovery status
            recovery_status = await self._calculate_recovery(
                {
                    "heart_rate": heart_rate,
                    "fatigue_level": fatigue_level,
                    "movement_metrics": movement_metrics,
                }
            )

            return BiometricData(
                heart_rate=heart_rate,
                fatigue_level=fatigue_level,
                movement_metrics=movement_metrics,
                recovery_status=recovery_status,
            )

        except Exception as e:
            logger.error(f"Error processing biometric data: {e}")
            return None

    async def _process_heart_rate(self, hr_data: list[float]) -> dict[str, float]:
        """Process heart rate data.

        Args:
            hr_data: List of heart rate measurements

        Returns:
            Processed heart rate features
        """
        try:
            if not hr_data:
                return {
                    "mean_hr": 0.0,
                    "max_hr": 0.0,
                    "min_hr": 0.0,
                    "hr_variability": 0.0,
                    "fatigue_indicator": 0.0,
                }

            hr_array = np.array(hr_data)

            # Calculate basic statistics
            mean_hr = float(np.mean(hr_array))
            max_hr = float(np.max(hr_array))
            min_hr = float(np.min(hr_array))

            # Calculate heart rate variability
            hr_variability = float(np.std(hr_array))

            # Calculate fatigue indicator
            fatigue_indicator = self._calculate_hr_fatigue(hr_array)

            return {
                "mean_hr": mean_hr,
                "max_hr": max_hr,
                "min_hr": min_hr,
                "hr_variability": hr_variability,
                "fatigue_indicator": fatigue_indicator,
            }

        except Exception as e:
            logger.error(f"Error processing heart rate data: {e}")
            return {
                "mean_hr": 0.0,
                "max_hr": 0.0,
                "min_hr": 0.0,
                "hr_variability": 0.0,
                "fatigue_indicator": 0.0,
            }

    def _calculate_hr_fatigue(self, hr_data: np.ndarray) -> float:
        """Calculate fatigue indicator from heart rate data.

        Args:
            hr_data: Heart rate data array

        Returns:
            Fatigue indicator (0-1)
        """
        try:
            # Calculate trend
            trend = self._calculate_trend(hr_data)

            # Normalize to 0-1 range
            fatigue = np.clip((trend + 1) / 2, 0, 1)

            return float(fatigue)

        except Exception as e:
            logger.error(f"Error calculating HR fatigue: {e}")
            return 0.0

    def _calculate_trend(self, data: np.ndarray) -> float:
        """Calculate trend in data.

        Args:
            data: Input data array

        Returns:
            Trend value (-1 to 1)
        """
        try:
            if len(data) < 2:
                return 0.0

            # Simple linear trend
            x = np.arange(len(data))
            slope = np.polyfit(x, data, 1)[0]

            # Normalize slope
            trend = np.tanh(slope / np.std(data) if np.std(data) > 0 else 0)

            return float(trend)

        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return 0.0

    async def _calculate_fatigue(self, fatigue_metrics: dict[str, Any]) -> float:
        """Calculate overall fatigue level.

        Args:
            fatigue_metrics: Fatigue-related metrics

        Returns:
            Fatigue level (0-1)
        """
        try:
            if not fatigue_metrics:
                return 0.0

            # Extract relevant metrics
            sleep_quality = fatigue_metrics.get("sleep_quality", 0.5)
            stress_level = fatigue_metrics.get("stress_level", 0.5)
            workload = fatigue_metrics.get("workload", 0.5)

            # Calculate weighted fatigue score
            fatigue_score = (
                0.4 * (1 - sleep_quality)  # Poor sleep increases fatigue
                + 0.3 * stress_level  # High stress increases fatigue
                + 0.3 * workload  # High workload increases fatigue
            )

            return float(np.clip(fatigue_score, 0, 1))

        except Exception as e:
            logger.error(f"Error calculating fatigue: {e}")
            return 0.0

    async def _process_movement(
        self, movement_data: dict[str, Any]
    ) -> dict[str, float]:
        """Process movement data.

        Args:
            movement_data: Raw movement data

        Returns:
            Processed movement metrics
        """
        try:
            if not movement_data:
                return {
                    "total_distance": 0.0,
                    "avg_speed": 0.0,
                    "max_speed": 0.0,
                    "acceleration_count": 0.0,
                }

            return {
                "total_distance": float(movement_data.get("total_distance", 0.0)),
                "avg_speed": float(movement_data.get("avg_speed", 0.0)),
                "max_speed": float(movement_data.get("max_speed", 0.0)),
                "acceleration_count": float(
                    movement_data.get("acceleration_count", 0.0)
                ),
            }

        except Exception as e:
            logger.error(f"Error processing movement data: {e}")
            return {
                "total_distance": 0.0,
                "avg_speed": 0.0,
                "max_speed": 0.0,
                "acceleration_count": 0.0,
            }

    async def _calculate_recovery(self, processed_data: dict[str, Any]) -> float:
        """Calculate recovery status.

        Args:
            processed_data: Processed biometric data

        Returns:
            Recovery status (0-1)
        """
        try:
            # Extract components
            hr_data = processed_data.get("heart_rate", {})
            fatigue_level = processed_data.get("fatigue_level", 0.5)
            movement = processed_data.get("movement_metrics", {})

            # Calculate recovery indicators
            hr_recovery = 1.0 - (hr_data.get("fatigue_indicator", 0.0))
            fatigue_recovery = 1.0 - fatigue_level
            movement_recovery = min(movement.get("avg_speed", 0.0) / 2.0, 1.0)

            # Weighted recovery score
            recovery_score = (
                0.4 * hr_recovery + 0.4 * fatigue_recovery + 0.2 * movement_recovery
            )

            return float(np.clip(recovery_score, 0, 1))

        except Exception as e:
            logger.error(f"Error calculating recovery: {e}")
            return 0.5
