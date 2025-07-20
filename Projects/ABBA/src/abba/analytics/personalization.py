"""Personalization engine for ABBA."""

from datetime import datetime
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ..core.logging import get_logger
from .models import UserPatterns

logger = get_logger(__name__)


class PersonalizationEngine:
    """Personalizes models based on user behavior patterns."""

    def __init__(self):
        """Initialize the personalization engine."""
        self.pattern_cache: dict[str, UserPatterns] = {}

    async def analyze_patterns(self, user_history: list[Any]) -> UserPatterns:
        """Analyze user betting patterns.

        Args:
            user_history: User's betting history

        Returns:
            User patterns analysis
        """
        try:
            if not user_history:
                return UserPatterns(
                    success_rate=0.0,
                    preferred_sports=[],
                    bet_sizes=[],
                    time_patterns={},
                    risk_tolerance=0.5,
                )

            # Calculate success rate
            successful_bets = [
                bet for bet in user_history if getattr(bet, "outcome", False)
            ]
            success_rate = (
                len(successful_bets) / len(user_history) if user_history else 0.0
            )

            # Analyze preferred sports
            sports_counts = {}
            for bet in user_history:
                sport = getattr(bet, "sport", "unknown")
                sports_counts[sport] = sports_counts.get(sport, 0) + 1

            preferred_sports = sorted(
                sports_counts.keys(), key=lambda x: sports_counts[x], reverse=True
            )

            # Analyze bet sizes
            bet_sizes = [getattr(bet, "amount", 0.0) for bet in user_history]

            # Analyze time patterns
            time_patterns = await self._analyze_time_patterns(user_history)

            # Calculate risk tolerance
            risk_tolerance = self._calculate_risk_tolerance(user_history)

            return UserPatterns(
                success_rate=success_rate,
                preferred_sports=preferred_sports,
                bet_sizes=bet_sizes,
                time_patterns=time_patterns,
                risk_tolerance=risk_tolerance,
            )

        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return UserPatterns(
                success_rate=0.0,
                preferred_sports=[],
                bet_sizes=[],
                time_patterns={},
                risk_tolerance=0.5,
            )

    async def _analyze_time_patterns(self, user_history: list[Any]) -> dict[str, Any]:
        """Analyze time-based patterns in user behavior.

        Args:
            user_history: User's betting history

        Returns:
            Time pattern analysis
        """
        try:
            patterns = {
                "hourly_distribution": {},
                "daily_distribution": {},
                "weekly_distribution": {},
            }

            for bet in user_history:
                timestamp = getattr(bet, "timestamp", datetime.now())

                # Hourly distribution
                hour = timestamp.hour
                patterns["hourly_distribution"][hour] = (
                    patterns["hourly_distribution"].get(hour, 0) + 1
                )

                # Daily distribution
                day = timestamp.strftime("%A")
                patterns["daily_distribution"][day] = (
                    patterns["daily_distribution"].get(day, 0) + 1
                )

                # Weekly distribution
                week = timestamp.isocalendar()[1]
                patterns["weekly_distribution"][week] = (
                    patterns["weekly_distribution"].get(week, 0) + 1
                )

            return patterns

        except Exception as e:
            logger.error(f"Error analyzing time patterns: {e}")
            return {}

    def _calculate_risk_tolerance(self, user_history: list[Any]) -> float:
        """Calculate user's risk tolerance.

        Args:
            user_history: User's betting history

        Returns:
            Risk tolerance score (0-1)
        """
        try:
            if not user_history:
                return 0.5

            # Calculate average bet size relative to user's typical bet
            bet_sizes = [getattr(bet, "amount", 0.0) for bet in user_history]
            avg_bet_size = np.mean(bet_sizes) if bet_sizes else 0.0

            # Calculate variance in bet sizes (higher variance = higher risk tolerance)
            bet_variance = np.var(bet_sizes) if len(bet_sizes) > 1 else 0.0

            # Normalize to 0-1 range
            size_factor = min(avg_bet_size / 100.0, 1.0)  # Normalize by $100
            variance_factor = min(bet_variance / 1000.0, 1.0)  # Normalize by $1000

            risk_tolerance = (size_factor + variance_factor) / 2

            return float(np.clip(risk_tolerance, 0, 1))

        except Exception as e:
            logger.error(f"Error calculating risk tolerance: {e}")
            return 0.5

    async def create_model(
        self, patterns: UserPatterns
    ) -> RandomForestClassifier | None:
        """Create a personalized model based on user patterns.

        Args:
            patterns: User behavior patterns

        Returns:
            Personalized model or None
        """
        try:
            # Create a Random Forest model with parameters based on user patterns
            n_estimators = max(50, int(100 * patterns.risk_tolerance))
            max_depth = max(5, int(10 * patterns.risk_tolerance))

            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )

            return model

        except Exception as e:
            logger.error(f"Error creating personalized model: {e}")
            return None

    async def train_model(
        self, model: RandomForestClassifier, user_history: list[Any]
    ) -> bool:
        """Train the personalized model.

        Args:
            model: Model to train
            user_history: Training data

        Returns:
            Success status
        """
        try:
            if not user_history:
                logger.warning("No training data available")
                return False

            # Prepare training data
            X, y = await self._prepare_training_data(user_history)

            if len(X) == 0 or len(y) == 0:
                logger.warning("No valid training data")
                return False

            # Train the model
            model.fit(X, y)

            logger.info(f"Personalized model trained with {len(X)} samples")
            return True

        except Exception as e:
            logger.error(f"Error training personalized model: {e}")
            return False

    async def _prepare_training_data(self, user_history: list[Any]) -> tuple:
        """Prepare training data from user history.

        Args:
            user_history: User's betting history

        Returns:
            Tuple of (features, labels)
        """
        try:
            features = []
            labels = []

            for bet in user_history:
                # Extract features (simplified)
                bet_features = [
                    getattr(bet, "amount", 0.0),
                    getattr(bet, "odds", 1.0),
                    getattr(bet, "confidence", 0.5),
                ]

                # Add time-based features
                timestamp = getattr(bet, "timestamp", datetime.now())
                bet_features.extend(
                    [
                        timestamp.hour / 24.0,  # Hour of day (normalized)
                        timestamp.weekday() / 7.0,  # Day of week (normalized)
                    ]
                )

                features.append(bet_features)
                labels.append(1 if getattr(bet, "outcome", False) else 0)

            return np.array(features), np.array(labels)

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([])
