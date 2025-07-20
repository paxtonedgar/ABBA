"""Analytics data models."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class Prediction:
    """Container for ensemble prediction results."""

    value: float
    confidence: float
    error_margin: float
    model_count: int
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class BiometricData:
    """Biometric data structure."""

    heart_rate: dict[str, float]
    fatigue_level: float
    movement_metrics: dict[str, float]
    recovery_status: float
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class UserPatterns:
    """User betting patterns."""

    success_rate: float
    preferred_sports: list[str]
    bet_sizes: list[float]
    time_patterns: dict[str, Any]
    risk_tolerance: float


@dataclass
class ModelPerformance:
    """Model performance metrics."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float
    model_type: str
    training_date: datetime
