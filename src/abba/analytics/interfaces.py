"""Protocol interfaces for analytics components."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class PredictionModel(Protocol):
    """Protocol for prediction models."""
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        ...
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on input data."""
        ...


@runtime_checkable
class ClassificationModel(Protocol):
    """Protocol for classification models."""
    
    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        ...


@runtime_checkable
class DataProcessor(Protocol):
    """Protocol for data processors."""
    
    async def process(self, data: dict) -> dict:
        """Process input data."""
        ...


@runtime_checkable
class AgentInterface(Protocol):
    """Protocol for agent implementations."""
    
    async def execute(self, task: dict) -> dict:
        """Execute a task and return results."""
        ...
    
    async def validate_input(self, data: dict) -> bool:
        """Validate input data."""
        ...
    
    async def handle_error(self, error: Exception) -> dict:
        """Handle errors gracefully."""
        ...


class DatabaseInterface(ABC):
    """Abstract interface for database operations."""
    
    @abstractmethod
    async def get_bets(self, **kwargs) -> list:
        """Get bets from database."""
        ...
    
    @abstractmethod
    async def save_bet(self, bet: dict) -> bool:
        """Save a bet to database."""
        ...
    
    @abstractmethod
    async def update_bet(self, bet_id: str, updates: dict) -> bool:
        """Update a bet in database."""
        ...


class ModelRegistry:
    """Registry for model types and their strategies."""
    
    def __init__(self):
        self._models: dict[str, type[PredictionModel]] = {}
    
    def register(self, name: str, model_class: type[PredictionModel]):
        """Register a model class."""
        self._models[name] = model_class
    
    def create(self, name: str, **kwargs) -> PredictionModel:
        """Create a model instance."""
        if name not in self._models:
            raise ValueError(f"Unknown model type: {name}")
        return self._models[name](**kwargs)
    
    def list_models(self) -> list[str]:
        """List available model types."""
        return list(self._models.keys())
    
    def is_registered(self, name: str) -> bool:
        """Check if a model type is registered."""
        return name in self._models


class Thresholds:
    """Configuration thresholds for system parameters."""
    
    def __init__(self):
        self.bias_threshold: float = 0.15
        self.risk_threshold: float = 0.05
        self.ethical_violation_threshold: float = 0.1
        self.max_bet_amount: float = 100.0
        self.min_ev_threshold: float = 0.05
        self.max_drawdown: float = 0.20


class SportConfig:
    """Configuration for sport-specific parameters."""
    
    def __init__(self, season_length: int, playoff_teams: int, data_sources: list[str]):
        self.season_length = season_length
        self.playoff_teams = playoff_teams
        self.data_sources = data_sources
    
    @classmethod
    def mlb_config(cls) -> 'SportConfig':
        """Get MLB configuration."""
        return cls(
            season_length=162,
            playoff_teams=12,
            data_sources=["pybaseball", "mlb_api"]
        )
    
    @classmethod
    def nhl_config(cls) -> 'SportConfig':
        """Get NHL configuration."""
        return cls(
            season_length=82,
            playoff_teams=16,
            data_sources=["nhl_api"]
        ) 