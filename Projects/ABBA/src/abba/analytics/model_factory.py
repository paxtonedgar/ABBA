"""Factory for creating prediction models."""

from typing import Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from .interfaces import PredictionModel, ModelRegistry


class SklearnModelAdapter:
    """Adapter for sklearn models to implement PredictionModel protocol."""
    
    def __init__(self, model: Any):
        self._model = model
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Make predictions using the underlying sklearn model."""
        return self._model.predict(data)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the underlying sklearn model."""
        self._model.fit(X, y)
    
    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """Predict class probabilities if supported."""
        if hasattr(self._model, 'predict_proba'):
            return self._model.predict_proba(data)
        raise NotImplementedError("Model does not support probability predictions")
    
    @property
    def model(self) -> Any:
        """Access the underlying sklearn model."""
        return self._model


class ModelFactory:
    """Factory for creating prediction models."""
    
    def __init__(self):
        self.registry = ModelRegistry()
        self._register_default_models()
    
    def _register_default_models(self):
        """Register default sklearn models."""
        self.registry.register("random_forest", 
            lambda **kwargs: SklearnModelAdapter(RandomForestClassifier(**kwargs)))
        self.registry.register("gradient_boosting",
            lambda **kwargs: SklearnModelAdapter(GradientBoostingClassifier(**kwargs)))
        self.registry.register("logistic_regression",
            lambda **kwargs: SklearnModelAdapter(LogisticRegression(**kwargs)))
        self.registry.register("neural_network",
            lambda **kwargs: SklearnModelAdapter(MLPClassifier(**kwargs)))
    
    def create_model(self, model_type: str, **kwargs) -> PredictionModel:
        """Create a model instance."""
        return self.registry.create(model_type, **kwargs)
    
    def list_available_models(self) -> list[str]:
        """List available model types."""
        return self.registry.list_models()
    
    def is_model_available(self, model_type: str) -> bool:
        """Check if a model type is available."""
        return self.registry.is_registered(model_type)
    
    def register_custom_model(self, name: str, model_class: type[PredictionModel]):
        """Register a custom model class."""
        self.registry.register(name, model_class)
    
    def get_model_defaults(self, model_type: str) -> dict[str, Any]:
        """Get default parameters for a model type."""
        defaults = {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": None,
                "random_state": 42
            },
            "gradient_boosting": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42
            },
            "logistic_regression": {
                "random_state": 42,
                "max_iter": 1000
            },
            "neural_network": {
                "hidden_layer_sizes": (100, 50),
                "max_iter": 1000,
                "random_state": 42
            }
        }
        return defaults.get(model_type, {})
    
    def create_model_with_defaults(self, model_type: str, **overrides) -> PredictionModel:
        """Create a model with default parameters and optional overrides."""
        defaults = self.get_model_defaults(model_type)
        defaults.update(overrides)
        return self.create_model(model_type, **defaults)


class EnsembleFactory:
    """Factory for creating ensemble models."""
    
    def __init__(self, model_factory: ModelFactory):
        self.model_factory = model_factory
    
    def create_ensemble(self, model_types: list[str], **kwargs) -> dict[str, PredictionModel]:
        """Create an ensemble of models."""
        ensemble = {}
        for model_type in model_types:
            try:
                ensemble[model_type] = self.model_factory.create_model_with_defaults(model_type)
            except ValueError as e:
                raise ValueError(f"Failed to create model {model_type}: {e}")
        return ensemble
    
    def create_weighted_ensemble(self, model_configs: list[dict]) -> dict[str, PredictionModel]:
        """Create an ensemble with custom configurations."""
        ensemble = {}
        for config in model_configs:
            model_type = config.get("type")
            model_params = config.get("params", {})
            model_name = config.get("name", model_type)
            
            if not model_type:
                raise ValueError("Model type is required in ensemble configuration")
            
            try:
                ensemble[model_name] = self.model_factory.create_model(model_type, **model_params)
            except ValueError as e:
                raise ValueError(f"Failed to create model {model_name}: {e}")
        
        return ensemble 