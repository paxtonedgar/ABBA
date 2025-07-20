"""Advanced Analytics Manager for ABBA."""

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from ..core.logging import get_logger
from .biometrics import BiometricsProcessor
from .ensemble import EnsembleManager
from .graph import GraphAnalyzer
from .models import BiometricData, Prediction
from .personalization import PersonalizationEngine

logger = get_logger(__name__)


class AdvancedAnalyticsManager:
    """Manages advanced analytics including biometrics and personalization."""

    def __init__(self, config: dict[str, Any], db_manager: Any):
        """Initialize the analytics manager.

        Args:
            config: Configuration dictionary
            db_manager: Database manager instance
        """
        self.config = config
        self.db_manager = db_manager
        self.biometrics_processor = BiometricsProcessor()
        self.personalization_engine = PersonalizationEngine()
        self.ensemble_manager = EnsembleManager()
        self.graph_analyzer = GraphAnalyzer()

        # Model storage
        self.models_dir = Path(config.get("model_cache_dir", "./models"))
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize models
        self.models: dict[str, Any] = {}
        self.scalers: dict[str, StandardScaler] = {}

        logger.info("Advanced Analytics Manager initialized")

    async def integrate_biometrics(self, player_data: dict[str, Any]) -> np.ndarray:
        """Integrate real-time biometric data for predictions.

        Args:
            player_data: Raw biometric data

        Returns:
            Feature matrix for predictions
        """
        try:
            logger.info("Integrating biometric data for predictions")

            # Process biometric data
            processed_data = await self.biometrics_processor.process(player_data)

            if not processed_data:
                logger.warning("No biometric data to process")
                return np.array([])

            # Convert to feature matrix
            feature_matrix = self._extract_biometric_features(processed_data)

            logger.info(
                f"Biometric integration completed. Feature matrix shape: {feature_matrix.shape}"
            )

            return feature_matrix

        except Exception as e:
            logger.error(f"Error integrating biometrics: {e}")
            return np.array([])

    def _extract_biometric_features(self, features: BiometricData) -> np.ndarray:
        """Extract features from biometric data.

        Args:
            features: Processed biometric data

        Returns:
            Feature matrix
        """
        try:
            feature_vector = []

            # Heart rate features
            hr_features = features.heart_rate
            feature_vector.extend(
                [
                    hr_features.get("mean_hr", 0.0),
                    hr_features.get("max_hr", 0.0),
                    hr_features.get("min_hr", 0.0),
                    hr_features.get("hr_variability", 0.0),
                    hr_features.get("fatigue_indicator", 0.0),
                ]
            )

            # Fatigue level
            feature_vector.append(features.fatigue_level)

            # Movement metrics
            movement = features.movement_metrics
            feature_vector.extend(
                [
                    movement.get("total_distance", 0.0),
                    movement.get("avg_speed", 0.0),
                    movement.get("max_speed", 0.0),
                    movement.get("acceleration_count", 0.0),
                ]
            )

            # Recovery status
            feature_vector.append(features.recovery_status)

            return np.array(feature_vector).reshape(1, -1)

        except Exception as e:
            logger.error(f"Error extracting biometric features: {e}")
            return np.array([])

    async def personalize_models(self, user_history: list[Any]) -> Any | None:
        """Create user-specific models based on historical data.

        Args:
            user_history: User's betting history

        Returns:
            Personalized model or None
        """
        try:
            logger.info(
                f"Personalizing models for user with {len(user_history)} historical bets"
            )

            if not user_history:
                logger.warning("No user history available for personalization")
                return None

            # Analyze user patterns
            user_patterns = await self.personalization_engine.analyze_patterns(
                user_history
            )

            # Create personalized model
            personalized_model = await self.personalization_engine.create_model(
                user_patterns
            )

            if personalized_model:
                # Train on user-specific data
                await self.personalization_engine.train_model(
                    personalized_model, user_history
                )

                logger.info("Personalized model created and trained successfully")
                return personalized_model
            else:
                logger.warning("Failed to create personalized model")
                return None

        except Exception as e:
            logger.error(f"Error personalizing models: {e}")
            return None

    async def ensemble_predictions(
        self, models: list[Any], data: np.ndarray
    ) -> Prediction | None:
        """Combine multiple models with error bars.

        Args:
            models: List of trained models
            data: Input data for prediction

        Returns:
            Ensemble prediction with confidence intervals
        """
        try:
            if not models or data.size == 0:
                logger.warning("No models or data available for ensemble prediction")
                return None

            logger.info(f"Running ensemble prediction with {len(models)} models")

            # Get predictions from all models
            predictions = []
            for i, model in enumerate(models):
                try:
                    pred = await self._get_model_prediction(model, data)
                    if pred is not None:
                        predictions.append(pred)
                        logger.debug(f"Model {i} prediction: {pred}")
                except Exception as e:
                    logger.warning(f"Model {i} failed to predict: {e}")

            if not predictions:
                logger.error("No valid predictions from any model")
                return None

            # Combine predictions using ensemble methods
            ensemble_prediction = await self.ensemble_manager.combine_predictions(
                predictions
            )

            # Calculate error bars
            error_bars = await self.ensemble_manager.calculate_error_bars(predictions)

            prediction = Prediction(
                value=ensemble_prediction,
                confidence=error_bars["confidence"],
                error_margin=error_bars["margin"],
                model_count=len(predictions),
            )

            logger.info(
                f"Ensemble prediction completed: {prediction.value:.4f} ± {prediction.error_margin:.4f}"
            )

            return prediction

        except Exception as e:
            logger.error(f"Error in ensemble predictions: {e}")
            return None

    async def _get_model_prediction(self, model: Any, data: np.ndarray) -> float | None:
        """Get prediction from a single model.

        Args:
            model: Trained model
            data: Input data

        Returns:
            Model prediction or None
        """
        try:
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(data)[0][1]  # Probability of positive class
            else:
                pred = model.predict(data)[0]

            return float(pred)

        except Exception as e:
            logger.error(f"Error getting model prediction: {e}")
            return None

    async def create_ensemble_model(
        self, model_types: list[str] | None = None
    ) -> dict[str, Any]:
        """Create an ensemble of models.

        Args:
            model_types: List of model types to include

        Returns:
            Ensemble configuration
        """
        if model_types is None:
            model_types = ["random_forest", "gradient_boosting", "logistic_regression"]

        ensemble = {}

        for model_type in model_types:
            model = await self._create_single_model(model_type)
            if model is not None:
                ensemble[model_type] = model

        return ensemble

    async def _create_single_model(self, model_type: str) -> Any | None:
        """Create a single model of specified type.

        Args:
            model_type: Type of model to create

        Returns:
            Model instance or None
        """
        try:
            if model_type == "random_forest":
                return RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type == "gradient_boosting":
                return GradientBoostingClassifier(n_estimators=100, random_state=42)
            else:
                logger.warning(f"Unknown model type: {model_type}")
                return None

        except Exception as e:
            logger.error(f"Error creating model {model_type}: {e}")
            return None
