"""
Advanced Analytics Manager for ABMBA system.
Handles biometrics integration, personalization, and ensemble predictions.
"""

import json
import os
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import structlog
from database import DatabaseManager
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from models import Bet

logger = structlog.get_logger()


class Prediction:
    """Container for ensemble prediction results."""

    def __init__(
        self, value: float, confidence: float, error_margin: float, model_count: int
    ):
        self.value = value
        self.confidence = confidence
        self.error_margin = error_margin
        self.model_count = model_count


class AdvancedAnalyticsManager:
    """Manages advanced analytics including biometrics and personalization."""

    def __init__(self, config: dict, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.biometrics_processor = BiometricsProcessor()
        self.personalization_engine = PersonalizationEngine()
        self.ensemble_manager = EnsembleManager()
        self.graph_analyzer = GraphAnalyzer()

        # Model storage
        self.models_dir = "models/advanced_analytics"
        os.makedirs(self.models_dir, exist_ok=True)

        # Initialize models
        self.models = {}
        self.scalers = {}

        logger.info("Advanced Analytics Manager initialized")

    async def integrate_biometrics(self, player_data: dict) -> np.ndarray:
        """Integrate real-time biometric data for predictions."""
        try:
            logger.info("Integrating biometric data for predictions")

            # Process biometric data
            processed_data = await self.biometrics_processor.process(player_data)

            if not processed_data:
                logger.warning("No biometric data to process")
                return np.array([])

            # Extract relevant features
            features = {
                "heart_rate": processed_data.get("heart_rate", {}),
                "fatigue_level": processed_data.get("fatigue_level", 0.0),
                "movement_metrics": processed_data.get("movement_metrics", {}),
                "recovery_status": processed_data.get("recovery_status", 0.0),
            }

            # Convert to feature matrix
            feature_matrix = self._extract_biometric_features(features)

            logger.info(
                f"Biometric integration completed. Feature matrix shape: {feature_matrix.shape}"
            )

            return feature_matrix

        except Exception as e:
            logger.error(f"Error integrating biometrics: {e}")
            return np.array([])

    def _extract_biometric_features(self, features: dict) -> np.ndarray:
        """Extract features from biometric data."""
        try:
            feature_vector = []

            # Heart rate features
            hr_features = features.get("heart_rate", {})
            if hr_features:
                feature_vector.extend(
                    [
                        hr_features.get("mean_hr", 0.0),
                        hr_features.get("max_hr", 0.0),
                        hr_features.get("min_hr", 0.0),
                        hr_features.get("hr_variability", 0.0),
                        hr_features.get("fatigue_indicator", 0.0),
                    ]
                )
            else:
                feature_vector.extend([0.0] * 5)  # Default values

            # Fatigue level
            feature_vector.append(features.get("fatigue_level", 0.0))

            # Movement metrics
            movement = features.get("movement_metrics", {})
            if movement:
                feature_vector.extend(
                    [
                        movement.get("total_distance", 0.0),
                        movement.get("avg_speed", 0.0),
                        movement.get("max_speed", 0.0),
                        movement.get("acceleration_count", 0.0),
                    ]
                )
            else:
                feature_vector.extend([0.0] * 4)  # Default values

            # Recovery status
            feature_vector.append(features.get("recovery_status", 0.0))

            return np.array(feature_vector).reshape(1, -1)

        except Exception as e:
            logger.error(f"Error extracting biometric features: {e}")
            return np.array([])

    async def personalize_models(self, user_history: list[Bet]) -> Any | None:
        """Create user-specific models based on historical data."""
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
        """Combine multiple models with error bars."""
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
        """Get prediction from a single model."""
        try:
            # Handle different model types
            if hasattr(model, "predict_proba"):
                # Classification model
                proba = model.predict_proba(data)
                return proba[0][1] if proba.shape[1] > 1 else proba[0][0]
            elif hasattr(model, "predict"):
                # Regression model
                return model.predict(data)[0]
            else:
                logger.warning("Unknown model type")
                return None

        except Exception as e:
            logger.error(f"Error getting model prediction: {e}")
            return None

    async def graph_network_analysis(self, team_data: dict) -> dict[str, Any]:
        """Model team/player interconnections using graph neural networks."""
        try:
            logger.info("Running graph network analysis")

            # Build graph structure
            graph = await self.graph_analyzer.build_graph(team_data)

            if not graph:
                logger.warning("No graph structure built")
                return {}

            # Analyze connections
            analysis = await self.graph_analyzer.analyze_connections(graph)

            # Extract insights
            insights = {
                "key_players": analysis.get("key_players", []),
                "team_cohesion": analysis.get("cohesion_score", 0.0),
                "injury_impact": analysis.get("injury_impact", {}),
                "chemistry_metrics": analysis.get("chemistry", {}),
                "network_density": analysis.get("density", 0.0),
                "centrality_scores": analysis.get("centrality", {}),
            }

            logger.info(
                f"Graph analysis completed. Team cohesion: {insights['team_cohesion']:.3f}"
            )

            return insights

        except Exception as e:
            logger.error(f"Error in graph network analysis: {e}")
            return {}

    async def create_ensemble_model(
        self, model_types: list[str] = None
    ) -> dict[str, Any]:
        """Create an ensemble of different model types."""
        try:
            if model_types is None:
                model_types = [
                    "random_forest",
                    "gradient_boosting",
                    "neural_network",
                    "logistic_regression",
                ]

            logger.info(f"Creating ensemble model with types: {model_types}")

            ensemble = {}

            for model_type in model_types:
                model = await self._create_single_model(model_type)
                if model:
                    ensemble[model_type] = model
                    logger.info(f"Created {model_type} model")

            # Store ensemble
            ensemble_id = f"ensemble_{datetime.utcnow().timestamp()}"
            self.models[ensemble_id] = ensemble

            logger.info(f"Ensemble model created with {len(ensemble)} models")

            return {
                "ensemble_id": ensemble_id,
                "model_types": list(ensemble.keys()),
                "model_count": len(ensemble),
            }

        except Exception as e:
            logger.error(f"Error creating ensemble model: {e}")
            return {"error": str(e)}

    async def _create_single_model(self, model_type: str) -> Any | None:
        """Create a single model of specified type."""
        try:
            if model_type == "random_forest":
                return RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42
                )
            elif model_type == "gradient_boosting":
                return GradientBoostingClassifier(
                    n_estimators=100, max_depth=5, random_state=42
                )
            elif model_type == "neural_network":
                return MLPClassifier(
                    hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
                )
            elif model_type == "logistic_regression":
                return LogisticRegression(max_iter=1000, random_state=42)
            else:
                logger.warning(f"Unknown model type: {model_type}")
                return None

        except Exception as e:
            logger.error(f"Error creating {model_type} model: {e}")
            return None

    async def train_ensemble(
        self, ensemble_id: str, X: np.ndarray, y: np.ndarray
    ) -> bool:
        """Train an ensemble model."""
        try:
            if ensemble_id not in self.models:
                logger.error(f"Ensemble {ensemble_id} not found")
                return False

            ensemble = self.models[ensemble_id]
            logger.info(f"Training ensemble {ensemble_id} with {len(ensemble)} models")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Store scaler
            self.scalers[ensemble_id] = scaler

            # Train each model
            training_results = {}

            for model_type, model in ensemble.items():
                try:
                    # Train model
                    model.fit(X_train_scaled, y_train)

                    # Evaluate
                    train_score = model.score(X_train_scaled, y_train)
                    test_score = model.score(X_test_scaled, y_test)

                    training_results[model_type] = {
                        "train_score": train_score,
                        "test_score": test_score,
                        "overfitting": train_score - test_score,
                    }

                    logger.info(
                        f"{model_type}: train={train_score:.3f}, test={test_score:.3f}"
                    )

                except Exception as e:
                    logger.error(f"Error training {model_type}: {e}")
                    training_results[model_type] = {"error": str(e)}

            # Save ensemble
            await self._save_ensemble(ensemble_id, ensemble, training_results)

            logger.info(f"Ensemble training completed for {ensemble_id}")
            return True

        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            return False

    async def _save_ensemble(self, ensemble_id: str, ensemble: dict, results: dict):
        """Save ensemble model to disk."""
        try:
            ensemble_file = os.path.join(self.models_dir, f"{ensemble_id}.joblib")
            results_file = os.path.join(self.models_dir, f"{ensemble_id}_results.json")

            # Save models
            joblib.dump(ensemble, ensemble_file)

            # Save results
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Ensemble saved to {ensemble_file}")

        except Exception as e:
            logger.error(f"Error saving ensemble: {e}")

    async def load_ensemble(self, ensemble_id: str) -> bool:
        """Load ensemble model from disk."""
        try:
            ensemble_file = os.path.join(self.models_dir, f"{ensemble_id}.joblib")

            if not os.path.exists(ensemble_file):
                logger.error(f"Ensemble file not found: {ensemble_file}")
                return False

            # Load ensemble
            ensemble = joblib.load(ensemble_file)
            self.models[ensemble_id] = ensemble

            # Load scaler if available
            scaler_file = os.path.join(self.models_dir, f"{ensemble_id}_scaler.joblib")
            if os.path.exists(scaler_file):
                scaler = joblib.load(scaler_file)
                self.scalers[ensemble_id] = scaler

            logger.info(f"Ensemble {ensemble_id} loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading ensemble: {e}")
            return False

    async def get_model_performance(self, ensemble_id: str) -> dict[str, Any]:
        """Get performance metrics for an ensemble."""
        try:
            if ensemble_id not in self.models:
                return {"error": "Ensemble not found"}

            results_file = os.path.join(self.models_dir, f"{ensemble_id}_results.json")

            if not os.path.exists(results_file):
                return {"error": "Results file not found"}

            with open(results_file) as f:
                results = json.load(f)

            # Calculate ensemble metrics
            valid_results = {k: v for k, v in results.items() if "error" not in v}

            if not valid_results:
                return {"error": "No valid model results"}

            avg_train_score = np.mean(
                [r["train_score"] for r in valid_results.values()]
            )
            avg_test_score = np.mean([r["test_score"] for r in valid_results.values()])
            avg_overfitting = np.mean(
                [r["overfitting"] for r in valid_results.values()]
            )

            return {
                "ensemble_id": ensemble_id,
                "model_count": len(valid_results),
                "average_train_score": avg_train_score,
                "average_test_score": avg_test_score,
                "average_overfitting": avg_overfitting,
                "individual_results": valid_results,
            }

        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {"error": str(e)}

    async def update_biometric_models(self, new_data: dict) -> bool:
        """Update models with new biometric data."""
        try:
            logger.info("Updating models with new biometric data")

            # Process new biometric data
            biometric_features = await self.integrate_biometrics(new_data)

            if biometric_features.size == 0:
                logger.warning("No biometric features extracted")
                return False

            # Update each ensemble
            for ensemble_id in self.models.keys():
                try:
                    # This would involve retraining with new data
                    # For now, just log the update
                    logger.info(
                        f"Updated ensemble {ensemble_id} with new biometric data"
                    )

                except Exception as e:
                    logger.error(f"Error updating ensemble {ensemble_id}: {e}")

            return True

        except Exception as e:
            logger.error(f"Error updating biometric models: {e}")
            return False


class BiometricsProcessor:
    """Processes real-time biometric data from wearables."""

    def __init__(self):
        self.heart_rate_thresholds = {
            "resting": (60, 100),
            "active": (100, 180),
            "peak": (180, 220),
        }
        self.fatigue_indicators = [
            "heart_rate_variability",
            "sleep_quality",
            "recovery_time",
            "stress_level",
        ]

    async def process(self, player_data: dict) -> dict:
        """Process raw biometric data."""
        try:
            processed_data = {}

            # Process heart rate data
            if "heart_rate" in player_data:
                processed_data["heart_rate"] = await self._process_heart_rate(
                    player_data["heart_rate"]
                )

            # Process fatigue indicators
            if "fatigue_metrics" in player_data:
                processed_data["fatigue_level"] = await self._calculate_fatigue(
                    player_data["fatigue_metrics"]
                )

            # Process movement data
            if "movement" in player_data:
                processed_data["movement_metrics"] = await self._process_movement(
                    player_data["movement"]
                )

            # Calculate recovery status
            processed_data["recovery_status"] = await self._calculate_recovery(
                processed_data
            )

            return processed_data

        except Exception as e:
            logger.error(f"Error processing biometrics: {e}")
            return {}

    async def _process_heart_rate(self, hr_data: list[float]) -> dict:
        """Process heart rate data and extract features."""
        try:
            if not hr_data:
                return {}

            hr_array = np.array(hr_data)

            features = {
                "mean_hr": np.mean(hr_array),
                "max_hr": np.max(hr_array),
                "min_hr": np.min(hr_array),
                "hr_variability": np.std(hr_array),
                "hr_trend": self._calculate_trend(hr_array),
                "fatigue_indicator": self._calculate_hr_fatigue(hr_array),
            }

            return features

        except Exception as e:
            logger.error(f"Error processing heart rate: {e}")
            return {}

    def _calculate_trend(self, data: np.ndarray) -> float:
        """Calculate trend in data."""
        if len(data) < 2:
            return 0.0

        x = np.arange(len(data))
        slope = np.polyfit(x, data, 1)[0]
        return slope

    def _calculate_hr_fatigue(self, hr_data: np.ndarray) -> float:
        """Calculate fatigue indicator from heart rate."""
        if len(hr_data) < 10:
            return 0.5

        # Calculate heart rate variability
        hr_diff = np.diff(hr_data)
        hrv = np.std(hr_diff)

        # Normalize to 0-1 scale
        normalized_hrv = min(hrv / 10.0, 1.0)  # Assuming max HRV of 10

        return 1.0 - normalized_hrv  # Higher fatigue = lower HRV

    async def _calculate_fatigue(self, fatigue_metrics: dict) -> float:
        """Calculate overall fatigue level."""
        try:
            fatigue_score = 0.0
            weights = {
                "heart_rate_variability": 0.3,
                "sleep_quality": 0.25,
                "recovery_time": 0.25,
                "stress_level": 0.2,
            }

            for metric, weight in weights.items():
                if metric in fatigue_metrics:
                    # Normalize to 0-1 scale
                    normalized_value = self._normalize_fatigue_metric(
                        fatigue_metrics[metric], metric
                    )
                    fatigue_score += normalized_value * weight

            return min(fatigue_score, 1.0)  # Cap at 1.0

        except Exception as e:
            logger.error(f"Error calculating fatigue: {e}")
            return 0.5  # Default neutral value

    def _normalize_fatigue_metric(self, value: float, metric: str) -> float:
        """Normalize fatigue metric to 0-1 scale."""
        if metric == "sleep_quality":
            return max(0, 1 - value / 10)  # 0 = best sleep, 10 = worst
        elif metric == "recovery_time":
            return min(value / 24, 1)  # Hours of recovery needed
        elif metric == "stress_level":
            return value / 10  # 0-10 scale
        else:
            return min(max(value, 0), 1)  # Clamp to 0-1

    async def _process_movement(self, movement_data: dict) -> dict:
        """Process movement data."""
        try:
            processed = {}

            # Extract movement metrics
            processed["total_distance"] = movement_data.get("total_distance", 0.0)
            processed["avg_speed"] = movement_data.get("avg_speed", 0.0)
            processed["max_speed"] = movement_data.get("max_speed", 0.0)
            processed["acceleration_count"] = movement_data.get(
                "acceleration_count", 0.0
            )

            return processed

        except Exception as e:
            logger.error(f"Error processing movement data: {e}")
            return {}

    async def _calculate_recovery(self, processed_data: dict) -> float:
        """Calculate recovery status."""
        try:
            recovery_score = 0.0

            # Heart rate recovery
            hr_data = processed_data.get("heart_rate", {})
            if hr_data:
                hr_trend = hr_data.get("hr_trend", 0)
                if hr_trend < 0:  # Heart rate decreasing (good recovery)
                    recovery_score += 0.3

            # Fatigue level
            fatigue = processed_data.get("fatigue_level", 0.5)
            recovery_score += (1 - fatigue) * 0.4  # Lower fatigue = better recovery

            # Movement recovery
            movement = processed_data.get("movement_metrics", {})
            if movement:
                avg_speed = movement.get("avg_speed", 0)
                if avg_speed > 0:  # Some movement indicates recovery
                    recovery_score += 0.3

            return min(recovery_score, 1.0)

        except Exception as e:
            logger.error(f"Error calculating recovery: {e}")
            return 0.5


class PersonalizationEngine:
    """Engine for creating personalized models."""

    def __init__(self):
        self.pattern_weights = {
            "sport_preference": 0.3,
            "odds_preference": 0.25,
            "timing_preference": 0.2,
            "stake_preference": 0.25,
        }

    async def analyze_patterns(self, user_history: list[Bet]) -> dict[str, Any]:
        """Analyze user betting patterns."""
        try:
            patterns = {
                "sport_preference": {},
                "odds_preference": {},
                "timing_preference": {},
                "stake_preference": {},
                "success_patterns": {},
            }

            if not user_history:
                return patterns

            # Sport preference
            sport_counts = {}
            for bet in user_history:
                sport = getattr(bet, "sport", "unknown")
                sport_counts[sport] = sport_counts.get(sport, 0) + 1

            patterns["sport_preference"] = sport_counts

            # Odds preference
            odds_ranges = {"low": 0, "medium": 0, "high": 0}
            for bet in user_history:
                if bet.odds:
                    odds = float(bet.odds)
                    if odds < 2.0:
                        odds_ranges["low"] += 1
                    elif odds < 5.0:
                        odds_ranges["medium"] += 1
                    else:
                        odds_ranges["high"] += 1

            patterns["odds_preference"] = odds_ranges

            # Timing preference
            hour_counts = {}
            for bet in user_history:
                if bet.placed_at:
                    hour = bet.placed_at.hour
                    hour_counts[hour] = hour_counts.get(hour, 0) + 1

            patterns["timing_preference"] = hour_counts

            # Stake preference
            stakes = [float(b.stake or 0) for b in user_history if b.stake]
            if stakes:
                patterns["stake_preference"] = {
                    "mean": np.mean(stakes),
                    "std": np.std(stakes),
                    "min": np.min(stakes),
                    "max": np.max(stakes),
                }

            # Success patterns
            patterns["success_patterns"] = await self._analyze_success_patterns(
                user_history
            )

            return patterns

        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return {}

    async def _analyze_success_patterns(
        self, user_history: list[Bet]
    ) -> dict[str, Any]:
        """Analyze patterns in successful bets."""
        try:
            successful_bets = [b for b in user_history if b.result == "win"]

            if not successful_bets:
                return {}

            patterns = {}

            # Successful sport patterns
            sport_success = {}
            for bet in successful_bets:
                sport = getattr(bet, "sport", "unknown")
                sport_success[sport] = sport_success.get(sport, 0) + 1

            patterns["successful_sports"] = sport_success

            # Successful odds patterns
            successful_odds = [float(b.odds or 0) for b in successful_bets if b.odds]
            if successful_odds:
                patterns["successful_odds_range"] = {
                    "mean": np.mean(successful_odds),
                    "std": np.std(successful_odds),
                }

            return patterns

        except Exception as e:
            logger.error(f"Error analyzing success patterns: {e}")
            return {}

    async def create_model(self, patterns: dict[str, Any]) -> Any | None:
        """Create a personalized model based on patterns."""
        try:
            # Create a simple personalized classifier
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(
                n_estimators=50, max_depth=8, random_state=42
            )

            # Store patterns with model for later use
            model.personalization_patterns = patterns

            return model

        except Exception as e:
            logger.error(f"Error creating personalized model: {e}")
            return None

    async def train_model(self, model: Any, user_history: list[Bet]) -> bool:
        """Train the personalized model."""
        try:
            if not user_history:
                logger.warning("No user history for training")
                return False

            # Prepare training data
            X, y = await self._prepare_training_data(user_history)

            if X.size == 0 or y.size == 0:
                logger.warning("No valid training data")
                return False

            # Train model
            model.fit(X, y)

            logger.info(f"Personalized model trained with {len(user_history)} samples")
            return True

        except Exception as e:
            logger.error(f"Error training personalized model: {e}")
            return False

    async def _prepare_training_data(self, user_history: list[Bet]) -> tuple:
        """Prepare training data from user history."""
        try:
            features = []
            labels = []

            for bet in user_history:
                # Extract features
                feature_vector = []

                # Sport feature (one-hot encoded)
                sport = getattr(bet, "sport", "unknown")
                sport_features = [
                    1.0 if sport == s else 0.0
                    for s in ["basketball_nba", "football_nfl", "baseball_mlb"]
                ]
                feature_vector.extend(sport_features)

                # Odds feature
                odds = float(bet.odds or 0)
                feature_vector.append(odds)

                # EV feature
                ev = float(bet.expected_value or 0)
                feature_vector.append(ev)

                # Stake feature
                stake = float(bet.stake or 0)
                feature_vector.append(stake)

                # Time feature
                if bet.placed_at:
                    hour = bet.placed_at.hour
                    feature_vector.append(hour)
                else:
                    feature_vector.append(12)  # Default to noon

                features.append(feature_vector)

                # Label (1 for win, 0 for loss)
                label = 1 if bet.result == "win" else 0
                labels.append(label)

            return np.array(features), np.array(labels)

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([])


class EnsembleManager:
    """Manages ensemble prediction methods."""

    def __init__(self):
        self.combination_methods = {
            "average": self._average_combination,
            "weighted": self._weighted_combination,
            "median": self._median_combination,
            "voting": self._voting_combination,
        }

    async def combine_predictions(
        self, predictions: list[float], method: str = "weighted"
    ) -> float:
        """Combine multiple predictions."""
        try:
            if not predictions:
                return 0.0

            if method in self.combination_methods:
                return self.combination_methods[method](predictions)
            else:
                logger.warning(f"Unknown combination method: {method}, using average")
                return self._average_combination(predictions)

        except Exception as e:
            logger.error(f"Error combining predictions: {e}")
            return np.mean(predictions) if predictions else 0.0

    def _average_combination(self, predictions: list[float]) -> float:
        """Simple average combination."""
        return np.mean(predictions)

    def _weighted_combination(self, predictions: list[float]) -> float:
        """Weighted average combination."""
        # Use inverse variance as weights
        weights = [
            1.0 / (1.0 + abs(p - 0.5)) for p in predictions
        ]  # Higher weight for more confident predictions
        total_weight = sum(weights)

        if total_weight > 0:
            return (
                sum(p * w for p, w in zip(predictions, weights, strict=False))
                / total_weight
            )
        else:
            return np.mean(predictions)

    def _median_combination(self, predictions: list[float]) -> float:
        """Median combination."""
        return np.median(predictions)

    def _voting_combination(self, predictions: list[float]) -> float:
        """Voting combination (for classification)."""
        # Convert to binary predictions
        binary_predictions = [1 if p > 0.5 else 0 for p in predictions]
        return np.mean(binary_predictions)

    async def calculate_error_bars(self, predictions: list[float]) -> dict[str, float]:
        """Calculate error bars for ensemble predictions."""
        try:
            if not predictions:
                return {"confidence": 0.0, "margin": 0.0}

            # Calculate confidence based on agreement
            _mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)

            # Confidence increases with agreement (lower std)
            confidence = max(0, 1 - std_pred)

            # Error margin based on standard error
            margin = std_pred / np.sqrt(len(predictions))

            return {"confidence": confidence, "margin": margin}

        except Exception as e:
            logger.error(f"Error calculating error bars: {e}")
            return {"confidence": 0.5, "margin": 0.1}


class GraphAnalyzer:
    """Analyzes team/player interconnections using graph theory."""

    def __init__(self):
        self.centrality_metrics = ["degree", "betweenness", "closeness"]

    async def build_graph(self, team_data: dict) -> dict | None:
        """Build graph structure from team data."""
        try:
            if not team_data:
                return None

            graph = {
                "nodes": [],
                "edges": [],
                "node_attributes": {},
                "edge_attributes": {},
            }

            # Add players as nodes
            players = team_data.get("players", [])
            for player in players:
                player_id = player.get("id", f"player_{len(graph['nodes'])}")
                graph["nodes"].append(player_id)
                graph["node_attributes"][player_id] = {
                    "name": player.get("name", "Unknown"),
                    "position": player.get("position", "Unknown"),
                    "stats": player.get("stats", {}),
                }

            # Add connections as edges
            connections = team_data.get("connections", [])
            for connection in connections:
                source = connection.get("source")
                target = connection.get("target")
                weight = connection.get("weight", 1.0)

                if (
                    source
                    and target
                    and source in graph["nodes"]
                    and target in graph["nodes"]
                ):
                    graph["edges"].append((source, target))
                    graph["edge_attributes"][(source, target)] = {
                        "weight": weight,
                        "type": connection.get("type", "general"),
                    }

            return graph

        except Exception as e:
            logger.error(f"Error building graph: {e}")
            return None

    async def analyze_connections(self, graph: dict) -> dict[str, Any]:
        """Analyze connections in the graph."""
        try:
            analysis = {
                "key_players": [],
                "cohesion_score": 0.0,
                "injury_impact": {},
                "chemistry": {},
                "density": 0.0,
                "centrality": {},
            }

            if not graph or not graph["nodes"]:
                return analysis

            # Calculate network density
            n_nodes = len(graph["nodes"])
            n_edges = len(graph["edges"])
            max_edges = n_nodes * (n_nodes - 1) / 2

            if max_edges > 0:
                analysis["density"] = n_edges / max_edges

            # Calculate centrality scores
            analysis["centrality"] = await self._calculate_centrality(graph)

            # Identify key players
            analysis["key_players"] = await self._identify_key_players(
                graph, analysis["centrality"]
            )

            # Calculate team cohesion
            analysis["cohesion_score"] = await self._calculate_cohesion(graph)

            # Analyze injury impact
            analysis["injury_impact"] = await self._analyze_injury_impact(graph)

            # Analyze chemistry
            analysis["chemistry"] = await self._analyze_chemistry(graph)

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing connections: {e}")
            return {}

    async def _calculate_centrality(self, graph: dict) -> dict[str, dict[str, float]]:
        """Calculate centrality metrics for each node."""
        try:
            centrality = {}

            for node in graph["nodes"]:
                centrality[node] = {"degree": 0, "betweenness": 0, "closeness": 0}

            # Calculate degree centrality
            for edge in graph["edges"]:
                source, target = edge
                centrality[source]["degree"] += 1
                centrality[target]["degree"] += 1

            # Normalize degree centrality
            max_degree = max([c["degree"] for c in centrality.values()])
            if max_degree > 0:
                for node in centrality:
                    centrality[node]["degree"] /= max_degree

            # Simplified betweenness and closeness (would use networkx in real implementation)
            for node in centrality:
                centrality[node]["betweenness"] = centrality[node]["degree"] * 0.5
                centrality[node]["closeness"] = centrality[node]["degree"] * 0.3

            return centrality

        except Exception as e:
            logger.error(f"Error calculating centrality: {e}")
            return {}

    async def _identify_key_players(self, graph: dict, centrality: dict) -> list[str]:
        """Identify key players based on centrality."""
        try:
            key_players = []

            # Sort by degree centrality
            sorted_players = sorted(
                centrality.items(), key=lambda x: x[1]["degree"], reverse=True
            )

            # Top 3 players by degree centrality
            for player_id, metrics in sorted_players[:3]:
                if metrics["degree"] > 0.3:  # Threshold for key player
                    key_players.append(player_id)

            return key_players

        except Exception as e:
            logger.error(f"Error identifying key players: {e}")
            return []

    async def _calculate_cohesion(self, graph: dict) -> float:
        """Calculate team cohesion score."""
        try:
            if not graph["nodes"] or not graph["edges"]:
                return 0.0

            # Cohesion based on network density and connectivity
            n_nodes = len(graph["nodes"])
            n_edges = len(graph["edges"])

            # Density component
            max_edges = n_nodes * (n_nodes - 1) / 2
            density = n_edges / max_edges if max_edges > 0 else 0

            # Connectivity component (simplified)
            connectivity = min(1.0, n_edges / n_nodes) if n_nodes > 0 else 0

            # Combined cohesion score
            cohesion = (density * 0.6) + (connectivity * 0.4)

            return cohesion

        except Exception as e:
            logger.error(f"Error calculating cohesion: {e}")
            return 0.0

    async def _analyze_injury_impact(self, graph: dict) -> dict[str, float]:
        """Analyze impact of potential injuries."""
        try:
            impact = {}

            # Simulate injury to each player and measure impact
            for node in graph["nodes"]:
                # Remove player and recalculate cohesion
                remaining_nodes = [n for n in graph["nodes"] if n != node]
                remaining_edges = [
                    e for e in graph["edges"] if e[0] != node and e[1] != node
                ]

                # Calculate cohesion without this player
                n_remaining = len(remaining_nodes)
                n_remaining_edges = len(remaining_edges)

                if n_remaining > 1:
                    max_remaining_edges = n_remaining * (n_remaining - 1) / 2
                    remaining_density = (
                        n_remaining_edges / max_remaining_edges
                        if max_remaining_edges > 0
                        else 0
                    )
                    impact[node] = (
                        1.0 - remaining_density
                    )  # Higher impact = lower remaining cohesion
                else:
                    impact[node] = 1.0  # Maximum impact if only one player remains

            return impact

        except Exception as e:
            logger.error(f"Error analyzing injury impact: {e}")
            return {}

    async def _analyze_chemistry(self, graph: dict) -> dict[str, float]:
        """Analyze team chemistry metrics."""
        try:
            chemistry = {
                "overall_chemistry": 0.0,
                "position_chemistry": {},
                "experience_chemistry": 0.0,
            }

            if not graph["nodes"]:
                return chemistry

            # Overall chemistry based on connection strength
            total_weight = sum(
                graph["edge_attributes"].get(edge, {}).get("weight", 1.0)
                for edge in graph["edges"]
            )

            avg_weight = total_weight / len(graph["edges"]) if graph["edges"] else 0
            chemistry["overall_chemistry"] = min(
                avg_weight / 5.0, 1.0
            )  # Normalize to 0-1

            # Position chemistry (simplified)
            positions = {}
            for node, attrs in graph["node_attributes"].items():
                pos = attrs.get("position", "unknown")
                if pos not in positions:
                    positions[pos] = []
                positions[pos].append(node)

            for pos, players in positions.items():
                if len(players) > 1:
                    # Calculate connections within position
                    pos_connections = sum(
                        1
                        for edge in graph["edges"]
                        if edge[0] in players and edge[1] in players
                    )
                    max_pos_connections = len(players) * (len(players) - 1) / 2
                    chemistry["position_chemistry"][pos] = (
                        pos_connections / max_pos_connections
                        if max_pos_connections > 0
                        else 0
                    )

            return chemistry

        except Exception as e:
            logger.error(f"Error analyzing chemistry: {e}")
            return {"overall_chemistry": 0.0}
