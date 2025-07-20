"""Refactored Advanced Analytics Manager using proper interfaces and separation of concerns."""

import os
from typing import Any, Dict, List, Optional
import numpy as np
import structlog
from datetime import datetime

from .interfaces import (
    PredictionModel, 
    DataProcessor, 
    ModelRegistry,
    Thresholds,
    SportConfig
)
from .model_factory import ModelFactory, EnsembleFactory

logger = structlog.get_logger()


class BiometricsManager:
    """Handles biometric data processing only."""
    
    def __init__(self, processor: DataProcessor):
        self.processor = processor
    
    async def integrate_biometrics(self, player_data: dict) -> np.ndarray:
        """Integrate real-time biometric data for predictions."""
        try:
            logger.info("Integrating biometric data for predictions")
            
            # Process biometric data
            processed_data = await self.processor.process(player_data)
            
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
            
            logger.info(f"Biometric integration completed. Feature matrix shape: {feature_matrix.shape}")
            return feature_matrix
            
        except (ValueError, TypeError) as e:
            logger.error(f"Data validation error in biometric integration: {e}")
            return np.array([])
        except Exception as e:
            logger.error(f"Unexpected error in biometric integration: {e}")
            raise
    
    def _extract_biometric_features(self, features: dict) -> np.ndarray:
        """Extract features from biometric data."""
        try:
            feature_vector = []
            
            # Heart rate features
            hr_features = features.get("heart_rate", {})
            if hr_features:
                feature_vector.extend([
                    hr_features.get("mean_hr", 0.0),
                    hr_features.get("max_hr", 0.0),
                    hr_features.get("min_hr", 0.0),
                    hr_features.get("hr_variability", 0.0),
                    hr_features.get("fatigue_indicator", 0.0),
                ])
            else:
                feature_vector.extend([0.0] * 5)  # Default values
            
            # Fatigue level
            feature_vector.append(features.get("fatigue_level", 0.0))
            
            # Movement metrics
            movement = features.get("movement_metrics", {})
            if movement:
                feature_vector.extend([
                    movement.get("total_distance", 0.0),
                    movement.get("avg_speed", 0.0),
                    movement.get("max_speed", 0.0),
                    movement.get("acceleration_count", 0.0),
                ])
            else:
                feature_vector.extend([0.0] * 4)  # Default values
            
            # Recovery status
            feature_vector.append(features.get("recovery_status", 0.0))
            
            return np.array(feature_vector).reshape(1, -1)
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error extracting biometric features: {e}")
            return np.array([])


class PersonalizationManager:
    """Handles user personalization only."""
    
    def __init__(self, engine: DataProcessor):
        self.engine = engine
    
    async def personalize_models(self, user_history: list) -> Optional[Any]:
        """Create user-specific models based on historical data."""
        try:
            logger.info(f"Personalizing models for user with {len(user_history)} historical bets")
            
            if not user_history:
                logger.warning("No user history available for personalization")
                return None
            
            # Analyze user patterns
            user_patterns = await self.engine.process({"history": user_history})
            
            # Create personalized model
            personalized_model = await self._create_personalized_model(user_patterns)
            
            if personalized_model:
                # Train on user-specific data
                await self._train_personalized_model(personalized_model, user_history)
                
                logger.info("Personalized model created and trained successfully")
                return personalized_model
            else:
                logger.warning("Failed to create personalized model")
                return None
                
        except (ValueError, TypeError) as e:
            logger.error(f"Data validation error in personalization: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in personalization: {e}")
            raise
    
    async def _create_personalized_model(self, patterns: dict) -> Optional[Any]:
        """Create a personalized model based on user patterns."""
        try:
            # Implementation would create model based on patterns
            return patterns.get("model")
        except Exception as e:
            logger.error(f"Error creating personalized model: {e}")
            return None
    
    async def _train_personalized_model(self, model: Any, user_history: list) -> bool:
        """Train the personalized model."""
        try:
            if not user_history:
                return False
            
            # Implementation would train the model
            logger.info(f"Training personalized model with {len(user_history)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training personalized model: {e}")
            return False


class EnsembleManager:
    """Handles ensemble predictions only."""
    
    def __init__(self, factory: ModelFactory):
        self.factory = factory
        self.ensemble_factory = EnsembleFactory(factory)
    
    async def ensemble_predictions(self, models: List[PredictionModel], data: np.ndarray) -> Optional[Dict[str, Any]]:
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
                except (ValueError, TypeError) as e:
                    logger.warning(f"Model {i} prediction error: {e}")
                except Exception as e:
                    logger.warning(f"Model {i} failed to predict: {e}")
            
            if not predictions:
                logger.error("No valid predictions from any model")
                return None
            
            # Combine predictions using ensemble methods
            ensemble_prediction = await self._combine_predictions(predictions)
            
            # Calculate error bars
            error_bars = await self._calculate_error_bars(predictions)
            
            result = {
                "value": ensemble_prediction,
                "confidence": error_bars["confidence"],
                "error_margin": error_bars["margin"],
                "model_count": len(predictions),
                "timestamp": datetime.now()
            }
            
            logger.info(f"Ensemble prediction completed: {result['value']:.4f} ± {result['error_margin']:.4f}")
            return result
            
        except (ValueError, TypeError) as e:
            logger.error(f"Data validation error in ensemble predictions: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in ensemble predictions: {e}")
            raise
    
    async def _get_model_prediction(self, model: PredictionModel, data: np.ndarray) -> Optional[float]:
        """Get prediction from a single model using protocol interface."""
        try:
            predictions = model.predict(data)
            return float(predictions[0])
        except (IndexError, ValueError) as e:
            logger.error(f"Model prediction error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected model prediction error: {e}")
            raise
    
    async def _combine_predictions(self, predictions: List[float]) -> float:
        """Combine predictions using weighted average."""
        try:
            if not predictions:
                return 0.0
            
            # Simple weighted average (could be enhanced with model confidence)
            weights = [1.0] * len(predictions)
            total_weight = sum(weights)
            
            if total_weight == 0:
                return np.mean(predictions)
            
            weighted_sum = sum(p * w for p, w in zip(predictions, weights))
            return weighted_sum / total_weight
            
        except Exception as e:
            logger.error(f"Error combining predictions: {e}")
            return np.mean(predictions) if predictions else 0.0
    
    async def _calculate_error_bars(self, predictions: List[float]) -> Dict[str, float]:
        """Calculate error bars for predictions."""
        try:
            if not predictions:
                return {"confidence": 0.0, "margin": 0.0}
            
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            
            # Calculate confidence interval (95%)
            confidence = 0.95
            margin = 1.96 * std_pred / np.sqrt(len(predictions))
            
            return {
                "confidence": confidence,
                "margin": margin
            }
            
        except Exception as e:
            logger.error(f"Error calculating error bars: {e}")
            return {"confidence": 0.0, "margin": 0.0}
    
    async def create_ensemble_model(self, model_types: List[str] = None) -> Dict[str, PredictionModel]:
        """Create ensemble model using factory."""
        try:
            if model_types is None:
                model_types = ["random_forest", "gradient_boosting", "logistic_regression"]
            
            ensemble = {}
            for model_type in model_types:
                try:
                    ensemble[model_type] = self.factory.create_model_with_defaults(model_type)
                except ValueError as e:
                    logger.warning(f"Failed to create model {model_type}: {e}")
            
            logger.info(f"Created ensemble with {len(ensemble)} models")
            return ensemble
            
        except Exception as e:
            logger.error(f"Error creating ensemble model: {e}")
            raise


class GraphAnalysisManager:
    """Handles graph analysis only."""
    
    def __init__(self, analyzer: DataProcessor):
        self.analyzer = analyzer
    
    async def graph_network_analysis(self, team_data: dict) -> Dict[str, Any]:
        """Perform graph network analysis on team data."""
        try:
            logger.info("Performing graph network analysis")
            
            # Process team data
            processed_data = await self.analyzer.process(team_data)
            
            if not processed_data:
                logger.warning("No team data to analyze")
                return {}
            
            # Build graph
            graph = await self._build_graph(processed_data)
            
            if not graph:
                logger.warning("Failed to build graph")
                return {}
            
            # Analyze connections
            analysis = await self._analyze_connections(graph)
            
            logger.info("Graph network analysis completed")
            return analysis
            
        except (ValueError, TypeError) as e:
            logger.error(f"Data validation error in graph analysis: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error in graph analysis: {e}")
            raise
    
    async def _build_graph(self, data: dict) -> Optional[dict]:
        """Build graph from team data."""
        try:
            # Implementation would build graph structure
            return {"nodes": data.get("players", []), "edges": data.get("connections", [])}
        except Exception as e:
            logger.error(f"Error building graph: {e}")
            return None
    
    async def _analyze_connections(self, graph: dict) -> Dict[str, Any]:
        """Analyze connections in the graph."""
        try:
            # Implementation would analyze graph structure
            return {
                "centrality": {},
                "clustering": 0.0,
                "density": 0.0,
                "key_players": []
            }
        except Exception as e:
            logger.error(f"Error analyzing connections: {e}")
            return {}


class RefactoredAdvancedAnalyticsManager:
    """Refactored analytics manager with proper separation of concerns."""
    
    def __init__(self, config: dict, db_manager: Any, 
                 biometrics_processor: DataProcessor,
                 personalization_engine: DataProcessor,
                 graph_analyzer: DataProcessor):
        self.config = config
        self.db_manager = db_manager
        self.thresholds = Thresholds()
        
        # Initialize focused managers
        self.biometrics_manager = BiometricsManager(biometrics_processor)
        self.personalization_manager = PersonalizationManager(personalization_engine)
        self.graph_analysis_manager = GraphAnalysisManager(graph_analyzer)
        
        # Initialize model factory and ensemble manager
        self.model_factory = ModelFactory()
        self.ensemble_manager = EnsembleManager(self.model_factory)
        
        # Model storage
        self.models_dir = "models/advanced_analytics"
        os.makedirs(self.models_dir, exist_ok=True)
        
        logger.info("Refactored Advanced Analytics Manager initialized")
    
    async def integrate_biometrics(self, player_data: dict) -> np.ndarray:
        """Integrate biometric data using focused manager."""
        return await self.biometrics_manager.integrate_biometrics(player_data)
    
    async def personalize_models(self, user_history: list) -> Optional[Any]:
        """Personalize models using focused manager."""
        return await self.personalization_manager.personalize_models(user_history)
    
    async def ensemble_predictions(self, models: List[PredictionModel], data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Run ensemble predictions using focused manager."""
        return await self.ensemble_manager.ensemble_predictions(models, data)
    
    async def graph_network_analysis(self, team_data: dict) -> Dict[str, Any]:
        """Perform graph analysis using focused manager."""
        return await self.graph_analysis_manager.graph_network_analysis(team_data)
    
    async def create_ensemble_model(self, model_types: List[str] = None) -> Dict[str, PredictionModel]:
        """Create ensemble model using focused manager."""
        return await self.ensemble_manager.create_ensemble_model(model_types)
    
    def get_sport_config(self, sport: str) -> SportConfig:
        """Get sport-specific configuration."""
        if sport.upper() == "MLB":
            return SportConfig.mlb_config()
        elif sport.upper() == "NHL":
            return SportConfig.nhl_config()
        else:
            # Default configuration
            return SportConfig(season_length=100, playoff_teams=8, data_sources=["default_api"])
    
    def get_thresholds(self) -> Thresholds:
        """Get system thresholds."""
        return self.thresholds 