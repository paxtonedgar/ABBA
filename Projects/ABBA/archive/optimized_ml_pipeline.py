"""
Optimized ML Pipeline for ABBA System
Advanced machine learning pipeline with incremental learning and model versioning.
"""

import json
import os
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import structlog
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

logger = structlog.get_logger()


class OptimizedMLPipeline:
    """Optimized ML pipeline with incremental learning and model versioning."""

    def __init__(self, config: dict, db_manager):
        self.config = config
        self.db_manager = db_manager
        self.model_registry = {}
        self.model_versions = {}
        self.feature_importance_cache = {}
        self.performance_history = {}

        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        # Initialize model types
        self.model_types = {
            'xgboost': xgb.XGBClassifier,
            'random_forest': RandomForestClassifier,
            'ensemble': self._create_ensemble_model
        }

        logger.info("OptimizedMLPipeline initialized")

    def _create_ensemble_model(self):
        """Create ensemble model combining multiple algorithms."""
        from sklearn.ensemble import VotingClassifier

        models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42))
        ]

        return VotingClassifier(estimators=models, voting='soft')

    async def train_models_incrementally(self, new_data: pd.DataFrame, sport: str) -> dict[str, Any]:
        """Train models incrementally with new data."""
        logger.info(f"Training models incrementally for {sport} with {len(new_data)} new records")

        results = {
            'models_updated': 0,
            'performance_improvements': {},
            'errors': []
        }

        try:
            # Prepare features and target
            features, target = self._prepare_training_data(new_data, sport)

            if len(features) < 50:  # Minimum data requirement
                logger.warning(f"Insufficient data for incremental training: {len(features)} records")
                return results

            # Load existing models
            existing_models = await self._load_existing_models(sport)

            # Update each model type
            for model_name, model_class in self.model_types.items():
                try:
                    model_key = f"{sport}_{model_name}"

                    if model_key in existing_models:
                        # Incremental update
                        updated_model = await self._update_model_incrementally(
                            existing_models[model_key], features, target, model_name
                        )
                    else:
                        # Train new model
                        updated_model = await self._train_new_model(
                            model_class, features, target, model_name
                        )

                    # Evaluate performance
                    performance = await self._evaluate_model(updated_model, features, target)

                    # Save updated model
                    await self._save_model(updated_model, model_key, performance, sport)

                    results['models_updated'] += 1
                    results['performance_improvements'][model_name] = performance

                    logger.info(f"Updated {model_name} model for {sport}")

                except Exception as e:
                    error_msg = f"Error updating {model_name} model: {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)

            return results

        except Exception as e:
            logger.error(f"Error in incremental training: {e}")
            results['errors'].append(str(e))
            return results

    async def _update_model_incrementally(self, existing_model, features: pd.DataFrame,
                                        target: pd.Series, model_name: str):
        """Update existing model with new data."""
        if model_name == 'xgboost':
            # XGBoost supports incremental learning
            existing_model.fit(features, target, xgb_model=existing_model)
            return existing_model
        elif model_name == 'random_forest':
            # Random Forest doesn't support incremental learning, so we retrain
            # In a real implementation, you might use online learning algorithms
            new_model = RandomForestClassifier(n_estimators=100, random_state=42)
            new_model.fit(features, target)
            return new_model
        else:
            # For ensemble models, retrain
            new_model = self._create_ensemble_model()
            new_model.fit(features, target)
            return new_model

    async def _train_new_model(self, model_class, features: pd.DataFrame,
                             target: pd.Series, model_name: str):
        """Train a new model from scratch."""
        if callable(model_class):
            model = model_class()
        else:
            model = model_class

        model.fit(features, target)
        return model

    async def _evaluate_model(self, model, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
        """Evaluate model performance."""
        try:
            # Cross-validation
            cv_scores = cross_val_score(model, features, target, cv=5, scoring='accuracy')

            # Predictions
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)[:, 1] if hasattr(model, 'predict_proba') else None

            # Calculate metrics
            performance = {
                'accuracy': accuracy_score(target, predictions),
                'precision': precision_score(target, predictions, average='weighted'),
                'recall': recall_score(target, predictions, average='weighted'),
                'f1_score': f1_score(target, predictions, average='weighted'),
                'cv_accuracy_mean': cv_scores.mean(),
                'cv_accuracy_std': cv_scores.std()
            }

            if probabilities is not None:
                performance['roc_auc'] = roc_auc_score(target, probabilities)

            return performance

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {'error': str(e)}

    async def _save_model(self, model, model_key: str, performance: dict, sport: str):
        """Save model with versioning."""
        try:
            # Generate version
            version = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

            # Save model file
            model_filename = f"models/{model_key}_{version}.joblib"
            joblib.dump(model, model_filename)

            # Save metadata
            metadata = {
                'model_key': model_key,
                'version': version,
                'sport': sport,
                'performance': performance,
                'created_at': datetime.utcnow().isoformat(),
                'feature_count': len(getattr(model, 'feature_importances_', []))
            }

            # Store in database
            await self._store_model_metadata(metadata)

            # Update registry
            self.model_registry[model_key] = {
                'model': model,
                'version': version,
                'performance': performance,
                'filename': model_filename
            }

            logger.info(f"Saved model {model_key} version {version}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")

    async def _store_model_metadata(self, metadata: dict):
        """Store model metadata in database."""
        try:
            # Store in database using raw SQL for now
            import sqlite3
            conn = sqlite3.connect('abmba.db')
            cursor = conn.cursor()

            # Create model_metadata table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    id VARCHAR(50) PRIMARY KEY,
                    model_key VARCHAR(100) NOT NULL,
                    version VARCHAR(50) NOT NULL,
                    sport VARCHAR(20) NOT NULL,
                    performance TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    feature_count INTEGER,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)

            # Insert metadata
            cursor.execute("""
                INSERT INTO model_metadata 
                (id, model_key, version, sport, performance, created_at, feature_count, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"{metadata['model_key']}_{metadata['version']}",
                metadata['model_key'],
                metadata['version'],
                metadata['sport'],
                json.dumps(metadata['performance']),
                metadata['created_at'],
                metadata['feature_count'],
                True
            ))

            conn.commit()
            conn.close()

            logger.debug(f"Stored model metadata for {metadata['model_key']}")

        except Exception as e:
            logger.error(f"Error storing model metadata: {e}")

    async def _load_existing_models(self, sport: str) -> dict[str, Any]:
        """Load existing models from storage."""
        try:
            models = {}

            # Load from model registry first
            for model_key in self.model_registry:
                if sport in model_key:
                    models[model_key] = self.model_registry[model_key]['model']

            # Load from disk if not in registry
            model_files = [f for f in os.listdir('models') if f.startswith(sport) and f.endswith('.joblib')]

            for model_file in model_files:
                model_key = model_file.replace('.joblib', '')
                if model_key not in models:
                    try:
                        model = joblib.load(f"models/{model_file}")
                        models[model_key] = model
                        logger.info(f"Loaded model from disk: {model_file}")
                    except Exception as e:
                        logger.error(f"Error loading model {model_file}: {e}")

            return models

        except Exception as e:
            logger.error(f"Error loading existing models: {e}")
            return {}

    def _prepare_training_data(self, data: pd.DataFrame, sport: str) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training."""
        # Select numeric features
        feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target and metadata columns
        exclude_columns = ['target', 'event_id', 'created_at', 'updated_at']
        feature_columns = [col for col in feature_columns if col not in exclude_columns]

        features = data[feature_columns].fillna(0)

        # Create target (simplified)
        if 'target' in data.columns:
            target = data['target']
        else:
            # Create mock target for demonstration
            target = pd.Series(np.random.randint(0, 2, size=len(data)))

        return features, target

    async def predict_with_ensemble(self, features: pd.DataFrame, sport: str) -> dict[str, Any]:
        """Make predictions using ensemble of models."""
        try:
            predictions = {}
            confidences = {}

            # Get available models for this sport
            available_models = [key for key in self.model_registry.keys() if sport in key]

            if not available_models:
                logger.warning(f"No models available for {sport}")
                return self._get_mock_prediction()

            # Get predictions from each model
            for model_key in available_models:
                model_info = self.model_registry[model_key]
                model = model_info['model']

                try:
                    pred = model.predict(features)[0]
                    prob = model.predict_proba(features)[0, 1] if hasattr(model, 'predict_proba') else 0.5

                    predictions[model_key] = pred
                    confidences[model_key] = prob

                except Exception as e:
                    logger.error(f"Error getting prediction from {model_key}: {e}")

            # Ensemble prediction
            if predictions:
                ensemble_pred = np.mean(list(predictions.values()))
                ensemble_conf = np.mean(list(confidences.values()))

                return {
                    'prediction': ensemble_pred,
                    'confidence': ensemble_conf,
                    'model_predictions': predictions,
                    'model_confidences': confidences,
                    'ensemble_method': 'mean'
                }
            else:
                return self._get_mock_prediction()

        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return self._get_mock_prediction()

    def _get_mock_prediction(self) -> dict[str, Any]:
        """Return mock prediction when models are unavailable."""
        return {
            'prediction': 0.5,
            'confidence': 0.6,
            'model_predictions': {'mock_model': 0.5},
            'model_confidences': {'mock_model': 0.6},
            'ensemble_method': 'mock'
        }

    async def get_model_performance_history(self, sport: str, model_key: str = None) -> dict[str, Any]:
        """Get performance history for models."""
        try:
            import sqlite3
            conn = sqlite3.connect('abmba.db')
            cursor = conn.cursor()

            # Create table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    id VARCHAR(50) PRIMARY KEY,
                    model_key VARCHAR(100) NOT NULL,
                    version VARCHAR(50) NOT NULL,
                    sport VARCHAR(20) NOT NULL,
                    performance TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    feature_count INTEGER,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)

            # Query performance history
            if model_key:
                cursor.execute("""
                    SELECT version, performance, created_at 
                    FROM model_metadata 
                    WHERE sport = ? AND model_key = ? 
                    ORDER BY created_at DESC
                """, (sport, model_key))
            else:
                cursor.execute("""
                    SELECT model_key, version, performance, created_at 
                    FROM model_metadata 
                    WHERE sport = ? 
                    ORDER BY created_at DESC
                """, (sport,))

            results = cursor.fetchall()
            conn.close()

            performance_history = []
            for result in results:
                if model_key:
                    version, performance_json, created_at = result
                    performance = json.loads(performance_json)
                    performance_history.append({
                        'version': version,
                        'performance': performance,
                        'created_at': created_at
                    })
                else:
                    model_key_result, version, performance_json, created_at = result
                    performance = json.loads(performance_json)
                    performance_history.append({
                        'model_key': model_key_result,
                        'version': version,
                        'performance': performance,
                        'created_at': created_at
                    })

            return {
                'sport': sport,
                'model_key': model_key,
                'performance_history': performance_history,
                'total_models': len(performance_history)
            }

        except Exception as e:
            logger.error(f"Error getting model performance history: {e}")
            return {'error': str(e)}

    async def compare_model_versions(self, sport: str, model_key: str) -> dict[str, Any]:
        """Compare different versions of a model."""
        try:
            history = await self.get_model_performance_history(sport, model_key)

            if 'error' in history:
                return history

            if len(history['performance_history']) < 2:
                return {'message': 'Not enough versions to compare'}

            # Compare latest two versions
            latest = history['performance_history'][0]
            previous = history['performance_history'][1]

            comparison = {
                'model_key': model_key,
                'sport': sport,
                'latest_version': latest['version'],
                'previous_version': previous['version'],
                'improvements': {}
            }

            # Calculate improvements
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                if metric in latest['performance'] and metric in previous['performance']:
                    improvement = latest['performance'][metric] - previous['performance'][metric]
                    comparison['improvements'][metric] = {
                        'change': improvement,
                        'percentage_change': (improvement / previous['performance'][metric]) * 100 if previous['performance'][metric] > 0 else 0
                    }

            return comparison

        except Exception as e:
            logger.error(f"Error comparing model versions: {e}")
            return {'error': str(e)}

    async def get_feature_importance(self, sport: str, model_key: str) -> dict[str, Any]:
        """Get feature importance for a specific model."""
        try:
            if model_key in self.model_registry:
                model = self.model_registry[model_key]['model']

                if hasattr(model, 'feature_importances_'):
                    # Get feature names (this would need to be stored with the model)
                    feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]

                    # Create feature importance dictionary
                    importance_dict = dict(zip(feature_names, model.feature_importances_, strict=False))
                    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

                    return {
                        'model_key': model_key,
                        'sport': sport,
                        'feature_importance': importance_dict,
                        'top_features': list(importance_dict.keys())[:10]
                    }
                else:
                    return {'message': 'Model does not support feature importance'}
            else:
                return {'error': 'Model not found in registry'}

        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {'error': str(e)}
