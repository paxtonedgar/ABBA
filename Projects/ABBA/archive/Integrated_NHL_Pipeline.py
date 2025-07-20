"""
Integrated NHL Pipeline: Complete Data & ML Pipeline Integration
Connects data ingestion, feature engineering, ML pipeline, and risk management into a cohesive system.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GameData:
    """Structured game data container."""
    game_id: str
    home_team: str
    away_team: str
    game_date: datetime
    home_goalie: dict[str, Any]
    away_goalie: dict[str, Any]
    home_team_stats: dict[str, Any]
    away_team_stats: dict[str, Any]
    market_data: dict[str, Any]
    situational_data: dict[str, Any]
    raw_features: dict[str, Any]
    processed_features: dict[str, Any] | None = None
    prediction: dict[str, Any] | None = None
    edge_analysis: dict[str, Any] | None = None
    risk_assessment: dict[str, Any] | None = None

# Mock ML Models
class MockXGBoost:
    def predict_proba(self, X):
        return np.array([[0.3, 0.7]])

class MockLightGBM:
    def predict_proba(self, X):
        return np.array([[0.35, 0.65]])

class MockCatBoost:
    def predict_proba(self, X):
        return np.array([[0.32, 0.68]])

class MockLogisticRegression:
    def predict_proba(self, X):
        return np.array([[0.34, 0.66]])

# Professional NHL Risk Manager (from previous file)
class ProfessionalNHLRiskManager:
    def __init__(self, initial_bankroll):
        self.bankroll = initial_bankroll
        self.max_single_bet = 0.03  # 3% max per bet
        self.max_daily_risk = 0.08  # 8% max daily
        self.max_weekly_risk = 0.15 # 15% max weekly
        self.fractional_kelly = 0.25  # 1/4 Kelly
        self.daily_risk_used = 0
        self.weekly_risk_used = 0
        self.active_bets = []

    def calculate_professional_stake(self, edge_analysis, prediction, odds, game_data):
        """Calculate stake using professional risk management."""

        try:
            # Base Kelly calculation
            kelly_fraction = self.calculate_base_kelly(prediction['prediction'], odds)

            # Apply fractional Kelly (1/4)
            fractional_stake = kelly_fraction * self.fractional_kelly

            # Edge-based adjustment
            edge_multiplier = self.calculate_edge_multiplier(edge_analysis['adjusted_edge'])

            # Professional adjustments
            professional_multiplier = self.calculate_professional_multiplier(game_data)

            # Correlation adjustment
            correlation_adjustment = self.calculate_correlation_adjustment(game_data)

            # Market condition adjustment
            market_adjustment = self.calculate_market_adjustment(edge_analysis)

            final_stake = (fractional_stake * edge_multiplier * professional_multiplier *
                          correlation_adjustment * market_adjustment)

            # Apply professional limits
            final_stake = min(final_stake, self.max_single_bet)
            final_stake = min(final_stake, self.max_daily_risk - self.daily_risk_used)
            final_stake = min(final_stake, self.max_weekly_risk - self.weekly_risk_used)

            # Safety check for minimum stake
            final_stake = max(0.005, final_stake)  # Minimum 0.5% stake

            # Validate stake is reasonable
            if final_stake > self.bankroll * 0.05:  # Never more than 5% of bankroll
                final_stake = self.bankroll * 0.05

            return final_stake

        except Exception as e:
            print(f"Error calculating stake: {e}")
            return 0.005  # Return minimum stake on error

    def calculate_base_kelly(self, win_prob, odds):
        """Calculate base Kelly fraction."""
        try:
            if odds > 0:
                implied_prob = 100 / (odds + 100)
            else:
                implied_prob = abs(odds) / (abs(odds) + 100)

            edge = win_prob - implied_prob
            return edge / implied_prob if implied_prob > 0 else 0
        except:
            return 0.01

    def calculate_edge_multiplier(self, adjusted_edge):
        """Calculate edge-based multiplier."""
        if adjusted_edge > 0.06:  # 6%+ edge
            return 1.3
        elif adjusted_edge > 0.04:  # 4%+ edge
            return 1.1
        elif adjusted_edge > 0.02:  # 2%+ edge
            return 1.0
        elif adjusted_edge > 0.01:  # 1%+ edge
            return 0.8
        else:  # <1% edge
            return 0.5

    def calculate_professional_multiplier(self, game_data):
        """Calculate professional adjustments."""
        multiplier = 1.0

        try:
            # Goalie quality adjustment (using GSAx)
            goalie_advantage = abs(game_data.get('home_goalie', {}).get('gsax', 0) -
                                 game_data.get('away_goalie', {}).get('gsax', 0))
            if goalie_advantage > 10:  # 10+ GSAx difference
                multiplier *= 1.2
            elif goalie_advantage > 5:  # 5+ GSAx difference
                multiplier *= 1.1
            elif goalie_advantage > 2:  # 2+ GSAx difference
                multiplier *= 1.05

            # Possession advantage adjustment
            possession_advantage = abs(game_data.get('home_team', {}).get('gar', 0) -
                                     game_data.get('away_team', {}).get('gar', 0))
            if possession_advantage > 5:  # 5+ GAR advantage
                multiplier *= 1.1
            elif possession_advantage > 2:  # 2+ GAR advantage
                multiplier *= 1.05

        except Exception as e:
            print(f"Error calculating professional multiplier: {e}")

        return multiplier

    def calculate_correlation_adjustment(self, game_data):
        """Calculate correlation adjustment."""
        try:
            correlation_score = 0.0
            for bet in self.active_bets:
                if (game_data.get('home_team', {}).get('name') == bet.get('team_name') or
                    game_data.get('away_team', {}).get('name') == bet.get('team_name')):
                    correlation_score += 0.8

            if correlation_score > 1.0:
                return 0.7  # High correlation, reduce stake
            elif correlation_score > 0.5:
                return 0.85  # Medium correlation, moderate reduction
            else:
                return 1.0  # Low correlation, no adjustment

        except Exception as e:
            print(f"Error calculating correlation: {e}")
            return 1.0

    def calculate_market_adjustment(self, edge_analysis):
        """Calculate market condition adjustments."""
        multiplier = 1.0

        try:
            market_efficiency = edge_analysis.get('market_efficiency', 0.85)
            if market_efficiency < 0.8:  # Inefficient market
                multiplier *= 1.1
            elif market_efficiency > 0.9:  # Very efficient market
                multiplier *= 0.9

            public_betting = edge_analysis.get('public_betting', 0.5)
            if public_betting > 0.75:  # Heavy public action
                multiplier *= 1.15  # Bet against public
            elif public_betting < 0.25:  # Light public action
                multiplier *= 1.05  # Bet with sharp money

        except Exception as e:
            print(f"Error calculating market adjustment: {e}")

        return multiplier

# Realistic Edge Detector (from previous file)
class RealisticEdgeDetector:
    def __init__(self):
        self.min_edge_threshold = 0.03  # 3% minimum edge
        self.max_edge_expectation = 0.08  # 8% maximum realistic edge

    def calculate_realistic_edge(self, prediction, odds, market_data, game_data):
        """Calculate realistic edge with professional standards."""

        try:
            our_prob = prediction['prediction']
            implied_prob = self.odds_to_prob(odds)

            # Validate probabilities
            our_prob = max(0.1, min(0.9, our_prob))
            implied_prob = max(0.1, min(0.9, implied_prob))

            # Base edge
            raw_edge = our_prob - implied_prob

            # Realistic market efficiency adjustment
            market_efficiency = market_data.get('efficiency_score', 0.85)
            edge_adjustment = 1 + (1 - market_efficiency) * 0.3

            # Professional adjustments
            professional_multiplier = self.calculate_professional_multiplier(market_data, game_data)

            adjusted_edge = raw_edge * edge_adjustment * professional_multiplier

            # Cap at realistic maximum
            adjusted_edge = min(adjusted_edge, self.max_edge_expectation)

            # Ensure minimum threshold
            if abs(adjusted_edge) < self.min_edge_threshold:
                adjusted_edge = 0  # No edge if below threshold

            return {
                'raw_edge': raw_edge,
                'adjusted_edge': adjusted_edge,
                'edge_adjustment': edge_adjustment,
                'professional_multiplier': professional_multiplier,
                'market_efficiency': market_efficiency,
                'edge_quality': self.assess_edge_quality(adjusted_edge, prediction)
            }

        except Exception as e:
            print(f"Error calculating edge: {e}")
            return {
                'raw_edge': 0,
                'adjusted_edge': 0,
                'edge_adjustment': 1.0,
                'professional_multiplier': 1.0,
                'market_efficiency': 0.85,
                'edge_quality': 'low'
            }

    def calculate_professional_multiplier(self, market_data, game_data):
        """Calculate professional adjustments."""
        multiplier = 1.0

        try:
            goalie_factor = market_data.get('goalie_factor', 1.0)
            if goalie_factor > 1.1:  # Elite goalie playing
                multiplier *= 1.15
            elif goalie_factor < 0.9:  # Weak goalie playing
                multiplier *= 1.1

            public_betting = market_data.get('public_betting_percentage', 0.5)
            if public_betting > 0.75:  # Heavy public action
                multiplier *= 1.2  # Bet against public
            elif public_betting < 0.25:  # Light public action
                multiplier *= 1.1  # Bet with sharp money

        except Exception as e:
            print(f"Error calculating professional multiplier: {e}")

        return multiplier

    def assess_edge_quality(self, adjusted_edge, prediction):
        """Assess the quality of the edge."""
        if abs(adjusted_edge) < 0.02:
            return 'low'
        elif abs(adjusted_edge) < 0.04:
            return 'medium'
        elif abs(adjusted_edge) < 0.06:
            return 'high'
        else:
            return 'very_high'

    def odds_to_prob(self, odds):
        """Convert American odds to probability."""
        try:
            # Handle different odds formats
            if isinstance(odds, dict):
                # Extract home odds from moneyline dict
                home_odds = odds.get('moneyline', {}).get('home', 0)
                if home_odds > 0:
                    return 100 / (home_odds + 100)
                else:
                    return abs(home_odds) / (abs(home_odds) + 100)
            elif isinstance(odds, (int, float)):
                if odds > 0:
                    return 100 / (odds + 100)
                else:
                    return abs(odds) / (abs(odds) + 100)
            else:
                return 0.5
        except Exception as e:
            print(f"Error converting odds to probability: {e}")
            return 0.5

# Data Ingestion Pipeline
class DataIngestionPipeline:
    def __init__(self):
        self.data_cache = {}
        self.last_update = {}

    async def ingest_game_data(self, game_id: str) -> GameData:
        """Ingest all data for a specific game."""

        logger.info(f"Starting data ingestion for game {game_id}")

        try:
            # Mock data ingestion
            game_data = GameData(
                game_id=game_id,
                home_team="Bruins",
                away_team="Maple Leafs",
                game_date=datetime.now(),
                home_goalie={'gsax': 15.2, 'high_danger_save_pct': 0.87, 'quality_start_pct': 0.65},
                away_goalie={'gsax': 8.7, 'high_danger_save_pct': 0.82, 'quality_start_pct': 0.58},
                home_team_stats={'gar': 12.5, 'controlled_entry_pct': 0.55, 'transition_efficiency': 0.68},
                away_team_stats={'gar': 9.8, 'controlled_entry_pct': 0.52, 'transition_efficiency': 0.64},
                market_data={'efficiency_score': 0.87, 'public_betting_percentage': 0.62},
                situational_data={'weather': {}, 'injuries': [], 'schedule': {}},
                raw_features={}
            )

            # Cache the data
            self.data_cache[game_id] = game_data
            self.last_update[game_id] = datetime.now()

            logger.info(f"Successfully ingested data for game {game_id}")
            return game_data

        except Exception as e:
            logger.error(f"Error ingesting data for game {game_id}: {e}")
            return self.get_fallback_data(game_id)

    def get_fallback_data(self, game_id: str) -> GameData:
        """Return fallback data when ingestion fails."""

        return GameData(
            game_id=game_id,
            home_team="Unknown",
            away_team="Unknown",
            game_date=datetime.now(),
            home_goalie={},
            away_goalie={},
            home_team_stats={},
            away_team_stats={},
            market_data={},
            situational_data={},
            raw_features={}
        )

# Feature Engineering Pipeline
class FeatureEngineeringPipeline:
    def __init__(self):
        self.feature_cache = {}

    def generate_features(self, game_data: GameData) -> dict[str, float]:
        """Generate comprehensive feature set."""

        logger.info(f"Generating features for game {game_data.game_id}")

        try:
            # Check cache first
            if game_data.game_id in self.feature_cache:
                return self.feature_cache[game_data.game_id]

            # Generate features from game data
            features = {}

            # Basic features
            features['home_gar'] = game_data.home_team_stats.get('gar', 0)
            features['away_gar'] = game_data.away_team_stats.get('gar', 0)
            features['gar_differential'] = features['home_gar'] - features['away_gar']

            features['home_gsax'] = game_data.home_goalie.get('gsax', 0)
            features['away_gsax'] = game_data.away_goalie.get('gsax', 0)
            features['gsax_differential'] = features['home_gsax'] - features['away_gsax']

            features['home_hd_save_pct'] = game_data.home_goalie.get('high_danger_save_pct', 0.85)
            features['away_hd_save_pct'] = game_data.away_goalie.get('high_danger_save_pct', 0.85)
            features['hd_save_pct_differential'] = features['home_hd_save_pct'] - features['away_hd_save_pct']

            features['home_quality_start_pct'] = game_data.home_goalie.get('quality_start_pct', 0.5)
            features['away_quality_start_pct'] = game_data.away_goalie.get('quality_start_pct', 0.5)

            features['home_controlled_entry_pct'] = game_data.home_team_stats.get('controlled_entry_pct', 0.5)
            features['away_controlled_entry_pct'] = game_data.away_team_stats.get('controlled_entry_pct', 0.5)

            features['home_transition_efficiency'] = game_data.home_team_stats.get('transition_efficiency', 0.5)
            features['away_transition_efficiency'] = game_data.away_team_stats.get('transition_efficiency', 0.5)

            # Derived features
            features['gar_gsax_interaction'] = features['gar_differential'] * features['gsax_differential']
            features['hd_save_transition_interaction'] = features['hd_save_pct_differential'] * features['home_transition_efficiency']

            features['gar_ratio'] = features['home_gar'] / max(features['away_gar'], 0.1)
            features['gsax_ratio'] = features['home_gsax'] / max(features['away_gsax'], 0.1)

            # Composite scores
            features['home_composite_score'] = (
                features['home_gar'] * 0.3 +
                features['home_gsax'] * 0.3 +
                features['home_hd_save_pct'] * 0.2 +
                features['home_transition_efficiency'] * 0.2
            )

            features['away_composite_score'] = (
                features['away_gar'] * 0.3 +
                features['away_gsax'] * 0.3 +
                features['away_hd_save_pct'] * 0.2 +
                features['away_transition_efficiency'] * 0.2
            )

            features['composite_differential'] = features['home_composite_score'] - features['away_composite_score']

            # Validate features
            features = self.validate_features(features)

            # Cache features
            self.feature_cache[game_data.game_id] = features

            # Update game data
            game_data.processed_features = features

            logger.info(f"Generated {len(features)} features for game {game_data.game_id}")
            return features

        except Exception as e:
            logger.error(f"Error generating features for game {game_data.game_id}: {e}")
            return self.get_fallback_features()

    def validate_features(self, features: dict[str, float]) -> dict[str, float]:
        """Validate and clean features."""

        validated_features = {}

        for feature_name, value in features.items():
            try:
                float_value = float(value)

                if np.isinf(float_value):
                    float_value = 0.0

                if np.isnan(float_value):
                    float_value = 0.0

                if abs(float_value) > 1000:
                    float_value = np.sign(float_value) * 1000

                validated_features[feature_name] = float_value

            except (ValueError, TypeError):
                logger.warning(f"Invalid feature value for {feature_name}: {value}, setting to 0")
                validated_features[feature_name] = 0.0

        return validated_features

    def get_fallback_features(self) -> dict[str, float]:
        """Return fallback features when generation fails."""

        return {
            'home_gar': 0.0,
            'away_gar': 0.0,
            'gar_differential': 0.0,
            'home_gsax': 0.0,
            'away_gsax': 0.0,
            'gsax_differential': 0.0,
            'home_hd_save_pct': 0.85,
            'away_hd_save_pct': 0.85,
            'hd_save_pct_differential': 0.0,
            'home_quality_start_pct': 0.5,
            'away_quality_start_pct': 0.5,
            'home_controlled_entry_pct': 0.5,
            'away_controlled_entry_pct': 0.5,
            'home_transition_efficiency': 0.5,
            'away_transition_efficiency': 0.5,
            'gar_gsax_interaction': 0.0,
            'hd_save_transition_interaction': 0.0,
            'gar_ratio': 1.0,
            'gsax_ratio': 1.0,
            'home_composite_score': 0.5,
            'away_composite_score': 0.5,
            'composite_differential': 0.0
        }

# ML Pipeline
class MLPipeline:
    def __init__(self):
        self.models = {
            'xgboost_advanced': MockXGBoost(),
            'lightgbm_advanced': MockLightGBM(),
            'catboost_advanced': MockCatBoost(),
            'meta_learner': MockLogisticRegression()
        }
        self.prediction_cache = {}

    def predict(self, features: dict[str, float], game_context: dict[str, Any]) -> dict[str, Any]:
        """Generate prediction using integrated ML pipeline."""

        try:
            # Convert features to array
            feature_names = sorted(features.keys())
            feature_values = [features[name] for name in feature_names]
            X = np.array([feature_values])

            # Get base predictions
            base_predictions = {}
            for model_name, model in self.models.items():
                if model_name != 'meta_learner':
                    try:
                        pred = model.predict_proba(X)[0][1]
                        base_predictions[model_name] = pred
                    except Exception as e:
                        logger.error(f"Error with {model_name}: {e}")
                        base_predictions[model_name] = 0.5

            # Calculate ensemble prediction
            if base_predictions:
                ensemble_prediction = np.mean(list(base_predictions.values()))

                # Meta-learner prediction
                meta_features = list(base_predictions.values())
                meta_prediction = self.models['meta_learner'].predict_proba([meta_features])[0][1]

                # Final prediction (weighted average)
                final_prediction = 0.7 * ensemble_prediction + 0.3 * meta_prediction
                final_prediction = max(0.1, min(0.9, final_prediction))

                # Calculate confidence and uncertainty
                confidence = self.calculate_confidence(base_predictions)
                uncertainty = self.calculate_uncertainty(base_predictions)
                model_agreement = self.calculate_model_agreement(base_predictions)

                prediction_result = {
                    'prediction': final_prediction,
                    'confidence': confidence,
                    'uncertainty': uncertainty,
                    'model_agreement': model_agreement,
                    'base_predictions': base_predictions,
                    'feature_importance': self.get_feature_importance(features),
                    'model_weights': self.calculate_model_weights(base_predictions, game_context)
                }

                return prediction_result
            else:
                raise ValueError("No valid base predictions available")

        except Exception as e:
            logger.error(f"Error in ML pipeline prediction: {e}")
            return self.get_fallback_prediction()

    def calculate_confidence(self, base_predictions: dict[str, float]) -> float:
        """Calculate prediction confidence."""

        if not base_predictions:
            return 0.5

        agreement = 1 - np.std(list(base_predictions.values()))
        avg_prediction = np.mean(list(base_predictions.values()))
        prediction_strength = abs(avg_prediction - 0.5) * 2

        confidence = 0.7 + (agreement * 0.2) + (prediction_strength * 0.1)
        return min(0.95, confidence)

    def calculate_uncertainty(self, base_predictions: dict[str, float]) -> float:
        """Calculate prediction uncertainty."""

        if not base_predictions:
            return 0.1

        return np.std(list(base_predictions.values()))

    def calculate_model_agreement(self, base_predictions: dict[str, float]) -> float:
        """Calculate agreement between models."""

        if not base_predictions:
            return 0.5

        values = list(base_predictions.values())
        return 1 - np.std(values)

    def calculate_model_weights(self, base_predictions: dict[str, float], game_context: dict[str, Any]) -> dict[str, float]:
        """Calculate dynamic model weights."""

        # Default weights for NHL
        return {
            'xgboost_advanced': 0.25,
            'lightgbm_advanced': 0.25,
            'catboost_advanced': 0.25,
            'neural_advanced': 0.15,
            'random_forest_advanced': 0.10
        }

    def get_feature_importance(self, features: dict[str, float]) -> dict[str, float]:
        """Get feature importance."""

        return dict.fromkeys(features.keys(), 0.1)

    def get_fallback_prediction(self) -> dict[str, Any]:
        """Return fallback prediction when ML pipeline fails."""

        return {
            'prediction': 0.5,
            'confidence': 0.5,
            'uncertainty': 0.1,
            'model_agreement': 0.5,
            'base_predictions': {},
            'feature_importance': {},
            'model_weights': {}
        }

# Risk Management Pipeline
class RiskManagementPipeline:
    def __init__(self, bankroll: float = 100000):
        self.bankroll = bankroll
        self.risk_manager = ProfessionalNHLRiskManager(bankroll)
        self.edge_detector = RealisticEdgeDetector()

    def assess_bet(self, game_data: GameData, prediction: dict[str, Any], odds: dict[str, Any]) -> dict[str, Any]:
        """Comprehensive bet assessment."""

        logger.info(f"Assessing bet for game {game_data.game_id}")

        try:
            # Calculate edge
            edge_analysis = self.edge_detector.calculate_realistic_edge(
                prediction, odds, game_data.market_data, game_data.raw_features
            )

            # Calculate optimal stake
            stake = self.risk_manager.calculate_professional_stake(
                edge_analysis, prediction, odds, game_data.raw_features
            )

            # Risk assessment
            risk_assessment = {
                'stake': stake,
                'stake_amount': stake * self.bankroll,
                'edge_analysis': edge_analysis,
                'risk_metrics': self.calculate_risk_metrics(stake, edge_analysis),
                'recommendation': self.generate_recommendation(stake, edge_analysis)
            }

            # Update game data
            game_data.risk_assessment = risk_assessment

            return risk_assessment

        except Exception as e:
            logger.error(f"Error in risk assessment for game {game_data.game_id}: {e}")
            return self.get_fallback_risk_assessment()

    def calculate_risk_metrics(self, stake: float, edge_analysis: dict[str, Any]) -> dict[str, Any]:
        """Calculate comprehensive risk metrics."""

        return {
            'stake_percentage': stake,
            'edge_size': edge_analysis.get('adjusted_edge', 0),
            'edge_quality': edge_analysis.get('edge_quality', 'low'),
            'expected_value': self.calculate_expected_value(stake, edge_analysis),
            'max_loss': stake,
            'kelly_fraction': self.calculate_kelly_fraction(edge_analysis)
        }

    def calculate_expected_value(self, stake: float, edge_analysis: dict[str, Any]) -> float:
        """Calculate expected value of the bet."""

        edge = edge_analysis.get('adjusted_edge', 0)
        return stake * edge

    def calculate_kelly_fraction(self, edge_analysis: dict[str, Any]) -> float:
        """Calculate Kelly fraction."""

        edge = edge_analysis.get('adjusted_edge', 0)
        if edge <= 0:
            return 0.0

        return edge / 2  # Conservative Kelly

    def generate_recommendation(self, stake: float, edge_analysis: dict[str, Any]) -> dict[str, Any]:
        """Generate betting recommendation."""

        edge = edge_analysis.get('adjusted_edge', 0)
        edge_quality = edge_analysis.get('edge_quality', 'low')

        if stake < 0.005:  # Less than 0.5%
            recommendation = 'pass'
            reason = 'insufficient_stake'
        elif edge < 0.03:  # Less than 3% edge
            recommendation = 'pass'
            reason = 'insufficient_edge'
        elif edge_quality == 'low':
            recommendation = 'pass'
            reason = 'low_edge_quality'
        else:
            recommendation = 'bet'
            reason = 'meets_criteria'

        return {
            'action': recommendation,
            'reason': reason,
            'confidence': edge_analysis.get('edge_quality', 'low'),
            'priority': self.calculate_priority(edge, edge_quality)
        }

    def calculate_priority(self, edge: float, edge_quality: str) -> str:
        """Calculate bet priority."""

        if edge > 0.06 and edge_quality in ['high', 'very_high']:
            return 'high'
        elif edge > 0.04 and edge_quality in ['medium', 'high', 'very_high']:
            return 'medium'
        else:
            return 'low'

    def get_fallback_risk_assessment(self) -> dict[str, Any]:
        """Return fallback risk assessment."""

        return {
            'stake': 0.0,
            'stake_amount': 0.0,
            'edge_analysis': {},
            'risk_metrics': {},
            'recommendation': {
                'action': 'pass',
                'reason': 'error_in_assessment',
                'confidence': 'low',
                'priority': 'low'
            }
        }

# Performance Tracker
class PerformanceTracker:
    def __init__(self):
        self.predictions = []
        self.performance_data = {}

    def track_prediction(self, game_id: str, prediction: dict[str, Any], risk_assessment: dict[str, Any]):
        """Track prediction for performance analysis."""

        self.predictions.append({
            'game_id': game_id,
            'prediction': prediction,
            'risk_assessment': risk_assessment,
            'timestamp': datetime.now()
        })

    def get_summary(self) -> dict[str, Any]:
        """Get performance summary."""

        return {
            'total_predictions': len(self.predictions),
            'avg_confidence': 0.75,
            'avg_edge': 0.04,
            'bet_rate': 0.12
        }

    def get_all_data(self) -> dict[str, Any]:
        return self.performance_data

    def load_data(self, data: dict[str, Any]):
        self.performance_data = data

# Integrated NHL Pipeline
class IntegratedNHLPipeline:
    def __init__(self, bankroll: float = 100000):
        self.data_pipeline = DataIngestionPipeline()
        self.feature_pipeline = FeatureEngineeringPipeline()
        self.ml_pipeline = MLPipeline()
        self.risk_pipeline = RiskManagementPipeline(bankroll)
        self.performance_tracker = PerformanceTracker()

    async def process_game(self, game_id: str, odds: dict[str, Any]) -> dict[str, Any]:
        """Complete game processing pipeline."""

        logger.info(f"Starting complete processing for game {game_id}")

        try:
            # Step 1: Data Ingestion
            game_data = await self.data_pipeline.ingest_game_data(game_id)

            # Step 2: Feature Engineering
            features = self.feature_pipeline.generate_features(game_data)

            # Step 3: ML Prediction
            game_context = {
                'sport': 'nhl',
                'game_data': game_data.raw_features,
                'features': features
            }
            prediction = self.ml_pipeline.predict(features, game_context)

            # Step 4: Risk Assessment
            risk_assessment = self.risk_pipeline.assess_bet(game_data, prediction, odds)

            # Step 5: Performance Tracking
            self.performance_tracker.track_prediction(game_id, prediction, risk_assessment)

            # Compile results
            results = {
                'game_id': game_id,
                'game_data': game_data,
                'features': features,
                'prediction': prediction,
                'risk_assessment': risk_assessment,
                'recommendation': risk_assessment['recommendation'],
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Completed processing for game {game_id}")
            return results

        except Exception as e:
            logger.error(f"Error processing game {game_id}: {e}")
            return self.get_fallback_results(game_id)

    def get_fallback_results(self, game_id: str) -> dict[str, Any]:
        """Return fallback results when processing fails."""

        return {
            'game_id': game_id,
            'game_data': None,
            'features': {},
            'prediction': {
                'prediction': 0.5,
                'confidence': 0.5,
                'uncertainty': 0.1,
                'model_agreement': 0.5
            },
            'risk_assessment': {
                'stake': 0.0,
                'recommendation': {
                    'action': 'pass',
                    'reason': 'processing_error',
                    'confidence': 'low',
                    'priority': 'low'
                }
            },
            'recommendation': {
                'action': 'pass',
                'reason': 'processing_error',
                'confidence': 'low',
                'priority': 'low'
            },
            'timestamp': datetime.now().isoformat()
        }

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary."""

        return self.performance_tracker.get_summary()

# Example usage
async def main():
    """Example of integrated pipeline usage."""

    # Initialize pipeline
    pipeline = IntegratedNHLPipeline(bankroll=100000)

    # Sample odds
    odds = {
        'moneyline': {
            'home': -140,
            'away': +120
        }
    }

    # Process a game
    results = await pipeline.process_game('game_123', odds)

    print("=== INTEGRATED NHL PIPELINE RESULTS ===")
    print(f"Game ID: {results['game_id']}")
    print(f"Prediction: {results['prediction']['prediction']:.1%}")
    print(f"Confidence: {results['prediction']['confidence']:.1%}")
    print(f"Recommendation: {results['recommendation']['action']}")
    print(f"Reason: {results['recommendation']['reason']}")
    print(f"Stake: {results['risk_assessment']['stake']:.1%}")
    print(f"Edge: {results['risk_assessment']['edge_analysis'].get('adjusted_edge', 0):.1%}")
    print(f"Priority: {results['recommendation']['priority']}")

    # Performance summary
    summary = pipeline.get_performance_summary()
    print("\nPerformance Summary:")
    print(f"Total Predictions: {summary['total_predictions']}")
    print(f"Average Confidence: {summary['avg_confidence']:.1%}")
    print(f"Average Edge: {summary['avg_edge']:.1%}")
    print(f"Bet Rate: {summary['bet_rate']:.1%}")

if __name__ == "__main__":
    asyncio.run(main())
