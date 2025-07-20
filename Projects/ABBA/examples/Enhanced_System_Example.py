"""
Enhanced System Example: Real-Time Data Processing & Advanced Ensemble Modeling
Demonstrates the improved MLB and NHL betting systems with real-time capabilities.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np


# Mock ML models for demonstration
class MockModel:
    def predict_proba(self, X):
        return np.array([[0.3, 0.7]])  # Mock prediction


class XGBClassifier(MockModel):
    pass


class LGBMClassifier(MockModel):
    pass


class CatBoostClassifier(MockModel):
    pass


class MLPClassifier(MockModel):
    pass


class SVC(MockModel):
    pass


class RandomForestClassifier(MockModel):
    pass


@dataclass
class RealTimeSignal:
    """Real-time market signal."""

    signal_type: str
    value: float
    confidence: float
    timestamp: datetime
    source: str


class RealTimeDataProcessor:
    """Real-time data processing for live betting opportunities."""

    def __init__(self):
        self.data_streams = {
            "odds": OddsStream(),
            "lineup": LineupStream(),
            "weather": WeatherStream(),
            "injury": InjuryStream(),
            "social": SocialMediaStream(),
        }
        self.signal_history = []

    async def process_real_time_signals(self, game_id: str) -> dict[str, Any]:
        """Process real-time signals for live betting opportunities."""

        signals = {}

        # Process all data streams concurrently
        tasks = [
            self.data_streams["odds"].get_movement_signals(game_id),
            self.data_streams["lineup"].get_confirmation_signals(game_id),
            self.data_streams["weather"].get_impact_signals(game_id),
            self.data_streams["injury"].get_last_minute_signals(game_id),
            self.data_streams["social"].get_sentiment_signals(game_id),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Odds movement analysis
        if not isinstance(results[0], Exception):
            odds_signals = results[0]
            signals["odds_steam"] = odds_signals.get("steam_move", False)
            signals["reverse_movement"] = odds_signals.get("reverse_movement", False)
            signals["sharp_action"] = odds_signals.get("sharp_action", False)
            signals["line_movement"] = odds_signals.get("line_movement", 0)

        # Lineup confirmation
        if not isinstance(results[1], Exception):
            lineup_signals = results[1]
            signals["lineup_confirmed"] = lineup_signals.get("confirmed", False)
            signals["key_player_status"] = lineup_signals.get("key_player_status", {})

        # Weather updates
        if not isinstance(results[2], Exception):
            weather_signals = results[2]
            signals["weather_impact"] = weather_signals.get("impact_score", 0)

        # Injury updates
        if not isinstance(results[3], Exception):
            injury_signals = results[3]
            signals["injury_impact"] = injury_signals.get("impact_score", 0)

        # Social media sentiment
        if not isinstance(results[4], Exception):
            social_signals = results[4]
            signals["public_sentiment"] = social_signals.get("sentiment_score", 0)

        # Store signal history
        self.signal_history.append(
            {"game_id": game_id, "signals": signals, "timestamp": datetime.now()}
        )

        return signals


class AdvancedEnsemblePredictor:
    """Advanced ensemble predictor with dynamic weighting."""

    def __init__(self):
        self.models = {
            "xgboost": XGBClassifier(),
            "lightgbm": LGBMClassifier(),
            "catboost": CatBoostClassifier(),
            "neural": MLPClassifier(),
            "svm": SVC(),
            "random_forest": RandomForestClassifier(),
        }
        self.performance_tracker = ModelPerformanceTracker()
        self.context_analyzer = ContextAnalyzer()

    def predict_with_dynamic_weighting(
        self, features: dict, game_context: dict
    ) -> dict[str, Any]:
        """Predict with dynamic model weighting based on context."""

        # Get individual predictions
        predictions = {}
        for name, model in self.models.items():
            try:
                X = np.array([list(features.values())])
                predictions[name] = model.predict_proba(X)[0][1]
            except Exception as e:
                print(f"Error with {name} model: {e}")
                predictions[name] = 0.5  # Default prediction

        # Calculate dynamic weights
        recent_weights = self.performance_tracker.get_recent_weights(30)
        context_weights = self.context_analyzer.calculate_context_weights(game_context)
        feature_weights = self.calculate_feature_alignment(features)
        uncertainty_weights = self.calculate_uncertainty_weights(predictions)

        # Combine weights
        final_weights = self.combine_weights(
            [recent_weights, context_weights, feature_weights, uncertainty_weights]
        )

        # Weighted ensemble prediction
        weighted_prediction = sum(
            predictions[model] * final_weights[model] for model in predictions
        )

        return {
            "prediction": weighted_prediction,
            "confidence": self.calculate_confidence(final_weights, predictions),
            "model_weights": final_weights,
            "uncertainty": self.calculate_uncertainty(predictions),
            "individual_predictions": predictions,
        }

    def calculate_feature_alignment(self, features: dict) -> dict[str, float]:
        """Calculate feature importance alignment weights."""

        # Simulate feature importance scores (in real implementation, these would be from SHAP)
        feature_importance = {
            "xgboost": {
                "pitching": 0.4,
                "batting": 0.3,
                "situational": 0.2,
                "weather": 0.1,
            },
            "lightgbm": {
                "pitching": 0.35,
                "batting": 0.35,
                "situational": 0.2,
                "weather": 0.1,
            },
            "catboost": {
                "pitching": 0.45,
                "batting": 0.25,
                "situational": 0.2,
                "weather": 0.1,
            },
            "neural": {
                "pitching": 0.3,
                "batting": 0.3,
                "situational": 0.25,
                "weather": 0.15,
            },
            "svm": {
                "pitching": 0.4,
                "batting": 0.3,
                "situational": 0.2,
                "weather": 0.1,
            },
            "random_forest": {
                "pitching": 0.35,
                "batting": 0.35,
                "situational": 0.2,
                "weather": 0.1,
            },
        }

        # Calculate alignment scores
        alignment_weights = {}
        for model, importance in feature_importance.items():
            alignment_score = 0
            for feature_type, weight in importance.items():
                # Simulate feature type detection
                if feature_type in str(features).lower():
                    alignment_score += weight
            alignment_weights[model] = alignment_score

        # Normalize weights
        total = sum(alignment_weights.values())
        if total > 0:
            alignment_weights = {k: v / total for k, v in alignment_weights.items()}

        return alignment_weights

    def calculate_uncertainty_weights(
        self, predictions: dict[str, float]
    ) -> dict[str, float]:
        """Calculate uncertainty-based weights."""

        # Models with lower variance get higher weights
        mean_pred = np.mean(list(predictions.values()))
        variances = {
            model: abs(pred - mean_pred) for model, pred in predictions.items()
        }

        # Invert variances (lower variance = higher weight)
        total_variance = sum(variances.values())
        if total_variance > 0:
            uncertainty_weights = {
                model: (total_variance - var) / total_variance
                for model, var in variances.items()
            }
        else:
            uncertainty_weights = {
                model: 1.0 / len(predictions) for model in predictions
            }

        return uncertainty_weights

    def combine_weights(self, weight_sets: list[dict[str, float]]) -> dict[str, float]:
        """Combine multiple weight sets."""

        combined_weights = {}
        for model in weight_sets[0].keys():
            # Weighted average of all weight sets
            weights = [ws.get(model, 0) for ws in weight_sets]
            combined_weights[model] = np.mean(weights)

        # Normalize
        total = sum(combined_weights.values())
        if total > 0:
            combined_weights = {k: v / total for k, v in combined_weights.items()}

        return combined_weights

    def calculate_confidence(
        self, weights: dict[str, float], predictions: dict[str, float]
    ) -> float:
        """Calculate confidence based on model agreement and weights."""

        # Model agreement
        predictions_list = list(predictions.values())
        agreement = 1 - np.std(predictions_list)

        # Weight quality
        weight_quality = 1 - np.std(list(weights.values()))

        # Combined confidence
        confidence = 0.7 + (agreement * 0.2) + (weight_quality * 0.1)

        return min(0.95, confidence)

    def calculate_uncertainty(self, predictions: dict[str, float]) -> float:
        """Calculate prediction uncertainty."""

        predictions_list = list(predictions.values())
        uncertainty = np.std(predictions_list)

        return uncertainty


class EnhancedBettingSystem:
    """Enhanced betting system with real-time capabilities."""

    def __init__(self, sport: str, initial_bankroll: float = 100000):
        self.sport = sport
        self.bankroll = initial_bankroll
        self.real_time_processor = RealTimeDataProcessor()
        self.ensemble_predictor = AdvancedEnsemblePredictor()
        self.risk_manager = AdvancedRiskManager(initial_bankroll)
        self.performance_monitor = AdvancedPerformanceMonitor()

        # Sport-specific configuration
        self.config = self.get_sport_config(sport)

    def get_sport_config(self, sport: str) -> dict[str, Any]:
        """Get sport-specific configuration."""

        if sport == "mlb":
            return {
                "min_edge_threshold": 0.03,
                "max_single_bet": 0.10,
                "max_daily_risk": 0.25,
                "max_weekly_risk": 0.50,
                "min_confidence": 0.75,
                "min_model_agreement": 0.7,
                "features": ["pitching", "batting", "situational", "weather"],
            }
        elif sport == "nhl":
            return {
                "min_edge_threshold": 0.04,
                "max_single_bet": 0.12,
                "max_daily_risk": 0.30,
                "max_weekly_risk": 0.60,
                "min_confidence": 0.80,
                "min_model_agreement": 0.75,
                "min_goalie_advantage": 0.02,
                "features": [
                    "goaltending",
                    "possession",
                    "special_teams",
                    "situational",
                ],
            }
        else:
            raise ValueError(f"Unsupported sport: {sport}")

    async def analyze_game_enhanced(
        self, game_data: dict, real_time_signals: dict = None
    ) -> dict[str, Any]:
        """Enhanced game analysis with real-time signals."""

        print(f"=== ENHANCED {self.sport.upper()} GAME ANALYSIS ===")
        print(
            f"Game: {game_data.get('home_team', 'Unknown')} vs {game_data.get('away_team', 'Unknown')}"
        )
        print(f"Timestamp: {datetime.now()}")
        print()

        # Extract base features
        base_features = self.extract_base_features(game_data)

        # Apply real-time signal adjustments
        if real_time_signals:
            adjusted_features = self.apply_real_time_adjustments(
                base_features, real_time_signals
            )
            print("=== REAL-TIME SIGNAL ADJUSTMENTS ===")
            for signal_type, value in real_time_signals.items():
                if isinstance(value, (int, float)) and value != 0:
                    print(f"{signal_type}: {value}")
        else:
            adjusted_features = base_features

        # Generate ensemble prediction
        game_context = {
            "sport": self.sport,
            "game_data": game_data,
            "real_time_signals": real_time_signals,
        }

        prediction = self.ensemble_predictor.predict_with_dynamic_weighting(
            adjusted_features, game_context
        )

        print("\n=== ENHANCED ENSEMBLE PREDICTION ===")
        print(f"Home Win Probability: {prediction['prediction']:.1%}")
        print(f"Away Win Probability: {1 - prediction['prediction']:.1%}")
        print(f"Model Confidence: {prediction['confidence']:.1%}")
        print(f"Prediction Uncertainty: {prediction['uncertainty']:.3f}")
        print("Model Weights:")
        for model, weight in prediction["model_weights"].items():
            print(f"  {model}: {weight:.1%}")

        return prediction

    def extract_base_features(self, game_data: dict) -> dict[str, float]:
        """Extract base features for the sport."""

        if self.sport == "mlb":
            return self.extract_mlb_features(game_data)
        elif self.sport == "nhl":
            return self.extract_nhl_features(game_data)
        else:
            return {}

    def extract_mlb_features(self, game_data: dict) -> dict[str, float]:
        """Extract MLB-specific features."""

        features = {}

        # Pitching features
        features["home_pitcher_era"] = game_data.get("home_pitcher", {}).get("era", 4.0)
        features["away_pitcher_era"] = game_data.get("away_pitcher", {}).get("era", 4.0)
        features["home_pitcher_velocity"] = game_data.get("home_pitcher", {}).get(
            "avg_velocity", 92.0
        )
        features["away_pitcher_velocity"] = game_data.get("away_pitcher", {}).get(
            "avg_velocity", 92.0
        )

        # Batting features
        features["home_team_woba"] = game_data.get("home_team", {}).get("woba", 0.320)
        features["away_team_woba"] = game_data.get("away_team", {}).get("woba", 0.320)

        # Situational features
        features["park_factor"] = game_data.get("park_factor", 1.0)
        features["home_advantage"] = 1.0
        features["rest_advantage"] = game_data.get("home_rest_days", 1) - game_data.get(
            "away_rest_days", 1
        )

        # Weather features
        features["temperature"] = game_data.get("weather", {}).get("temperature", 70)
        features["wind_speed"] = game_data.get("weather", {}).get("wind_speed", 5)

        return features

    def extract_nhl_features(self, game_data: dict) -> dict[str, float]:
        """Extract NHL-specific features."""

        features = {}

        # Goaltending features
        features["home_goalie_save_pct"] = game_data.get("home_goalie", {}).get(
            "save_percentage", 0.910
        )
        features["away_goalie_save_pct"] = game_data.get("away_goalie", {}).get(
            "save_percentage", 0.910
        )
        features["home_goalie_gsaa"] = game_data.get("home_goalie", {}).get("gsaa", 0.0)
        features["away_goalie_gsaa"] = game_data.get("away_goalie", {}).get("gsaa", 0.0)

        # Possession features
        features["home_corsi_percentage"] = game_data.get("home_team", {}).get(
            "corsi_for_percentage", 50.0
        )
        features["away_corsi_percentage"] = game_data.get("away_team", {}).get(
            "corsi_for_percentage", 50.0
        )
        features["home_xgf_percentage"] = game_data.get("home_team", {}).get(
            "xgf_percentage", 50.0
        )
        features["away_xgf_percentage"] = game_data.get("away_team", {}).get(
            "xgf_percentage", 50.0
        )

        # Special teams features
        features["home_power_play_pct"] = game_data.get("home_team", {}).get(
            "power_play_percentage", 20.0
        )
        features["away_power_play_pct"] = game_data.get("away_team", {}).get(
            "power_play_percentage", 20.0
        )
        features["home_penalty_kill_pct"] = game_data.get("home_team", {}).get(
            "penalty_kill_percentage", 80.0
        )
        features["away_penalty_kill_pct"] = game_data.get("away_team", {}).get(
            "penalty_kill_percentage", 80.0
        )

        # Situational features
        features["home_advantage"] = 1.0
        features["rest_advantage"] = game_data.get("home_rest_days", 1) - game_data.get(
            "away_rest_days", 1
        )
        features["back_to_back_penalty"] = (
            1 if game_data.get("away_back_to_back", False) else 0
        )

        return features

    def apply_real_time_adjustments(
        self, features: dict, signals: dict
    ) -> dict[str, float]:
        """Apply real-time signal adjustments to features."""

        adjusted_features = features.copy()

        # Odds movement adjustments
        if "line_movement" in signals:
            movement = signals["line_movement"]
            if abs(movement) > 10:  # Significant line movement
                adjustment_factor = 1 + (movement / 100) * 0.1
                for key in adjusted_features:
                    if "home" in key:
                        adjusted_features[key] *= adjustment_factor

        # Injury impact adjustments
        if "injury_impact" in signals:
            injury_impact = signals["injury_impact"]
            if injury_impact > 0.5:  # Significant injury impact
                for key in adjusted_features:
                    if "away" in key:
                        adjusted_features[key] *= 1 - injury_impact * 0.1

        # Weather impact adjustments
        if "weather_impact" in signals:
            weather_impact = signals["weather_impact"]
            if self.sport == "mlb" and abs(weather_impact) > 0.3:
                # Weather affects MLB more than NHL
                for key in adjusted_features:
                    if "weather" in key or "park" in key:
                        adjusted_features[key] *= 1 + weather_impact * 0.05

        # Public sentiment adjustments
        if "public_sentiment" in signals:
            sentiment = signals["public_sentiment"]
            if abs(sentiment) > 0.5:  # Strong public sentiment
                # Adjust against public sentiment (contrarian approach)
                for key in adjusted_features:
                    if "home" in key:
                        adjusted_features[key] *= 1 - sentiment * 0.05

        return adjusted_features


class ModelPerformanceTracker:
    """Track model performance for dynamic weighting."""

    def __init__(self):
        self.performance_history = {}

    def get_recent_weights(self, days: int) -> dict[str, float]:
        """Get recent performance weights."""

        # Simulate recent performance (in real implementation, this would be actual data)
        recent_performance = {
            "xgboost": 0.85,
            "lightgbm": 0.82,
            "catboost": 0.88,
            "neural": 0.80,
            "svm": 0.78,
            "random_forest": 0.83,
        }

        # Normalize weights
        total = sum(recent_performance.values())
        weights = {k: v / total for k, v in recent_performance.items()}

        return weights


class ContextAnalyzer:
    """Analyze game context for model weighting."""

    def calculate_context_weights(self, game_context: dict) -> dict[str, float]:
        """Calculate context-based weights."""

        sport = game_context.get("sport", "mlb")

        if sport == "mlb":
            # MLB context weights
            return {
                "xgboost": 0.20,
                "lightgbm": 0.18,
                "catboost": 0.22,
                "neural": 0.15,
                "svm": 0.12,
                "random_forest": 0.13,
            }
        elif sport == "nhl":
            # NHL context weights
            return {
                "xgboost": 0.18,
                "lightgbm": 0.20,
                "catboost": 0.19,
                "neural": 0.17,
                "svm": 0.13,
                "random_forest": 0.13,
            }
        else:
            # Default weights
            return dict.fromkeys(
                ["xgboost", "lightgbm", "catboost", "neural", "svm", "random_forest"],
                1.0 / 6,
            )


class AdvancedRiskManager:
    """Advanced risk manager with portfolio optimization."""

    def __init__(self, bankroll: float):
        self.bankroll = bankroll
        self.portfolio_state = {}

    def calculate_optimal_stake(
        self, bet_analysis: dict, portfolio_state: dict
    ) -> float:
        """Calculate optimal stake with advanced risk management."""

        # Base Kelly calculation
        base_kelly = self.calculate_base_kelly(bet_analysis)

        # Portfolio optimization
        portfolio_adjustment = self.calculate_portfolio_adjustment(
            bet_analysis, portfolio_state
        )

        # Correlation adjustment
        correlation_adjustment = self.calculate_correlation_adjustment(
            bet_analysis, portfolio_state
        )

        # VaR adjustment
        var_adjustment = self.calculate_var_adjustment(bet_analysis, portfolio_state)

        # Combine adjustments
        final_stake = (
            base_kelly * portfolio_adjustment * correlation_adjustment * var_adjustment
        )

        return min(final_stake, 0.15)  # Cap at 15%

    def calculate_base_kelly(self, bet_analysis: dict) -> float:
        """Calculate base Kelly stake."""

        win_prob = bet_analysis.get("win_probability", 0.5)
        odds = bet_analysis.get("odds", 0)

        if odds > 0:
            b = odds / 100
        else:
            b = 100 / abs(odds)

        p = win_prob
        q = 1 - p

        kelly = (b * p - q) / b

        return max(0, kelly)

    def calculate_portfolio_adjustment(
        self, bet_analysis: dict, portfolio_state: dict
    ) -> float:
        """Calculate portfolio optimization adjustment."""

        # Simulate portfolio adjustment
        current_exposure = portfolio_state.get("current_exposure", 0)
        max_exposure = portfolio_state.get("max_exposure", 0.5)

        if current_exposure < max_exposure * 0.3:
            return 1.2  # Increase stake when portfolio is under-allocated
        elif current_exposure > max_exposure * 0.8:
            return 0.7  # Decrease stake when portfolio is over-allocated
        else:
            return 1.0

    def calculate_correlation_adjustment(
        self, bet_analysis: dict, portfolio_state: dict
    ) -> float:
        """Calculate correlation adjustment."""

        # Simulate correlation analysis
        correlation = portfolio_state.get("correlation", 0)

        if correlation < 0.2:
            return 1.1  # Low correlation, increase stake
        elif correlation > 0.6:
            return 0.8  # High correlation, decrease stake
        else:
            return 1.0

    def calculate_var_adjustment(
        self, bet_analysis: dict, portfolio_state: dict
    ) -> float:
        """Calculate Value at Risk adjustment."""

        # Simulate VaR calculation
        current_var = portfolio_state.get("current_var", 0.1)
        max_var = portfolio_state.get("max_var", 0.2)

        if current_var < max_var * 0.5:
            return 1.1  # Low VaR, increase stake
        elif current_var > max_var * 0.9:
            return 0.8  # High VaR, decrease stake
        else:
            return 1.0


class AdvancedPerformanceMonitor:
    """Advanced performance monitoring system."""

    def __init__(self):
        self.metrics_history = []

    def calculate_metrics(self, bet_history: list[dict]) -> dict[str, Any]:
        """Calculate comprehensive performance metrics."""

        if not bet_history:
            return {}

        # Basic metrics
        total_bets = len(bet_history)
        wins = sum(1 for bet in bet_history if bet.get("result") == "win")
        win_rate = wins / total_bets if total_bets > 0 else 0

        # Risk-adjusted metrics
        returns = [bet.get("return", 0) for bet in bet_history]
        avg_return = np.mean(returns) if returns else 0
        std_return = np.std(returns) if returns else 0
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0

        # Advanced metrics
        max_drawdown = self.calculate_max_drawdown(returns)
        information_ratio = self.calculate_information_ratio(returns)

        return {
            "total_bets": total_bets,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "information_ratio": information_ratio,
        }

    def calculate_max_drawdown(self, returns: list[float]) -> float:
        """Calculate maximum drawdown."""

        if not returns:
            return 0

        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        return abs(np.min(drawdown))

    def calculate_information_ratio(self, returns: list[float]) -> float:
        """Calculate information ratio."""

        if not returns:
            return 0

        avg_return = np.mean(returns)
        tracking_error = np.std(returns)

        return avg_return / tracking_error if tracking_error > 0 else 0


# Mock data stream classes
class OddsStream:
    async def get_movement_signals(self, game_id: str) -> dict[str, Any]:
        return {
            "steam_move": np.random.choice([True, False], p=[0.1, 0.9]),
            "reverse_movement": np.random.choice([True, False], p=[0.15, 0.85]),
            "sharp_action": np.random.choice([True, False], p=[0.2, 0.8]),
            "line_movement": np.random.normal(0, 15),
        }


class LineupStream:
    async def get_confirmation_signals(self, game_id: str) -> dict[str, Any]:
        return {
            "confirmed": np.random.choice([True, False], p=[0.8, 0.2]),
            "key_player_status": {"player1": "active", "player2": "questionable"},
        }


class WeatherStream:
    async def get_impact_signals(self, game_id: str) -> dict[str, Any]:
        return {"impact_score": np.random.normal(0, 0.3)}


class InjuryStream:
    async def get_last_minute_signals(self, game_id: str) -> dict[str, Any]:
        return {"impact_score": np.random.choice([0, 0.3, 0.7], p=[0.7, 0.2, 0.1])}


class SocialMediaStream:
    async def get_sentiment_signals(self, game_id: str) -> dict[str, Any]:
        return {"sentiment_score": np.random.normal(0, 0.4)}


# Example usage
async def main():
    """Example of enhanced system usage."""

    # Sample game data
    sample_game = {
        "home_team": "Boston Bruins",
        "away_team": "Toronto Maple Leafs",
        "home_goalie": {"save_percentage": 0.925, "gsaa": 12.5},
        "away_goalie": {"save_percentage": 0.895, "gsaa": -2.1},
        "home_team": {
            "corsi_for_percentage": 52.8,
            "xgf_percentage": 52.5,
            "power_play_percentage": 24.5,
            "penalty_kill_percentage": 87.2,
        },
        "away_team": {
            "corsi_for_percentage": 51.2,
            "xgf_percentage": 51.5,
            "power_play_percentage": 22.1,
            "penalty_kill_percentage": 82.5,
        },
        "home_rest_days": 2,
        "away_rest_days": 0,
        "away_back_to_back": True,
    }

    # Initialize enhanced system
    enhanced_system = EnhancedBettingSystem("nhl", initial_bankroll=100000)

    # Process real-time signals
    real_time_signals = (
        await enhanced_system.real_time_processor.process_real_time_signals("game_123")
    )

    # Analyze game with enhanced system
    prediction = await enhanced_system.analyze_game_enhanced(
        sample_game, real_time_signals
    )

    print("\n=== ENHANCED SYSTEM SUMMARY ===")
    print(f"Sport: {enhanced_system.sport.upper()}")
    print(f"Prediction: {prediction['prediction']:.1%}")
    print(f"Confidence: {prediction['confidence']:.1%}")
    print(f"Uncertainty: {prediction['uncertainty']:.3f}")
    print(f"Real-time signals processed: {len(real_time_signals)}")
    print(f"Models used: {len(prediction['model_weights'])}")


if __name__ == "__main__":
    asyncio.run(main())
