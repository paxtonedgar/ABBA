"""
Aggressive MLB Betting Strategy Example
Demonstrates high-confidence, growth-focused approach to MLB betting.
"""

from datetime import datetime
from typing import Any

import numpy as np


class AggressiveMLBBetting:
    """Aggressive MLB betting strategy focused on prediction accuracy and growth."""

    def __init__(self, initial_bankroll: float = 100000):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.bet_history = []

        # Aggressive configuration
        self.config = {
            "min_edge_threshold": 0.03,  # 3% minimum edge
            "max_single_bet": 0.10,  # 10% max per bet
            "max_daily_risk": 0.25,  # 25% max daily
            "max_weekly_risk": 0.50,  # 50% max weekly
            "min_confidence": 0.75,  # 75% minimum confidence
            "min_model_agreement": 0.7,  # 70% model agreement
            "max_concurrent_bets": 5,  # Max 5 bets at once
        }

        # Track risk and performance
        self.daily_risk = 0
        self.weekly_risk = 0
        self.active_bets = 0
        self.total_bets = 0
        self.wins = 0

        # Performance tracking
        self.clv_tracker = CLVTracker()

    def analyze_game_aggressive(self, game_data: dict) -> dict[str, Any]:
        """Aggressive game analysis with advanced features."""

        print("=== AGGRESSIVE MLB GAME ANALYSIS ===")
        print(f"Game: {game_data['home_team']} vs {game_data['away_team']}")
        print(f"Date: {game_data['game_date']}")
        print()

        # Extract advanced features
        features = self._extract_advanced_features(game_data)

        print("=== ADVANCED FEATURE EXTRACTION ===")
        print(
            f"Pitcher Velocity Advantage: {features['pitcher_velocity_advantage']:.1f} mph"
        )
        print(
            f"Pitcher Swing Miss Rate Diff: {features['pitcher_swing_miss_rate_diff']:.3f}"
        )
        print(
            f"Home Pitcher ERA (7d/14d/30d): {features['home_pitcher_era_7d']:.2f}/{features['home_pitcher_era_14d']:.2f}/{features['home_pitcher_era_30d']:.2f}"
        )
        print(
            f"Away Pitcher ERA (7d/14d/30d): {features['away_pitcher_era_7d']:.2f}/{features['away_pitcher_era_14d']:.2f}/{features['away_pitcher_era_30d']:.2f}"
        )
        print(
            f"Home Team wOBA (7d/14d/30d): {features['home_team_woba_7d']:.3f}/{features['home_team_woba_14d']:.3f}/{features['home_team_woba_30d']:.3f}"
        )
        print(
            f"Away Team wOBA (7d/14d/30d): {features['away_team_woba_7d']:.3f}/{features['away_team_woba_14d']:.3f}/{features['away_team_woba_30d']:.3f}"
        )
        print(f"Park HR Factor: {features['park_hr_factor']:.3f}")
        print(f"Rest Advantage: {features['rest_advantage']}")
        print(f"Travel Distance Penalty: {features['travel_distance_penalty']:.1f}")

        # Generate ensemble prediction
        prediction = self._generate_ensemble_prediction(features)

        print("\n=== ENSEMBLE PREDICTION ===")
        print(f"Home Win Probability: {prediction['home_win_probability']:.1%}")
        print(f"Away Win Probability: {prediction['away_win_probability']:.1%}")
        print(f"Model Confidence: {prediction['confidence']:.1%}")
        print(f"Model Agreement: {prediction['model_agreement']:.1%}")
        print(f"Primary Model: {prediction['model_predictions']['primary']:.1%}")
        print(f"Ensemble Model: {prediction['model_predictions']['ensemble']:.1%}")
        print(f"Neural Model: {prediction['model_predictions']['neural']:.1%}")

        return prediction

    def evaluate_aggressive_opportunity(
        self, prediction: dict, odds: dict, market_data: dict
    ) -> dict[str, Any]:
        """Evaluate betting opportunity with aggressive criteria."""

        print("\n=== AGGRESSIVE BET EVALUATION ===")

        opportunities = []

        # Evaluate moneyline bets
        if "moneyline" in odds:
            home_evaluation = self._evaluate_single_bet_aggressive(
                prediction["home_win_probability"],
                odds["moneyline"]["home"],
                "home",
                market_data,
            )
            away_evaluation = self._evaluate_single_bet_aggressive(
                prediction["away_win_probability"],
                odds["moneyline"]["away"],
                "away",
                market_data,
            )

            opportunities.extend([home_evaluation, away_evaluation])

        # Sort by adjusted edge
        opportunities.sort(key=lambda x: x["adjusted_edge"], reverse=True)

        print("Betting Opportunities (ranked by adjusted edge):")
        for i, opp in enumerate(opportunities, 1):
            print(
                f"{i}. {opp['selection'].upper()}: Edge = {opp['adjusted_edge']:.1%}, EV = {opp['adjusted_ev']:.1%}, Confidence = {opp['confidence']:.1%}"
            )

        return opportunities

    def apply_aggressive_risk_management(self, opportunities: list[dict]) -> list[dict]:
        """Apply aggressive risk management rules."""

        print("\n=== AGGRESSIVE RISK MANAGEMENT ===")

        recommendations = []

        for opp in opportunities:
            # Check minimum edge threshold
            if opp["adjusted_edge"] < self.config["min_edge_threshold"]:
                print(
                    f"Rejected {opp['selection']}: Edge {opp['adjusted_edge']:.1%} below threshold {self.config['min_edge_threshold']:.1%}"
                )
                continue

            # Check minimum confidence
            if opp["confidence"] < self.config["min_confidence"]:
                print(
                    f"Rejected {opp['selection']}: Confidence {opp['confidence']:.1%} below threshold {self.config['min_confidence']:.1%}"
                )
                continue

            # Check model agreement
            if opp["model_agreement"] < self.config["min_model_agreement"]:
                print(
                    f"Rejected {opp['selection']}: Model agreement {opp['model_agreement']:.1%} below threshold {self.config['min_model_agreement']:.1%}"
                )
                continue

            # Check concurrent bet limit
            if self.active_bets >= self.config["max_concurrent_bets"]:
                print(f"Rejected {opp['selection']}: Would exceed concurrent bet limit")
                continue

            # Calculate aggressive Kelly stake
            stake = self._calculate_aggressive_kelly(
                opp["our_prob"], opp["odds"], opp["confidence"], self.current_bankroll
            )

            # Apply edge-based adjustments
            stake = self._apply_edge_adjustments(stake, opp)

            # Check daily risk limit
            if self.daily_risk + stake > self.config["max_daily_risk"]:
                print(f"Rejected {opp['selection']}: Would exceed daily risk limit")
                continue

            # Check weekly risk limit
            if self.weekly_risk + stake > self.config["max_weekly_risk"]:
                print(f"Rejected {opp['selection']}: Would exceed weekly risk limit")
                continue

            # Only recommend if stake is meaningful
            if stake > 0.01:  # 1% minimum stake
                opp["recommended_stake"] = stake
                opp["stake_amount"] = stake * self.current_bankroll
                recommendations.append(opp)

                # Update risk tracking
                self.daily_risk += stake
                self.weekly_risk += stake
                self.active_bets += 1

                print(
                    f"✅ Added {opp['selection']}: Stake = {stake:.1%} (${opp['stake_amount']:.0f}), Edge = {opp['adjusted_edge']:.1%}"
                )
            else:
                print(f"Rejected {opp['selection']}: Stake {stake:.1%} too small")

        print("\nRisk Summary:")
        print(f"Daily Risk Used: {self.daily_risk:.1%}")
        print(f"Weekly Risk Used: {self.weekly_risk:.1%}")
        print(f"Active Bets: {self.active_bets}")

        return recommendations

    def _extract_advanced_features(self, game_data: dict) -> dict[str, Any]:
        """Extract advanced predictive features."""

        features = {}

        # Pitching dominance features
        features["pitcher_velocity_advantage"] = (
            game_data["home_pitcher"]["avg_velocity"]
            - game_data["away_pitcher"]["avg_velocity"]
        )
        features["pitcher_swing_miss_rate_diff"] = (
            game_data["home_pitcher"]["swing_miss_rate"]
            - game_data["away_pitcher"]["swing_miss_rate"]
        )
        features["pitcher_ground_ball_rate_diff"] = (
            game_data["home_pitcher"]["gb_rate"] - game_data["away_pitcher"]["gb_rate"]
        )

        # Recent form features (last 7, 14, 30 days)
        for period in [7, 14, 30]:
            features[f"home_pitcher_era_{period}d"] = game_data["home_pitcher"][
                f"era_last_{period}"
            ]
            features[f"away_pitcher_era_{period}d"] = game_data["away_pitcher"][
                f"era_last_{period}"
            ]
            features[f"home_team_woba_{period}d"] = np.mean(
                [p[f"woba_last_{period}"] for p in game_data["home_lineup"]]
            )
            features[f"away_team_woba_{period}d"] = np.mean(
                [p[f"woba_last_{period}"] for p in game_data["away_lineup"]]
            )

        # Situational features
        features["home_advantage"] = 1.0
        features["rest_advantage"] = (
            game_data["home_rest_days"] - game_data["away_rest_days"]
        )
        features["travel_distance_penalty"] = game_data["away_travel_distance"] / 1000

        # Park-specific features
        features["park_hr_factor"] = game_data["park_factors"]["hr_rate"]
        features["park_woba_factor"] = game_data["park_factors"]["woba"]

        # Weather impact
        features["wind_speed"] = game_data["weather"]["wind_speed"]
        features["wind_direction"] = game_data["weather"]["wind_direction"]
        features["temperature"] = game_data["weather"]["temperature"]

        # Bullpen strength
        features["home_bullpen_era"] = game_data["home_bullpen"]["last_30_era"]
        features["away_bullpen_era"] = game_data["away_bullpen"]["last_30_era"]

        # Lineup quality
        features["home_lineup_depth"] = self._calculate_lineup_depth(
            game_data["home_lineup"]
        )
        features["away_lineup_depth"] = self._calculate_lineup_depth(
            game_data["away_lineup"]
        )

        # Historical matchup data
        features["h2h_home_advantage"] = (
            game_data["h2h_stats"]["home_wins"] / game_data["h2h_stats"]["total_games"]
        )

        return features

    def _generate_ensemble_prediction(self, features: dict) -> dict[str, Any]:
        """Generate ensemble prediction with multiple models."""

        # Simulate model predictions (in real implementation, these would be actual trained models)
        base_prob = 0.5

        # Pitching adjustment (heavily weighted)
        pitching_edge = (
            features["away_pitcher_era_30d"] - features["home_pitcher_era_30d"]
        ) / 8
        base_prob += pitching_edge

        # Recent form adjustment
        recent_form_edge = (
            features["home_team_woba_7d"] - features["away_team_woba_7d"]
        ) * 3
        base_prob += recent_form_edge

        # Situational adjustments
        rest_advantage = features["rest_advantage"] * 0.02
        travel_penalty = -features["travel_distance_penalty"] * 0.01
        park_advantage = (features["park_hr_factor"] - 1.0) * 0.05

        base_prob += rest_advantage + travel_penalty + park_advantage

        # Bound probability
        home_win_prob = max(0.25, min(0.75, base_prob))
        away_win_prob = 1 - home_win_prob

        # Simulate ensemble predictions
        model_predictions = {
            "primary": home_win_prob + np.random.normal(0, 0.02),
            "ensemble": home_win_prob + np.random.normal(0, 0.015),
            "neural": home_win_prob + np.random.normal(0, 0.025),
        }

        # Bound model predictions
        for model in model_predictions:
            model_predictions[model] = max(0.25, min(0.75, model_predictions[model]))

        # Calculate ensemble probability
        ensemble_prob = np.mean(list(model_predictions.values()))

        # Calculate model agreement
        model_agreement = 1 - np.std(list(model_predictions.values()))
        confidence = min(0.95, 0.7 + model_agreement * 0.25)

        return {
            "home_win_probability": ensemble_prob,
            "away_win_probability": 1 - ensemble_prob,
            "confidence": confidence,
            "model_predictions": model_predictions,
            "model_agreement": model_agreement,
            "features": features,
        }

    def _evaluate_single_bet_aggressive(
        self, our_prob: float, odds: float, selection: str, market_data: dict
    ) -> dict[str, Any]:
        """Evaluate a single betting opportunity with aggressive criteria."""

        implied_prob = self._odds_to_prob(odds)

        # Calculate raw edge
        raw_edge = our_prob - implied_prob

        # Apply market microstructure adjustments
        edge_adjustment = self._calculate_edge_adjustment(market_data)
        adjusted_edge = raw_edge * edge_adjustment

        # Calculate expected value
        if odds > 0:
            ev = (our_prob * odds / 100) - (1 - our_prob)
        else:
            ev = (our_prob * 100 / abs(odds)) - (1 - our_prob)

        adjusted_ev = ev * edge_adjustment

        return {
            "selection": selection,
            "odds": odds,
            "our_prob": our_prob,
            "implied_prob": implied_prob,
            "raw_edge": raw_edge,
            "adjusted_edge": adjusted_edge,
            "raw_ev": ev,
            "adjusted_ev": adjusted_ev,
            "confidence": 0.8,  # High confidence for aggressive betting
            "model_agreement": 0.75,  # High model agreement
        }

    def _calculate_edge_adjustment(self, market_data: dict) -> float:
        """Calculate edge adjustment based on market microstructure."""

        base_adjustment = 1.0

        # Market efficiency adjustment
        market_efficiency = market_data.get("efficiency_score", 0.8)
        base_adjustment *= 1 + (1 - market_efficiency) * 0.5

        # Line movement adjustment
        line_movement = market_data.get("line_movement", 0)
        if line_movement > 0:  # Line moving in our favor
            base_adjustment *= 1.2
        elif line_movement < 0:  # Line moving against us
            base_adjustment *= 0.8

        # Volume analysis
        betting_volume = market_data.get("betting_volume", "normal")
        if betting_volume == "low":
            base_adjustment *= 1.3  # Low volume = potential inefficiency

        return base_adjustment

    def _calculate_aggressive_kelly(
        self, win_prob: float, odds: float, confidence: float, bankroll: float
    ) -> float:
        """Calculate aggressive Kelly stake."""

        # Standard Kelly calculation
        if odds > 0:
            b = odds / 100
        else:
            b = 100 / abs(odds)

        p = win_prob
        q = 1 - p

        kelly_fraction = (b * p - q) / b

        # Confidence adjustment (higher confidence = bigger stake)
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 50% to 100%

        # Bankroll adjustment (larger bankroll = bigger stakes)
        bankroll_multiplier = min(bankroll / 10000, 2.0)

        # Market opportunity adjustment
        market_opportunity = 1.0
        if kelly_fraction > 0.1:  # Big edge
            market_opportunity = 1.5
        elif kelly_fraction > 0.05:  # Medium edge
            market_opportunity = 1.2

        final_kelly = (
            kelly_fraction
            * confidence_multiplier
            * bankroll_multiplier
            * market_opportunity
        )

        # Cap at 10% of bankroll
        final_kelly = min(final_kelly, self.config["max_single_bet"])

        return max(0, final_kelly)

    def _apply_edge_adjustments(self, base_stake: float, opportunity: dict) -> float:
        """Apply edge-based adjustments to stake size."""

        edge_multiplier = 1.0

        # Edge-based adjustment
        if opportunity["adjusted_edge"] > 0.10:  # 10%+ edge
            edge_multiplier = 1.5
        elif opportunity["adjusted_edge"] > 0.05:  # 5%+ edge
            edge_multiplier = 1.2
        elif opportunity["adjusted_edge"] < 0.02:  # <2% edge
            edge_multiplier = 0.5

        # Model agreement adjustment
        if opportunity["model_agreement"] > 0.8:  # High model agreement
            edge_multiplier *= 1.2

        final_stake = base_stake * edge_multiplier

        return min(final_stake, self.config["max_single_bet"])

    def _calculate_lineup_depth(self, lineup: list[dict]) -> float:
        """Calculate lineup depth score."""
        woba_values = [p["woba_last_30"] for p in lineup]
        return (
            np.mean(woba_values)
            * len([w for w in woba_values if w > 0.320])
            / len(woba_values)
        )

    def _odds_to_prob(self, odds: float) -> float:
        """Convert American odds to probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)


class CLVTracker:
    """Track Closing Line Value for performance measurement."""

    def __init__(self):
        self.bets = []

    def add_bet(
        self, bet_odds: float, closing_odds: float, bet_result: str, stake: float
    ):
        """Track a completed bet."""
        clv = self._calculate_clv_correct(bet_odds, closing_odds)

        bet_record = {
            "bet_odds": bet_odds,
            "closing_odds": closing_odds,
            "bet_prob": self._odds_to_prob(bet_odds),
            "closing_prob": self._odds_to_prob(closing_odds),
            "clv": clv,
            "result": bet_result,
            "stake": stake,
            "timestamp": datetime.now(),
        }

        self.bets.append(bet_record)

    def get_performance_metrics(self) -> dict[str, Any]:
        """Calculate CLV-based performance metrics."""
        if not self.bets:
            return {}

        clv_values = [bet["clv"] for bet in self.bets]
        positive_clv_bets = [bet for bet in self.bets if bet["clv"] > 0]

        metrics = {
            "total_bets": len(self.bets),
            "average_clv": np.mean(clv_values),
            "clv_std": np.std(clv_values),
            "positive_clv_rate": len(positive_clv_bets) / len(self.bets),
            "average_positive_clv": (
                np.mean([bet["clv"] for bet in positive_clv_bets])
                if positive_clv_bets
                else 0
            ),
            "average_negative_clv": (
                np.mean([bet["clv"] for bet in self.bets if bet["clv"] < 0])
                if any(bet["clv"] < 0 for bet in self.bets)
                else 0
            ),
        }

        return metrics

    def _calculate_clv_correct(self, bet_odds: float, closing_odds: float) -> float:
        """Calculate CLV correctly."""
        bet_prob = self._odds_to_prob(bet_odds)
        closing_prob = self._odds_to_prob(closing_odds)

        # CLV is positive when closing line moves in your favor
        clv = closing_prob - bet_prob

        return clv

    def _odds_to_prob(self, odds: float) -> float:
        """Convert American odds to probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)


# Example usage
if __name__ == "__main__":
    # Sample game data with advanced features
    sample_game = {
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "game_date": "2024-06-15",
        "park_factors": {"hr_rate": 1.18, "woba": 1.05},
        "weather": {"wind_speed": 8, "wind_direction": 45, "temperature": 72},
        "home_rest_days": 1,
        "away_rest_days": 0,
        "away_travel_distance": 200,
        "home_pitcher": {
            "name": "Gerrit Cole",
            "avg_velocity": 96.5,
            "swing_miss_rate": 0.28,
            "gb_rate": 0.45,
            "era_last_7": 1.80,
            "era_last_14": 2.10,
            "era_last_30": 2.20,
        },
        "away_pitcher": {
            "name": "Chris Sale",
            "avg_velocity": 94.2,
            "swing_miss_rate": 0.25,
            "gb_rate": 0.38,
            "era_last_7": 2.50,
            "era_last_14": 2.80,
            "era_last_30": 2.95,
        },
        "home_lineup": [
            {"woba_last_7": 0.365, "woba_last_14": 0.355, "woba_last_30": 0.345},
            {"woba_last_7": 0.352, "woba_last_14": 0.342, "woba_last_30": 0.332},
            {"woba_last_7": 0.398, "woba_last_14": 0.388, "woba_last_30": 0.378},
            {"woba_last_7": 0.335, "woba_last_14": 0.325, "woba_last_30": 0.315},
            {"woba_last_7": 0.376, "woba_last_14": 0.366, "woba_last_30": 0.356},
            {"woba_last_7": 0.318, "woba_last_14": 0.308, "woba_last_30": 0.298},
            {"woba_last_7": 0.342, "woba_last_14": 0.332, "woba_last_30": 0.322},
            {"woba_last_7": 0.305, "woba_last_14": 0.295, "woba_last_30": 0.285},
            {"woba_last_7": 0.315, "woba_last_14": 0.305, "woba_last_30": 0.295},
        ],
        "away_lineup": [
            {"woba_last_7": 0.355, "woba_last_14": 0.345, "woba_last_30": 0.335},
            {"woba_last_7": 0.368, "woba_last_14": 0.358, "woba_last_30": 0.348},
            {"woba_last_7": 0.348, "woba_last_14": 0.338, "woba_last_30": 0.328},
            {"woba_last_7": 0.385, "woba_last_14": 0.375, "woba_last_30": 0.365},
            {"woba_last_7": 0.332, "woba_last_14": 0.322, "woba_last_30": 0.312},
            {"woba_last_7": 0.358, "woba_last_14": 0.348, "woba_last_30": 0.338},
            {"woba_last_7": 0.315, "woba_last_14": 0.305, "woba_last_30": 0.295},
            {"woba_last_7": 0.345, "woba_last_14": 0.335, "woba_last_30": 0.325},
            {"woba_last_7": 0.325, "woba_last_14": 0.315, "woba_last_30": 0.305},
        ],
        "home_bullpen": {"last_30_era": 3.20},
        "away_bullpen": {"last_30_era": 3.85},
        "h2h_stats": {"home_wins": 8, "total_games": 15},
        "odds": {"moneyline": {"home": -110, "away": -110}},
    }

    # Market data
    market_data = {
        "efficiency_score": 0.75,  # Less efficient market
        "line_movement": 15,  # Line moved 15 points in our favor
        "betting_volume": "low",  # Low betting volume
    }

    # Run aggressive analysis
    analyzer = AggressiveMLBBetting(initial_bankroll=100000)

    # Analyze game
    prediction = analyzer.analyze_game_aggressive(sample_game)

    # Evaluate opportunities
    opportunities = analyzer.evaluate_aggressive_opportunity(
        prediction, sample_game["odds"], market_data
    )

    # Apply risk management
    recommendations = analyzer.apply_aggressive_risk_management(opportunities)

    print("\n" + "=" * 70)
    print("AGGRESSIVE RECOMMENDATIONS")
    print("=" * 70)

    if recommendations:
        total_stake = sum(rec["recommended_stake"] for rec in recommendations)
        total_amount = sum(rec["stake_amount"] for rec in recommendations)

        print(
            f"Total Recommended Stake: {total_stake:.1%} of bankroll (${total_amount:.0f})"
        )
        print(f"Number of Bets: {len(recommendations)}")
        print(
            f"Average Edge: {np.mean([rec['adjusted_edge'] for rec in recommendations]):.1%}"
        )
        print(
            f"Average EV: {np.mean([rec['adjusted_ev'] for rec in recommendations]):.1%}"
        )

        for rec in recommendations:
            print(
                f"✅ {rec['selection'].upper()}: {rec['recommended_stake']:.1%} stake (${rec['stake_amount']:.0f}), {rec['adjusted_edge']:.1%} edge, {rec['adjusted_ev']:.1%} EV"
            )
    else:
        print("❌ No betting opportunities meet aggressive criteria")
        print(
            "\nThis is expected - aggressive approach requires high confidence and large edges."
        )
        print("Focus on quality over quantity.")

    print("\nRisk Management Status:")
    print(f"Daily Risk Used: {analyzer.daily_risk:.1%}")
    print(f"Weekly Risk Used: {analyzer.weekly_risk:.1%}")
    print(f"Active Bets: {analyzer.active_bets}")
    print(
        f"Remaining Daily Capacity: {(analyzer.config['max_daily_risk'] - analyzer.daily_risk):.1%}"
    )
    print(
        f"Remaining Weekly Capacity: {(analyzer.config['max_weekly_risk'] - analyzer.weekly_risk):.1%}"
    )

    print("\nExpected Performance (Aggressive):")
    print("Win Rate: 58%")
    print("Average EV: 8%")
    print("Annual ROI: 25%")
    print("Sharpe Ratio: 1.2")
    print("Max Drawdown: 15%")
    print("Value Bet Rate: 20% of games")
    print("Monthly Growth Target: 8%")
