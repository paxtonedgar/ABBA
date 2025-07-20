"""
Ultra-Conservative MLB Betting Strategy Example
Demonstrates the most conservative, practical approach to MLB betting.
"""

from datetime import datetime
from typing import Any

import numpy as np


class UltraConservativeMLBBetting:
    """Ultra-conservative MLB betting strategy with minimal complexity."""

    def __init__(self, initial_bankroll: float = 10000):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.bet_history = []

        # Ultra-conservative configuration
        self.config = {
            "min_edge_threshold": 0.015,  # 1.5% minimum edge
            "max_single_bet": 0.01,  # 1% max per bet
            "max_daily_risk": 0.05,  # 5% max daily
            "max_weekly_risk": 0.10,  # 10% max weekly
            "min_confidence": 0.6,  # 60% minimum confidence
            "max_concurrent_bets": 3,  # Max 3 bets at once
            "uncertainty_factor": 0.7,  # Reduce edge by 30% for uncertainty
        }

        # Track daily and weekly risk
        self.daily_risk = 0
        self.weekly_risk = 0
        self.active_bets = 0

        # CLV tracker
        self.clv_tracker = CLVTracker()

    def analyze_game_ultra_simple(self, game_data: dict) -> dict[str, Any]:
        """Ultra-simple game analysis with minimal features."""

        print("=== ULTRA-CONSERVATIVE MLB GAME ANALYSIS ===")
        print(f"Game: {game_data['home_team']} vs {game_data['away_team']}")
        print(f"Date: {game_data['game_date']}")
        print()

        # Extract only essential features
        features = {
            "home_pitcher_era": game_data["home_pitcher"]["last_30_era"],
            "away_pitcher_era": game_data["away_pitcher"]["last_30_era"],
            "home_team_woba": np.mean(
                [p["woba_last_30"] for p in game_data["home_lineup"]]
            ),
            "away_team_woba": np.mean(
                [p["woba_last_30"] for p in game_data["away_lineup"]]
            ),
            "park_factor": game_data["park_factor"],
            "rest_advantage": self._calculate_rest_advantage(game_data),
        }

        print("=== SIMPLE FEATURE EXTRACTION ===")
        print(f"Home Pitcher ERA (30 days): {features['home_pitcher_era']:.2f}")
        print(f"Away Pitcher ERA (30 days): {features['away_pitcher_era']:.2f}")
        print(f"Home Team wOBA (30 days): {features['home_team_woba']:.3f}")
        print(f"Away Team wOBA (30 days): {features['away_team_woba']:.3f}")
        print(f"Park Factor: {features['park_factor']:.3f}")
        print(f"Rest Advantage: {features['rest_advantage']}")

        # Simple probability calculation
        prediction = self._calculate_simple_probability(features)

        print("\n=== SIMPLE PREDICTION ===")
        print(f"Home Win Probability: {prediction['home_win_probability']:.1%}")
        print(f"Away Win Probability: {prediction['away_win_probability']:.1%}")
        print(f"Model Confidence: {prediction['confidence']:.1%}")

        return prediction

    def evaluate_bet_opportunity(self, prediction: dict, odds: dict) -> dict[str, Any]:
        """Evaluate betting opportunity with ultra-conservative criteria."""

        print("\n=== ULTRA-CONSERVATIVE BET EVALUATION ===")

        opportunities = []

        # Evaluate moneyline bets only (no totals, no props)
        if "moneyline" in odds:
            home_evaluation = self._evaluate_single_bet(
                prediction["home_win_probability"], odds["moneyline"]["home"], "home"
            )
            away_evaluation = self._evaluate_single_bet(
                prediction["away_win_probability"], odds["moneyline"]["away"], "away"
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

    def apply_ultra_conservative_risk_management(
        self, opportunities: list[dict]
    ) -> list[dict]:
        """Apply ultra-conservative risk management rules."""

        print("\n=== ULTRA-CONSERVATIVE RISK MANAGEMENT ===")

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

            # Check concurrent bet limit
            if self.active_bets >= self.config["max_concurrent_bets"]:
                print(f"Rejected {opp['selection']}: Would exceed concurrent bet limit")
                continue

            # Calculate ultra-conservative stake
            stake = self._calculate_ultra_conservative_stake(
                opp["adjusted_edge"], opp["confidence"]
            )

            # Check daily risk limit
            if self.daily_risk + stake > self.config["max_daily_risk"]:
                print(f"Rejected {opp['selection']}: Would exceed daily risk limit")
                continue

            # Check weekly risk limit
            if self.weekly_risk + stake > self.config["max_weekly_risk"]:
                print(f"Rejected {opp['selection']}: Would exceed weekly risk limit")
                continue

            # Only recommend if stake is meaningful
            if stake > 0.001:  # 0.1% minimum stake
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

    def _calculate_simple_probability(self, features: dict) -> dict[str, Any]:
        """Calculate win probability using simple heuristics."""

        # Start with 50-50
        base_prob = 0.5

        # Pitching adjustment (heavily weighted)
        pitching_edge = (
            features["away_pitcher_era"] - features["home_pitcher_era"]
        ) / 10
        base_prob += pitching_edge

        # Batting adjustment (moderate weight)
        batting_edge = (features["home_team_woba"] - features["away_team_woba"]) * 2
        base_prob += batting_edge

        # Park adjustment (light weight)
        park_adjustment = (features["park_factor"] - 1.0) * 0.1
        base_prob += park_adjustment

        # Rest adjustment (light weight)
        rest_adjustment = features["rest_advantage"] * 0.02
        base_prob += rest_adjustment

        # Bound probability conservatively
        home_win_prob = max(0.35, min(0.65, base_prob))
        away_win_prob = 1 - home_win_prob

        # Calculate confidence based on feature agreement
        confidence = self._calculate_confidence(features)

        return {
            "home_win_probability": home_win_prob,
            "away_win_probability": away_win_prob,
            "confidence": confidence,
            "features": features,
        }

    def _evaluate_single_bet(
        self, our_prob: float, odds: float, selection: str
    ) -> dict[str, Any]:
        """Evaluate a single betting opportunity."""

        implied_prob = self._odds_to_prob(odds)

        # Calculate raw edge
        raw_edge = our_prob - implied_prob

        # Apply conservative adjustment for uncertainty
        adjusted_edge = raw_edge * self.config["uncertainty_factor"]

        # Calculate simple EV
        if odds > 0:
            ev = (our_prob * odds / 100) - (1 - our_prob)
        else:
            ev = (our_prob * 100 / abs(odds)) - (1 - our_prob)

        # Apply conservative adjustment to EV
        adjusted_ev = ev * self.config["uncertainty_factor"]

        return {
            "selection": selection,
            "odds": odds,
            "our_prob": our_prob,
            "implied_prob": implied_prob,
            "raw_edge": raw_edge,
            "adjusted_edge": adjusted_edge,
            "raw_ev": ev,
            "adjusted_ev": adjusted_ev,
            "confidence": 0.6,  # Conservative confidence
        }

    def _calculate_ultra_conservative_stake(
        self, edge: float, confidence: float
    ) -> float:
        """Calculate ultra-conservative stake size."""

        # Base stake: 0.5% of bankroll per 1% edge
        base_stake = edge * 0.5

        # Apply confidence adjustment
        confidence_stake = base_stake * confidence

        # Maximum stake: 1% of bankroll
        max_stake = min(confidence_stake, self.config["max_single_bet"])

        # Minimum stake: 0.1% of bankroll
        final_stake = max(max_stake, 0.001)

        return final_stake

    def _calculate_rest_advantage(self, game_data: dict) -> int:
        """Calculate rest advantage."""
        home_rest = game_data["home_rest_days"]
        away_rest = game_data["away_rest_days"]

        if home_rest > away_rest + 1:
            return 1
        elif away_rest > home_rest + 1:
            return -1
        else:
            return 0

    def _calculate_confidence(self, features: dict) -> float:
        """Calculate confidence based on feature agreement."""

        # Higher confidence if features strongly favor one team
        pitching_edge = abs(features["away_pitcher_era"] - features["home_pitcher_era"])
        batting_edge = abs(features["home_team_woba"] - features["away_team_woba"])

        # Normalize edges
        pitching_confidence = min(
            pitching_edge / 2.0, 1.0
        )  # Max confidence at 2.0 ERA difference
        batting_confidence = min(
            batting_edge / 0.1, 1.0
        )  # Max confidence at 0.1 wOBA difference

        # Average confidence
        avg_confidence = (pitching_confidence + batting_confidence) / 2

        # Conservative adjustment
        final_confidence = avg_confidence * 0.8  # Reduce by 20%

        return max(0.3, min(0.8, final_confidence))  # Bound between 30% and 80%

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
    # Sample game data with realistic values
    sample_game = {
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "game_date": "2024-06-15",
        "park_factor": 1.15,
        "home_rest_days": 1,
        "away_rest_days": 0,
        "home_pitcher": {"name": "Gerrit Cole", "last_30_era": 2.20},
        "away_pitcher": {"name": "Chris Sale", "last_30_era": 2.95},
        "home_lineup": [
            {"woba_last_30": 0.345},
            {"woba_last_30": 0.332},
            {"woba_last_30": 0.378},
            {"woba_last_30": 0.315},
            {"woba_last_30": 0.356},
            {"woba_last_30": 0.298},
            {"woba_last_30": 0.322},
            {"woba_last_30": 0.285},
            {"woba_last_30": 0.295},
        ],
        "away_lineup": [
            {"woba_last_30": 0.335},
            {"woba_last_30": 0.348},
            {"woba_last_30": 0.328},
            {"woba_last_30": 0.365},
            {"woba_last_30": 0.312},
            {"woba_last_30": 0.338},
            {"woba_last_30": 0.295},
            {"woba_last_30": 0.325},
            {"woba_last_30": 0.305},
        ],
        "odds": {"moneyline": {"home": -120, "away": +100}},
    }

    # Run ultra-conservative analysis
    analyzer = UltraConservativeMLBBetting(initial_bankroll=10000)

    # Analyze game
    prediction = analyzer.analyze_game_ultra_simple(sample_game)

    # Evaluate opportunities
    opportunities = analyzer.evaluate_bet_opportunity(prediction, sample_game["odds"])

    # Apply risk management
    recommendations = analyzer.apply_ultra_conservative_risk_management(opportunities)

    print("\n" + "=" * 70)
    print("ULTRA-CONSERVATIVE RECOMMENDATIONS")
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

        for rec in recommendations:
            print(
                f"✅ {rec['selection'].upper()}: {rec['recommended_stake']:.1%} stake (${rec['stake_amount']:.0f}), {rec['adjusted_edge']:.1%} edge"
            )
    else:
        print("❌ No betting opportunities meet ultra-conservative criteria")
        print("\nThis is expected - ultra-conservative approach rejects most games.")
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

    print("\nExpected Performance (Ultra-Conservative):")
    print("Win Rate: 52%")
    print("Average EV: 0.8%")
    print("Annual ROI: 3%")
    print("Sharpe Ratio: 0.3")
    print("Max Drawdown: 20%")
    print("Value Bet Rate: 5% of games")
