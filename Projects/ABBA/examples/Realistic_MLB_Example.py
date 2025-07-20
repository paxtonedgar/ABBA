"""
Realistic MLB Betting Strategy Example
Demonstrates a more conservative, sustainable approach to MLB betting.
"""

from datetime import datetime
from typing import Any

import numpy as np


class RealisticMLBBetting:
    """Realistic MLB betting strategy with conservative risk management."""

    def __init__(self, initial_bankroll: float = 10000):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.bet_history = []

        # Conservative configuration
        self.config = {
            "min_ev_threshold": 0.01,  # 1% minimum EV
            "max_risk_per_bet": 0.02,  # 2% max risk per bet
            "kelly_fraction": 0.25,  # Conservative Kelly fraction
            "min_confidence": 0.7,  # 70% minimum confidence
            "max_daily_risk": 0.10,  # 10% max daily risk
            "max_weekly_risk": 0.20,  # 20% max weekly risk
            "uncertainty_factor": 0.25,  # Uncertainty adjustment
        }

        # Track daily and weekly risk
        self.daily_risk = 0
        self.weekly_risk = 0
        self.last_reset_date = datetime.now().date()

    def analyze_game_conservative(self, game_data: dict) -> dict[str, Any]:
        """Conservative game analysis with realistic expectations."""

        print("=== CONSERVATIVE MLB GAME ANALYSIS ===")
        print(f"Game: {game_data['home_team']} vs {game_data['away_team']}")
        print(f"Date: {game_data['game_date']}")
        print()

        # Simplified analysis focusing on key factors
        pitching_analysis = self._analyze_pitching_simple(game_data)
        batting_analysis = self._analyze_batting_simple(game_data)
        situational_analysis = self._analyze_situational_simple(game_data)

        # Generate conservative predictions
        predictions = self._generate_conservative_predictions(
            pitching_analysis, batting_analysis, situational_analysis
        )

        # Evaluate betting opportunities conservatively
        opportunities = self._evaluate_opportunities_conservative(
            predictions, game_data["odds"]
        )

        # Apply strict risk management
        recommendations = self._apply_conservative_risk_management(opportunities)

        return {
            "game_info": game_data,
            "analysis": {
                "pitching": pitching_analysis,
                "batting": batting_analysis,
                "situational": situational_analysis,
            },
            "predictions": predictions,
            "opportunities": opportunities,
            "recommendations": recommendations,
        }

    def _analyze_pitching_simple(self, game_data: dict) -> dict[str, Any]:
        """Simplified pitching analysis focusing on key metrics."""

        home_pitcher = game_data["home_pitcher"]
        away_pitcher = game_data["away_pitcher"]

        print("=== SIMPLIFIED PITCHING ANALYSIS ===")

        # Focus on recent performance (last 30 days)
        home_pitcher_score = self._calculate_pitcher_score(home_pitcher, "home")
        away_pitcher_score = self._calculate_pitcher_score(away_pitcher, "away")

        advantage = home_pitcher_score - away_pitcher_score

        print(f"Home Pitcher ({home_pitcher['name']}):")
        print(f"  Recent ERA: {home_pitcher['last_30_era']:.2f}")
        print(f"  K/9: {home_pitcher['k_per_9_last_30']:.1f}")
        print(f"  BB/9: {home_pitcher['bb_per_9_last_30']:.1f}")
        print(f"  Score: {home_pitcher_score:.2f}")

        print(f"\nAway Pitcher ({away_pitcher['name']}):")
        print(f"  Recent ERA: {away_pitcher['last_30_era']:.2f}")
        print(f"  K/9: {away_pitcher['k_per_9_last_30']:.1f}")
        print(f"  BB/9: {away_pitcher['bb_per_9_last_30']:.1f}")
        print(f"  Score: {away_pitcher_score:.2f}")

        print(
            f"\nPitching Advantage: {'home' if advantage > 0 else 'away'} by {abs(advantage):.2f}"
        )

        return {
            "home_score": home_pitcher_score,
            "away_score": away_pitcher_score,
            "advantage": advantage,
            "confidence": self._calculate_pitching_confidence(
                home_pitcher, away_pitcher
            ),
        }

    def _analyze_batting_simple(self, game_data: dict) -> dict[str, Any]:
        """Simplified batting analysis focusing on key metrics."""

        home_lineup = game_data["home_lineup"]
        away_lineup = game_data["away_lineup"]

        print("\n=== SIMPLIFIED BATTING ANALYSIS ===")

        # Calculate lineup strength using recent performance
        home_batting_score = self._calculate_lineup_score(home_lineup, "home")
        away_batting_score = self._calculate_lineup_score(away_lineup, "away")

        advantage = home_batting_score - away_batting_score

        print("Home Team Batting:")
        print(f"  Recent wOBA: {np.mean([p['woba_last_30'] for p in home_lineup]):.3f}")
        print(f"  ISO: {np.mean([p['iso_last_30'] for p in home_lineup]):.3f}")
        print(f"  Score: {home_batting_score:.3f}")

        print("\nAway Team Batting:")
        print(f"  Recent wOBA: {np.mean([p['woba_last_30'] for p in away_lineup]):.3f}")
        print(f"  ISO: {np.mean([p['iso_last_30'] for p in away_lineup]):.3f}")
        print(f"  Score: {away_batting_score:.3f}")

        print(
            f"\nBatting Advantage: {'home' if advantage > 0 else 'away'} by {abs(advantage):.3f}"
        )

        return {
            "home_score": home_batting_score,
            "away_score": away_batting_score,
            "advantage": advantage,
            "confidence": self._calculate_batting_confidence(home_lineup, away_lineup),
        }

    def _analyze_situational_simple(self, game_data: dict) -> dict[str, Any]:
        """Simplified situational analysis."""

        print("\n=== SIMPLIFIED SITUATIONAL ANALYSIS ===")

        situational_factors = {
            "park_factor": game_data["park_factor"],
            "weather_impact": self._calculate_weather_impact_simple(
                game_data["weather"]
            ),
            "rest_advantage": self._calculate_rest_advantage_simple(game_data),
            "bullpen_advantage": self._calculate_bullpen_advantage_simple(game_data),
        }

        # Calculate overall situational advantage
        total_advantage = (
            (situational_factors["park_factor"] - 1.0) * 0.5
            + situational_factors["weather_impact"] * 0.3
            + (
                1
                if situational_factors["rest_advantage"] == "home"
                else -1 if situational_factors["rest_advantage"] == "away" else 0
            )
            * 0.1
            + (
                1
                if situational_factors["bullpen_advantage"]["winner"] == "home"
                else -1
            )
            * situational_factors["bullpen_advantage"]["advantage"]
            * 0.1
        )

        print(f"Park Factor: {situational_factors['park_factor']:.3f}")
        print(f"Weather Impact: {situational_factors['weather_impact']:.3f}")
        print(f"Rest Advantage: {situational_factors['rest_advantage']}")
        print(
            f"Bullpen Advantage: {situational_factors['bullpen_advantage']['winner']}"
        )
        print(f"Total Situational Advantage: {total_advantage:.3f}")

        return {
            "factors": situational_factors,
            "total_advantage": total_advantage,
            "confidence": 0.8,  # High confidence in situational factors
        }

    def _generate_conservative_predictions(
        self, pitching: dict, batting: dict, situational: dict
    ) -> dict[str, Any]:
        """Generate conservative predictions with uncertainty quantification."""

        print("\n=== CONSERVATIVE MODEL PREDICTIONS ===")

        # Base probability (50-50)
        base_prob = 0.5

        # Adjust for pitching advantage (weighted heavily)
        pitching_weight = 0.4
        pitching_adjustment = (pitching["advantage"] / 10) * pitching_weight

        # Adjust for batting advantage (moderate weight)
        batting_weight = 0.2
        batting_adjustment = (batting["advantage"] / 0.1) * batting_weight

        # Adjust for situational factors (light weight)
        situational_weight = 0.1
        situational_adjustment = situational["total_advantage"] * situational_weight

        # Calculate raw probability
        raw_prob = (
            base_prob
            + pitching_adjustment
            + batting_adjustment
            + situational_adjustment
        )

        # Apply conservative bounds
        home_win_prob = max(0.35, min(0.65, raw_prob))
        away_win_prob = 1 - home_win_prob

        # Calculate confidence based on agreement between factors
        confidence = self._calculate_overall_confidence(pitching, batting, situational)

        # Calculate expected runs (simplified)
        expected_runs = 9.0  # Base runs per game
        expected_runs += (
            pitching["home_score"] + pitching["away_score"] - 6.0
        ) * 0.5  # Pitching adjustment
        expected_runs += (
            batting["home_score"] + batting["away_score"] - 0.7
        ) * 10  # Batting adjustment
        expected_runs *= situational["factors"]["park_factor"]  # Park adjustment

        expected_runs = max(6.0, min(12.0, expected_runs))

        print(f"Home Win Probability: {home_win_prob:.1%}")
        print(f"Away Win Probability: {away_win_prob:.1%}")
        print(f"Expected Total Runs: {expected_runs:.1f}")
        print(f"Model Confidence: {confidence:.1%}")

        return {
            "home_win_probability": home_win_prob,
            "away_win_probability": away_win_prob,
            "expected_total_runs": expected_runs,
            "confidence": confidence,
        }

    def _evaluate_opportunities_conservative(
        self, predictions: dict, odds: dict
    ) -> list[dict]:
        """Evaluate betting opportunities with conservative criteria."""

        print("\n=== CONSERVATIVE OPPORTUNITY EVALUATION ===")

        opportunities = []

        # Evaluate moneyline bets
        if "moneyline" in odds:
            home_ml_ev = self._calculate_ev_conservative(
                predictions["home_win_probability"], odds["moneyline"]["home"]
            )
            away_ml_ev = self._calculate_ev_conservative(
                predictions["away_win_probability"], odds["moneyline"]["away"]
            )

            opportunities.extend(
                [
                    {
                        "bet_type": "moneyline",
                        "selection": "home",
                        "odds": odds["moneyline"]["home"],
                        "implied_prob": self._odds_to_prob(odds["moneyline"]["home"]),
                        "our_prob": predictions["home_win_probability"],
                        "expected_value": home_ml_ev,
                        "confidence": predictions["confidence"],
                        "kelly_stake": self._calculate_conservative_kelly(
                            predictions["home_win_probability"],
                            odds["moneyline"]["home"],
                        ),
                    },
                    {
                        "bet_type": "moneyline",
                        "selection": "away",
                        "odds": odds["moneyline"]["away"],
                        "implied_prob": self._odds_to_prob(odds["moneyline"]["away"]),
                        "our_prob": predictions["away_win_probability"],
                        "expected_value": away_ml_ev,
                        "confidence": predictions["confidence"],
                        "kelly_stake": self._calculate_conservative_kelly(
                            predictions["away_win_probability"],
                            odds["moneyline"]["away"],
                        ),
                    },
                ]
            )

        # Evaluate total runs (only if confidence is high)
        if "total" in odds and predictions["confidence"] > 0.8:
            over_ev = self._calculate_total_ev_conservative(
                predictions["expected_total_runs"], odds["total"]["line"], "over"
            )
            under_ev = self._calculate_total_ev_conservative(
                predictions["expected_total_runs"], odds["total"]["line"], "under"
            )

            opportunities.extend(
                [
                    {
                        "bet_type": "total",
                        "selection": "over",
                        "line": odds["total"]["line"],
                        "odds": odds["total"]["over"],
                        "expected_value": over_ev,
                        "confidence": predictions["confidence"]
                        * 0.9,  # Lower confidence for totals
                        "kelly_stake": self._calculate_conservative_kelly(
                            0.5, odds["total"]["over"]
                        ),
                    },
                    {
                        "bet_type": "total",
                        "selection": "under",
                        "line": odds["total"]["line"],
                        "odds": odds["total"]["under"],
                        "expected_value": under_ev,
                        "confidence": predictions["confidence"] * 0.9,
                        "kelly_stake": self._calculate_conservative_kelly(
                            0.5, odds["total"]["under"]
                        ),
                    },
                ]
            )

        # Sort by expected value
        opportunities.sort(key=lambda x: x["expected_value"], reverse=True)

        print("Betting Opportunities (ranked by EV):")
        for i, opp in enumerate(opportunities, 1):
            print(
                f"{i}. {opp['bet_type'].upper()} - {opp['selection']}: EV = {opp['expected_value']:.1%}, Kelly = {opp['kelly_stake']:.1%}, Confidence = {opp['confidence']:.1%}"
            )

        return opportunities

    def _apply_conservative_risk_management(
        self, opportunities: list[dict]
    ) -> list[dict]:
        """Apply conservative risk management rules."""

        print("\n=== CONSERVATIVE RISK MANAGEMENT ===")

        # Reset daily/weekly risk if needed
        self._reset_risk_tracking()

        recommendations = []

        for opp in opportunities:
            # Check minimum EV threshold
            if opp["expected_value"] < self.config["min_ev_threshold"]:
                print(
                    f"Rejected {opp['bet_type']} - {opp['selection']}: EV {opp['expected_value']:.1%} below threshold {self.config['min_ev_threshold']:.1%}"
                )
                continue

            # Check minimum confidence
            if opp["confidence"] < self.config["min_confidence"]:
                print(
                    f"Rejected {opp['bet_type']} - {opp['selection']}: Confidence {opp['confidence']:.1%} below threshold {self.config['min_confidence']:.1%}"
                )
                continue

            # Apply conservative Kelly with uncertainty adjustment
            kelly_stake = (
                opp["kelly_stake"]
                * self.config["kelly_fraction"]
                * self.config["uncertainty_factor"]
            )

            # Check maximum risk per bet
            if kelly_stake > self.config["max_risk_per_bet"]:
                kelly_stake = self.config["max_risk_per_bet"]

            # Check daily risk limit
            if self.daily_risk + kelly_stake > self.config["max_daily_risk"]:
                print(
                    f"Rejected {opp['bet_type']} - {opp['selection']}: Would exceed daily risk limit"
                )
                continue

            # Check weekly risk limit
            if self.weekly_risk + kelly_stake > self.config["max_weekly_risk"]:
                print(
                    f"Rejected {opp['bet_type']} - {opp['selection']}: Would exceed weekly risk limit"
                )
                continue

            # Only recommend if stake is meaningful
            if kelly_stake > 0.005:  # 0.5% minimum stake
                opp["recommended_stake"] = kelly_stake
                opp["stake_amount"] = kelly_stake * self.current_bankroll
                recommendations.append(opp)

                # Update risk tracking
                self.daily_risk += kelly_stake
                self.weekly_risk += kelly_stake

                print(
                    f"✅ Added {opp['bet_type']} - {opp['selection']}: Stake = {kelly_stake:.1%} (${opp['stake_amount']:.0f}), EV = {opp['expected_value']:.1%}"
                )
            else:
                print(
                    f"Rejected {opp['bet_type']} - {opp['selection']}: Stake {kelly_stake:.1%} too small"
                )

        print("\nRisk Summary:")
        print(f"Daily Risk Used: {self.daily_risk:.1%}")
        print(f"Weekly Risk Used: {self.weekly_risk:.1%}")

        return recommendations

    # Helper methods
    def _calculate_pitcher_score(self, pitcher: dict, location: str) -> float:
        """Calculate pitcher score based on recent performance."""
        # Base score from ERA
        era_score = 6.0 - pitcher["last_30_era"]  # Lower ERA = higher score

        # K/BB ratio bonus
        k_bb_ratio = pitcher["k_per_9_last_30"] / max(pitcher["bb_per_9_last_30"], 1.0)
        k_bb_bonus = min(k_bb_ratio - 2.0, 1.0)  # Cap at 1.0

        # Location adjustment
        location_adjustment = 0
        if location == "home" and "home_era" in pitcher:
            location_adjustment = (pitcher["era"] - pitcher["home_era"]) * 0.5
        elif location == "away" and "away_era" in pitcher:
            location_adjustment = (pitcher["era"] - pitcher["away_era"]) * 0.5

        return era_score + k_bb_bonus + location_adjustment

    def _calculate_lineup_score(self, lineup: list[dict], location: str) -> float:
        """Calculate lineup score based on recent performance."""
        # Average wOBA
        avg_woba = np.mean([p["woba_last_30"] for p in lineup])

        # ISO (power)
        avg_iso = np.mean([p["iso_last_30"] for p in lineup])

        # Location adjustment
        location_adjustment = 0
        if location == "home":
            avg_home_woba = np.mean(
                [p.get("home_woba", p["woba_last_30"]) for p in lineup]
            )
            location_adjustment = (avg_home_woba - avg_woba) * 0.5
        elif location == "away":
            avg_away_woba = np.mean(
                [p.get("away_woba", p["woba_last_30"]) for p in lineup]
            )
            location_adjustment = (avg_away_woba - avg_woba) * 0.5

        return avg_woba + avg_iso * 0.5 + location_adjustment

    def _calculate_weather_impact_simple(self, weather: dict) -> float:
        """Calculate simple weather impact."""
        temp_factor = (weather["temperature"] - 70) / 200  # Small impact
        wind_factor = weather["wind_speed"] / 200  # Small impact
        return temp_factor + wind_factor

    def _calculate_rest_advantage_simple(self, game_data: dict) -> str:
        """Calculate rest advantage."""
        home_rest = game_data["home_rest_days"]
        away_rest = game_data["away_rest_days"]

        if home_rest > away_rest + 1:
            return "home"
        elif away_rest > home_rest + 1:
            return "away"
        else:
            return "even"

    def _calculate_bullpen_advantage_simple(self, game_data: dict) -> dict:
        """Calculate bullpen advantage."""
        home_bullpen_era = game_data["home_bullpen_era"]
        away_bullpen_era = game_data["away_bullpen_era"]

        advantage = away_bullpen_era - home_bullpen_era

        if advantage > 0.5:
            return {"winner": "home", "advantage": advantage}
        elif advantage < -0.5:
            return {"winner": "away", "advantage": abs(advantage)}
        else:
            return {"winner": "even", "advantage": 0}

    def _calculate_pitching_confidence(
        self, home_pitcher: dict, away_pitcher: dict
    ) -> float:
        """Calculate confidence in pitching analysis."""
        # Higher confidence if both pitchers have good recent data
        home_confidence = min(
            home_pitcher["last_30_era"] / 6.0, 1.0
        )  # Lower ERA = higher confidence
        away_confidence = min(away_pitcher["last_30_era"] / 6.0, 1.0)

        return (home_confidence + away_confidence) / 2

    def _calculate_batting_confidence(
        self, home_lineup: list[dict], away_lineup: list[dict]
    ) -> float:
        """Calculate confidence in batting analysis."""
        # Higher confidence if lineups are strong
        home_avg_woba = np.mean([p["woba_last_30"] for p in home_lineup])
        away_avg_woba = np.mean([p["woba_last_30"] for p in away_lineup])

        # Normalize to 0-1 range
        home_confidence = min(home_avg_woba / 0.4, 1.0)
        away_confidence = min(away_avg_woba / 0.4, 1.0)

        return (home_confidence + away_confidence) / 2

    def _calculate_overall_confidence(
        self, pitching: dict, batting: dict, situational: dict
    ) -> float:
        """Calculate overall model confidence."""
        # Weighted average of component confidences
        pitching_weight = 0.5
        batting_weight = 0.3
        situational_weight = 0.2

        overall_confidence = (
            pitching["confidence"] * pitching_weight
            + batting["confidence"] * batting_weight
            + situational["confidence"] * situational_weight
        )

        return overall_confidence

    def _calculate_ev_conservative(self, our_prob: float, odds: float) -> float:
        """Calculate expected value with conservative adjustments."""
        # Standard EV calculation
        if odds > 0:
            ev = (our_prob * odds / 100) - (1 - our_prob)
        else:
            ev = (our_prob * 100 / abs(odds)) - (1 - our_prob)

        # Apply conservative adjustment (reduce EV by 20% to account for uncertainty)
        return ev * 0.8

    def _calculate_conservative_kelly(self, win_prob: float, odds: float) -> float:
        """Calculate conservative Kelly stake."""
        if odds > 0:
            b = odds / 100
        else:
            b = 100 / abs(odds)

        p = win_prob
        q = 1 - p

        kelly = (b * p - q) / b

        # Apply conservative adjustments
        kelly = (
            kelly * self.config["kelly_fraction"] * self.config["uncertainty_factor"]
        )

        return max(0, kelly)

    def _calculate_total_ev_conservative(
        self, expected_runs: float, line: float, direction: str
    ) -> float:
        """Calculate expected value for total bets with conservative approach."""
        # Simplified calculation with high uncertainty
        if direction == "over":
            prob_over = max(0.1, min(0.9, (expected_runs - line + 0.5) / 2))
            ev = prob_over - 0.5
        else:
            prob_under = max(0.1, min(0.9, (line - expected_runs + 0.5) / 2))
            ev = prob_under - 0.5

        # Apply conservative adjustment
        return ev * 0.6  # 40% reduction for totals uncertainty

    def _odds_to_prob(self, odds: float) -> float:
        """Convert American odds to probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)

    def _reset_risk_tracking(self):
        """Reset daily and weekly risk tracking."""
        current_date = datetime.now().date()

        # Reset daily risk
        if current_date != self.last_reset_date:
            self.daily_risk = 0
            self.last_reset_date = current_date

        # Reset weekly risk (simplified - in practice would track actual week)
        if self.weekly_risk > self.config["max_weekly_risk"]:
            self.weekly_risk = 0


# Example usage with realistic data
if __name__ == "__main__":
    # Sample game data with realistic values
    realistic_game = {
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "game_date": "2024-06-15",
        "park": "Yankee Stadium",
        "park_factor": 1.15,
        "weather": {
            "temperature": 75,
            "wind_speed": 8,
            "wind_direction": "out to right",
        },
        "home_rest_days": 1,
        "away_rest_days": 0,
        "home_bullpen_era": 3.45,
        "away_bullpen_era": 3.78,
        "home_pitcher": {
            "name": "Gerrit Cole",
            "era": 2.50,
            "last_30_era": 2.20,
            "k_per_9_last_30": 9.8,
            "bb_per_9_last_30": 2.1,
            "home_era": 2.30,
        },
        "away_pitcher": {
            "name": "Chris Sale",
            "era": 3.15,
            "last_30_era": 2.95,
            "k_per_9_last_30": 10.2,
            "bb_per_9_last_30": 2.8,
            "away_era": 3.25,
        },
        "home_lineup": [
            {"woba_last_30": 0.345, "iso_last_30": 0.180, "home_woba": 0.355},
            {"woba_last_30": 0.332, "iso_last_30": 0.165, "home_woba": 0.340},
            {"woba_last_30": 0.378, "iso_last_30": 0.220, "home_woba": 0.385},
            {"woba_last_30": 0.315, "iso_last_30": 0.140, "home_woba": 0.320},
            {"woba_last_30": 0.356, "iso_last_30": 0.190, "home_woba": 0.365},
            {"woba_last_30": 0.298, "iso_last_30": 0.120, "home_woba": 0.305},
            {"woba_last_30": 0.322, "iso_last_30": 0.150, "home_woba": 0.330},
            {"woba_last_30": 0.285, "iso_last_30": 0.110, "home_woba": 0.290},
            {"woba_last_30": 0.295, "iso_last_30": 0.115, "home_woba": 0.300},
        ],
        "away_lineup": [
            {"woba_last_30": 0.335, "iso_last_30": 0.170, "away_woba": 0.325},
            {"woba_last_30": 0.348, "iso_last_30": 0.185, "away_woba": 0.338},
            {"woba_last_30": 0.328, "iso_last_30": 0.160, "away_woba": 0.318},
            {"woba_last_30": 0.365, "iso_last_30": 0.210, "away_woba": 0.355},
            {"woba_last_30": 0.312, "iso_last_30": 0.145, "away_woba": 0.302},
            {"woba_last_30": 0.338, "iso_last_30": 0.175, "away_woba": 0.328},
            {"woba_last_30": 0.295, "iso_last_30": 0.125, "away_woba": 0.285},
            {"woba_last_30": 0.325, "iso_last_30": 0.155, "away_woba": 0.315},
            {"woba_last_30": 0.305, "iso_last_30": 0.135, "away_woba": 0.295},
        ],
        "odds": {
            "moneyline": {"home": -120, "away": +100},
            "total": {"line": 8.5, "over": -110, "under": -110},
        },
    }

    # Run conservative analysis
    analyzer = RealisticMLBBetting(initial_bankroll=10000)
    results = analyzer.analyze_game_conservative(realistic_game)

    print("\n" + "=" * 60)
    print("FINAL CONSERVATIVE RECOMMENDATIONS")
    print("=" * 60)

    if results["recommendations"]:
        total_stake = sum(
            rec["recommended_stake"] for rec in results["recommendations"]
        )
        total_amount = sum(rec["stake_amount"] for rec in results["recommendations"])

        print(
            f"Total Recommended Stake: {total_stake:.1%} of bankroll (${total_amount:.0f})"
        )
        print(f"Number of Bets: {len(results['recommendations'])}")
        print(
            f"Expected Portfolio EV: {np.mean([rec['expected_value'] for rec in results['recommendations']]):.1%}"
        )

        for rec in results["recommendations"]:
            print(
                f"✅ {rec['bet_type'].upper()} - {rec['selection']}: {rec['recommended_stake']:.1%} stake (${rec['stake_amount']:.0f}), {rec['expected_value']:.1%} EV"
            )
    else:
        print("❌ No betting opportunities meet conservative criteria")
        print("\nThis is expected in efficient markets - most games offer no value.")
        print("The strategy prioritizes quality over quantity.")

    print("\nRisk Management Status:")
    print(f"Daily Risk Used: {analyzer.daily_risk:.1%}")
    print(f"Weekly Risk Used: {analyzer.weekly_risk:.1%}")
    print(
        f"Remaining Daily Capacity: {(analyzer.config['max_daily_risk'] - analyzer.daily_risk):.1%}"
    )
    print(
        f"Remaining Weekly Capacity: {(analyzer.config['max_weekly_risk'] - analyzer.weekly_risk):.1%}"
    )
