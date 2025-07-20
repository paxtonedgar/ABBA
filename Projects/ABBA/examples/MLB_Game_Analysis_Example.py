"""
Practical Example: MLB Game Analysis and Betting Decision Process
Demonstrates how the ABBA system analyzes a real MLB game and determines betting opportunities.
"""

from typing import Any

import numpy as np


class MLBBettingAnalysis:
    """Example of how the system analyzes an MLB game for betting opportunities."""

    def __init__(self):
        self.config = {
            "min_ev_threshold": 0.02,  # 2% minimum expected value
            "max_risk_per_bet": 0.05,  # 5% max risk per bet
            "kelly_fraction": 0.25,  # Conservative Kelly fraction
            "min_confidence": 0.65,  # Minimum model confidence
        }

    def analyze_game(self, game_data: dict) -> dict[str, Any]:
        """
        Complete analysis of an MLB game for betting opportunities.

        Args:
            game_data: Dictionary containing game information, lineups, odds, etc.

        Returns:
            Dictionary with analysis results and betting recommendations
        """

        print("=== MLB GAME ANALYSIS ===")
        print(f"Game: {game_data['home_team']} vs {game_data['away_team']}")
        print(f"Date: {game_data['game_date']}")
        print(f"Park: {game_data['park']}")
        print(f"Weather: {game_data['weather']}")
        print()

        # Step 1: Collect and analyze data
        pitching_analysis = self._analyze_pitching_matchup(game_data)
        batting_analysis = self._analyze_batting_matchup(game_data)
        situational_analysis = self._analyze_situational_factors(game_data)

        # Step 2: Generate model predictions
        predictions = self._generate_predictions(
            pitching_analysis, batting_analysis, situational_analysis
        )

        # Step 3: Evaluate betting opportunities
        betting_opportunities = self._evaluate_betting_opportunities(
            predictions, game_data["odds"]
        )

        # Step 4: Apply risk management
        final_recommendations = self._apply_risk_management(betting_opportunities)

        return {
            "game_info": game_data,
            "pitching_analysis": pitching_analysis,
            "batting_analysis": batting_analysis,
            "situational_analysis": situational_analysis,
            "predictions": predictions,
            "betting_opportunities": betting_opportunities,
            "final_recommendations": final_recommendations,
        }

    def _analyze_pitching_matchup(self, game_data: dict) -> dict[str, Any]:
        """Analyze the pitching matchup using Statcast data."""

        home_pitcher = game_data["home_pitcher"]
        away_pitcher = game_data["away_pitcher"]

        print("=== PITCHING ANALYSIS ===")

        # Home pitcher analysis
        home_pitcher_stats = {
            "era": home_pitcher["era"],
            "avg_velocity": home_pitcher["avg_velocity"],
            "spin_rate": home_pitcher["avg_spin_rate"],
            "strikeout_rate": home_pitcher["k_per_9"],
            "walk_rate": home_pitcher["bb_per_9"],
            "home_era": home_pitcher["home_era"],
            "recent_form": home_pitcher["last_5_games_era"],
            "pitch_quality_score": self._calculate_pitch_quality(home_pitcher),
        }

        # Away pitcher analysis
        away_pitcher_stats = {
            "era": away_pitcher["era"],
            "avg_velocity": away_pitcher["avg_velocity"],
            "spin_rate": away_pitcher["avg_spin_rate"],
            "strikeout_rate": away_pitcher["k_per_9"],
            "walk_rate": away_pitcher["bb_per_9"],
            "away_era": away_pitcher["away_era"],
            "recent_form": away_pitcher["last_5_games_era"],
            "pitch_quality_score": self._calculate_pitch_quality(away_pitcher),
        }

        # Pitching advantage calculation
        pitching_advantage = self._calculate_pitching_advantage(
            home_pitcher_stats, away_pitcher_stats
        )

        print(f"Home Pitcher ({home_pitcher['name']}):")
        print(f"  ERA: {home_pitcher_stats['era']:.2f}")
        print(f"  Avg Velocity: {home_pitcher_stats['avg_velocity']:.1f} mph")
        print(f"  Recent Form: {home_pitcher_stats['recent_form']:.2f} ERA")
        print(f"  Pitch Quality Score: {home_pitcher_stats['pitch_quality_score']:.2f}")

        print(f"\nAway Pitcher ({away_pitcher['name']}):")
        print(f"  ERA: {away_pitcher_stats['era']:.2f}")
        print(f"  Avg Velocity: {away_pitcher_stats['avg_velocity']:.1f} mph")
        print(f"  Recent Form: {away_pitcher_stats['recent_form']:.2f} ERA")
        print(f"  Pitch Quality Score: {away_pitcher_stats['pitch_quality_score']:.2f}")

        print(
            f"\nPitching Advantage: {pitching_advantage['winner']} by {pitching_advantage['advantage']:.2f}"
        )

        return {
            "home_pitcher": home_pitcher_stats,
            "away_pitcher": away_pitcher_stats,
            "advantage": pitching_advantage,
        }

    def _analyze_batting_matchup(self, game_data: dict) -> dict[str, Any]:
        """Analyze the batting matchup using advanced metrics."""

        home_lineup = game_data["home_lineup"]
        away_lineup = game_data["away_lineup"]

        print("\n=== BATTING ANALYSIS ===")

        # Home team batting analysis
        home_batting_stats = {
            "avg_exit_velocity": np.mean([p["avg_exit_velocity"] for p in home_lineup]),
            "barrel_percentage": np.mean([p["barrel_percentage"] for p in home_lineup]),
            "hard_hit_percentage": np.mean(
                [p["hard_hit_percentage"] for p in home_lineup]
            ),
            "woba": np.mean([p["woba"] for p in home_lineup]),
            "home_woba": np.mean([p["home_woba"] for p in home_lineup]),
            "recent_form": np.mean([p["last_30_games_woba"] for p in home_lineup]),
        }

        # Away team batting analysis
        away_batting_stats = {
            "avg_exit_velocity": np.mean([p["avg_exit_velocity"] for p in away_lineup]),
            "barrel_percentage": np.mean([p["barrel_percentage"] for p in away_lineup]),
            "hard_hit_percentage": np.mean(
                [p["hard_hit_percentage"] for p in away_lineup]
            ),
            "woba": np.mean([p["woba"] for p in away_lineup]),
            "away_woba": np.mean([p["away_woba"] for p in away_lineup]),
            "recent_form": np.mean([p["last_30_games_woba"] for p in away_lineup]),
        }

        # Batting advantage calculation
        batting_advantage = self._calculate_batting_advantage(
            home_batting_stats, away_batting_stats
        )

        print("Home Team Batting:")
        print(f"  Avg Exit Velocity: {home_batting_stats['avg_exit_velocity']:.1f} mph")
        print(f"  Barrel %: {home_batting_stats['barrel_percentage']:.1f}%")
        print(f"  wOBA: {home_batting_stats['woba']:.3f}")
        print(f"  Recent Form: {home_batting_stats['recent_form']:.3f}")

        print("\nAway Team Batting:")
        print(f"  Avg Exit Velocity: {away_batting_stats['avg_exit_velocity']:.1f} mph")
        print(f"  Barrel %: {away_batting_stats['barrel_percentage']:.1f}%")
        print(f"  wOBA: {away_batting_stats['woba']:.3f}")
        print(f"  Recent Form: {away_batting_stats['recent_form']:.3f}")

        print(
            f"\nBatting Advantage: {batting_advantage['winner']} by {batting_advantage['advantage']:.3f}"
        )

        return {
            "home_batting": home_batting_stats,
            "away_batting": away_batting_stats,
            "advantage": batting_advantage,
        }

    def _analyze_situational_factors(self, game_data: dict) -> dict[str, Any]:
        """Analyze situational factors affecting the game."""

        print("\n=== SITUATIONAL ANALYSIS ===")

        situational_factors = {
            "park_factor": game_data["park_factor"],
            "weather_impact": self._calculate_weather_impact(game_data["weather"]),
            "rest_advantage": self._calculate_rest_advantage(game_data),
            "bullpen_strength": self._calculate_bullpen_advantage(game_data),
            "lineup_strength": self._calculate_lineup_strength(game_data),
        }

        print(f"Park Factor: {situational_factors['park_factor']:.3f}")
        print(f"Weather Impact: {situational_factors['weather_impact']:.3f}")
        print(f"Rest Advantage: {situational_factors['rest_advantage']}")
        print(f"Bullpen Advantage: {situational_factors['bullpen_strength']['winner']}")
        print(f"Lineup Strength: {situational_factors['lineup_strength']['winner']}")

        return situational_factors

    def _generate_predictions(
        self, pitching: dict, batting: dict, situational: dict
    ) -> dict[str, Any]:
        """Generate model predictions for the game."""

        print("\n=== MODEL PREDICTIONS ===")

        # Combine all factors to generate predictions
        home_win_prob = self._calculate_win_probability(
            pitching["advantage"], batting["advantage"], situational
        )

        away_win_prob = 1 - home_win_prob

        # Calculate expected runs
        expected_runs = self._calculate_expected_runs(
            pitching["home_pitcher"],
            pitching["away_pitcher"],
            batting["home_batting"],
            batting["away_batting"],
            situational,
        )

        predictions = {
            "home_win_probability": home_win_prob,
            "away_win_probability": away_win_prob,
            "expected_total_runs": expected_runs,
            "confidence": self._calculate_confidence(pitching, batting, situational),
        }

        print(f"Home Win Probability: {home_win_prob:.1%}")
        print(f"Away Win Probability: {away_win_prob:.1%}")
        print(f"Expected Total Runs: {expected_runs:.1f}")
        print(f"Model Confidence: {predictions['confidence']:.1%}")

        return predictions

    def _evaluate_betting_opportunities(
        self, predictions: dict, odds: dict
    ) -> list[dict]:
        """Evaluate all available betting opportunities."""

        print("\n=== BETTING OPPORTUNITY EVALUATION ===")

        opportunities = []

        # Evaluate moneyline bets
        if "moneyline" in odds:
            home_ml_ev = self._calculate_ev(
                predictions["home_win_probability"], odds["moneyline"]["home"]
            )
            away_ml_ev = self._calculate_ev(
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
                        "kelly_stake": self._calculate_kelly_stake(
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
                        "kelly_stake": self._calculate_kelly_stake(
                            predictions["away_win_probability"],
                            odds["moneyline"]["away"],
                        ),
                    },
                ]
            )

        # Evaluate run line bets
        if "run_line" in odds:
            home_rl_ev = self._calculate_run_line_ev(
                predictions, odds["run_line"]["home"]
            )
            away_rl_ev = self._calculate_run_line_ev(
                predictions, odds["run_line"]["away"]
            )

            opportunities.extend(
                [
                    {
                        "bet_type": "run_line",
                        "selection": "home",
                        "odds": odds["run_line"]["home"],
                        "expected_value": home_rl_ev,
                        "kelly_stake": self._calculate_kelly_stake(
                            0.5, odds["run_line"]["home"]
                        ),
                    },
                    {
                        "bet_type": "run_line",
                        "selection": "away",
                        "odds": odds["run_line"]["away"],
                        "expected_value": away_rl_ev,
                        "kelly_stake": self._calculate_kelly_stake(
                            0.5, odds["run_line"]["away"]
                        ),
                    },
                ]
            )

        # Evaluate total runs bets
        if "total" in odds:
            over_ev = self._calculate_total_ev(
                predictions["expected_total_runs"], odds["total"]["line"], "over"
            )
            under_ev = self._calculate_total_ev(
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
                        "kelly_stake": self._calculate_kelly_stake(
                            0.5, odds["total"]["over"]
                        ),
                    },
                    {
                        "bet_type": "total",
                        "selection": "under",
                        "line": odds["total"]["line"],
                        "odds": odds["total"]["under"],
                        "expected_value": under_ev,
                        "kelly_stake": self._calculate_kelly_stake(
                            0.5, odds["total"]["under"]
                        ),
                    },
                ]
            )

        # Sort opportunities by expected value
        opportunities.sort(key=lambda x: x["expected_value"], reverse=True)

        print("Betting Opportunities (ranked by EV):")
        for i, opp in enumerate(opportunities, 1):
            print(
                f"{i}. {opp['bet_type'].upper()} - {opp['selection']}: EV = {opp['expected_value']:.1%}, Kelly = {opp['kelly_stake']:.1%}"
            )

        return opportunities

    def _apply_risk_management(self, opportunities: list[dict]) -> list[dict]:
        """Apply risk management rules to filter and size bets."""

        print("\n=== RISK MANAGEMENT ===")

        recommendations = []

        for opp in opportunities:
            # Check minimum EV threshold
            if opp["expected_value"] < self.config["min_ev_threshold"]:
                continue

            # Check minimum confidence
            if opp.get("confidence", 0.5) < self.config["min_confidence"]:
                continue

            # Apply Kelly Criterion with fractional Kelly
            kelly_stake = opp["kelly_stake"] * self.config["kelly_fraction"]

            # Check maximum risk per bet
            if kelly_stake > self.config["max_risk_per_bet"]:
                kelly_stake = self.config["max_risk_per_bet"]

            # Only recommend if stake is meaningful
            if kelly_stake > 0.005:  # 0.5% minimum stake
                opp["recommended_stake"] = kelly_stake
                recommendations.append(opp)
                print(
                    f"Debug: Added {opp['bet_type']} - {opp['selection']} with stake {kelly_stake:.1%}"
                )
            else:
                print(
                    f"Debug: Rejected {opp['bet_type']} - {opp['selection']} due to low stake {kelly_stake:.1%}"
                )

        print("Final Recommendations:")
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(
                    f"{i}. {rec['bet_type'].upper()} - {rec['selection']}: Stake = {rec['recommended_stake']:.1%}, EV = {rec['expected_value']:.1%}"
                )
        else:
            print("No recommendations meet criteria")
            print(f"Debug: EV threshold = {self.config['min_ev_threshold']:.1%}")
            print(f"Debug: Min confidence = {self.config['min_confidence']:.1%}")
            for opp in opportunities[:3]:  # Show first 3 opportunities
                print(
                    f"Debug: {opp['bet_type']} - {opp['selection']}: EV = {opp['expected_value']:.1%}, Kelly = {opp['kelly_stake']:.1%}"
                )

        return recommendations

    # Helper methods for calculations
    def _calculate_pitch_quality(self, pitcher: dict) -> float:
        """Calculate composite pitch quality score."""
        # Simplified calculation - in practice would use more sophisticated metrics
        return (
            (100 - pitcher["era"] * 10)
            + (pitcher["avg_velocity"] - 90)
            + (pitcher["k_per_9"] - 7)
        )

    def _calculate_pitching_advantage(self, home: dict, away: dict) -> dict:
        """Calculate pitching advantage between teams."""
        home_score = home["pitch_quality_score"]
        away_score = away["pitch_quality_score"]

        if home_score > away_score:
            return {"winner": "home", "advantage": home_score - away_score}
        else:
            return {"winner": "away", "advantage": away_score - home_score}

    def _calculate_batting_advantage(self, home: dict, away: dict) -> dict:
        """Calculate batting advantage between teams."""
        home_score = home["woba"] * 1000  # Scale for comparison
        away_score = away["woba"] * 1000

        if home_score > away_score:
            return {"winner": "home", "advantage": home_score - away_score}
        else:
            return {"winner": "away", "advantage": away_score - home_score}

    def _calculate_weather_impact(self, weather: dict) -> float:
        """Calculate weather impact on scoring."""
        # Simplified - in practice would use more sophisticated weather models
        temp_factor = (weather["temperature"] - 70) / 100
        wind_factor = weather["wind_speed"] / 100
        return temp_factor + wind_factor

    def _calculate_rest_advantage(self, game_data: dict) -> str:
        """Calculate rest advantage between teams."""
        home_rest = game_data["home_rest_days"]
        away_rest = game_data["away_rest_days"]

        if home_rest > away_rest:
            return "home"
        elif away_rest > home_rest:
            return "away"
        else:
            return "even"

    def _calculate_bullpen_advantage(self, game_data: dict) -> dict:
        """Calculate bullpen advantage."""
        home_bullpen_era = game_data["home_bullpen_era"]
        away_bullpen_era = game_data["away_bullpen_era"]

        if home_bullpen_era < away_bullpen_era:
            return {"winner": "home", "advantage": away_bullpen_era - home_bullpen_era}
        else:
            return {"winner": "away", "advantage": home_bullpen_era - away_bullpen_era}

    def _calculate_lineup_strength(self, game_data: dict) -> dict:
        """Calculate lineup strength advantage."""
        home_woba = np.mean([p["woba"] for p in game_data["home_lineup"]])
        away_woba = np.mean([p["woba"] for p in game_data["away_lineup"]])

        if home_woba > away_woba:
            return {"winner": "home", "advantage": home_woba - away_woba}
        else:
            return {"winner": "away", "advantage": away_woba - home_woba}

    def _calculate_win_probability(
        self, pitching: dict, batting: dict, situational: dict
    ) -> float:
        """Calculate win probability based on all factors."""
        # Simplified calculation - in practice would use trained ML model
        base_prob = 0.5

        # Adjust for pitching advantage (more weight on pitching)
        if pitching["winner"] == "home":
            base_prob += pitching["advantage"] * 0.02
        else:
            base_prob -= pitching["advantage"] * 0.02

        # Adjust for batting advantage (less weight on batting)
        if batting["winner"] == "home":
            base_prob += batting["advantage"] * 0.05
        else:
            base_prob -= batting["advantage"] * 0.05

        # Adjust for situational factors
        base_prob += situational["weather_impact"] * 0.02

        # Adjust for rest advantage
        if situational["rest_advantage"] == "home":
            base_prob += 0.02
        elif situational["rest_advantage"] == "away":
            base_prob -= 0.02

        return max(0.1, min(0.9, base_prob))

    def _calculate_expected_runs(
        self,
        home_pitcher: dict,
        away_pitcher: dict,
        home_batting: dict,
        away_batting: dict,
        situational: dict,
    ) -> float:
        """Calculate expected total runs."""
        # Simplified calculation
        avg_runs_per_game = 9.0

        # Adjust for pitching
        pitching_factor = (home_pitcher["era"] + away_pitcher["era"]) / 2 / 4.0

        # Adjust for batting
        batting_factor = (home_batting["woba"] + away_batting["woba"]) / 2 / 0.320

        # Adjust for park and weather
        park_factor = situational["park_factor"]
        weather_factor = 1 + situational["weather_impact"]

        expected_runs = (
            avg_runs_per_game
            * pitching_factor
            * batting_factor
            * park_factor
            * weather_factor
        )

        return max(3.0, min(15.0, expected_runs))

    def _calculate_confidence(
        self, pitching: dict, batting: dict, situational: dict
    ) -> float:
        """Calculate model confidence in predictions."""
        # Simplified confidence calculation
        confidence = 0.7  # Base confidence

        # Higher confidence with larger advantages
        if pitching["advantage"]["advantage"] > 5:
            confidence += 0.1
        if batting["advantage"]["advantage"] > 0.05:
            confidence += 0.1

        return min(0.95, confidence)

    def _calculate_ev(self, our_prob: float, odds: float) -> float:
        """Calculate expected value."""
        if odds > 0:
            return (our_prob * odds / 100) - (1 - our_prob)
        else:
            return (our_prob * 100 / abs(odds)) - (1 - our_prob)

    def _odds_to_prob(self, odds: float) -> float:
        """Convert American odds to probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)

    def _calculate_kelly_stake(self, win_prob: float, odds: float) -> float:
        """Calculate Kelly Criterion stake."""
        if odds > 0:
            b = odds / 100
        else:
            b = 100 / abs(odds)

        p = win_prob
        q = 1 - p

        kelly = (b * p - q) / b
        return max(0, kelly)

    def _calculate_run_line_ev(self, predictions: dict, odds: float) -> float:
        """Calculate expected value for run line bets."""
        # Simplified - assumes 50% probability for run line
        return self._calculate_ev(0.5, odds)

    def _calculate_total_ev(
        self, expected_runs: float, line: float, direction: str
    ) -> float:
        """Calculate expected value for total bets."""
        # Simplified calculation
        if direction == "over":
            prob_over = max(0.1, min(0.9, (expected_runs - line + 0.5) / 2))
            return prob_over - 0.5
        else:
            prob_under = max(0.1, min(0.9, (line - expected_runs + 0.5) / 2))
            return prob_under - 0.5


# Example usage
if __name__ == "__main__":
    # Sample game data
    sample_game = {
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "game_date": "2024-06-15",
        "park": "Yankee Stadium",
        "park_factor": 1.15,  # HR-friendly park
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
            "avg_velocity": 95.2,
            "avg_spin_rate": 2200,
            "k_per_9": 9.8,
            "bb_per_9": 2.1,
            "home_era": 2.30,
            "last_5_games_era": 2.20,
        },
        "away_pitcher": {
            "name": "Chris Sale",
            "era": 3.15,
            "avg_velocity": 94.8,
            "avg_spin_rate": 2250,
            "k_per_9": 10.2,
            "bb_per_9": 2.8,
            "away_era": 3.25,
            "last_5_games_era": 2.95,
        },
        "home_lineup": [
            {
                "avg_exit_velocity": 89.5,
                "barrel_percentage": 8.2,
                "hard_hit_percentage": 42.1,
                "woba": 0.345,
                "home_woba": 0.355,
                "last_30_games_woba": 0.350,
            },
            {
                "avg_exit_velocity": 88.2,
                "barrel_percentage": 7.8,
                "hard_hit_percentage": 38.9,
                "woba": 0.332,
                "home_woba": 0.340,
                "last_30_games_woba": 0.335,
            },
            {
                "avg_exit_velocity": 91.1,
                "barrel_percentage": 9.1,
                "hard_hit_percentage": 45.2,
                "woba": 0.378,
                "home_woba": 0.385,
                "last_30_games_woba": 0.380,
            },
            {
                "avg_exit_velocity": 87.8,
                "barrel_percentage": 6.9,
                "hard_hit_percentage": 36.7,
                "woba": 0.315,
                "home_woba": 0.320,
                "last_30_games_woba": 0.318,
            },
            {
                "avg_exit_velocity": 90.3,
                "barrel_percentage": 8.5,
                "hard_hit_percentage": 41.8,
                "woba": 0.356,
                "home_woba": 0.365,
                "last_30_games_woba": 0.360,
            },
            {
                "avg_exit_velocity": 86.9,
                "barrel_percentage": 6.2,
                "hard_hit_percentage": 35.1,
                "woba": 0.298,
                "home_woba": 0.305,
                "last_30_games_woba": 0.302,
            },
            {
                "avg_exit_velocity": 88.7,
                "barrel_percentage": 7.1,
                "hard_hit_percentage": 37.8,
                "woba": 0.322,
                "home_woba": 0.330,
                "last_30_games_woba": 0.325,
            },
            {
                "avg_exit_velocity": 85.4,
                "barrel_percentage": 5.8,
                "hard_hit_percentage": 33.2,
                "woba": 0.285,
                "home_woba": 0.290,
                "last_30_games_woba": 0.288,
            },
            {
                "avg_exit_velocity": 87.1,
                "barrel_percentage": 6.5,
                "hard_hit_percentage": 34.9,
                "woba": 0.295,
                "home_woba": 0.300,
                "last_30_games_woba": 0.298,
            },
        ],
        "away_lineup": [
            {
                "avg_exit_velocity": 88.1,
                "barrel_percentage": 7.5,
                "hard_hit_percentage": 39.8,
                "woba": 0.335,
                "away_woba": 0.325,
                "last_30_games_woba": 0.330,
            },
            {
                "avg_exit_velocity": 89.2,
                "barrel_percentage": 8.1,
                "hard_hit_percentage": 41.2,
                "woba": 0.348,
                "away_woba": 0.338,
                "last_30_games_woba": 0.342,
            },
            {
                "avg_exit_velocity": 87.5,
                "barrel_percentage": 7.2,
                "hard_hit_percentage": 38.1,
                "woba": 0.328,
                "away_woba": 0.318,
                "last_30_games_woba": 0.322,
            },
            {
                "avg_exit_velocity": 90.8,
                "barrel_percentage": 8.8,
                "hard_hit_percentage": 43.5,
                "woba": 0.365,
                "away_woba": 0.355,
                "last_30_games_woba": 0.360,
            },
            {
                "avg_exit_velocity": 86.3,
                "barrel_percentage": 6.8,
                "hard_hit_percentage": 36.2,
                "woba": 0.312,
                "away_woba": 0.302,
                "last_30_games_woba": 0.308,
            },
            {
                "avg_exit_velocity": 88.9,
                "barrel_percentage": 7.6,
                "hard_hit_percentage": 39.5,
                "woba": 0.338,
                "away_woba": 0.328,
                "last_30_games_woba": 0.332,
            },
            {
                "avg_exit_velocity": 85.7,
                "barrel_percentage": 6.1,
                "hard_hit_percentage": 34.8,
                "woba": 0.295,
                "away_woba": 0.285,
                "last_30_games_woba": 0.290,
            },
            {
                "avg_exit_velocity": 87.8,
                "barrel_percentage": 7.3,
                "hard_hit_percentage": 38.7,
                "woba": 0.325,
                "away_woba": 0.315,
                "last_30_games_woba": 0.320,
            },
            {
                "avg_exit_velocity": 86.1,
                "barrel_percentage": 6.4,
                "hard_hit_percentage": 35.6,
                "woba": 0.305,
                "away_woba": 0.295,
                "last_30_games_woba": 0.300,
            },
        ],
        "odds": {
            "moneyline": {"home": -130, "away": +110},
            "run_line": {"home": -110, "away": -110},
            "total": {"line": 8.5, "over": -110, "under": -110},
        },
    }

    # Run analysis
    analyzer = MLBBettingAnalysis()
    results = analyzer.analyze_game(sample_game)

    print("\n" + "=" * 50)
    print("FINAL RECOMMENDATIONS SUMMARY")
    print("=" * 50)

    if results["final_recommendations"]:
        total_stake = sum(
            rec["recommended_stake"] for rec in results["final_recommendations"]
        )
        print(f"Total Recommended Stake: {total_stake:.1%} of bankroll")
        print(f"Number of Bets: {len(results['final_recommendations'])}")

        for rec in results["final_recommendations"]:
            print(
                f"✅ {rec['bet_type'].upper()} - {rec['selection']}: {rec['recommended_stake']:.1%} stake, {rec['expected_value']:.1%} EV"
            )
    else:
        print("❌ No betting opportunities meet our criteria")
        print("This could be due to:")
        print("- Insufficient expected value")
        print("- Low model confidence")
        print("- Risk management constraints")
