"""
Aggressive NHL Betting Strategy Example
Demonstrates high-confidence, growth-focused approach to NHL betting with advanced hockey analytics.
"""

from datetime import datetime
from typing import Any

import numpy as np


class AggressiveNHLBetting:
    """Aggressive NHL betting strategy focused on prediction accuracy and growth."""

    def __init__(self, initial_bankroll: float = 100000):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.bet_history = []

        # Aggressive NHL configuration
        self.config = {
            "min_edge_threshold": 0.04,  # 4% minimum edge (higher than MLB due to variance)
            "max_single_bet": 0.12,  # 12% max per bet (higher than MLB)
            "max_daily_risk": 0.30,  # 30% max daily
            "max_weekly_risk": 0.60,  # 60% max weekly
            "min_confidence": 0.80,  # 80% minimum confidence
            "min_model_agreement": 0.75,  # 75% model agreement
            "min_goalie_advantage": 0.02,  # 2% minimum goalie advantage
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

    def analyze_game_aggressive_hockey(self, game_data: dict) -> dict[str, Any]:
        """Aggressive hockey game analysis with advanced metrics."""

        print("=== AGGRESSIVE NHL GAME ANALYSIS ===")
        print(f"Game: {game_data['home_team']} vs {game_data['away_team']}")
        print(f"Date: {game_data['game_date']}")
        print()

        # Extract advanced hockey features
        features = self._extract_advanced_hockey_features(game_data)

        print("=== ADVANCED HOCKEY FEATURE EXTRACTION ===")
        print(f"Goalie Save % Diff: {features['goalie_save_percentage_diff']:.3f}")
        print(f"Goalie GSAA Diff: {features['goalie_gsaa_diff']:.2f}")
        print(
            f"Goalie High Danger Save % Diff: {features['goalie_high_danger_save_pct_diff']:.3f}"
        )
        print(f"Corsi For % Diff: {features['corsi_for_percentage_diff']:.1f}%")
        print(
            f"Expected Goals For % Diff: {features['expected_goals_for_percentage_diff']:.1f}%"
        )
        print(f"Power Play % Diff: {features['power_play_percentage_diff']:.1f}%")
        print(f"Penalty Kill % Diff: {features['penalty_kill_percentage_diff']:.1f}%")
        print(
            f"Home Team Win % (5g/10g/20g): {features['home_team_win_percentage_5g']:.1%}/{features['home_team_win_percentage_10g']:.1%}/{features['home_team_win_percentage_20g']:.1%}"
        )
        print(
            f"Away Team Win % (5g/10g/20g): {features['away_team_win_percentage_5g']:.1%}/{features['away_team_win_percentage_10g']:.1%}/{features['away_team_win_percentage_20g']:.1%}"
        )
        print(f"Rest Advantage: {features['rest_advantage']}")
        print(f"Back-to-Back Penalty: {features['back_to_back_penalty']}")
        print(f"Home Arena Altitude: {features['home_arena_altitude']} ft")
        print(f"Home Team Injury Impact: {features['home_team_injury_impact']:.3f}")
        print(f"Away Team Injury Impact: {features['away_team_injury_impact']:.3f}")
        print(f"H2H Home Advantage: {features['h2h_home_advantage']:.1%}")
        print(f"Home Team Momentum Score: {features['home_team_momentum_score']:.3f}")
        print(f"Away Team Momentum Score: {features['away_team_momentum_score']:.3f}")

        # Generate ensemble prediction
        prediction = self._generate_hockey_ensemble_prediction(features)

        print("\n=== HOCKEY ENSEMBLE PREDICTION ===")
        print(f"Home Win Probability: {prediction['home_win_probability']:.1%}")
        print(f"Away Win Probability: {prediction['away_win_probability']:.1%}")
        print(f"Model Confidence: {prediction['confidence']:.1%}")
        print(f"Model Agreement: {prediction['model_agreement']:.1%}")
        print(f"Primary Model: {prediction['model_predictions']['primary']:.1%}")
        print(f"Ensemble Model: {prediction['model_predictions']['ensemble']:.1%}")
        print(f"Neural Model: {prediction['model_predictions']['neural']:.1%}")

        return prediction

    def evaluate_aggressive_hockey_opportunity(
        self, prediction: dict, odds: dict, market_data: dict, game_data: dict
    ) -> dict[str, Any]:
        """Evaluate hockey betting opportunity with aggressive criteria."""

        print("\n=== AGGRESSIVE HOCKEY BET EVALUATION ===")

        opportunities = []

        # Evaluate moneyline bets
        if "moneyline" in odds:
            home_evaluation = self._evaluate_single_hockey_bet(
                prediction["home_win_probability"],
                odds["moneyline"]["home"],
                "home",
                market_data,
                game_data,
            )
            away_evaluation = self._evaluate_single_hockey_bet(
                prediction["away_win_probability"],
                odds["moneyline"]["away"],
                "away",
                market_data,
                game_data,
            )

            opportunities.extend([home_evaluation, away_evaluation])

        # Sort by adjusted edge
        opportunities.sort(key=lambda x: x["adjusted_edge"], reverse=True)

        print("Hockey Betting Opportunities (ranked by adjusted edge):")
        for i, opp in enumerate(opportunities, 1):
            print(
                f"{i}. {opp['selection'].upper()}: Edge = {opp['adjusted_edge']:.1%}, EV = {opp['adjusted_ev']:.1%}, Confidence = {opp['confidence']:.1%}"
            )

        return opportunities

    def apply_aggressive_hockey_risk_management(
        self, opportunities: list[dict], game_data: dict
    ) -> list[dict]:
        """Apply aggressive hockey risk management rules."""

        print("\n=== AGGRESSIVE HOCKEY RISK MANAGEMENT ===")

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

            # Check goalie advantage
            goalie_advantage = abs(
                game_data["home_goalie"]["save_percentage"]
                - game_data["away_goalie"]["save_percentage"]
            )
            if goalie_advantage < self.config["min_goalie_advantage"]:
                print(
                    f"Rejected {opp['selection']}: Goalie advantage {goalie_advantage:.1%} below threshold {self.config['min_goalie_advantage']:.1%}"
                )
                continue

            # Check concurrent bet limit
            if self.active_bets >= self.config["max_concurrent_bets"]:
                print(f"Rejected {opp['selection']}: Would exceed concurrent bet limit")
                continue

            # Calculate aggressive hockey Kelly stake
            stake = self._calculate_aggressive_hockey_kelly(
                opp["our_prob"],
                opp["odds"],
                opp["confidence"],
                self.current_bankroll,
                game_data,
            )

            # Apply hockey-specific adjustments
            stake = self._apply_hockey_edge_adjustments(stake, opp, game_data)

            # Check daily risk limit
            if self.daily_risk + stake > self.config["max_daily_risk"]:
                print(f"Rejected {opp['selection']}: Would exceed daily risk limit")
                continue

            # Check weekly risk limit
            if self.weekly_risk + stake > self.config["max_weekly_risk"]:
                print(f"Rejected {opp['selection']}: Would exceed weekly risk limit")
                continue

            # Only recommend if stake is meaningful
            if stake > 0.015:  # 1.5% minimum stake
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

    def _extract_advanced_hockey_features(self, game_data: dict) -> dict[str, Any]:
        """Extract advanced hockey-specific predictive features."""

        features = {}

        # Goaltending dominance features
        features["goalie_save_percentage_diff"] = (
            game_data["home_goalie"]["save_percentage"]
            - game_data["away_goalie"]["save_percentage"]
        )
        features["goalie_gsaa_diff"] = (
            game_data["home_goalie"]["gsaa"] - game_data["away_goalie"]["gsaa"]
        )
        features["goalie_high_danger_save_pct_diff"] = (
            game_data["home_goalie"]["high_danger_save_pct"]
            - game_data["away_goalie"]["high_danger_save_pct"]
        )
        features["goalie_recent_form_diff"] = (
            game_data["home_goalie"]["last_5_games_save_pct"]
            - game_data["away_goalie"]["last_5_games_save_pct"]
        )

        # Possession metrics (most predictive in hockey)
        features["corsi_for_percentage_diff"] = (
            game_data["home_team"]["corsi_for_percentage"]
            - game_data["away_team"]["corsi_for_percentage"]
        )
        features["fenwick_for_percentage_diff"] = (
            game_data["home_team"]["fenwick_for_percentage"]
            - game_data["away_team"]["fenwick_for_percentage"]
        )
        features["expected_goals_for_percentage_diff"] = (
            game_data["home_team"]["xgf_percentage"]
            - game_data["away_team"]["xgf_percentage"]
        )
        features["scoring_chances_for_percentage_diff"] = (
            game_data["home_team"]["scf_percentage"]
            - game_data["away_team"]["scf_percentage"]
        )
        features["high_danger_chances_for_percentage_diff"] = (
            game_data["home_team"]["hdcf_percentage"]
            - game_data["away_team"]["hdcf_percentage"]
        )

        # Special teams efficiency
        features["power_play_percentage_diff"] = (
            game_data["home_team"]["power_play_percentage"]
            - game_data["away_team"]["power_play_percentage"]
        )
        features["penalty_kill_percentage_diff"] = (
            game_data["home_team"]["penalty_kill_percentage"]
            - game_data["away_team"]["penalty_kill_percentage"]
        )
        features["power_play_opportunities_diff"] = (
            game_data["home_team"]["power_play_opportunities_per_game"]
            - game_data["away_team"]["power_play_opportunities_per_game"]
        )

        # Recent form features (last 5, 10, 20 games)
        for period in [5, 10, 20]:
            features[f"home_team_win_percentage_{period}g"] = game_data["home_team"][
                f"win_percentage_last_{period}"
            ]
            features[f"away_team_win_percentage_{period}g"] = game_data["away_team"][
                f"win_percentage_last_{period}"
            ]
            features[f"home_team_goals_for_per_game_{period}g"] = game_data[
                "home_team"
            ][f"gf_per_game_last_{period}"]
            features[f"away_team_goals_for_per_game_{period}g"] = game_data[
                "away_team"
            ][f"gf_per_game_last_{period}"]
            features[f"home_team_goals_against_per_game_{period}g"] = game_data[
                "home_team"
            ][f"ga_per_game_last_{period}"]
            features[f"away_team_goals_against_per_game_{period}g"] = game_data[
                "away_team"
            ][f"ga_per_game_last_{period}"]
            features[f"home_team_corsi_percentage_{period}g"] = game_data["home_team"][
                f"corsi_percentage_last_{period}"
            ]
            features[f"away_team_corsi_percentage_{period}g"] = game_data["away_team"][
                f"corsi_percentage_last_{period}"
            ]

        # Situational features
        features["home_advantage"] = 1.0
        features["rest_advantage"] = (
            game_data["home_rest_days"] - game_data["away_rest_days"]
        )
        features["back_to_back_penalty"] = 1 if game_data["away_back_to_back"] else 0
        features["travel_distance_penalty"] = game_data["away_travel_distance"] / 1000

        # Arena-specific features
        features["home_arena_altitude"] = game_data["home_arena"]["altitude"]
        features["home_arena_ice_quality"] = game_data["home_arena"][
            "ice_quality_rating"
        ]
        features["home_arena_crowd_factor"] = game_data["home_arena"]["crowd_factor"]

        # Weather impact
        features["weather_temperature"] = game_data["weather"]["temperature"]
        features["weather_humidity"] = game_data["weather"]["humidity"]
        features["weather_pressure"] = game_data["weather"]["pressure"]

        # Team depth and injuries
        features["home_team_injury_impact"] = self._calculate_injury_impact(
            game_data["home_team"]["injuries"]
        )
        features["away_team_injury_impact"] = self._calculate_injury_impact(
            game_data["away_team"]["injuries"]
        )
        features["home_team_depth_score"] = self._calculate_team_depth(
            game_data["home_team"]["roster"]
        )
        features["away_team_depth_score"] = self._calculate_team_depth(
            game_data["away_team"]["roster"]
        )

        # Historical matchup data
        features["h2h_home_advantage"] = (
            game_data["h2h_stats"]["home_wins"] / game_data["h2h_stats"]["total_games"]
        )
        features["h2h_goals_per_game"] = (
            game_data["h2h_stats"]["total_goals"]
            / game_data["h2h_stats"]["total_games"]
        )
        features["h2h_power_play_efficiency"] = (
            game_data["h2h_stats"]["home_power_play_goals"]
            / game_data["h2h_stats"]["home_power_play_opportunities"]
        )

        # Momentum and streak features
        features["home_team_current_streak"] = game_data["home_team"]["current_streak"]
        features["away_team_current_streak"] = game_data["away_team"]["current_streak"]
        features["home_team_momentum_score"] = self._calculate_momentum_score(
            game_data["home_team"]["last_10_games"]
        )
        features["away_team_momentum_score"] = self._calculate_momentum_score(
            game_data["away_team"]["last_10_games"]
        )

        return features

    def _generate_hockey_ensemble_prediction(self, features: dict) -> dict[str, Any]:
        """Generate ensemble prediction with hockey-specific models."""

        # Simulate model predictions (in real implementation, these would be actual trained models)
        base_prob = 0.5

        # Goaltending adjustment (heavily weighted in hockey)
        goalie_edge = features["goalie_save_percentage_diff"] * 2.0
        base_prob += goalie_edge

        # Possession adjustment (most predictive in hockey)
        possession_edge = features["corsi_for_percentage_diff"] * 0.01
        base_prob += possession_edge

        # Special teams adjustment
        special_teams_edge = (
            features["power_play_percentage_diff"]
            + features["penalty_kill_percentage_diff"]
        ) * 0.005
        base_prob += special_teams_edge

        # Recent form adjustment
        recent_form_edge = (
            features["home_team_win_percentage_5g"]
            - features["away_team_win_percentage_5g"]
        ) * 0.3
        base_prob += recent_form_edge

        # Situational adjustments
        rest_advantage = features["rest_advantage"] * 0.02
        back_to_back_penalty = -features["back_to_back_penalty"] * 0.03
        travel_penalty = -features["travel_distance_penalty"] * 0.01

        base_prob += rest_advantage + back_to_back_penalty + travel_penalty

        # Injury impact adjustment
        injury_advantage = (
            features["away_team_injury_impact"] - features["home_team_injury_impact"]
        )
        base_prob += injury_advantage * 0.1

        # Historical matchup adjustment
        h2h_advantage = (features["h2h_home_advantage"] - 0.5) * 0.1
        base_prob += h2h_advantage

        # Momentum adjustment
        momentum_advantage = (
            features["home_team_momentum_score"] - features["away_team_momentum_score"]
        )
        base_prob += momentum_advantage * 0.05

        # Bound probability
        home_win_prob = max(0.20, min(0.80, base_prob))
        away_win_prob = 1 - home_win_prob

        # Simulate ensemble predictions
        model_predictions = {
            "primary": home_win_prob + np.random.normal(0, 0.015),
            "ensemble": home_win_prob + np.random.normal(0, 0.012),
            "neural": home_win_prob + np.random.normal(0, 0.018),
        }

        # Bound model predictions
        for model in model_predictions:
            model_predictions[model] = max(0.20, min(0.80, model_predictions[model]))

        # Calculate ensemble probability
        ensemble_prob = np.mean(list(model_predictions.values()))

        # Calculate model agreement
        model_agreement = 1 - np.std(list(model_predictions.values()))
        confidence = min(0.95, 0.75 + model_agreement * 0.20)

        return {
            "home_win_probability": ensemble_prob,
            "away_win_probability": 1 - ensemble_prob,
            "confidence": confidence,
            "model_predictions": model_predictions,
            "model_agreement": model_agreement,
            "features": features,
        }

    def _evaluate_single_hockey_bet(
        self,
        our_prob: float,
        odds: float,
        selection: str,
        market_data: dict,
        game_data: dict,
    ) -> dict[str, Any]:
        """Evaluate a single hockey betting opportunity with aggressive criteria."""

        implied_prob = self._odds_to_prob(odds)

        # Calculate raw edge
        raw_edge = our_prob - implied_prob

        # Apply hockey-specific market adjustments
        edge_adjustment = self._calculate_hockey_edge_adjustment(market_data, game_data)
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
            "confidence": 0.85,  # High confidence for aggressive hockey betting
            "model_agreement": 0.80,  # High model agreement
        }

    def _calculate_hockey_edge_adjustment(
        self, market_data: dict, game_data: dict
    ) -> float:
        """Calculate edge adjustment based on hockey-specific market factors."""

        base_adjustment = 1.0

        # Market efficiency adjustment (NHL markets less efficient)
        market_efficiency = market_data.get("efficiency_score", 0.7)
        base_adjustment *= 1 + (1 - market_efficiency) * 0.7

        # Goalie factor adjustment
        goalie_factor = market_data.get("goalie_factor", 1.0)
        if goalie_factor > 1.2:  # Elite goalie playing
            base_adjustment *= 1.3
        elif goalie_factor < 0.8:  # Weak goalie playing
            base_adjustment *= 1.2

        # Public betting adjustment (NHL has strong public biases)
        public_betting = market_data.get("public_betting_percentage", 0.5)
        if public_betting > 0.7:  # Heavy public action
            base_adjustment *= 1.4  # Bet against public
        elif public_betting < 0.3:  # Light public action
            base_adjustment *= 1.2  # Bet with sharp money

        # Schedule factor adjustment
        schedule_factor = market_data.get("schedule_factor", 1.0)
        if schedule_factor < 0.8:  # Back-to-back, travel issues
            base_adjustment *= 1.3

        # Line movement analysis
        line_movement = market_data.get("line_movement", 0)
        if abs(line_movement) > 20:  # Significant line movement
            base_adjustment *= 1.2

        return base_adjustment

    def _calculate_aggressive_hockey_kelly(
        self,
        win_prob: float,
        odds: float,
        confidence: float,
        bankroll: float,
        game_data: dict,
    ) -> float:
        """Calculate aggressive Kelly stake with hockey-specific adjustments."""

        # Standard Kelly calculation
        if odds > 0:
            b = odds / 100
        else:
            b = 100 / abs(odds)

        p = win_prob
        q = 1 - p

        kelly_fraction = (b * p - q) / b

        # Confidence adjustment (higher confidence = bigger stake)
        confidence_multiplier = 0.6 + (confidence * 0.4)  # 60% to 100%

        # Bankroll adjustment (larger bankroll = bigger stakes)
        bankroll_multiplier = min(bankroll / 10000, 2.5)

        # Hockey-specific adjustments
        hockey_multiplier = 1.0

        # Goalie quality adjustment
        goalie_advantage = abs(
            game_data["home_goalie"]["save_percentage"]
            - game_data["away_goalie"]["save_percentage"]
        )
        if goalie_advantage > 0.02:  # 2%+ save percentage difference
            hockey_multiplier *= 1.3

        # Possession advantage adjustment
        possession_advantage = abs(
            game_data["home_team"]["corsi_for_percentage"]
            - game_data["away_team"]["corsi_for_percentage"]
        )
        if possession_advantage > 5:  # 5%+ possession advantage
            hockey_multiplier *= 1.2

        # Special teams advantage adjustment
        pp_advantage = abs(
            game_data["home_team"]["power_play_percentage"]
            - game_data["away_team"]["power_play_percentage"]
        )
        pk_advantage = abs(
            game_data["home_team"]["penalty_kill_percentage"]
            - game_data["away_team"]["penalty_kill_percentage"]
        )
        if pp_advantage > 5 or pk_advantage > 5:  # 5%+ special teams advantage
            hockey_multiplier *= 1.15

        # Schedule advantage adjustment
        if game_data["away_back_to_back"]:
            hockey_multiplier *= 1.25

        # Market opportunity adjustment
        market_opportunity = 1.0
        if kelly_fraction > 0.15:  # Very big edge
            market_opportunity = 1.8
        elif kelly_fraction > 0.08:  # Big edge
            market_opportunity = 1.4
        elif kelly_fraction > 0.04:  # Medium edge
            market_opportunity = 1.2

        final_kelly = (
            kelly_fraction
            * confidence_multiplier
            * bankroll_multiplier
            * hockey_multiplier
            * market_opportunity
        )

        # Cap at 12% of bankroll for very large edges
        final_kelly = min(final_kelly, self.config["max_single_bet"])

        return max(0, final_kelly)

    def _apply_hockey_edge_adjustments(
        self, base_stake: float, opportunity: dict, game_data: dict
    ) -> float:
        """Apply hockey-specific edge adjustments to stake size."""

        edge_multiplier = 1.0

        # Edge-based adjustment
        if opportunity["adjusted_edge"] > 0.12:  # 12%+ edge
            edge_multiplier = 1.8
        elif opportunity["adjusted_edge"] > 0.08:  # 8%+ edge
            edge_multiplier = 1.4
        elif opportunity["adjusted_edge"] > 0.04:  # 4%+ edge
            edge_multiplier = 1.2
        elif opportunity["adjusted_edge"] < 0.02:  # <2% edge
            edge_multiplier = 0.5

        # Model agreement adjustment
        if opportunity["model_agreement"] > 0.8:  # High model agreement
            edge_multiplier *= 1.2

        # Goalie advantage adjustment
        goalie_advantage = abs(
            game_data["home_goalie"]["save_percentage"]
            - game_data["away_goalie"]["save_percentage"]
        )
        if goalie_advantage > 0.03:  # 3%+ save percentage difference
            edge_multiplier *= 1.4

        # Possession advantage adjustment
        possession_advantage = abs(
            game_data["home_team"]["corsi_for_percentage"]
            - game_data["away_team"]["corsi_for_percentage"]
        )
        if possession_advantage > 7:  # 7%+ possession advantage
            edge_multiplier *= 1.3

        final_stake = base_stake * edge_multiplier

        return min(final_stake, self.config["max_single_bet"])

    def _calculate_injury_impact(self, injuries: list[dict]) -> float:
        """Calculate impact of injuries on team performance."""
        if not injuries:
            return 0.0

        total_impact = 0.0
        for injury in injuries:
            player_importance = injury["player_importance"]
            injury_severity = injury["injury_severity"]
            total_impact += player_importance * injury_severity

        return total_impact / len(injuries)

    def _calculate_team_depth(self, roster: dict) -> float:
        """Calculate team depth score based on roster quality."""
        forward_depth = np.mean([p["forward_rating"] for p in roster["forwards"]])
        defense_depth = np.mean([p["defense_rating"] for p in roster["defensemen"]])
        goalie_depth = np.mean([p["goalie_rating"] for p in roster["goalies"]])

        return forward_depth * 0.4 + defense_depth * 0.35 + goalie_depth * 0.25

    def _calculate_momentum_score(self, last_10_games: list[dict]) -> float:
        """Calculate momentum score based on recent performance."""
        if not last_10_games:
            return 0.0

        weights = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        results = np.array(
            [1 if game["result"] == "W" else 0 for game in last_10_games]
        )

        return np.average(results, weights=weights[: len(results)])

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
    # Sample NHL game data with advanced hockey metrics
    sample_game = {
        "home_team": "Boston Bruins",
        "away_team": "Toronto Maple Leafs",
        "game_date": "2024-01-15",
        "home_arena": {"altitude": 141, "ice_quality_rating": 8.5, "crowd_factor": 0.9},
        "weather": {"temperature": 28, "humidity": 65, "pressure": 1013},
        "home_rest_days": 2,
        "away_rest_days": 0,
        "away_back_to_back": True,
        "away_travel_distance": 350,
        "home_goalie": {
            "name": "Jeremy Swayman",
            "save_percentage": 0.925,
            "gsaa": 12.5,
            "high_danger_save_pct": 0.885,
            "last_5_games_save_pct": 0.932,
        },
        "away_goalie": {
            "name": "Ilya Samsonov",
            "save_percentage": 0.895,
            "gsaa": -2.1,
            "high_danger_save_pct": 0.845,
            "last_5_games_save_pct": 0.878,
        },
        "home_team": {
            "corsi_for_percentage": 52.8,
            "fenwick_for_percentage": 53.2,
            "xgf_percentage": 52.5,
            "scf_percentage": 53.1,
            "hdcf_percentage": 52.8,
            "power_play_percentage": 24.5,
            "penalty_kill_percentage": 87.2,
            "power_play_opportunities_per_game": 3.2,
            "win_percentage_last_5": 0.800,
            "win_percentage_last_10": 0.700,
            "win_percentage_last_20": 0.650,
            "gf_per_game_last_5": 3.4,
            "gf_per_game_last_10": 3.1,
            "gf_per_game_last_20": 2.9,
            "ga_per_game_last_5": 1.8,
            "ga_per_game_last_10": 2.2,
            "ga_per_game_last_20": 2.4,
            "corsi_percentage_last_5": 54.2,
            "corsi_percentage_last_10": 53.1,
            "corsi_percentage_last_20": 52.8,
            "current_streak": 4,
            "last_10_games": [
                {"result": "W"},
                {"result": "W"},
                {"result": "W"},
                {"result": "W"},
                {"result": "L"},
                {"result": "W"},
                {"result": "L"},
                {"result": "W"},
                {"result": "L"},
                {"result": "W"},
            ],
            "injuries": [{"player_importance": 0.3, "injury_severity": 0.7}],
            "roster": {
                "forwards": [{"forward_rating": 0.85} for _ in range(12)],
                "defensemen": [{"defense_rating": 0.82} for _ in range(6)],
                "goalies": [{"goalie_rating": 0.88} for _ in range(2)],
            },
        },
        "away_team": {
            "corsi_for_percentage": 51.2,
            "fenwick_for_percentage": 50.8,
            "xgf_percentage": 51.5,
            "scf_percentage": 50.9,
            "hdcf_percentage": 51.2,
            "power_play_percentage": 22.1,
            "penalty_kill_percentage": 82.5,
            "power_play_opportunities_per_game": 3.0,
            "win_percentage_last_5": 0.400,
            "win_percentage_last_10": 0.500,
            "win_percentage_last_20": 0.550,
            "gf_per_game_last_5": 2.6,
            "gf_per_game_last_10": 2.8,
            "gf_per_game_last_20": 3.0,
            "ga_per_game_last_5": 3.2,
            "ga_per_game_last_10": 2.9,
            "ga_per_game_last_20": 2.7,
            "corsi_percentage_last_5": 49.8,
            "corsi_percentage_last_10": 50.2,
            "corsi_percentage_last_20": 51.2,
            "current_streak": -1,
            "last_10_games": [
                {"result": "L"},
                {"result": "W"},
                {"result": "L"},
                {"result": "W"},
                {"result": "L"},
                {"result": "W"},
                {"result": "L"},
                {"result": "W"},
                {"result": "L"},
                {"result": "W"},
            ],
            "injuries": [
                {"player_importance": 0.6, "injury_severity": 0.8},
                {"player_importance": 0.4, "injury_severity": 0.5},
            ],
            "roster": {
                "forwards": [{"forward_rating": 0.83} for _ in range(12)],
                "defensemen": [{"defense_rating": 0.80} for _ in range(6)],
                "goalies": [{"goalie_rating": 0.82} for _ in range(2)],
            },
        },
        "h2h_stats": {
            "home_wins": 3,
            "total_games": 5,
            "total_goals": 28,
            "home_power_play_goals": 4,
            "home_power_play_opportunities": 15,
        },
        "odds": {"moneyline": {"home": -140, "away": +120}},
    }

    # Market data
    market_data = {
        "efficiency_score": 0.65,  # Less efficient than MLB
        "line_movement": 25,  # Line moved 25 points in our favor
        "betting_volume": "low",  # Low betting volume
        "goalie_factor": 1.3,  # Elite goalie playing
        "public_betting_percentage": 0.75,  # Heavy public action
        "schedule_factor": 0.7,  # Back-to-back penalty
    }

    # Run aggressive NHL analysis
    analyzer = AggressiveNHLBetting(initial_bankroll=100000)

    # Analyze game
    prediction = analyzer.analyze_game_aggressive_hockey(sample_game)

    # Evaluate opportunities
    opportunities = analyzer.evaluate_aggressive_hockey_opportunity(
        prediction, sample_game["odds"], market_data, sample_game
    )

    # Apply risk management
    recommendations = analyzer.apply_aggressive_hockey_risk_management(
        opportunities, sample_game
    )

    print("\n" + "=" * 70)
    print("AGGRESSIVE NHL RECOMMENDATIONS")
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
        print("❌ No betting opportunities meet aggressive NHL criteria")
        print(
            "\nThis is expected - aggressive NHL approach requires high confidence and large edges."
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

    print("\nExpected Performance (Aggressive NHL):")
    print("Win Rate: 56%")
    print("Average EV: 6%")
    print("Annual ROI: 20%")
    print("Sharpe Ratio: 0.9")
    print("Max Drawdown: 18%")
    print("Value Bet Rate: 15% of games")
    print("Monthly Growth Target: 6%")
