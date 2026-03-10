"""Tests for engine components -- math correctness is critical here."""

import math

import numpy as np
import pytest

from abba.engine.ensemble import EnsembleEngine
from abba.engine.features import FeatureEngine
from abba.engine.graph import GraphEngine
from abba.engine.kelly import KellyEngine
from abba.engine.value import ValueEngine


class TestEnsembleEngine:
    def setup_method(self):
        self.engine = EnsembleEngine()

    def test_average_combine(self):
        result = self.engine.combine([0.6, 0.7, 0.65], method="average")
        assert abs(result.value - 0.65) < 0.001
        assert result.model_count == 3

    def test_median_combine(self):
        result = self.engine.combine([0.4, 0.9, 0.6], method="median")
        assert abs(result.value - 0.6) < 0.001

    def test_voting_combine(self):
        # 3 out of 4 predict > 0.5
        result = self.engine.combine([0.6, 0.7, 0.3, 0.8], method="voting")
        assert abs(result.value - 0.75) < 0.001  # 3/4

    def test_weighted_prefers_consensus(self):
        """Weighted should favor predictions closer to the group mean."""
        result = self.engine.combine([0.60, 0.62, 0.61, 0.90], method="weighted")
        # The outlier (0.90) should pull less weight
        assert result.value < 0.70  # closer to the cluster than to the outlier

    def test_explicit_weights(self):
        result = self.engine.combine([0.4, 0.8], method="weighted", weights=[3.0, 1.0])
        # 3/4 * 0.4 + 1/4 * 0.8 = 0.5
        assert abs(result.value - 0.5) < 0.001

    def test_confidence_high_agreement(self):
        result = self.engine.combine([0.60, 0.61, 0.59, 0.60])
        assert result.confidence > 0.8

    def test_confidence_low_agreement(self):
        result = self.engine.combine([0.2, 0.8, 0.3, 0.9])
        assert result.confidence < 0.5

    def test_error_margin_shrinks_with_more_models(self):
        r3 = self.engine.combine([0.55, 0.60, 0.57])
        r6 = self.engine.combine([0.55, 0.60, 0.57, 0.56, 0.59, 0.58])
        assert r6.error_margin < r3.error_margin

    def test_empty_predictions(self):
        result = self.engine.combine([])
        assert result.value == 0.0
        assert result.model_count == 0

    def test_single_prediction(self):
        result = self.engine.combine([0.65])
        assert result.value == 0.65
        assert result.model_count == 1

    def test_validate_good_ensemble(self):
        v = self.engine.validate([0.55, 0.58, 0.56, 0.57])
        assert v["valid"] is True

    def test_validate_bad_ensemble(self):
        v = self.engine.validate([0.1, 0.9, 0.2, 0.8])
        assert v["valid"] is False


class TestKellyEngine:
    def setup_method(self):
        self.kelly = KellyEngine(kelly_fraction=0.5, max_bet_pct=0.05, min_edge=0.02, min_ev=0.03)

    def test_basic_kelly(self):
        """Fair coin at 2.0 odds: full Kelly = 0.0, so half-Kelly = 0.0 (no edge)."""
        result = self.kelly.calculate(0.50, 2.0, 10000)
        assert result.fraction == 0.0  # No edge, no bet

    def test_positive_ev(self):
        """60% win prob at 2.0 odds: edge = 0.10, EV = 0.20."""
        result = self.kelly.calculate(0.60, 2.0, 10000)
        # Full Kelly: (1*0.6 - 0.4)/1 = 0.20
        # Half Kelly: 0.10
        # But capped at 0.05
        assert result.fraction == 0.05  # hit the cap
        assert result.recommended_stake == 500.0
        assert result.expected_value > 0

    def test_no_bet_below_min_edge(self):
        """Edge below min_edge threshold: no bet."""
        # 51% at 2.0: edge = 0.01 (below min_edge=0.02), EV = 0.02 (below min_ev=0.03)
        result = self.kelly.calculate(0.51, 2.0, 10000)
        assert result.fraction == 0.0

    def test_negative_ev_no_bet(self):
        """40% at 2.0 odds: negative EV, should never bet."""
        result = self.kelly.calculate(0.40, 2.0, 10000)
        assert result.fraction == 0.0
        assert result.expected_value < 0

    def test_high_odds_high_prob(self):
        """Strong edge: 70% at 2.5 odds."""
        result = self.kelly.calculate(0.70, 2.5, 10000)
        # Full Kelly: (1.5*0.7 - 0.3)/1.5 = (1.05-0.3)/1.5 = 0.50
        # Half Kelly: 0.25
        # Capped at 0.05
        assert result.fraction == 0.05
        assert result.expected_value > 0
        assert result.edge > 0.2

    def test_american_to_decimal(self):
        assert abs(self.kelly.american_to_decimal(150) - 2.50) < 0.01
        assert abs(self.kelly.american_to_decimal(-150) - 1.667) < 0.01
        assert abs(self.kelly.american_to_decimal(100) - 2.0) < 0.01
        assert abs(self.kelly.american_to_decimal(-100) - 2.0) < 0.01

    def test_implied_probability(self):
        assert abs(self.kelly.implied_probability(2.0) - 0.5) < 0.001
        assert abs(self.kelly.implied_probability(3.0) - 0.333) < 0.001


class TestValueEngine:
    def setup_method(self):
        self.value = ValueEngine(min_ev=0.03, min_edge=0.02)

    def test_find_value_basic(self):
        games = [{"game_id": "g1", "home_team": "NYY", "away_team": "BOS"}]
        predictions = {"g1": 0.65}  # We think home wins 65%
        odds = [{
            "game_id": "g1", "sportsbook": "DK",
            "home_odds": 2.10,  # implied 47.6%
            "away_odds": 1.75,  # implied 57.1%
        }]
        opps = self.value.find_value(games, predictions, odds)
        # Home bet: prob=0.65, odds=2.10, EV=0.65*1.10-0.35=0.365 (big edge)
        assert len(opps) > 0
        home_opp = next((o for o in opps if o["selection"] == "home"), None)
        assert home_opp is not None
        assert home_opp["expected_value"] > 0.3

    def test_no_value_when_odds_fair(self):
        games = [{"game_id": "g1", "home_team": "NYY", "away_team": "BOS"}]
        predictions = {"g1": 0.50}
        odds = [{
            "game_id": "g1", "sportsbook": "DK",
            "home_odds": 1.91,  # implied 52.4% (with vig)
            "away_odds": 1.91,
        }]
        opps = self.value.find_value(games, predictions, odds)
        # 50% prob at 1.91 odds: EV = 0.5*0.91 - 0.5 = -0.045 (negative)
        assert len(opps) == 0

    def test_ev_calculation_math(self):
        """EV = p * (odds - 1) - (1 - p)"""
        ev = self.value._calculate_ev(0.60, 2.0)
        expected = 0.60 * 1.0 - 0.40  # = 0.20
        assert abs(ev - expected) < 0.001

    def test_compare_odds(self):
        odds = [
            {"game_id": "g1", "sportsbook": "DK", "home_odds": 1.90, "away_odds": 1.95},
            {"game_id": "g1", "sportsbook": "FD", "home_odds": 1.95, "away_odds": 1.90},
        ]
        comparison = self.value.compare_odds(odds, "g1")
        assert comparison["best_home"]["sportsbook"] == "FD"
        assert comparison["best_away"]["sportsbook"] == "DK"


class TestFeatureEngine:
    def setup_method(self):
        self.features = FeatureEngine()

    def test_build_features(self):
        home = {"stats": {"wins": 90, "losses": 72, "runs_scored": 750, "runs_allowed": 680}}
        away = {"stats": {"wins": 70, "losses": 92, "runs_scored": 620, "runs_allowed": 720}}
        f = self.features.build_features(home, away, sport="MLB")

        assert f["home_win_pct"] == pytest.approx(90 / 162, abs=0.001)
        assert f["away_win_pct"] == pytest.approx(70 / 162, abs=0.001)
        assert f["home_run_diff_per_game"] > 0  # positive run diff
        assert f["away_run_diff_per_game"] < 0  # negative run diff

    def test_weather_impact(self):
        home = {"stats": {"wins": 80, "losses": 82, "runs_scored": 700, "runs_allowed": 700}}
        away = {"stats": {"wins": 80, "losses": 82, "runs_scored": 700, "runs_allowed": 700}}
        weather = {"temperature": 40, "wind_speed": 20, "precipitation_chance": 0.3}
        f = self.features.build_features(home, away, weather, sport="MLB")

        assert f["temp_impact"] < 0  # cold
        assert f["wind_impact"] > 0.5  # high wind
        assert f["precip_risk"] == 0.3

    def test_predictions_are_bounded(self):
        """All model predictions should be in (0, 1)."""
        home = {"stats": {"wins": 100, "losses": 62, "runs_scored": 900, "runs_allowed": 500}}
        away = {"stats": {"wins": 50, "losses": 112, "runs_scored": 400, "runs_allowed": 900}}
        f = self.features.build_features(home, away, sport="MLB")
        preds = self.features.predict_from_features(f)

        assert len(preds) == 4
        for p in preds:
            assert 0.01 <= p <= 0.99

    def test_log5_method(self):
        """log5: strong home vs weak away should predict high home win prob."""
        home = {"stats": {"wins": 100, "losses": 62, "runs_scored": 850, "runs_allowed": 650}}
        away = {"stats": {"wins": 55, "losses": 107, "runs_scored": 550, "runs_allowed": 800}}
        f = self.features.build_features(home, away, sport="MLB")
        preds = self.features.predict_from_features(f)

        # Strong home team: most models should predict > 0.6
        avg = sum(preds) / len(preds)
        assert avg > 0.55

    def test_even_matchup(self):
        """Even teams should predict close to 0.5."""
        stats = {"stats": {"wins": 81, "losses": 81, "runs_scored": 700, "runs_allowed": 700}}
        f = self.features.build_features(stats, stats, sport="MLB")
        preds = self.features.predict_from_features(f)
        avg = sum(preds) / len(preds)
        assert 0.45 <= avg <= 0.60  # slight home advantage is expected


class TestGraphEngine:
    def setup_method(self):
        self.graph = GraphEngine()

    def test_basic_analysis(self):
        team = {
            "players": ["A", "B", "C", "D"],
            "relationships": [
                {"player1_idx": 0, "player2_idx": 1, "weight": 1.0},
                {"player1_idx": 1, "player2_idx": 2, "weight": 1.0},
                {"player1_idx": 2, "player2_idx": 3, "weight": 1.0},
                {"player1_idx": 0, "player2_idx": 3, "weight": 1.0},
            ],
        }
        result = self.graph.analyze_team(team)

        assert result["player_count"] == 4
        assert result["relationship_count"] == 4
        assert 0 <= result["network_density"] <= 1
        assert 0 <= result["clustering_coefficient"] <= 1
        assert result["key_player_count"] >= 1

    def test_fully_connected(self):
        """Fully connected graph should have density = 1.0."""
        team = {
            "players": ["A", "B", "C"],
            "relationships": [
                {"player1_idx": 0, "player2_idx": 1, "weight": 1.0},
                {"player1_idx": 0, "player2_idx": 2, "weight": 1.0},
                {"player1_idx": 1, "player2_idx": 2, "weight": 1.0},
            ],
        }
        result = self.graph.analyze_team(team)
        assert abs(result["network_density"] - 1.0) < 0.01
        assert abs(result["clustering_coefficient"] - 1.0) < 0.01

    def test_star_topology(self):
        """Star: one central node connected to all others. Center should be key player."""
        team = {
            "players": ["center", "leaf1", "leaf2", "leaf3", "leaf4"],
            "relationships": [
                {"player1_idx": 0, "player2_idx": 1, "weight": 1.0},
                {"player1_idx": 0, "player2_idx": 2, "weight": 1.0},
                {"player1_idx": 0, "player2_idx": 3, "weight": 1.0},
                {"player1_idx": 0, "player2_idx": 4, "weight": 1.0},
            ],
        }
        result = self.graph.analyze_team(team)
        center = result["players"][0]
        assert center["is_key_player"] is True
        assert center["degree_centrality"] == 1.0

    def test_betweenness_centrality_bridge(self):
        """Node connecting two clusters should have high betweenness."""
        # A-B-C connected, D-E-F connected, C-D is bridge
        team = {
            "players": ["A", "B", "C", "D", "E", "F"],
            "relationships": [
                {"player1_idx": 0, "player2_idx": 1, "weight": 1.0},
                {"player1_idx": 1, "player2_idx": 2, "weight": 1.0},
                {"player1_idx": 0, "player2_idx": 2, "weight": 1.0},
                {"player1_idx": 2, "player2_idx": 3, "weight": 1.0},  # bridge
                {"player1_idx": 3, "player2_idx": 4, "weight": 1.0},
                {"player1_idx": 4, "player2_idx": 5, "weight": 1.0},
                {"player1_idx": 3, "player2_idx": 5, "weight": 1.0},
            ],
        }
        result = self.graph.analyze_team(team)
        # Nodes 2 and 3 (the bridge) should have highest betweenness
        node_2 = result["players"][2]
        node_3 = result["players"][3]
        other_betweenness = [p["betweenness_centrality"] for i, p in enumerate(result["players"]) if i not in (2, 3)]
        assert node_2["betweenness_centrality"] >= max(other_betweenness) or \
               node_3["betweenness_centrality"] >= max(other_betweenness)

    def test_too_few_players(self):
        result = self.graph.analyze_team({"players": ["A"], "relationships": []})
        assert "error" in result
