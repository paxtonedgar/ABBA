"""Tests for the NHL Elo rating system."""

from __future__ import annotations

import math

import pytest

from abba.engine.elo import EloRatings


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def elo() -> EloRatings:
    """Fresh EloRatings instance with default settings."""
    return EloRatings()


# ------------------------------------------------------------------
# Basic rating behaviour
# ------------------------------------------------------------------

class TestInitialRatings:
    def test_unseen_team_gets_initial_rating(self, elo: EloRatings) -> None:
        assert elo.get_rating("BOS") == 1500

    def test_default_initial_rating_value(self) -> None:
        custom = EloRatings(initial_rating=1600)
        assert custom.get_rating("NYR") == 1600

    def test_get_all_ratings_empty(self, elo: EloRatings) -> None:
        assert elo.get_all_ratings() == {}


class TestPrediction:
    def test_home_team_gets_advantage(self, elo: EloRatings) -> None:
        """With equal ratings, the home team should be favoured."""
        pred = elo.predict("BOS", "NYR")
        assert pred["home_win_prob"] > 0.5
        assert pred["away_win_prob"] < 0.5

    def test_probabilities_sum_to_one(self, elo: EloRatings) -> None:
        pred = elo.predict("BOS", "NYR")
        assert math.isclose(
            pred["home_win_prob"] + pred["away_win_prob"], 1.0, abs_tol=1e-12
        )

    def test_probability_in_valid_range(self, elo: EloRatings) -> None:
        pred = elo.predict("BOS", "NYR")
        assert 0.0 <= pred["home_win_prob"] <= 1.0
        assert 0.0 <= pred["away_win_prob"] <= 1.0

    def test_prediction_includes_ratings(self, elo: EloRatings) -> None:
        pred = elo.predict("BOS", "NYR")
        assert pred["home_rating"] == 1500
        assert pred["away_rating"] == 1500


class TestUpdate:
    def test_winner_gains_loser_drops(self, elo: EloRatings) -> None:
        elo.update("BOS", "NYR", home_score=4, away_score=2)
        assert elo.get_rating("BOS") > 1500
        assert elo.get_rating("NYR") < 1500

    def test_zero_sum(self, elo: EloRatings) -> None:
        """Total Elo across all teams must stay constant."""
        elo.update("BOS", "NYR", home_score=3, away_score=1)
        elo.update("TOR", "MTL", home_score=2, away_score=5)
        total = sum(elo.get_all_ratings().values())
        expected_total = 4 * 1500  # four teams
        assert math.isclose(total, expected_total, abs_tol=1e-9)

    def test_update_returns_shift_info(self, elo: EloRatings) -> None:
        result = elo.update("BOS", "NYR", home_score=3, away_score=1)
        assert "home_pre" in result
        assert "home_post" in result
        assert "away_pre" in result
        assert "away_post" in result
        assert "shift" in result
        assert result["home_post"] > result["home_pre"]

    def test_away_win_lowers_home(self, elo: EloRatings) -> None:
        result = elo.update("BOS", "NYR", home_score=1, away_score=4)
        assert result["home_post"] < result["home_pre"]
        assert result["away_post"] > result["away_pre"]


class TestMarginOfVictory:
    def test_bigger_margin_bigger_shift(self, elo: EloRatings) -> None:
        """A blowout should move ratings more than a one-goal game."""
        elo1 = EloRatings()
        elo1.update("BOS", "NYR", home_score=2, away_score=1)
        shift_small = elo1.get_rating("BOS") - 1500

        elo2 = EloRatings()
        elo2.update("BOS", "NYR", home_score=7, away_score=1)
        shift_large = elo2.get_rating("BOS") - 1500

        assert shift_large > shift_small

    def test_mov_multiplier_at_least_one(self, elo: EloRatings) -> None:
        """The multiplier should never shrink a one-goal update."""
        mult = EloRatings._margin_of_victory_multiplier(goal_diff=1, elo_diff=0)
        assert mult >= 1.0

    def test_mov_multiplier_dampened_by_elo_gap(self) -> None:
        """Large rating gaps should dampen the MOV multiplier."""
        mult_close = EloRatings._margin_of_victory_multiplier(goal_diff=3, elo_diff=0)
        mult_far = EloRatings._margin_of_victory_multiplier(goal_diff=3, elo_diff=300)
        assert mult_close > mult_far


class TestSeasonReset:
    def test_season_reset_moves_toward_initial(self, elo: EloRatings) -> None:
        # Push BOS above 1500
        elo.update("BOS", "NYR", home_score=5, away_score=0)
        pre_reset = elo.get_rating("BOS")
        assert pre_reset > 1500

        elo.season_reset()
        post_reset = elo.get_rating("BOS")

        # Should be closer to 1500 now
        assert abs(post_reset - 1500) < abs(pre_reset - 1500)

    def test_season_reset_exact_fraction(self, elo: EloRatings) -> None:
        """Rating should move exactly 1/3 toward 1500."""
        elo._ratings["BOS"] = 1650
        elo.season_reset()
        assert math.isclose(elo.get_rating("BOS"), 1600.0, abs_tol=1e-9)

    def test_season_reset_below_mean(self, elo: EloRatings) -> None:
        elo._ratings["BUF"] = 1350
        elo.season_reset()
        assert math.isclose(elo.get_rating("BUF"), 1400.0, abs_tol=1e-9)


class TestInitializeFromGames:
    def test_processes_multiple_games(self, elo: EloRatings) -> None:
        games = [
            {"home_team": "BOS", "away_team": "NYR", "home_score": 3, "away_score": 1},
            {"home_team": "NYR", "away_team": "BOS", "home_score": 4, "away_score": 2},
            {"home_team": "BOS", "away_team": "NYR", "home_score": 2, "away_score": 1},
        ]
        ratings = elo.initialize_from_games(games)
        assert "BOS" in ratings
        assert "NYR" in ratings
        # After 3 games, ratings should have moved from initial
        assert ratings["BOS"] != 1500 or ratings["NYR"] != 1500

    def test_season_boundary_triggers_reset(self) -> None:
        elo = EloRatings()
        games = [
            {"home_team": "BOS", "away_team": "NYR", "home_score": 6, "away_score": 0, "season": 2024},
            {"home_team": "BOS", "away_team": "NYR", "home_score": 6, "away_score": 0, "season": 2024},
            # Season change should trigger reset before this game
            {"home_team": "BOS", "away_team": "NYR", "home_score": 3, "away_score": 2, "season": 2025},
        ]

        # Compare to a run without season boundaries
        elo_no_season = EloRatings()
        games_no_season = [
            {"home_team": "BOS", "away_team": "NYR", "home_score": 6, "away_score": 0},
            {"home_team": "BOS", "away_team": "NYR", "home_score": 6, "away_score": 0},
            {"home_team": "BOS", "away_team": "NYR", "home_score": 3, "away_score": 2},
        ]

        ratings_with = elo.initialize_from_games(games)
        ratings_without = elo_no_season.initialize_from_games(games_no_season)

        # The season reset should pull BOS closer to 1500, so its final
        # rating with seasons should be lower than without.
        assert ratings_with["BOS"] < ratings_without["BOS"]

    def test_returns_dict(self, elo: EloRatings) -> None:
        games = [
            {"home_team": "BOS", "away_team": "NYR", "home_score": 3, "away_score": 1},
        ]
        result = elo.initialize_from_games(games)
        assert isinstance(result, dict)
