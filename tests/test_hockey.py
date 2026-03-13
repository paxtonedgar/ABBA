"""Tests for NHL-specific analytics engine -- comprehensive hockey math.

Tests Corsi, Fenwick, xG, goaltender models, special teams, rest effects,
score-state adjustment, cap analysis, season review, and playoff odds.
"""

import math

import numpy as np
import pytest

from abba.engine.hockey import HockeyAnalytics


@pytest.fixture
def hockey():
    return HockeyAnalytics()


class TestCorsi:
    def test_basic_corsi(self, hockey):
        result = hockey.corsi(
            shots_for=30, blocks_for=10, missed_for=8,
            shots_against=25, blocks_against=8, missed_against=7,
        )
        assert result["corsi_for"] == 48  # 30+10+8
        assert result["corsi_against"] == 40  # 25+8+7
        assert result["corsi_pct"] == pytest.approx(54.55, abs=0.1)  # 48/88*100
        assert result["corsi_rel"] > 0  # above 50%

    def test_corsi_even(self, hockey):
        result = hockey.corsi(
            shots_for=30, blocks_for=10, missed_for=10,
            shots_against=30, blocks_against=10, missed_against=10,
        )
        assert result["corsi_pct"] == pytest.approx(50.0, abs=0.01)
        assert result["corsi_rel"] == pytest.approx(0.0, abs=0.01)

    def test_corsi_per60(self, hockey):
        result = hockey.corsi(
            shots_for=30, blocks_for=10, missed_for=10,
            shots_against=25, blocks_against=8, missed_against=7,
            minutes_5v5=50.0,
        )
        # CF = 50, per60 = 50/50*60 = 60
        assert result["corsi_for_per60"] == pytest.approx(60.0, abs=0.1)

    def test_corsi_zero_shots(self, hockey):
        result = hockey.corsi(0, 0, 0, 0, 0, 0)
        assert result["corsi_pct"] == 50.0  # neutral when no data


class TestFenwick:
    def test_basic_fenwick(self, hockey):
        result = hockey.fenwick(
            shots_for=30, missed_for=8,
            shots_against=25, missed_against=7,
        )
        assert result["fenwick_for"] == 38  # 30+8
        assert result["fenwick_against"] == 32  # 25+7
        assert result["fenwick_pct"] == pytest.approx(54.29, abs=0.1)

    def test_fenwick_excludes_blocks(self, hockey):
        """Fenwick should not include blocked shots."""
        result = hockey.fenwick(
            shots_for=30, missed_for=8,
            shots_against=25, missed_against=7,
        )
        corsi = hockey.corsi(30, 100, 8, 25, 100, 7)
        assert result["fenwick_for"] == 38
        assert corsi["corsi_for"] == 138  # includes 100 blocks

    def test_fenwick_zero_shots(self, hockey):
        result = hockey.fenwick(0, 0, 0, 0)
        assert result["fenwick_pct"] == 50.0


class TestExpectedGoals:
    def test_xg_close_shot_calibrated(self, hockey):
        """10ft straight-on wrist shot should be ~9% (published coefficient)."""
        shots = [{"distance": 10, "angle": 0, "shot_type": "wrist"}]
        result = hockey.expected_goals(shots)
        # z = -1.9963 + (-0.0316 * 10) + (-0.0081 * 0) = -2.3123
        # p = 1/(1+exp(2.3123)) = 0.0901
        assert result["xg_total"] == pytest.approx(0.090, abs=0.005)

    def test_xg_mid_range_calibrated(self, hockey):
        """30ft straight-on wrist shot should be ~5%."""
        shots = [{"distance": 30, "angle": 0, "shot_type": "wrist"}]
        result = hockey.expected_goals(shots)
        # z = -1.9963 + (-0.0316 * 30) = -2.9443
        # p = 1/(1+exp(2.9443)) = 0.0501
        assert result["xg_total"] == pytest.approx(0.050, abs=0.005)

    def test_xg_far_shots_low(self, hockey):
        """50ft+ shots should be very low xG."""
        shots = [{"distance": 55, "angle": 40, "shot_type": "slap"}]
        result = hockey.expected_goals(shots)
        assert result["xg_total"] < 0.025

    def test_xg_rebound_bonus(self, hockey):
        """Rebounds add ~0.41 in log-odds, roughly 1.5x odds ratio."""
        base = hockey.expected_goals([{"distance": 15, "angle": 10}])
        rebound = hockey.expected_goals([{"distance": 15, "angle": 10, "is_rebound": True}])
        assert rebound["xg_total"] > base["xg_total"] * 1.3

    def test_xg_pp_boost(self, hockey):
        """Power play adds ~0.41 in log-odds."""
        even = hockey.expected_goals([{"distance": 20, "angle": 10, "strength": "even"}])
        pp = hockey.expected_goals([{"distance": 20, "angle": 10, "strength": "pp"}])
        assert pp["xg_total"] > even["xg_total"]

    def test_xg_tip_highest_type(self, hockey):
        """Tips/deflections should have highest shot type coefficient."""
        tip = hockey.expected_goals([{"distance": 15, "angle": 10, "shot_type": "tip"}])
        wrist = hockey.expected_goals([{"distance": 15, "angle": 10, "shot_type": "wrist"}])
        slap = hockey.expected_goals([{"distance": 15, "angle": 10, "shot_type": "slap"}])
        assert tip["xg_total"] > wrist["xg_total"] > slap["xg_total"]

    def test_xg_empty_shots(self, hockey):
        result = hockey.expected_goals([])
        assert result["xg_total"] == 0.0
        assert result["shot_count"] == 0

    def test_xg_never_exceeds_one(self, hockey):
        """Even extreme shots stay within [0, 1]."""
        shots = [{"distance": 0, "angle": 0, "is_rebound": True, "strength": "pp", "shot_type": "tip"}]
        result = hockey.expected_goals(shots)
        assert 0 < result["shots"][0]["xg"] < 1.0

    def test_xg_multiple_shots_accumulate(self, hockey):
        """Total xG should be sum of individual shots."""
        shots = [
            {"distance": 10, "angle": 5},
            {"distance": 20, "angle": 15},
            {"distance": 40, "angle": 30},
        ]
        result = hockey.expected_goals(shots)
        assert result["shot_count"] == 3
        individual_sum = sum(s["xg"] for s in result["shots"])
        assert result["xg_total"] == pytest.approx(individual_sum, abs=0.001)

    def test_xg_negative_distance_clamped(self, hockey):
        """Negative distance should be clamped to 0."""
        shots = [{"distance": -5, "angle": 0}]
        result = hockey.expected_goals(shots)
        assert result["shots"][0]["distance"] == 0.0
        assert result["xg_total"] > 0

    def test_xg_angle_reduces_probability(self, hockey):
        """Higher angles should produce lower xG."""
        straight = hockey.expected_goals([{"distance": 20, "angle": 0}])
        angled = hockey.expected_goals([{"distance": 20, "angle": 60}])
        assert straight["xg_total"] > angled["xg_total"]


class TestGoaltenderModel:
    def test_basic_goalie_metrics(self, hockey):
        result = hockey.goaltender_metrics(
            saves=1500, shots_against=1650, goals_against=150,
            xg_against=160.0, games_played=55, minutes_played=3300.0,
            quality_starts=35, shutouts=4,
        )
        assert result["save_pct"] == pytest.approx(1500 / 1650, abs=0.001)
        assert result["gaa"] == pytest.approx(150 / 3300 * 60, abs=0.01)
        assert result["shutouts"] == 4

    def test_gsaa_formula_correct(self, hockey):
        """GSAA = (1 - league_avg_sv) * SA - GA. Positive = good."""
        result = hockey.goaltender_metrics(
            saves=1550, shots_against=1650, goals_against=100,
            xg_against=140.0, games_played=55, minutes_played=3300.0,
        )
        # Expected GA at league avg (0.907): (1-0.907)*1650 = 0.093*1650 = 153.45
        # GSAA = 153.45 - 100 = 53.45
        assert result["gsaa"] == pytest.approx(53.45, abs=0.5)

    def test_gsaa_negative_for_bad_goalie(self, hockey):
        """Below-average goalie should have negative GSAA."""
        result = hockey.goaltender_metrics(
            saves=1400, shots_against=1650, goals_against=250,
            xg_against=200.0, games_played=55, minutes_played=3300.0,
        )
        # Expected GA: 0.093*1650 = 153.45, Actual GA: 250
        # GSAA = 153.45 - 250 = -96.55
        assert result["gsaa"] < 0

    def test_xgsaa_outperforming(self, hockey):
        """Positive xGSAA means goalie outperforms shot quality."""
        result = hockey.goaltender_metrics(
            saves=1500, shots_against=1650, goals_against=130,
            xg_against=160.0, games_played=55, minutes_played=3300.0,
        )
        # xGSAA = xGA - GA = 160 - 130 = 30
        assert result["xgsaa"] == pytest.approx(30.0, abs=0.1)

    def test_goaltender_matchup_calibrated(self, hockey):
        """0.020 Sv% diff should produce ~7.5% edge (not 30%)."""
        result = hockey.goaltender_matchup_edge(
            starter_sv_pct=0.925, opponent_sv_pct=0.905,
            starter_gsaa=15.0, opponent_gsaa=-5.0,
        )
        assert result["goaltender_edge"] > 0
        # 0.020 Sv% diff: sv_edge = 2 * 0.0375 = 0.075
        # Should be in the realistic 5-10% range, not 30%
        assert abs(result["goaltender_edge"]) < 0.15

    def test_goaltender_matchup_even(self, hockey):
        """Equal goalies should have near-zero edge."""
        result = hockey.goaltender_matchup_edge(
            starter_sv_pct=0.910, opponent_sv_pct=0.910,
            starter_gsaa=0.0, opponent_gsaa=0.0,
        )
        assert abs(result["goaltender_edge"]) < 0.01


class TestRegressionToMean:
    def test_early_season_heavy_regression(self, hockey):
        """After 10 games, .700 team should regress significantly toward .500."""
        result = hockey.regress_to_mean(0.700, 0.500, 10, k=55)
        # weight = 10/65 = 0.154. result = 0.5 + 0.2*0.154 = 0.531
        assert result == pytest.approx(0.531, abs=0.005)

    def test_full_season_less_regression(self, hockey):
        """After 82 games, .700 team should regress less."""
        result = hockey.regress_to_mean(0.700, 0.500, 82, k=55)
        # weight = 82/137 = 0.599. result = 0.5 + 0.2*0.599 = 0.620
        assert result == pytest.approx(0.620, abs=0.005)

    def test_average_team_unchanged(self, hockey):
        """A .500 team should stay at .500 regardless of sample size."""
        assert hockey.regress_to_mean(0.500, 0.500, 10) == pytest.approx(0.500)
        assert hockey.regress_to_mean(0.500, 0.500, 82) == pytest.approx(0.500)


class TestSpecialTeams:
    def test_basic_special_teams(self, hockey):
        result = hockey.special_teams_rating(
            pp_goals=45, pp_opportunities=200,
            pk_kills=170, pk_times_shorthanded=200,
        )
        assert result["pp_pct"] == pytest.approx(22.5, abs=0.1)
        assert result["pk_pct"] == pytest.approx(85.0, abs=0.1)

    def test_special_teams_above_avg(self, hockey):
        """Good PP and PK should have positive index."""
        result = hockey.special_teams_rating(
            pp_goals=50, pp_opportunities=200,  # 25% PP
            pk_kills=172, pk_times_shorthanded=200,  # 86% PK
        )
        assert result["pp_above_avg"] > 0
        assert result["pk_above_avg"] > 0
        assert result["special_teams_index"] > 0

    def test_special_teams_below_avg(self, hockey):
        result = hockey.special_teams_rating(
            pp_goals=30, pp_opportunities=200,  # 15%
            pk_kills=140, pk_times_shorthanded=200,  # 70%
        )
        assert result["pp_above_avg"] < 0
        assert result["pk_above_avg"] < 0


class TestRestAdvantage:
    def test_back_to_back_penalty(self, hockey):
        """B2B team should have fatigue disadvantage."""
        result = hockey.rest_advantage(
            home_rest_days=2, away_rest_days=0,
            home_is_back_to_back=False, away_is_back_to_back=True,
        )
        assert result["rest_edge"] > 0  # favors rested home team
        assert result["away_fatigue_total"] > result["home_fatigue_total"]

    def test_both_rested_neutral(self, hockey):
        result = hockey.rest_advantage(
            home_rest_days=2, away_rest_days=2,
            home_is_back_to_back=False, away_is_back_to_back=False,
        )
        assert abs(result["rest_edge"]) < 0.01

    def test_travel_penalty(self, hockey):
        """Long travel should add fatigue."""
        short = hockey.rest_advantage(
            home_rest_days=1, away_rest_days=1,
            home_is_back_to_back=False, away_is_back_to_back=False,
            home_travel_km=0, away_travel_km=100,
        )
        long_travel = hockey.rest_advantage(
            home_rest_days=1, away_rest_days=1,
            home_is_back_to_back=False, away_is_back_to_back=False,
            home_travel_km=0, away_travel_km=4000,
        )
        assert long_travel["away_fatigue_total"] > short["away_fatigue_total"]


class TestScoreAdjustedCorsi:
    def test_score_adjustment_direction(self, hockey):
        """Leading teams should get CF boost, trailing teams CA boost."""
        result = hockey.score_adjusted_corsi(
            corsi_leading={"cf": 100, "ca": 80},
            corsi_trailing={"cf": 120, "ca": 100},
            corsi_tied={"cf": 110, "ca": 90},
            minutes_leading=20.0, minutes_trailing=15.0, minutes_tied=25.0,
        )
        # Leading CF boosted, trailing CF reduced -> net positive adjustment for good teams
        assert result["adj_corsi_pct"] > 50

    def test_tied_minimal_adjustment(self, hockey):
        """If entire game is tied, adjustment should be near-neutral."""
        result = hockey.score_adjusted_corsi(
            corsi_leading={"cf": 0, "ca": 0},
            corsi_trailing={"cf": 0, "ca": 0},
            corsi_tied={"cf": 100, "ca": 100},
            minutes_leading=0, minutes_trailing=0, minutes_tied=60,
        )
        # Tied factors: CF*0.972, CA*1.029 -> 97.2 / (97.2+102.9) = 48.6%
        assert result["adj_corsi_pct"] == pytest.approx(48.6, abs=0.5)


class TestCapAnalysis:
    def test_basic_cap(self, hockey):
        roster = [
            {"name": "Star Player", "position": "C", "cap_hit": 10_000_000, "contract_years_remaining": 5, "status": "active"},
            {"name": "Depth Player", "position": "LW", "cap_hit": 1_500_000, "contract_years_remaining": 1, "status": "active"},
            {"name": "Injured Player", "position": "RD", "cap_hit": 4_000_000, "contract_years_remaining": 3, "status": "ltir"},
        ]
        result = hockey.cap_analysis(roster, cap_ceiling=95_500_000)
        assert result["total_cap_hit"] == 15_500_000
        assert result["cap_space"] == 95_500_000 - 15_500_000
        assert result["expiring_contracts"] == 1  # depth player
        # LTIR relief: team has huge cap space (80M), so LTIR pool = max(4M - 80M, 0) = 0
        # When team is well under the cap, LTIR relief is minimal
        assert result["ltir_relief"] >= 0

    def test_cap_health_tight(self, hockey):
        roster = [{"name": f"Player {i}", "position": "C", "cap_hit": 4_300_000, "status": "active"} for i in range(22)]
        result = hockey.cap_analysis(roster, cap_ceiling=95_500_000)
        # 22 * 4.3M = 94.6M, space = 900K
        assert result["cap_health"] == "tight"

    def test_ltir_over_cap(self, hockey):
        """Team over the cap with LTIR should get relief."""
        roster = [
            *[{"name": f"P{i}", "position": "C", "cap_hit": 8_000_000, "status": "active"} for i in range(12)],
            {"name": "LTIR Guy", "position": "D", "cap_hit": 6_000_000, "status": "ltir"},
        ]
        result = hockey.cap_analysis(roster, cap_ceiling=95_500_000)
        # Total: 12*8M + 6M = 102M. Cap space = 95.5-102 = -6.5M
        # LTIR relief: max(6M - max(-6.5M, 0), 0) = max(6M, 0) = 6M
        assert result["ltir_relief"] == 6_000_000

    def test_empty_roster(self, hockey):
        result = hockey.cap_analysis([])
        assert "error" in result


class TestSeasonReview:
    def test_basic_review(self, hockey):
        stats = {
            "wins": 50, "losses": 22, "overtime_losses": 10,
            "goals_for": 280, "goals_against": 230,
            "power_play_percentage": 24.0, "penalty_kill_percentage": 82.0,
        }
        result = hockey.season_review(stats)
        assert result["record"] == "50-22-10"
        assert result["points"] == 110
        assert result["goal_differential"] == 50
        assert result["pythagorean_wins"] > 0

    def test_pythagorean_uses_205_exponent(self, hockey):
        """Verify Pythagorean expectation uses 2.05 exponent."""
        stats = {
            "wins": 41, "losses": 41, "overtime_losses": 0,
            "goals_for": 250, "goals_against": 200,
            "power_play_percentage": 20.0, "penalty_kill_percentage": 80.0,
        }
        result = hockey.season_review(stats)
        # 250^2.05 / (250^2.05 + 200^2.05) should be > 0.5
        expected_pyth_pct = 250**2.05 / (250**2.05 + 200**2.05)
        expected_pyth_wins = expected_pyth_pct * 82
        assert result["pythagorean_wins"] == pytest.approx(expected_pyth_wins, abs=0.5)

    def test_review_with_advanced(self, hockey):
        stats = {"wins": 45, "losses": 25, "overtime_losses": 12, "goals_for": 260, "goals_against": 240,
                 "power_play_percentage": 22.0, "penalty_kill_percentage": 80.0}
        advanced = {"corsi_pct": 54.0, "xgf_pct": 54.0}
        result = hockey.season_review(stats, advanced)
        assert result["analytics_grade"] == "elite"

    def test_review_luck_factor(self, hockey):
        """Pythagorean wins vs actual wins measures luck."""
        stats = {
            "wins": 50, "losses": 22, "overtime_losses": 10,
            "goals_for": 250, "goals_against": 250,  # even goals = ~41 pyth wins
            "power_play_percentage": 20.0, "penalty_kill_percentage": 80.0,
        }
        result = hockey.season_review(stats)
        assert result["luck_factor"] > 5  # winning much more than expected


class TestPlayoffProbability:
    def test_high_points_high_probability(self, hockey):
        result = hockey.playoff_probability(
            current_points=85, games_remaining=10, games_played=72,
            wildcard_cutline=95,
        )
        assert result["wildcard_probability"] > 0.2

    def test_mathematically_eliminated(self, hockey):
        """Can't reach cutline even winning out."""
        result = hockey.playoff_probability(
            current_points=50, games_remaining=10, games_played=72,
            wildcard_cutline=95,
        )
        # max possible = 50 + 20 = 70 < 95
        assert result["status"] == "eliminated"

    def test_simulation_uses_regression(self, hockey):
        """Early-season team should show regression to mean in probabilities."""
        # Team at .700 pace after 20 games
        result = hockey.playoff_probability(
            current_points=28, games_remaining=62, games_played=20,
            wildcard_cutline=95,
        )
        # After regression, true talent is ~.531 not .700
        # So projected points < 28 + 62*1.4 = 114.8 (raw pace)
        assert result["projected_points"] < 115  # raw pace projection
        assert result["true_talent_pts_pg"] < 1.4  # regressed down from 1.4 ppg

    def test_game_by_game_sim_variance(self, hockey):
        """Probability should not be 0 or 1 for borderline teams."""
        result = hockey.playoff_probability(
            current_points=75, games_remaining=20, games_played=62,
            wildcard_cutline=95,
        )
        # Borderline: needs 20 pts in 20 games (~.500 pace)
        assert 0.05 < result["wildcard_probability"] < 0.95

    def test_zero_games_error(self, hockey):
        result = hockey.playoff_probability(0, 82, 0)
        assert "error" in result


class TestNHLFeatures:
    def test_build_features(self, hockey):
        home = {"stats": {"wins": 50, "losses": 22, "overtime_losses": 10, "goals_for": 280, "goals_against": 230,
                          "power_play_percentage": 24.0, "penalty_kill_percentage": 82.0, "recent_form": 0.65}}
        away = {"stats": {"wins": 35, "losses": 35, "overtime_losses": 12, "goals_for": 240, "goals_against": 260,
                          "power_play_percentage": 19.0, "penalty_kill_percentage": 78.0, "recent_form": 0.45}}

        features = hockey.build_nhl_features(home, away)
        assert features["home_pts_pct"] > features["away_pts_pct"]
        assert features["home_goal_diff_pg"] > 0
        assert features["away_goal_diff_pg"] < 0
        assert "home_games_played" in features
        assert "home_gf_per_game" in features

    def test_features_with_advanced(self, hockey):
        home = {"stats": {"wins": 40, "losses": 30, "overtime_losses": 12, "goals_for": 250, "goals_against": 240,
                          "power_play_percentage": 22.0, "penalty_kill_percentage": 80.0}}
        away = {"stats": {"wins": 40, "losses": 30, "overtime_losses": 12, "goals_for": 250, "goals_against": 240,
                          "power_play_percentage": 22.0, "penalty_kill_percentage": 80.0}}
        home_adv = {"corsi_pct": 55.0, "xgf_pct": 54.0}
        away_adv = {"corsi_pct": 48.0, "xgf_pct": 47.0}

        features = hockey.build_nhl_features(home, away, home_advanced=home_adv, away_advanced=away_adv)
        assert features["home_corsi_pct"] > features["away_corsi_pct"]
        assert features["home_xgf_pct"] > features["away_xgf_pct"]

    def test_features_with_goalie(self, hockey):
        home = {"stats": {"wins": 40, "losses": 30, "overtime_losses": 12, "goals_for": 250, "goals_against": 240,
                          "power_play_percentage": 22.0, "penalty_kill_percentage": 80.0}}
        away = {"stats": {"wins": 40, "losses": 30, "overtime_losses": 12, "goals_for": 250, "goals_against": 240,
                          "power_play_percentage": 22.0, "penalty_kill_percentage": 80.0}}
        home_g = {"save_pct": 0.925, "gsaa": 15.0}
        away_g = {"save_pct": 0.900, "gsaa": -10.0}

        features = hockey.build_nhl_features(home, away, home_goalie=home_g, away_goalie=away_g)
        assert features["goaltender_edge"] > 0


class TestNHLPredictions:
    def test_predictions_bounded(self, hockey):
        features = {
            "home_pts_pct": 0.65, "away_pts_pct": 0.45,
            "home_goal_diff_pg": 0.5, "away_goal_diff_pg": -0.3,
            "home_recent_form": 0.7, "away_recent_form": 0.4,
            "home_games_played": 60, "away_games_played": 60,
            "home_gf_per_game": 3.4, "home_ga_per_game": 2.8,
            "away_gf_per_game": 2.7, "away_ga_per_game": 3.2,
            "home_corsi_pct": 0.54, "away_corsi_pct": 0.47,
            "home_xgf_pct": 0.55, "away_xgf_pct": 0.46,
            "goaltender_edge": 0.05, "home_st_edge": 0.05, "rest_edge": 0.02,
        }
        preds = hockey.predict_nhl_game(features)
        assert len(preds) == 4  # points_log5, pythagorean_possession, situational, goaltender
        for p in preds:
            assert 0.01 <= p <= 0.99

    def test_strong_home_predicts_high(self, hockey):
        features = {
            "home_pts_pct": 0.70, "away_pts_pct": 0.35,
            "home_goal_diff_pg": 1.0, "away_goal_diff_pg": -0.8,
            "home_recent_form": 0.8, "away_recent_form": 0.3,
            "home_games_played": 70, "away_games_played": 70,
            "home_gf_per_game": 3.8, "home_ga_per_game": 2.5,
            "away_gf_per_game": 2.3, "away_ga_per_game": 3.5,
            "home_corsi_pct": 0.56, "away_corsi_pct": 0.44,
            "home_xgf_pct": 0.57, "away_xgf_pct": 0.43,
            "goaltender_edge": 0.08, "home_st_edge": 0.06, "rest_edge": 0.03,
        }
        preds = hockey.predict_nhl_game(features)
        avg = sum(preds) / len(preds)
        assert avg > 0.58

    def test_even_matchup_near_54(self, hockey):
        """Even teams should predict near home ice advantage (~54%)."""
        features = {
            "home_pts_pct": 0.50, "away_pts_pct": 0.50,
            "home_goal_diff_pg": 0.0, "away_goal_diff_pg": 0.0,
            "home_recent_form": 0.50, "away_recent_form": 0.50,
            "home_games_played": 82, "away_games_played": 82,
            "home_gf_per_game": 3.0, "home_ga_per_game": 3.0,
            "away_gf_per_game": 3.0, "away_ga_per_game": 3.0,
            "home_corsi_pct": 0.50, "away_corsi_pct": 0.50,
            "home_xgf_pct": 0.50, "away_xgf_pct": 0.50,
            "goaltender_edge": 0.0, "home_st_edge": 0.0, "rest_edge": 0.0,
        }
        preds = hockey.predict_nhl_game(features)
        avg = sum(preds) / len(preds)
        assert 0.50 <= avg <= 0.58  # home ice advantage

    def test_models_diverse(self, hockey):
        """Core models should produce varied predictions."""
        features = {
            "home_pts_pct": 0.58, "away_pts_pct": 0.48,
            "home_goal_diff_pg": 0.3, "away_goal_diff_pg": -0.1,
            "home_recent_form": 0.60, "away_recent_form": 0.45,
            "home_games_played": 60, "away_games_played": 60,
            "home_gf_per_game": 3.2, "home_ga_per_game": 2.8,
            "away_gf_per_game": 2.9, "away_ga_per_game": 3.1,
            "home_corsi_pct": 0.53, "away_corsi_pct": 0.48,
            "home_xgf_pct": 0.52, "away_xgf_pct": 0.49,
            "goaltender_edge": 0.03, "home_st_edge": 0.02, "rest_edge": 0.01,
        }
        preds = hockey.predict_nhl_game(features)
        assert max(preds) - min(preds) > 0.01

    def test_early_season_regression_dampens_prediction(self, hockey):
        """10 games in, a .700 team should not predict as strongly as at 70 games."""
        base_features = {
            "home_pts_pct": 0.70, "away_pts_pct": 0.40,
            "home_goal_diff_pg": 0.8, "away_goal_diff_pg": -0.5,
            "home_recent_form": 0.70, "away_recent_form": 0.40,
            "home_gf_per_game": 3.5, "home_ga_per_game": 2.5,
            "away_gf_per_game": 2.5, "away_ga_per_game": 3.3,
            "home_corsi_pct": 0.55, "away_corsi_pct": 0.45,
            "home_xgf_pct": 0.54, "away_xgf_pct": 0.46,
            "goaltender_edge": 0.04, "home_st_edge": 0.03, "rest_edge": 0.0,
        }
        early = {**base_features, "home_games_played": 10, "away_games_played": 10}
        late = {**base_features, "home_games_played": 70, "away_games_played": 70}

        preds_early = hockey.predict_nhl_game(early)
        preds_late = hockey.predict_nhl_game(late)
        # Early season should be closer to 0.54 (home ice), late season more extreme
        avg_early = sum(preds_early) / len(preds_early)
        avg_late = sum(preds_late) / len(preds_late)
        assert avg_early < avg_late  # regression dampens early-season signal
