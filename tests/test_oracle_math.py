"""Oracle tests for critical math functions in ABBA.

Each test uses hand-computed expected values to verify the math is correct.
These are pure mathematical identities -- if they break, the formulas changed.

Marked with pytest.mark.contract so they run in the trust-check suite.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from abba.engine.hockey import HockeyAnalytics
from abba.engine.elo import EloRatings
from abba.engine.ensemble import EnsembleEngine
from abba.engine.confidence import _compute_confidence_interval, _BASE_CALIBRATION_ERROR
from abba.engine.kelly import KellyEngine
from abba.engine.features import FeatureEngine

TOL = 1e-4

hockey = HockeyAnalytics()
features_engine = FeatureEngine()


# ==========================================================================
# 1. Log5 formula
#    P(A beats B) = (pA - pA*pB) / (pA + pB - 2*pA*pB)
# ==========================================================================

class TestLog5:
    """Log5 head-to-head probability formula."""

    @pytest.mark.contract
    def test_log5_sixty_vs_forty(self):
        # pA=0.6, pB=0.4
        # num = 0.6 - 0.6*0.4 = 0.6 - 0.24 = 0.36
        # den = 0.6 + 0.4 - 2*0.6*0.4 = 1.0 - 0.48 = 0.52
        # P = 0.36 / 0.52 = 0.692307692...
        features = {
            "home_win_pct": 0.6, "away_win_pct": 0.4,
            "home_run_diff_per_game": 0, "away_run_diff_per_game": 0,
            "home_recent_form": 0.5, "away_recent_form": 0.5,
            "home_advantage": 0.54,
        }
        preds = features_engine.predict_from_features(features)
        # Model 1 = 0.7 * log5 + 0.3 * home_adv = 0.7 * 0.692308 + 0.3 * 0.54
        # = 0.484615 + 0.162 = 0.646615...
        expected_m1 = 0.7 * (0.36 / 0.52) + 0.3 * 0.54
        assert abs(preds[0] - expected_m1) < TOL

    @pytest.mark.contract
    def test_log5_equal_teams(self):
        # pA=0.5, pB=0.5
        # num = 0.5 - 0.25 = 0.25
        # den = 0.5 + 0.5 - 0.50 = 0.50
        # P = 0.25 / 0.50 = 0.5
        features = {
            "home_win_pct": 0.5, "away_win_pct": 0.5,
            "home_run_diff_per_game": 0, "away_run_diff_per_game": 0,
            "home_recent_form": 0.5, "away_recent_form": 0.5,
            "home_advantage": 0.54,
        }
        preds = features_engine.predict_from_features(features)
        # m1 = 0.7 * 0.5 + 0.3 * 0.54 = 0.35 + 0.162 = 0.512
        expected_m1 = 0.7 * 0.5 + 0.3 * 0.54
        assert abs(preds[0] - expected_m1) < TOL

    @pytest.mark.contract
    def test_log5_seventy_vs_thirty(self):
        # pA=0.7, pB=0.3
        # num = 0.7 - 0.7*0.3 = 0.7 - 0.21 = 0.49
        # den = 0.7 + 0.3 - 2*0.7*0.3 = 1.0 - 0.42 = 0.58
        # P = 0.49 / 0.58 = 0.844827586...
        # But features.py clamps to [0.01, 0.99] before log5 -- no effect here.
        features = {
            "home_win_pct": 0.7, "away_win_pct": 0.3,
            "home_run_diff_per_game": 0, "away_run_diff_per_game": 0,
            "home_recent_form": 0.5, "away_recent_form": 0.5,
            "home_advantage": 0.54,
        }
        preds = features_engine.predict_from_features(features)
        # m1 = 0.7 * (0.49/0.58) + 0.3 * 0.54
        expected_m1 = 0.7 * (0.49 / 0.58) + 0.3 * 0.54
        assert abs(preds[0] - expected_m1) < TOL

    @pytest.mark.contract
    def test_log5_edge_near_zero(self):
        # pA=0.01 (clamped), pB=0.99 (clamped)
        # num = 0.01 - 0.01*0.99 = 0.01 - 0.0099 = 0.0001
        # den = 0.01 + 0.99 - 2*0.01*0.99 = 1.0 - 0.0198 = 0.9802
        # P = 0.0001 / 0.9802 = 0.00010202...
        features = {
            "home_win_pct": 0.01, "away_win_pct": 0.99,
            "home_run_diff_per_game": 0, "away_run_diff_per_game": 0,
            "home_recent_form": 0.5, "away_recent_form": 0.5,
            "home_advantage": 0.54,
        }
        preds = features_engine.predict_from_features(features)
        raw_log5 = 0.0001 / 0.9802
        expected_m1 = 0.7 * raw_log5 + 0.3 * 0.54
        assert abs(preds[0] - expected_m1) < TOL

    @pytest.mark.contract
    def test_log5_nhl_predict_equal_teams(self):
        """Log5 in hockey.py predict_nhl_game: equal teams at 82 GP."""
        # With hp=0.5, ap=0.5, regressed: still 0.5 (no change).
        # log5(0.5, 0.5) = 0.5, plus home_boost=0.04 => m1 = 0.54
        features = {
            "home_pts_pct": 0.5, "away_pts_pct": 0.5,
            "home_goal_diff_pg": 0.0, "away_goal_diff_pg": 0.0,
            "home_recent_form": 0.5, "away_recent_form": 0.5,
            "goaltender_edge": 0.0, "home_st_edge": 0.0,
            "rest_edge": 0.0, "home_games_played": 82,
            "away_games_played": 82,
            "home_gf_per_game": 3.0, "home_ga_per_game": 3.0,
            "away_gf_per_game": 3.0, "away_ga_per_game": 3.0,
        }
        preds = hockey.predict_nhl_game(features)
        # m1 should be 0.5 + 0.04 = 0.54
        assert abs(preds[0] - 0.54) < TOL


# ==========================================================================
# 2. Pythagorean expectation (hockey exponent=2.05)
#    WinPct = GF^2.05 / (GF^2.05 + GA^2.05)
# ==========================================================================

class TestPythagorean:
    """Pythagorean win expectation with hockey exponent."""

    @pytest.mark.contract
    def test_pyth_250_220(self):
        # GF=250, GA=220, exp=2.05
        # 250^2.05 = exp(2.05 * ln(250))
        # 220^2.05 = exp(2.05 * ln(220))
        gf, ga, exp = 250, 220, 2.05
        gf_exp = gf ** exp
        ga_exp = ga ** exp
        expected = gf_exp / (gf_exp + ga_exp)
        # Compute via season_review
        stats = {"wins": 50, "losses": 25, "overtime_losses": 7,
                 "goals_for": 250, "goals_against": 220}
        review = hockey.season_review(stats)
        # pyth_wins = expected * gp where gp=82
        gp = 82
        pyth_wins = expected * gp
        assert abs(review["pythagorean_wins"] - round(pyth_wins, 1)) < 0.15

    @pytest.mark.contract
    def test_pyth_equal_goals(self):
        # GF=200, GA=200 => 0.5 exactly
        stats = {"wins": 41, "losses": 41, "overtime_losses": 0,
                 "goals_for": 200, "goals_against": 200}
        review = hockey.season_review(stats)
        # pyth_pct = 0.5, pyth_wins = 0.5 * 82 = 41.0
        assert abs(review["pythagorean_wins"] - 41.0) < 0.15

    @pytest.mark.contract
    def test_pyth_raw_formula(self):
        # Direct formula check: 300 GF, 250 GA
        gf, ga, exp = 300, 250, 2.05
        gf_e = gf ** exp  # 300^2.05
        ga_e = ga ** exp  # 250^2.05
        expected = gf_e / (gf_e + ga_e)
        # Verify via season_review
        stats = {"wins": 50, "losses": 25, "overtime_losses": 7,
                 "goals_for": 300, "goals_against": 250}
        review = hockey.season_review(stats)
        gp = 82
        assert abs(review["pythagorean_wins"] - round(expected * gp, 1)) < 0.15


# ==========================================================================
# 3. Elo win probability
#    P(A) = 1 / (1 + 10^((Rb - Ra) / 400))
# ==========================================================================

class TestEloWinProbability:
    """Elo logistic win probability formula."""

    @pytest.mark.contract
    def test_elo_equal_ratings(self):
        # Ra=1500, Rb=1500 => P = 1/(1+10^0) = 1/2 = 0.5
        p = EloRatings._win_probability(1500, 1500)
        assert abs(p - 0.5) < TOL

    @pytest.mark.contract
    def test_elo_1600_vs_1400(self):
        # Ra=1600, Rb=1400
        # exponent = (1400-1600)/400 = -200/400 = -0.5
        # P = 1/(1+10^(-0.5)) = 1/(1+0.316228) = 1/1.316228 = 0.75974...
        expected = 1.0 / (1.0 + math.pow(10.0, -0.5))
        p = EloRatings._win_probability(1600, 1400)
        assert abs(p - expected) < TOL
        assert abs(p - 0.7597) < 1e-3

    @pytest.mark.contract
    def test_elo_home_advantage(self):
        # Ra=1500+50=1550, Rb=1500 (home ice adds 50)
        # exponent = (1500-1550)/400 = -50/400 = -0.125
        # P = 1/(1+10^(-0.125)) = 1/(1+0.74989...) = 1/1.74989 = 0.57148...
        expected = 1.0 / (1.0 + math.pow(10.0, -0.125))
        p = EloRatings._win_probability(1550, 1500)
        assert abs(p - expected) < TOL

    @pytest.mark.contract
    def test_elo_predict_includes_home_advantage(self):
        """EloRatings.predict() adds home_advantage to home rating."""
        elo = EloRatings(k=4, home_advantage=50, initial_rating=1500)
        result = elo.predict("TeamA", "TeamB")
        # Both at 1500, home gets +50 => P(home) = _win_probability(1550, 1500)
        expected = EloRatings._win_probability(1550, 1500)
        assert abs(result["home_win_prob"] - expected) < TOL

    @pytest.mark.contract
    def test_elo_symmetry(self):
        # P(A beats B) + P(B beats A) = 1
        p_ab = EloRatings._win_probability(1600, 1400)
        p_ba = EloRatings._win_probability(1400, 1600)
        assert abs(p_ab + p_ba - 1.0) < TOL


# ==========================================================================
# 4. Elo margin-of-victory multiplier
#    ln(|goal_diff|+1) * (2.2 / (elo_diff*0.001 + 2.2))
# ==========================================================================

class TestEloMOVMultiplier:
    """Margin-of-victory multiplier for Elo updates."""

    @pytest.mark.contract
    def test_mov_goal_diff_3_elo_diff_100(self):
        # goal_diff=3, elo_diff=100
        # raw = ln(3+1) * (2.2 / (100*0.001 + 2.2))
        #     = ln(4) * (2.2 / 2.3)
        #     = 1.38629... * 0.95652...
        #     = 1.32597...
        # max(1.32597, 1.0) = 1.32597
        raw = math.log(4) * (2.2 / (100 * 0.001 + 2.2))
        expected = max(raw, 1.0)
        result = EloRatings._margin_of_victory_multiplier(3, 100)
        assert abs(result - expected) < TOL

    @pytest.mark.contract
    def test_mov_goal_diff_zero(self):
        # goal_diff=0 => returns 1.0 (special case in code)
        result = EloRatings._margin_of_victory_multiplier(0, 100)
        assert abs(result - 1.0) < TOL

    @pytest.mark.contract
    def test_mov_goal_diff_1_elo_diff_0(self):
        # goal_diff=1, elo_diff=0
        # raw = ln(2) * (2.2 / (0 + 2.2)) = ln(2) * 1.0 = 0.693147...
        # max(0.693147, 1.0) = 1.0  (clamped to minimum 1.0)
        result = EloRatings._margin_of_victory_multiplier(1, 0)
        assert abs(result - 1.0) < TOL

    @pytest.mark.contract
    def test_mov_blowout(self):
        # goal_diff=7, elo_diff=50
        # raw = ln(8) * (2.2 / (0.05 + 2.2)) = 2.07944 * (2.2/2.25) = 2.07944 * 0.97778
        #     = 2.03326...
        raw = math.log(8) * (2.2 / (50 * 0.001 + 2.2))
        expected = max(raw, 1.0)
        result = EloRatings._margin_of_victory_multiplier(7, 50)
        assert abs(result - expected) < TOL


# ==========================================================================
# 5. Goaltender matchup edge
#    sv_edge = (sv_diff / 0.01) * 0.0375
#    gsaa_edge = (gsaa_diff / 10.0) * 0.05
#    combined = 0.7 * clip(sv_edge) + 0.3 * clip(gsaa_edge)
# ==========================================================================

class TestGoaltenderMatchupEdge:
    """Goaltender matchup edge calculation."""

    @pytest.mark.contract
    def test_equal_goalies(self):
        # Equal goalies => edge = 0.0
        result = hockey.goaltender_matchup_edge(0.910, 0.910, 5.0, 5.0)
        assert abs(result["goaltender_edge"] - 0.0) < TOL

    @pytest.mark.contract
    def test_920_vs_900(self):
        # starter=0.920, opponent=0.900, gsaa: starter=10, opponent=0
        # sv_diff = 0.920 - 0.900 = 0.020
        # sv_edge = (0.020 / 0.01) * 0.0375 = 2.0 * 0.0375 = 0.075
        # gsaa_diff = 10 - 0 = 10
        # gsaa_edge = (10 / 10.0) * 0.05 = 0.05
        # clip(sv_edge, -0.5, 0.5) = 0.075
        # clip(gsaa_edge, -0.5, 0.5) = 0.05
        # combined = 0.7 * 0.075 + 0.3 * 0.05 = 0.0525 + 0.015 = 0.0675
        # clip(0.0675, -0.5, 0.5) = 0.0675
        result = hockey.goaltender_matchup_edge(0.920, 0.900, 10.0, 0.0)
        assert abs(result["goaltender_edge"] - 0.0675) < TOL

    @pytest.mark.contract
    def test_sv_edge_clamped(self):
        # Extreme case: starter=0.940, opponent=0.880
        # sv_diff = 0.060
        # sv_edge = (0.060 / 0.01) * 0.0375 = 6.0 * 0.0375 = 0.225
        # gsaa: both 0 => gsaa_edge = 0
        # clip(0.225, -0.5, 0.5) = 0.225 (within range)
        # combined = 0.7 * 0.225 + 0.3 * 0.0 = 0.1575
        result = hockey.goaltender_matchup_edge(0.940, 0.880, 0.0, 0.0)
        assert abs(result["goaltender_edge"] - 0.1575) < TOL

    @pytest.mark.contract
    def test_negative_edge(self):
        # starter=0.900, opponent=0.920 (starter is worse)
        # sv_diff = -0.020
        # sv_edge = (-0.020 / 0.01) * 0.0375 = -0.075
        # gsaa: starter=-5, opponent=10 => gsaa_diff = -15
        # gsaa_edge = (-15 / 10.0) * 0.05 = -0.075
        # combined = 0.7 * (-0.075) + 0.3 * (-0.075) = -0.075
        result = hockey.goaltender_matchup_edge(0.900, 0.920, -5.0, 10.0)
        assert abs(result["goaltender_edge"] - (-0.075)) < TOL


# ==========================================================================
# 6. Kelly criterion
#    f* = (b*p - q) / b, then fractional Kelly and safety caps
# ==========================================================================

class TestKellyCriterion:
    """Kelly criterion position sizing."""

    @pytest.mark.contract
    def test_kelly_60pct_at_2x_odds(self):
        # p=0.6, decimal_odds=2.0, bankroll=1000
        # b = 2.0 - 1.0 = 1.0
        # q = 0.4
        # full_kelly = (1.0*0.6 - 0.4) / 1.0 = 0.2
        # half_kelly = 0.2 * 0.5 = 0.1
        # implied_prob = 1/2.0 = 0.5
        # edge = 0.6 - 0.5 = 0.10
        # ev = 0.6 * 1.0 - 0.4 = 0.20
        # edge(0.10) >= min_edge(0.02) and ev(0.20) >= min_ev(0.03) => bet
        # fraction = min(0.1, 0.05) = 0.05 (capped at max_bet_pct)
        engine = KellyEngine(kelly_fraction=0.5, max_bet_pct=0.05, min_edge=0.02, min_ev=0.03)
        result = engine.calculate(0.6, 2.0, 1000.0)
        assert abs(result.fraction - 0.05) < TOL  # capped
        assert abs(result.expected_value - 0.20) < TOL
        assert abs(result.edge - 0.10) < TOL

    @pytest.mark.contract
    def test_kelly_60pct_uncapped(self):
        # Same but with max_bet_pct=1.0 to see raw half-Kelly
        # half_kelly = 0.1
        engine = KellyEngine(kelly_fraction=0.5, max_bet_pct=1.0, min_edge=0.02, min_ev=0.03)
        result = engine.calculate(0.6, 2.0, 1000.0)
        assert abs(result.fraction - 0.10) < TOL

    @pytest.mark.contract
    def test_kelly_negative_ev(self):
        # p=0.4, decimal_odds=2.0
        # b=1.0, q=0.6
        # full_kelly = (1.0*0.4 - 0.6)/1.0 = -0.2
        # fraction = max(0, -0.2 * 0.5) = 0.0
        # Also edge = 0.4 - 0.5 = -0.1 < min_edge => 0
        engine = KellyEngine()
        result = engine.calculate(0.4, 2.0, 1000.0)
        assert abs(result.fraction - 0.0) < TOL
        assert abs(result.recommended_stake - 0.0) < TOL

    @pytest.mark.contract
    def test_kelly_edge_below_minimum(self):
        # p=0.51, decimal_odds=2.0
        # implied=0.5, edge=0.01 < min_edge=0.02 => no bet
        engine = KellyEngine(min_edge=0.02, min_ev=0.03)
        result = engine.calculate(0.51, 2.0, 1000.0)
        assert abs(result.fraction - 0.0) < TOL


# ==========================================================================
# 7. Confidence interval width
#    half_width = calibration_error * sample_factor * feature_adjustments
#    ci_half = half_width * 1.28 (80% CI)
# ==========================================================================

class TestConfidenceInterval:
    """Confidence interval computation."""

    @pytest.mark.contract
    def test_ci_50gp_goalie_live(self):
        # min_gp=50, has_goalie=True, data_source="live"
        # base half_width = 0.08
        # sample_factor = sqrt(50/50) = 1.0, max(1.0, 1.0) = 1.0
        # no goalie penalty, no seed penalty
        # half_width = 0.08 * 1.0 = 0.08
        # ci_half = 0.08 * 1.28 = 0.1024
        # point=0.55 => lower=0.55-0.1024=0.4476, upper=0.55+0.1024=0.6524
        # width = 0.6524 - 0.4476 = 0.2048
        ci = _compute_confidence_interval(
            prediction_value=0.55,
            calibration_error=_BASE_CALIBRATION_ERROR,
            min_gp=50,
            has_goalie_data=True,
            data_source="live",
        )
        assert abs(ci["point"] - 0.55) < TOL
        assert abs(ci["width"] - 0.2048) < TOL
        assert abs(ci["lower"] - 0.4476) < TOL
        assert abs(ci["upper"] - 0.6524) < TOL

    @pytest.mark.contract
    def test_ci_10gp_no_goalie_seed(self):
        # min_gp=10, has_goalie=False, data_source="seed"
        # base half_width = 0.08
        # sample_factor = sqrt(50/10) = sqrt(5) = 2.23607, max(2.23607, 1.0) = 2.23607
        # half_width = 0.08 * 2.23607 = 0.17889
        # no goalie: *= 1.25 => 0.22361
        # seed: *= 2.0 => 0.44721
        # ci_half = 0.44721 * 1.28 = 0.57243
        # point=0.55 => lower = max(0.55 - 0.57243, 0) = 0.0
        # upper = min(0.55 + 0.57243, 1.0) = 1.0
        # width = 1.0 - 0.0 = 1.0
        ci = _compute_confidence_interval(
            prediction_value=0.55,
            calibration_error=_BASE_CALIBRATION_ERROR,
            min_gp=10,
            has_goalie_data=False,
            data_source="seed",
        )
        assert abs(ci["point"] - 0.55) < TOL
        # Clamped to [0, 1] so width is 1.0
        assert ci["lower"] == 0.0
        assert ci["upper"] == 1.0
        assert abs(ci["width"] - 1.0) < TOL

    @pytest.mark.contract
    def test_ci_wider_with_fewer_games(self):
        """Fewer games => strictly wider interval."""
        ci_50 = _compute_confidence_interval(0.55, _BASE_CALIBRATION_ERROR, 50, True, "live")
        ci_20 = _compute_confidence_interval(0.55, _BASE_CALIBRATION_ERROR, 20, True, "live")
        assert ci_20["width"] > ci_50["width"]

    @pytest.mark.contract
    def test_ci_wider_without_goalie(self):
        """Missing goalie data => wider interval (1.25x multiplier)."""
        ci_with = _compute_confidence_interval(0.55, _BASE_CALIBRATION_ERROR, 50, True, "live")
        ci_without = _compute_confidence_interval(0.55, _BASE_CALIBRATION_ERROR, 50, False, "live")
        # Without goalie: half_width *= 1.25, so width should be 1.25x
        assert abs(ci_without["width"] / ci_with["width"] - 1.25) < TOL


# ==========================================================================
# 8. Palmer-Tango regression to the mean
#    true_talent = league_avg + (observed - league_avg) * n / (n + k)
# ==========================================================================

class TestPalmerTangoRegression:
    """Palmer-Tango empirical Bayes regression to the mean."""

    @pytest.mark.contract
    def test_regress_82gp(self):
        # observed=0.6, avg=0.5, n=82, k=55
        # weight = 82 / (82+55) = 82/137 = 0.59854...
        # true_talent = 0.5 + (0.6 - 0.5) * 0.59854 = 0.5 + 0.059854 = 0.559854
        weight = 82.0 / 137.0
        expected = 0.5 + 0.1 * weight
        result = hockey.regress_to_mean(0.6, 0.5, 82, k=55)
        assert abs(result - expected) < TOL
        assert abs(result - 0.5599) < 1e-3

    @pytest.mark.contract
    def test_regress_10gp(self):
        # observed=0.6, avg=0.5, n=10, k=55
        # weight = 10 / 65 = 0.15385...
        # true_talent = 0.5 + 0.1 * 0.15385 = 0.5 + 0.015385 = 0.515385
        weight = 10.0 / 65.0
        expected = 0.5 + 0.1 * weight
        result = hockey.regress_to_mean(0.6, 0.5, 10, k=55)
        assert abs(result - expected) < TOL
        # More regressed than 82 GP case
        assert result < hockey.regress_to_mean(0.6, 0.5, 82, k=55)

    @pytest.mark.contract
    def test_regress_zero_games(self):
        # n=0 => weight=0 => true_talent = league_avg
        result = hockey.regress_to_mean(0.7, 0.5, 0, k=55)
        assert abs(result - 0.5) < TOL

    @pytest.mark.contract
    def test_regress_huge_sample(self):
        # n=10000 => weight ~ 10000/10055 ~ 0.9945
        # true_talent very close to observed
        result = hockey.regress_to_mean(0.6, 0.5, 10000, k=55)
        assert abs(result - 0.6) < 1e-3

    @pytest.mark.contract
    def test_regress_below_average(self):
        # observed=0.4 (below avg 0.5), n=82, k=55
        # true_talent = 0.5 + (0.4 - 0.5) * 82/137 = 0.5 - 0.05985 = 0.44015
        weight = 82.0 / 137.0
        expected = 0.5 + (-0.1) * weight
        result = hockey.regress_to_mean(0.4, 0.5, 82, k=55)
        assert abs(result - expected) < TOL


# ==========================================================================
# 9. xG sigmoid (logistic regression on shot quality)
#    z = intercept + coef_distance * dist + coef_angle * angle + shot_type
#    xg = 1 / (1 + exp(-z))
# ==========================================================================

class TestXGSigmoid:
    """Expected goals logistic model."""

    @pytest.mark.contract
    def test_xg_wrist_shot_20ft_center(self):
        # Wrist shot from 20ft, angle=0, even strength, no rebound/rush
        # z = -1.9963 + (-0.0316)*20 + (-0.0081)*0 + 0.0 (wrist)
        #   = -1.9963 + -0.632 = -2.6283
        # xg = 1/(1+exp(2.6283)) = 1/(1+13.849) = 0.06734...
        z = -1.9963 + (-0.0316) * 20.0
        expected_xg = 1.0 / (1.0 + math.exp(-z))
        result = hockey.expected_goals([{
            "distance": 20.0, "angle": 0.0, "shot_type": "wrist",
            "is_rebound": False, "is_rush": False, "strength": "even",
        }])
        assert abs(result["shots"][0]["xg"] - round(expected_xg, 4)) < 1e-4

    @pytest.mark.contract
    def test_xg_tip_rebound_close(self):
        # Tip-in from 5ft, rebound, even strength
        # z = -1.9963 + (-0.0316)*5 + 0.0*0 + 0.35 (tip) + 0.4133 (rebound)
        #   = -1.9963 - 0.158 + 0.35 + 0.4133 = -1.391
        # xg = 1/(1+exp(1.391)) = 1/(1+4.019) = 0.1993...
        z = -1.9963 + (-0.0316) * 5.0 + 0.35 + 0.4133
        expected_xg = 1.0 / (1.0 + math.exp(-z))
        result = hockey.expected_goals([{
            "distance": 5.0, "angle": 0.0, "shot_type": "tip",
            "is_rebound": True, "is_rush": False, "strength": "even",
        }])
        assert abs(result["shots"][0]["xg"] - round(expected_xg, 4)) < 1e-4

    @pytest.mark.contract
    def test_xg_overflow_clamp(self):
        # Extreme negative z (very far shot) — should not overflow
        # distance=200 => z = -1.9963 + (-0.0316)*200 = -1.9963 - 6.32 = -8.3163
        # xg ≈ 0.0002 (very small but not error)
        result = hockey.expected_goals([{
            "distance": 200.0, "angle": 0.0, "shot_type": "slap",
        }])
        xg = result["shots"][0]["xg"]
        assert xg >= 0.0
        assert xg < 0.01  # should be very small

    @pytest.mark.contract
    def test_xg_empty_shots(self):
        result = hockey.expected_goals([])
        assert result["xg_total"] == 0.0
        assert result["shot_count"] == 0


# ==========================================================================
# 10. Ensemble consensus-proximity weighting
#     Models closer to the group mean get higher weight.
#     Fallback: simple mean when std < 0.02.
# ==========================================================================

class TestEnsembleConsensusProximity:
    """Ensemble consensus-proximity weighting."""

    @pytest.mark.contract
    def test_near_consensus_falls_back_to_mean(self):
        # All predictions within std < 0.02 → simple mean
        # [0.55, 0.55, 0.56, 0.55] → std ≈ 0.005 < 0.02 → mean = 0.5525
        ensemble = EnsembleEngine()
        preds = [0.55, 0.55, 0.56, 0.55]
        result = ensemble._weighted_combine(np.array(preds), None)
        expected = np.mean(preds)
        assert abs(result - expected) < TOL

    @pytest.mark.contract
    def test_outlier_downweighted(self):
        # [0.55, 0.55, 0.55, 0.30] → mean = 0.4875, std > 0.02
        # The 0.30 outlier is far from mean → gets lower weight
        # Result should be closer to 0.55 than to 0.4875
        ensemble = EnsembleEngine()
        preds = np.array([0.55, 0.55, 0.55, 0.30])
        result = ensemble._weighted_combine(preds, None)
        simple_mean = float(np.mean(preds))  # 0.4875
        # Result should be pulled toward the consensus (0.55)
        assert result > simple_mean
        assert result < 0.55  # still influenced by 0.30

    @pytest.mark.contract
    def test_single_prediction_passthrough(self):
        ensemble = EnsembleEngine()
        result = ensemble._weighted_combine(np.array([0.65]), None)
        assert abs(result - 0.65) < TOL

    @pytest.mark.contract
    def test_explicit_weights_used(self):
        # With explicit weights, uses them directly (not consensus-proximity)
        ensemble = EnsembleEngine()
        preds = np.array([0.6, 0.4])
        weights = [0.75, 0.25]
        result = ensemble._weighted_combine(preds, weights)
        expected = 0.75 * 0.6 + 0.25 * 0.4  # 0.55
        assert abs(result - expected) < TOL


# ==========================================================================
# 11. Elo update integration
#     shift = K * MOV * (actual - expected)
# ==========================================================================

class TestEloUpdate:
    """Elo rating update with K-factor, MOV, and actual outcome."""

    @pytest.mark.contract
    def test_home_win_updates_ratings(self):
        # Home (1500+50=1550 effective) beats Away (1500)
        # expected = _win_probability(1550, 1500) ≈ 0.5715
        # actual = 1.0 (home won)
        # goal_diff = 3, elo_diff = |1500-1500| = 0 (raw ratings, no home adj in diff)
        # MOV = _margin_of_victory_multiplier(3, 0)
        #     = max(ln(4) * (2.2 / (0+2.2)), 1.0) = max(1.3863, 1.0) = 1.3863
        # shift = K * MOV * (1.0 - expected) = 4 * 1.3863 * (1.0 - 0.5715) = 4 * 1.3863 * 0.4285 = 2.376
        elo = EloRatings(k=4, home_advantage=50, initial_rating=1500)

        expected_prob = EloRatings._win_probability(1550, 1500)
        mov = EloRatings._margin_of_victory_multiplier(3, 0)
        expected_shift = 4 * mov * (1.0 - expected_prob)

        result = elo.update("HomeTeam", "AwayTeam", home_score=4, away_score=1)
        # Home should gain, away should lose
        assert result["home_post"] > 1500
        assert result["away_post"] < 1500
        # Verify shift magnitude
        assert abs(result["shift"] - expected_shift) < 0.1
