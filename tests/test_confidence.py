"""Tests for prediction confidence and accuracy metadata system.

Validates that every prediction response carries honest uncertainty signals
so the calling LLM doesn't launder noise into false precision.
"""

import time

import pytest

from abba.engine.confidence import (
    LLM_INTERPRETATION_GUIDE,
    PredictionConfidence,
    _compute_confidence_interval,
    _compute_reliability_grade,
    _generate_caveats,
    build_prediction_meta,
    build_workflow_meta,
)


# -----------------------------------------------------------------------
# Reliability grade assignment
# -----------------------------------------------------------------------

class TestReliabilityGrades:
    """Each tier (A through F) has clear criteria."""

    def test_grade_a_live_full_data_fresh(self):
        """A: live data, 50+ GP both teams, goalie data, <1hr stale."""
        grade = _compute_reliability_grade(
            data_source="live",
            home_gp=60,
            away_gp=55,
            has_goalie_data=True,
            staleness_seconds=1800.0,  # 30 minutes
        )
        assert grade == "A"

    def test_grade_b_live_30gp_fresh(self):
        """B: live data, 30+ GP, <24hr stale."""
        grade = _compute_reliability_grade(
            data_source="live",
            home_gp=35,
            away_gp=40,
            has_goalie_data=True,
            staleness_seconds=7200.0,  # 2 hours
        )
        assert grade == "B"

    def test_grade_b_50gp_but_no_goalie(self):
        """50+ GP but missing goalie data: not A, should be B if <24hr."""
        grade = _compute_reliability_grade(
            data_source="live",
            home_gp=60,
            away_gp=55,
            has_goalie_data=False,
            staleness_seconds=3600.0,
        )
        # No goalie data blocks grade A; 30+ GP and <24hr qualifies for B
        assert grade == "B"

    def test_grade_c_low_gp(self):
        """C: live data but <30 GP."""
        grade = _compute_reliability_grade(
            data_source="live",
            home_gp=20,
            away_gp=25,
            has_goalie_data=True,
            staleness_seconds=3600.0,
        )
        assert grade == "C"

    def test_grade_c_stale_data(self):
        """C: live data, good GP, but >24hr stale."""
        grade = _compute_reliability_grade(
            data_source="live",
            home_gp=50,
            away_gp=50,
            has_goalie_data=True,
            staleness_seconds=100000.0,  # >24hr
        )
        assert grade == "C"

    def test_grade_c_missing_goalie_and_low_gp(self):
        """C: live data, 20 GP, no goalie data -- both degrade quality."""
        grade = _compute_reliability_grade(
            data_source="live",
            home_gp=20,
            away_gp=50,
            has_goalie_data=False,
            staleness_seconds=1800.0,
        )
        assert grade == "C"

    def test_grade_d_seed_data(self):
        """D: seed data regardless of other factors."""
        grade = _compute_reliability_grade(
            data_source="seed",
            home_gp=70,
            away_gp=70,
            has_goalie_data=True,
            staleness_seconds=0.0,
        )
        assert grade == "D"

    def test_grade_d_very_low_gp(self):
        """D: <10 GP even with live data."""
        grade = _compute_reliability_grade(
            data_source="live",
            home_gp=8,
            away_gp=50,
            has_goalie_data=True,
            staleness_seconds=1800.0,
        )
        assert grade == "D"

    def test_grade_f_no_data(self):
        """F: no data at all."""
        grade = _compute_reliability_grade(
            data_source="none",
            home_gp=0,
            away_gp=0,
            has_goalie_data=False,
            staleness_seconds=0.0,
        )
        assert grade == "F"

    def test_grade_f_none_source_overrides_everything(self):
        """F even if GP and goalie data look good -- source is none."""
        grade = _compute_reliability_grade(
            data_source="none",
            home_gp=80,
            away_gp=80,
            has_goalie_data=True,
            staleness_seconds=0.0,
        )
        assert grade == "F"


# -----------------------------------------------------------------------
# Caveat generation
# -----------------------------------------------------------------------

class TestCaveatGeneration:

    def test_seed_data_caveat(self):
        caveats = _generate_caveats(
            data_source="seed",
            home_gp=50, away_gp=50,
            has_goalie_data=True,
            staleness_seconds=0.0,
        )
        assert any("seed data" in c.lower() for c in caveats)

    def test_small_sample_caveat_under_10(self):
        caveats = _generate_caveats(
            data_source="live",
            home_gp=8, away_gp=50,
            has_goalie_data=True,
            staleness_seconds=0.0,
        )
        assert any("8 games played" in c for c in caveats)
        assert any("regression" in c.lower() for c in caveats)

    def test_small_sample_caveat_under_30(self):
        caveats = _generate_caveats(
            data_source="live",
            home_gp=25, away_gp=40,
            has_goalie_data=True,
            staleness_seconds=0.0,
        )
        assert any("small sample" in c.lower() for c in caveats)

    def test_no_goalie_caveat(self):
        caveats = _generate_caveats(
            data_source="live",
            home_gp=50, away_gp=50,
            has_goalie_data=False,
            staleness_seconds=0.0,
        )
        assert any("goaltender" in c.lower() for c in caveats)

    def test_stale_data_caveat_48h(self):
        caveats = _generate_caveats(
            data_source="live",
            home_gp=50, away_gp=50,
            has_goalie_data=True,
            staleness_seconds=200000.0,  # >48hr
        )
        assert any("48+" in c for c in caveats)

    def test_stale_data_caveat_24h(self):
        caveats = _generate_caveats(
            data_source="live",
            home_gp=50, away_gp=50,
            has_goalie_data=True,
            staleness_seconds=90000.0,  # 25 hours
        )
        assert any("24+" in c for c in caveats)

    def test_no_caveats_when_data_is_good(self):
        caveats = _generate_caveats(
            data_source="live",
            home_gp=60, away_gp=60,
            has_goalie_data=True,
            staleness_seconds=1800.0,
        )
        assert caveats == []

    def test_extra_caveats_appended(self):
        caveats = _generate_caveats(
            data_source="live",
            home_gp=60, away_gp=60,
            has_goalie_data=True,
            staleness_seconds=1800.0,
            extra_caveats=["Starter scratched 30 minutes ago"],
        )
        assert "Starter scratched 30 minutes ago" in caveats

    def test_no_data_caveat(self):
        caveats = _generate_caveats(
            data_source="none",
            home_gp=0, away_gp=0,
            has_goalie_data=False,
            staleness_seconds=0.0,
        )
        assert any("no data" in c.lower() for c in caveats)


# -----------------------------------------------------------------------
# Confidence interval
# -----------------------------------------------------------------------

class TestConfidenceInterval:

    def test_ci_contains_point_estimate(self):
        ci = _compute_confidence_interval(
            prediction_value=0.55,
            calibration_error=0.08,
            min_gp=50,
            has_goalie_data=True,
            data_source="live",
        )
        assert ci["lower"] <= ci["point"] <= ci["upper"]

    def test_ci_wider_with_fewer_games(self):
        ci_many = _compute_confidence_interval(
            prediction_value=0.55,
            calibration_error=0.08,
            min_gp=60,
            has_goalie_data=True,
            data_source="live",
        )
        ci_few = _compute_confidence_interval(
            prediction_value=0.55,
            calibration_error=0.08,
            min_gp=10,
            has_goalie_data=True,
            data_source="live",
        )
        assert ci_few["width"] > ci_many["width"]

    def test_ci_wider_without_goalie_data(self):
        ci_with = _compute_confidence_interval(
            prediction_value=0.55,
            calibration_error=0.08,
            min_gp=50,
            has_goalie_data=True,
            data_source="live",
        )
        ci_without = _compute_confidence_interval(
            prediction_value=0.55,
            calibration_error=0.08,
            min_gp=50,
            has_goalie_data=False,
            data_source="live",
        )
        assert ci_without["width"] > ci_with["width"]

    def test_ci_wider_for_seed_data(self):
        ci_live = _compute_confidence_interval(
            prediction_value=0.55,
            calibration_error=0.08,
            min_gp=50,
            has_goalie_data=True,
            data_source="live",
        )
        ci_seed = _compute_confidence_interval(
            prediction_value=0.55,
            calibration_error=0.08,
            min_gp=50,
            has_goalie_data=True,
            data_source="seed",
        )
        assert ci_seed["width"] > ci_live["width"]

    def test_ci_bounded_zero_one(self):
        """CI should never go below 0 or above 1."""
        ci = _compute_confidence_interval(
            prediction_value=0.05,
            calibration_error=0.20,
            min_gp=5,
            has_goalie_data=False,
            data_source="seed",
        )
        assert ci["lower"] >= 0.0
        assert ci["upper"] <= 1.0

    def test_ci_massive_for_no_data(self):
        ci = _compute_confidence_interval(
            prediction_value=0.50,
            calibration_error=0.08,
            min_gp=0,
            has_goalie_data=False,
            data_source="none",
        )
        # With no data, interval should span nearly the whole range
        assert ci["width"] >= 0.8


# -----------------------------------------------------------------------
# build_prediction_meta (integration)
# -----------------------------------------------------------------------

class TestBuildPredictionMeta:

    def _make_features(self, home_gp: int = 60, away_gp: int = 55) -> dict:
        return {
            "home_games_played": home_gp,
            "away_games_played": away_gp,
            "home_pts_pct": 0.58,
            "away_pts_pct": 0.52,
        }

    def test_returns_all_required_keys(self):
        meta = build_prediction_meta(
            features=self._make_features(),
            prediction_value=0.55,
            data_source="live",
            last_refresh_ts=time.time() - 600,
            has_goalie_data=True,
        )
        assert "accuracy_history" in meta
        assert "data_freshness" in meta
        assert "sample_size" in meta
        assert "confidence_interval" in meta
        assert "reliability_grade" in meta
        assert "caveats" in meta
        assert "interpretation_guide" in meta

    def test_grade_a_scenario(self):
        meta = build_prediction_meta(
            features=self._make_features(60, 55),
            prediction_value=0.55,
            data_source="live",
            last_refresh_ts=time.time() - 600,
            has_goalie_data=True,
        )
        assert meta["reliability_grade"] == "A"
        assert meta["caveats"] == []

    def test_grade_d_seed_scenario(self):
        meta = build_prediction_meta(
            features=self._make_features(60, 55),
            prediction_value=0.55,
            data_source="seed",
        )
        assert meta["reliability_grade"] == "D"
        assert meta["data_freshness"] == "seed"
        assert any("seed" in c.lower() for c in meta["caveats"])

    def test_grade_f_no_data(self):
        meta = build_prediction_meta(
            features={"home_games_played": 0, "away_games_played": 0},
            prediction_value=0.50,
            data_source="none",
            has_goalie_data=False,
        )
        assert meta["reliability_grade"] == "F"

    def test_sample_size_in_output(self):
        meta = build_prediction_meta(
            features=self._make_features(42, 38),
            prediction_value=0.55,
            data_source="live",
            last_refresh_ts=time.time(),
        )
        assert meta["sample_size"]["home_games_played"] == 42
        assert meta["sample_size"]["away_games_played"] == 38

    def test_confidence_interval_present(self):
        meta = build_prediction_meta(
            features=self._make_features(),
            prediction_value=0.60,
            data_source="live",
            last_refresh_ts=time.time(),
        )
        ci = meta["confidence_interval"]
        assert "point" in ci
        assert "lower" in ci
        assert "upper" in ci
        assert "width" in ci
        assert ci["point"] == pytest.approx(0.60, abs=0.001)

    def test_extra_caveats_passed_through(self):
        meta = build_prediction_meta(
            features=self._make_features(),
            prediction_value=0.55,
            data_source="live",
            last_refresh_ts=time.time(),
            extra_caveats=["Backup goalie starting tonight"],
        )
        assert "Backup goalie starting tonight" in meta["caveats"]

    def test_custom_accuracy_history(self):
        custom = {"log_loss": 0.55, "brier_score": 0.20, "accuracy": 0.62,
                  "sample_size": 1200, "date_range": "2024-01-01 to 2024-06-30"}
        meta = build_prediction_meta(
            features=self._make_features(),
            prediction_value=0.55,
            data_source="live",
            last_refresh_ts=time.time(),
            accuracy_history=custom,
        )
        assert meta["accuracy_history"]["accuracy"] == 0.62
        assert meta["accuracy_history"]["sample_size"] == 1200


# -----------------------------------------------------------------------
# build_workflow_meta
# -----------------------------------------------------------------------

class TestBuildWorkflowMeta:

    def test_workflow_meta_basic(self):
        meta = build_workflow_meta(
            workflow_name="game_prediction",
            data_sources_used=["live_api"],
            steps_completed=4,
            steps_total=4,
            last_refresh_ts=time.time() - 300,
            min_games_played=55,
            has_goalie_data=True,
        )
        assert meta["workflow"] == "game_prediction"
        assert meta["reliability_grade"] in ("A", "B")
        assert "interpretation_guide" in meta

    def test_workflow_seed_degrades_grade(self):
        meta = build_workflow_meta(
            workflow_name="tonights_slate",
            data_sources_used=["live_api", "seed"],
            steps_completed=3,
            steps_total=3,
            min_games_played=50,
        )
        # Seed in data_sources -> overall_data_source = "seed" -> grade D
        assert meta["reliability_grade"] == "D"

    def test_workflow_incomplete_caveat(self):
        meta = build_workflow_meta(
            workflow_name="value_scan",
            data_sources_used=["live_api"],
            steps_completed=2,
            steps_total=5,
            last_refresh_ts=time.time(),
            min_games_played=50,
        )
        assert any("incomplete" in c.lower() for c in meta["caveats"])

    def test_workflow_no_sources_is_grade_f(self):
        meta = build_workflow_meta(
            workflow_name="test",
            data_sources_used=[],
            steps_completed=0,
            steps_total=3,
        )
        assert meta["reliability_grade"] == "F"


# -----------------------------------------------------------------------
# LLM Interpretation Guide
# -----------------------------------------------------------------------

class TestLLMInterpretationGuide:

    def test_guide_exists(self):
        assert isinstance(LLM_INTERPRETATION_GUIDE, dict)
        assert len(LLM_INTERPRETATION_GUIDE) > 0

    def test_required_keys_present(self):
        required_keys = [
            "unreliable_grades",
            "confidence_interval",
            "stale_data",
            "seed_data",
            "coin_flip_rule",
        ]
        for key in required_keys:
            assert key in LLM_INTERPRETATION_GUIDE, f"Missing key: {key}"

    def test_unreliable_grades_mentions_d_and_f(self):
        text = LLM_INTERPRETATION_GUIDE["unreliable_grades"].lower()
        assert "d" in text or "f" in text

    def test_seed_data_warns_not_real(self):
        text = LLM_INTERPRETATION_GUIDE["seed_data"].lower()
        assert "not" in text and "real" in text

    def test_coin_flip_rule_describes_uncertainty(self):
        text = LLM_INTERPRETATION_GUIDE["coin_flip_rule"].lower()
        assert "coin flip" in text or "uncertain" in text

    def test_guide_included_in_prediction_meta(self):
        meta = build_prediction_meta(
            features={"home_games_played": 50, "away_games_played": 50},
            prediction_value=0.55,
            data_source="live",
            last_refresh_ts=time.time(),
        )
        assert meta["interpretation_guide"] is LLM_INTERPRETATION_GUIDE


# -----------------------------------------------------------------------
# PredictionConfidence dataclass
# -----------------------------------------------------------------------

class TestPredictionConfidenceDataclass:

    def test_to_dict_round_trip(self):
        pc = PredictionConfidence(
            accuracy_history={"accuracy": 0.57},
            data_freshness=1200.5,
            sample_size={"home_games_played": 50, "away_games_played": 45},
            confidence_interval={"point": 0.55, "lower": 0.42, "upper": 0.68, "width": 0.26},
            reliability_grade="B",
            caveats=["Something to note"],
        )
        d = pc.to_dict()
        assert d["reliability_grade"] == "B"
        assert d["caveats"] == ["Something to note"]
        assert "interpretation_guide" in d

    def test_default_values(self):
        pc = PredictionConfidence()
        assert pc.reliability_grade == "F"
        assert pc.caveats == []
        assert isinstance(pc.accuracy_history, dict)
