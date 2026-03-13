"""Tests for the walk-forward backtest engine.

Uses synthetic game data — no internet required. Tests verify:
- No lookahead bias (pre-game stats don't include game outcome)
- Proper chronological ordering
- Running stats accumulate correctly
- CalibrationArtifact produced with valid metrics
- Report generation
"""

import numpy as np
import pytest

from abba.engine.backtest import BacktestResult, WalkForwardBacktest
from abba.engine.calibration import CalibrationArtifact


def _generate_synthetic_season(n_teams: int = 8, games_per_team: int = 30, seed: int = 42) -> list[dict]:
    """Generate a synthetic NHL season for testing.

    Creates round-robin schedule with realistic score distributions.
    Better teams (lower index) have higher win probability.
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    games = []
    game_num = 0

    for round_idx in range(games_per_team // (n_teams - 1) + 1):
        for i in range(n_teams):
            for j in range(i + 1, n_teams):
                if len([g for g in games if g["home_team"] == teams[i] or g["away_team"] == teams[i]]) >= games_per_team:
                    continue

                # Better teams (lower index) have higher strength
                home_strength = 3.0 + (n_teams - i) * 0.15
                away_strength = 3.0 + (n_teams - j) * 0.15
                home_strength += 0.2  # home ice

                home_score = int(rng.poisson(home_strength))
                away_score = int(rng.poisson(away_strength))

                # Avoid ties (NHL doesn't have ties)
                if home_score == away_score:
                    if rng.random() > 0.5:
                        home_score += 1
                    else:
                        away_score += 1

                game_num += 1
                games.append({
                    "game_id": f"syn-{game_num:04d}",
                    "date": f"2025-{10 + game_num // 100:02d}-{(game_num % 28) + 1:02d}",
                    "home_team": teams[i],
                    "away_team": teams[j],
                    "home_score": home_score,
                    "away_score": away_score,
                })

    # Sort by date (critical for walk-forward)
    games.sort(key=lambda g: g["date"])
    return games


class TestWalkForwardBacktest:
    @pytest.fixture
    def synthetic_games(self):
        return _generate_synthetic_season(n_teams=8, games_per_team=30)

    @pytest.fixture
    def backtest(self, synthetic_games):
        bt = WalkForwardBacktest()
        bt.run(synthetic_games)
        return bt

    def test_produces_results(self, backtest):
        """Backtest should produce results for all games."""
        assert len(backtest.results) > 0

    def test_no_lookahead_in_first_game(self, backtest):
        """First game should use neutral stats (no prior data exists)."""
        first = backtest.results[0]
        # With no prior data, prediction should be close to 0.5 + home ice
        assert 0.40 < first.predicted_home_prob < 0.65

    def test_predictions_bounded(self, backtest):
        """All predictions should be between 0.01 and 0.99."""
        for r in backtest.results:
            assert 0.01 <= r.predicted_home_prob <= 0.99, \
                f"Prediction {r.predicted_home_prob} out of bounds for {r.game_id}"

    def test_elo_updates_chronologically(self, backtest):
        """Elo ratings should change over time as games are processed."""
        first_5 = [r.elo_prob for r in backtest.results[:5]]
        last_5 = [r.elo_prob for r in backtest.results[-5:]]
        # After many games, predictions should be more varied than at start
        # (when all teams are at 1500)
        first_spread = max(first_5) - min(first_5)
        last_spread = max(last_5) - min(last_5)
        assert last_spread > first_spread or len(backtest.results) < 10

    def test_running_stats_accumulate(self, backtest, synthetic_games):
        """Running stats should reflect games played so far."""
        # Check a team's final running stats
        team = synthetic_games[0]["home_team"]
        stats = backtest._running_stats[team]
        total_gp = stats["wins"] + stats["losses"] + stats["overtime_losses"]
        assert total_gp > 0

    def test_individual_models_present(self, backtest):
        """Each result should have multiple individual model predictions."""
        for r in backtest.results:
            assert len(r.individual_models) >= 3, \
                f"Expected 3+ models, got {len(r.individual_models)} for {r.game_id}"

    def test_results_chronological(self, backtest):
        """Results should be in chronological order."""
        dates = [r.date for r in backtest.results]
        assert dates == sorted(dates)


class TestCalibrationArtifactFromBacktest:
    @pytest.fixture
    def artifact(self):
        games = _generate_synthetic_season(n_teams=8, games_per_team=40, seed=123)
        bt = WalkForwardBacktest()
        bt.run(games)
        return bt.evaluate(min_warmup=20)

    def test_is_calibration_artifact(self, artifact):
        assert isinstance(artifact, CalibrationArtifact)

    def test_has_valid_log_loss(self, artifact):
        """Log loss should be finite and positive."""
        assert 0 < artifact.log_loss < 2.0

    def test_has_valid_brier_score(self, artifact):
        assert 0 <= artifact.brier_score <= 1.0

    def test_has_valid_accuracy(self, artifact):
        assert 0 <= artifact.accuracy <= 1.0

    def test_has_valid_ece(self, artifact):
        assert 0 <= artifact.ece <= 1.0

    def test_has_temperature(self, artifact):
        """Temperature should be in reasonable range."""
        assert 0.1 <= artifact.temperature <= 10.0

    def test_has_baselines(self, artifact):
        assert "coin_flip_log_loss" in artifact.baselines
        assert "home_55_log_loss" in artifact.baselines

    def test_has_date_range(self, artifact):
        assert artifact.date_range != ""
        assert " to " in artifact.date_range

    def test_has_reliability_bins(self, artifact):
        assert len(artifact.reliability_bins) > 0

    def test_sample_size_correct(self, artifact):
        assert artifact.sample_size > 0


class TestBacktestReport:
    def test_report_generation(self):
        games = _generate_synthetic_season(n_teams=6, games_per_team=20, seed=99)
        bt = WalkForwardBacktest()
        bt.run(games)
        report = bt.report(min_warmup=10)

        assert "WALK-FORWARD" in report
        assert "Log loss" in report
        assert "Brier" in report
        assert "Baselines" in report
        assert "Calibration Bins" in report

    def test_insufficient_data_report(self):
        """Should handle case with too few games gracefully."""
        games = _generate_synthetic_season(n_teams=4, games_per_team=5, seed=77)
        bt = WalkForwardBacktest()
        bt.run(games)
        report = bt.report(min_warmup=100)  # more warmup than games
        assert "Insufficient" in report


class TestModelContract:
    """Verify the model contract types are importable and usable."""

    def test_imports(self):
        from abba.types import (
            MODEL_NEUTRAL_DEFAULTS,
            MODEL_OPTIONAL_FEATURES,
            MODEL_REQUIRED_FEATURES,
            ModelFeatures,
            PredictionProvenance,
            PredictionInput,
            PredictionOutput,
            build_prediction_input,
            split_model_features,
        )
        assert len(MODEL_REQUIRED_FEATURES) == 10
        assert len(MODEL_OPTIONAL_FEATURES) > 0
        assert all(k in MODEL_NEUTRAL_DEFAULTS for k in MODEL_OPTIONAL_FEATURES)

    def test_required_features_match_hockey_engine(self):
        """Required features in contract should match what build_nhl_features produces."""
        from abba.engine.hockey import HockeyAnalytics
        from abba.types import MODEL_REQUIRED_FEATURES

        hockey = HockeyAnalytics()
        features = hockey.build_nhl_features(
            {"stats": {"wins": 40, "losses": 30, "overtime_losses": 10,
                       "goals_for": 200, "goals_against": 180}},
            {"stats": {"wins": 35, "losses": 35, "overtime_losses": 10,
                       "goals_for": 190, "goals_against": 200}},
        )

        for req in MODEL_REQUIRED_FEATURES:
            assert req in features, f"Required feature '{req}' not in build_nhl_features output"

    def test_optional_features_have_neutral_defaults(self):
        """Every optional feature must have a defined neutral default."""
        from abba.types import MODEL_NEUTRAL_DEFAULTS, MODEL_OPTIONAL_FEATURES
        for feat in MODEL_OPTIONAL_FEATURES:
            assert feat in MODEL_NEUTRAL_DEFAULTS, \
                f"Optional feature '{feat}' missing from MODEL_NEUTRAL_DEFAULTS"

    def test_build_prediction_input_splits_required_and_optional_features(self):
        """Shared contract should preserve explicit required vs optional partitions."""
        from abba.types import build_prediction_input

        features = {
            "home_pts_pct": 0.61,
            "away_pts_pct": 0.54,
            "home_goal_diff_pg": 0.45,
            "away_goal_diff_pg": 0.10,
            "home_games_played": 70,
            "away_games_played": 69,
            "home_gf_per_game": 3.3,
            "home_ga_per_game": 2.7,
            "away_gf_per_game": 3.0,
            "away_ga_per_game": 2.9,
            "home_corsi_pct": 0.53,
            "away_xgf_pct": 0.49,
        }
        prediction_input = build_prediction_input(
            game_id="nhl-test",
            home_team="NYR",
            away_team="BOS",
            sport="NHL",
            season="2025-26",
            features=features,
            defaulted_features=["rest_edge"],
            provenance={"team_stats": {"status": "present", "source": "seed"}},
        )

        assert set(prediction_input["required_features"].keys()) == {
            "home_pts_pct", "away_pts_pct",
            "home_goal_diff_pg", "away_goal_diff_pg",
            "home_games_played", "away_games_played",
            "home_gf_per_game", "home_ga_per_game",
            "away_gf_per_game", "away_ga_per_game",
        }
        assert prediction_input["optional_features"]["home_corsi_pct"] == 0.53
        assert prediction_input["optional_features"]["away_xgf_pct"] == 0.49
        assert prediction_input["provenance"]["team_stats"]["status"] == "present"
