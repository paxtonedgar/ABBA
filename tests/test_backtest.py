"""ARCHIVED — Biased backtest (lookahead contamination).

*** DO NOT use these metrics for model validation or accuracy claims. ***

This backtest uses TODAY's standings to predict PAST games, which means
the model has indirect access to outcomes (lookahead bias). Reported metrics
are optimistic.

A leakage-free replacement is being built using the standings_snapshots table.
Once enough daily snapshots accumulate (30+ days), a proper walk-forward
backtest can be constructed that only uses data available BEFORE each game.

The infrastructure for this is in place:
  - Storage: standings_snapshots table (snapshot_date, team_id, stats)
  - Capture: refresh_data() auto-snapshots standings on each NHL refresh
  - Query: storage.get_standings_snapshot(date, team_id)

This file is kept for reference and to maintain test count continuity.
It documents what an honest backtest looks like structurally — calibration bins,
Brier score, log loss, baseline comparison — which the replacement will inherit.

Requires internet access to fetch from api-web.nhle.com (free, no auth).
"""

import json
import math
import urllib.request
import urllib.error
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

import numpy as np
import pytest
from scipy import stats as scipy_stats


# Skip if no internet
def _can_reach_nhl_api() -> bool:
    try:
        req = urllib.request.Request(
            "https://api-web.nhle.com/v1/standings/now",
            headers={"User-Agent": "ABBA-Backtest/1.0"},
        )
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception:
        return False


def _fetch_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "ABBA-Backtest/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def _fetch_completed_games(days_back: int = 5) -> list[dict]:
    """Fetch recent completed NHL games from the real API.

    NOTE: days_back is kept small (default 5) to limit lookahead bias.
    See module docstring for details.
    """
    games = []
    today = date.today()

    for offset in range(1, days_back + 1):
        game_date = (today - timedelta(days=offset)).isoformat()
        try:
            data = _fetch_json(f"https://api-web.nhle.com/v1/score/{game_date}")
            for game in data.get("games", []):
                if game.get("gameState") in ("FINAL", "OFF"):
                    home = game.get("homeTeam", {})
                    away = game.get("awayTeam", {})
                    games.append({
                        "game_id": f"nhl-{game.get('id', '')}",
                        "date": game_date,
                        "home_team": home.get("abbrev", ""),
                        "away_team": away.get("abbrev", ""),
                        "home_score": home.get("score", 0),
                        "away_score": away.get("score", 0),
                        "home_won": home.get("score", 0) > away.get("score", 0),
                        "period": game.get("periodDescriptor", {}).get("number", 3),
                    })
        except Exception:
            continue

    return games


def _fetch_standings() -> dict[str, dict]:
    """Fetch current standings for all teams.

    WARNING: These standings include results of games in our backtest window,
    introducing lookahead bias. See module docstring.
    """
    data = _fetch_json("https://api-web.nhle.com/v1/standings/now")
    teams = {}
    for entry in data.get("standings", []):
        abbrev = entry.get("teamAbbrev", {}).get("default", "")
        gp = entry.get("gamesPlayed", 1)
        teams[abbrev] = {
            "wins": entry.get("wins", 0),
            "losses": entry.get("losses", 0),
            "overtime_losses": entry.get("otLosses", 0),
            "points": entry.get("points", 0),
            "goals_for": entry.get("goalFor", 0),
            "goals_against": entry.get("goalAgainst", 0),
            "games_played": gp,
            "win_pct": entry.get("wins", 0) / max(gp, 1),
            "pts_pct": entry.get("pointPctg", 0.5),
            "l10_wins": entry.get("l10Wins", 0),
            "l10_losses": entry.get("l10Losses", 0),
            "home_wins": entry.get("homeWins", 0),
            "home_losses": entry.get("homeLosses", 0),
            "road_wins": entry.get("roadWins", 0),
            "road_losses": entry.get("roadLosses", 0),
            "power_play_percentage": 22.0,  # API doesn't give PP% in standings
            "penalty_kill_percentage": 80.0,
            "recent_form": entry.get("l10Wins", 0) / max(
                entry.get("l10Wins", 0) + entry.get("l10Losses", 0) + entry.get("l10OtLosses", 0), 1
            ),
        }
    return teams


def _log_loss(y_true: list[bool], y_pred: list[float], eps: float = 1e-7) -> float:
    """Binary log loss (proper scoring rule). Lower is better."""
    total = 0.0
    for actual, pred in zip(y_true, y_pred):
        p = np.clip(pred, eps, 1 - eps)
        if actual:
            total -= math.log(p)
        else:
            total -= math.log(1 - p)
    return total / len(y_true) if y_true else 0.0


def _brier_score(y_true: list[bool], y_pred: list[float]) -> float:
    """Brier score (mean squared error of probabilities). Lower is better."""
    total = 0.0
    for actual, pred in zip(y_true, y_pred):
        total += (float(actual) - pred) ** 2
    return total / len(y_true) if y_true else 0.0


def _calibration_bins(y_true: list[bool], y_pred: list[float], n_bins: int = 5) -> list[dict]:
    """Group predictions into bins and check actual win rate."""
    bins: dict[int, list] = defaultdict(list)
    for actual, pred in zip(y_true, y_pred):
        bin_idx = min(int(pred * n_bins), n_bins - 1)
        bins[bin_idx].append(actual)

    result = []
    for i in range(n_bins):
        if bins[i]:
            actual_rate = sum(bins[i]) / len(bins[i])
            expected_low = i / n_bins
            expected_high = (i + 1) / n_bins
            expected_mid = (expected_low + expected_high) / 2
            result.append({
                "bin": f"{expected_low:.0%}-{expected_high:.0%}",
                "count": len(bins[i]),
                "predicted_avg": round(expected_mid, 3),
                "actual_win_rate": round(actual_rate, 3),
                "calibration_error": round(abs(actual_rate - expected_mid), 3),
            })
    return result


@pytest.mark.skipif(not _can_reach_nhl_api(), reason="NHL API not reachable")
class TestBacktest:
    """Honest backtest against real completed NHL games."""

    @pytest.fixture(scope="class")
    def backtest_data(self):
        """Fetch real data once for all tests.

        Uses a 5-day lookback to limit lookahead bias (see module docstring).
        """
        games = _fetch_completed_games(days_back=5)
        standings = _fetch_standings()
        return {"games": games, "standings": standings}

    @pytest.fixture(scope="class")
    def predictions(self, backtest_data):
        """Run our prediction models against real completed games."""
        from abba.engine.hockey import HockeyAnalytics

        hockey = HockeyAnalytics()
        games = backtest_data["games"]
        standings = backtest_data["standings"]

        results = []
        for game in games:
            home = game["home_team"]
            away = game["away_team"]

            home_stats = standings.get(home)
            away_stats = standings.get(away)
            if not home_stats or not away_stats:
                continue

            features = hockey.build_nhl_features(
                {"stats": home_stats},
                {"stats": away_stats},
            )
            model_preds = hockey.predict_nhl_game(features)
            avg_pred = sum(model_preds) / len(model_preds)

            results.append({
                "game_id": game["game_id"],
                "date": game["date"],
                "home_team": home,
                "away_team": away,
                "home_won": game["home_won"],
                "home_score": game["home_score"],
                "away_score": game["away_score"],
                "predicted_home_prob": avg_pred,
                "individual_models": model_preds,
            })

        return results

    def test_has_enough_games(self, predictions):
        """Need at least 10 games for a meaningful (short-window) backtest."""
        assert len(predictions) >= 10, f"Only {len(predictions)} games, need 10+"
        print(f"\nBacktest: {len(predictions)} completed NHL games")

    def test_acknowledges_lookahead_bias(self, predictions):
        """Document that this backtest has a known lookahead bias.

        We use today's standings to predict games from the past 5 days.
        Those game results are already baked into the standings, which means
        the model has indirect access to the outcome. A proper fix would
        require daily standings snapshots, which the public NHL API does not
        expose.

        This test exists purely as documentation — it always passes but prints
        a warning so the bias is visible in every test run.
        """
        n = len(predictions)
        dates = sorted(set(p["date"] for p in predictions))
        print("\n*** LOOKAHEAD BIAS WARNING ***")
        print(f"  Standings fetched: today ({date.today().isoformat()})")
        print(f"  Games predicted:   {dates[0]} to {dates[-1]} ({n} games)")
        print("  Standings INCLUDE results of these games.")
        print("  Reported metrics are optimistic. See module docstring.")
        # Always passes — this is a documentation test.
        assert True

    def test_log_loss_better_than_coin_flip(self, predictions):
        """Our model should have lower log loss than predicting 50% every time."""
        y_true = [p["home_won"] for p in predictions]
        y_pred = [p["predicted_home_prob"] for p in predictions]

        our_log_loss = _log_loss(y_true, y_pred)
        coin_flip_log_loss = _log_loss(y_true, [0.5] * len(y_true))

        print(f"\nLog loss: model={our_log_loss:.4f}, coin_flip={coin_flip_log_loss:.4f}")
        # Model must strictly beat coin flip — no slack multiplier.
        assert our_log_loss < coin_flip_log_loss, \
            f"Model log loss ({our_log_loss:.4f}) worse than coin flip ({coin_flip_log_loss:.4f})"

    def test_log_loss_better_than_always_home(self, predictions):
        """Model should beat the naive 'home team always wins at 55%' baseline."""
        y_true = [p["home_won"] for p in predictions]
        y_pred = [p["predicted_home_prob"] for p in predictions]

        our_log_loss = _log_loss(y_true, y_pred)
        home_bias_log_loss = _log_loss(y_true, [0.55] * len(y_true))

        print(f"\nLog loss: model={our_log_loss:.4f}, home_55%={home_bias_log_loss:.4f}")
        # Model must strictly beat the home-bias baseline.
        assert our_log_loss < home_bias_log_loss, \
            f"Model ({our_log_loss:.4f}) not better than home bias ({home_bias_log_loss:.4f})"

    def test_brier_score(self, predictions):
        """Brier score should be reasonable (< 0.30 for sports)."""
        y_true = [p["home_won"] for p in predictions]
        y_pred = [p["predicted_home_prob"] for p in predictions]

        brier = _brier_score(y_true, y_pred)
        print(f"\nBrier score: {brier:.4f} (lower is better, random = 0.25)")
        assert brier < 0.30, f"Brier score {brier:.4f} too high"

    def test_accuracy(self, predictions):
        """Directional accuracy: how often does the model pick the winner?"""
        correct = sum(
            1 for p in predictions
            if (p["predicted_home_prob"] > 0.5) == p["home_won"]
        )
        accuracy = correct / len(predictions) if predictions else 0
        home_always_accuracy = sum(1 for p in predictions if p["home_won"]) / len(predictions)

        print(f"\nAccuracy: model={accuracy:.1%}, always_home={home_always_accuracy:.1%}")
        # 50% is random chance — the model must beat it.
        assert accuracy >= 0.50, f"Accuracy {accuracy:.1%} below random chance (50%)"

    def test_calibration(self, predictions):
        """Check if predictions are well-calibrated."""
        y_true = [p["home_won"] for p in predictions]
        y_pred = [p["predicted_home_prob"] for p in predictions]

        bins = _calibration_bins(y_true, y_pred, n_bins=5)
        print("\nCalibration:")
        for b in bins:
            print(f"  {b['bin']}: predicted {b['predicted_avg']:.0%}, "
                  f"actual {b['actual_win_rate']:.0%} ({b['count']} games)")

        # Weighted average calibration error must be < 15%.
        # We weight by bin count so a bin with 1 game doesn't dominate.
        if bins:
            total_games = sum(b["count"] for b in bins)
            avg_cal_error = sum(
                b["calibration_error"] * b["count"] for b in bins
            ) / total_games
            print(f"  Weighted avg calibration error: {avg_cal_error:.3f}")
            assert avg_cal_error < 0.15, f"Calibration error {avg_cal_error:.3f} too high (max 0.15)"

    def test_model_diversity(self, predictions):
        """The 6 models should produce diverse predictions (not all identical)."""
        spreads = []
        for p in predictions:
            models = p["individual_models"]
            spreads.append(max(models) - min(models))

        avg_spread = np.mean(spreads)
        print(f"\nModel diversity: avg spread = {avg_spread:.4f}")
        assert avg_spread > 0.01, "Models are too similar -- no ensemble benefit"

    def test_favorites_win_more(self, predictions):
        """Games where model is more confident should have higher win rate.

        When the model assigns >55% to either side, the favored side should
        win more than 45% of the time (weak but real signal).
        """
        confident = [p for p in predictions if abs(p["predicted_home_prob"] - 0.5) > 0.05]

        if len(confident) >= 5:
            conf_correct = sum(
                1 for p in confident
                if (p["predicted_home_prob"] > 0.5) == p["home_won"]
            ) / len(confident)

            print(f"\nConfident picks ({len(confident)}): {conf_correct:.1%} correct")
            assert conf_correct >= 0.45, (
                f"Favorites (>{55}% predicted) only won {conf_correct:.1%} of the time; "
                f"expected at least 45%"
            )
        else:
            print(f"\nOnly {len(confident)} confident picks — skipping assertion (need >= 5)")

    def test_summary_report(self, predictions):
        """Print a comprehensive backtest report and verify it contains key metrics."""
        y_true = [p["home_won"] for p in predictions]
        y_pred = [p["predicted_home_prob"] for p in predictions]

        n = len(predictions)
        correct = sum(1 for t, p in zip(y_true, y_pred) if (p > 0.5) == t)
        home_wins = sum(y_true)

        our_ll = _log_loss(y_true, y_pred)
        coin_ll = _log_loss(y_true, [0.5] * n)
        improvement = (coin_ll - our_ll) / coin_ll * 100

        lines = []
        lines.append("=" * 60)
        lines.append("ABBA NHL BACKTEST REPORT")
        lines.append("=" * 60)
        lines.append(f"Games analyzed:     {n}")
        lines.append(f"Date range:         {predictions[-1]['date']} to {predictions[0]['date']}")
        lines.append(f"Home win rate:      {home_wins/n:.1%}")
        lines.append(f"Model accuracy:     {correct/n:.1%}")
        lines.append(f"Log loss:           {our_ll:.4f}")
        lines.append(f"Brier score:        {_brier_score(y_true, y_pred):.4f}")
        lines.append(f"Avg prediction:     {np.mean(y_pred):.3f}")
        lines.append(f"Prediction std:     {np.std(y_pred):.3f}")
        lines.append(f"Models used:        6 (log5, pyth, corsi, xG, goalie, combined)")
        lines.append(f"Data source:        api-web.nhle.com (live)")
        lines.append(f"*** Lookahead bias: YES (see module docstring) ***")
        lines.append("")
        lines.append(f"Baselines:")
        lines.append(f"  Coin flip (50%):  {coin_ll:.4f} log loss")
        lines.append(f"  Home bias (55%):  {_log_loss(y_true, [0.55]*n):.4f} log loss")
        lines.append(f"  Always home:      {home_wins/n:.1%} accuracy")
        lines.append("")
        lines.append(f"Improvement over coin flip: {improvement:+.1f}%")
        lines.append("=" * 60)

        report = "\n".join(lines)
        print("\n" + report)

        # Assert the report contains key metrics
        assert "Games analyzed" in report
        assert "Log loss" in report
        assert "Model accuracy" in report
        assert "Improvement over coin flip" in report
        assert "Lookahead bias" in report

    def test_simulated_roi(self, predictions):
        """Simulate flat-bet ROI when the model finds an edge vs implied odds.

        For each game where the model's predicted probability exceeds the
        implied probability by more than 3 percentage points, we simulate a
        flat $100 bet on the model's pick. Implied odds are approximated from
        a 55% home-win base rate (no real line data available).

        We do NOT require positive ROI — that would be too strict for a model
        without real closing-line data. Instead we assert the model does not
        hemorrhage money (ROI > -30%).
        """
        BET_SIZE = 100.0
        EDGE_THRESHOLD = 0.03
        # Approximate implied probability: 55% home / 45% away (no real odds).
        IMPLIED_HOME = 0.55

        total_wagered = 0.0
        total_returned = 0.0
        bets_placed = 0

        for p in predictions:
            pred = p["predicted_home_prob"]

            # Check for edge on home side
            if pred - IMPLIED_HOME > EDGE_THRESHOLD:
                total_wagered += BET_SIZE
                bets_placed += 1
                if p["home_won"]:
                    # Fair-odds payout: BET_SIZE / implied_prob
                    total_returned += BET_SIZE / IMPLIED_HOME
            # Check for edge on away side
            elif (1 - pred) - (1 - IMPLIED_HOME) > EDGE_THRESHOLD:
                total_wagered += BET_SIZE
                bets_placed += 1
                if not p["home_won"]:
                    total_returned += BET_SIZE / (1 - IMPLIED_HOME)

        if total_wagered > 0:
            roi = (total_returned - total_wagered) / total_wagered * 100
        else:
            roi = 0.0

        print(f"\nSimulated ROI (flat $100 bets, >{EDGE_THRESHOLD:.0%} edge):")
        print(f"  Bets placed:   {bets_placed}")
        print(f"  Total wagered: ${total_wagered:,.0f}")
        print(f"  Total returned: ${total_returned:,.0f}")
        print(f"  ROI:           {roi:+.1f}%")
        if bets_placed == 0:
            print("  (No bets met the edge threshold — model is conservative)")

        # Model should not hemorrhage money. -30% ROI is the floor.
        assert roi > -30.0, f"Simulated ROI of {roi:+.1f}% is catastrophic (floor is -30%)"
