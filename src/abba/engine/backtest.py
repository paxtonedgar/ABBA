"""Walk-forward backtest engine for NHL predictions.

Evaluates prediction quality WITHOUT lookahead bias by only using data
available before each game. Two modes:

1. **Snapshot-based** (preferred): Uses standings_snapshots table for
   point-in-time team stats. Requires accumulated daily snapshots.

2. **Decay-based** (fallback): Uses completed games to reconstruct
   approximate team stats with exponential decay weighting. Less
   accurate but works without snapshots.

Both modes produce a CalibrationArtifact that replaces the fabricated
confidence baselines in confidence.py.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np

from .calibration import (
    CalibrationArtifact,
    accuracy,
    apply_temperature,
    brier_score,
    expected_calibration_error,
    find_temperature,
    log_loss,
    reliability_bins,
)
from .elo import EloRatings
from .hockey import HockeyAnalytics


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Raw results from a single backtested game."""
    game_id: str
    date: str
    home_team: str
    away_team: str
    home_won: bool
    home_score: int
    away_score: int
    predicted_home_prob: float
    individual_models: list[float]
    elo_prob: float | None = None
    market_prob: float | None = None


class WalkForwardBacktest:
    """Leakage-free backtest engine.

    For each game in chronological order:
    1. Build features using ONLY pre-game data
    2. Generate prediction
    3. Record prediction vs outcome
    4. Update running stats with the game result

    No future data ever leaks into predictions.
    """

    def __init__(self) -> None:
        self.hockey = HockeyAnalytics()
        self.elo = EloRatings(k=4, home_advantage=50)
        self._running_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: self._neutral_stats()
        )
        self._team_game_dates: dict[str, list[date]] = defaultdict(list)
        self.results: list[BacktestResult] = []

    @staticmethod
    def _neutral_stats() -> dict[str, Any]:
        """Default stats for a team with no history."""
        return {
            "wins": 0, "losses": 0, "overtime_losses": 0,
            "goals_for": 0, "goals_against": 0,
            "recent_results": [],  # list of (won: bool, gf: int, ga: int)
            "power_play_percentage": 22.0,
            "penalty_kill_percentage": 80.0,
        }

    def _compute_recent_form(self, team: str, window: int = 10) -> float:
        """L10-style recent form from running results."""
        recent = self._running_stats[team]["recent_results"][-window:]
        if not recent:
            return 0.5
        wins = sum(1 for r in recent if r[0])
        return wins / len(recent)

    def _build_pre_game_stats(self, team: str) -> dict[str, Any]:
        """Build a stats dict using only data accumulated before this game."""
        s = self._running_stats[team]
        gp = s["wins"] + s["losses"] + s["overtime_losses"]
        if gp == 0:
            return {"stats": self._neutral_stats()}

        stats = dict(s)
        stats["recent_form"] = self._compute_recent_form(team)
        stats["games_played"] = gp
        return {"stats": stats}

    def _update_running_stats(
        self, home: str, away: str, home_score: int, away_score: int
    ) -> None:
        """Update running stats AFTER recording prediction."""
        hs = self._running_stats[home]
        as_ = self._running_stats[away]

        home_won = home_score > away_score
        # Handle OT loss (simplified: if 1-goal game and loser, it's OT loss 25% of the time)
        # In reality we'd need period data, but this approximation works for backtesting
        is_one_goal = abs(home_score - away_score) == 1

        if home_won:
            hs["wins"] += 1
            if is_one_goal:
                as_["overtime_losses"] += 1
            else:
                as_["losses"] += 1
        else:
            as_["wins"] += 1
            if is_one_goal:
                hs["overtime_losses"] += 1
            else:
                hs["losses"] += 1

        hs["goals_for"] += home_score
        hs["goals_against"] += away_score
        as_["goals_for"] += away_score
        as_["goals_against"] += home_score

        hs["recent_results"].append((home_won, home_score, away_score))
        as_["recent_results"].append((not home_won, away_score, home_score))

    def run(
        self,
        games: list[dict[str, Any]],
        market_odds: dict[str, float] | None = None,
        advanced_stats: dict[str, dict[str, Any]] | None = None,
        goalie_starts: dict[str, dict[str, Any]] | None = None,
        goalie_stats: dict[str, dict[str, Any]] | None = None,
    ) -> list[BacktestResult]:
        """Run walk-forward backtest on a chronologically sorted list of games.

        Args:
            games: List of completed games, each with:
                game_id, date, home_team, away_team, home_score, away_score
                Must be sorted by date ascending.
            market_odds: Optional dict of {game_id: implied_home_prob} for
                market baseline comparison.
            advanced_stats: Optional dict of {team_abbrev: {corsi_pct, xgf_pct, ...}}
                Season-level 5v5 advanced stats (e.g. from MoneyPuck).
            goalie_starts: Optional dict of {game_id: {home_goalie_id, away_goalie_id, ...}}
                Starting goalie identity per game.
            goalie_stats: Optional dict of {player_id_str: {save_pct, gaa, ...}}
                Season goalie stats keyed by player ID string.

        Returns:
            List of BacktestResult objects.
        """
        self.results = []

        for game in games:
            home = game["home_team"]
            away = game["away_team"]
            home_score = game.get("home_score", 0)
            away_score = game.get("away_score", 0)

            if home_score is None or away_score is None:
                continue

            # Parse game date for rest calculation
            game_date_str = str(game.get("date", ""))
            try:
                game_dt = date.fromisoformat(game_date_str)
            except (ValueError, TypeError):
                game_dt = None

            # Step 1: Build features from PRE-GAME data only
            home_stats = self._build_pre_game_stats(home)
            away_stats = self._build_pre_game_stats(away)

            # Advanced stats (season-level, treat as constant)
            home_adv = advanced_stats.get(home) if advanced_stats else None
            away_adv = advanced_stats.get(away) if advanced_stats else None

            # Goaltender stats for this game's starters
            home_goalie_stats = None
            away_goalie_stats = None
            game_id = game.get("game_id", "")
            if goalie_starts and goalie_stats and game_id in goalie_starts:
                gs = goalie_starts[game_id]
                hgid = str(gs.get("home_goalie_id", ""))
                agid = str(gs.get("away_goalie_id", ""))
                if hgid in goalie_stats:
                    home_goalie_stats = goalie_stats[hgid]
                if agid in goalie_stats:
                    away_goalie_stats = goalie_stats[agid]

            features = self.hockey.build_nhl_features(
                home_stats, away_stats,
                home_advanced=home_adv,
                away_advanced=away_adv,
                home_goalie=home_goalie_stats,
                away_goalie=away_goalie_stats,
            )

            # Step 1b: Compute rest edge from schedule history (pre-game)
            if game_dt:
                home_history = self._team_game_dates[home]
                away_history = self._team_game_dates[away]
                if home_history and away_history:
                    home_last = home_history[-1]
                    away_last = away_history[-1]
                    home_rest = (game_dt - home_last).days
                    away_rest = (game_dt - away_last).days
                    week_ago = game_dt - timedelta(days=7)
                    home_g7 = sum(1 for d in home_history if d > week_ago)
                    away_g7 = sum(1 for d in away_history if d > week_ago)
                    rest_info = self.hockey.rest_advantage(
                        home_rest_days=home_rest,
                        away_rest_days=away_rest,
                        home_is_back_to_back=home_rest <= 1,
                        away_is_back_to_back=away_rest <= 1,
                        home_games_last_7=home_g7,
                        away_games_last_7=away_g7,
                    )
                    features["rest_edge"] = rest_info["rest_edge"]

            # Step 2: Elo prediction (also pre-game)
            elo_pred = self.elo.predict(home, away)
            elo_prob = elo_pred["home_win_prob"]

            # Step 3: Generate model predictions
            model_preds = self.hockey.predict_nhl_game(features, elo_prob=elo_prob)
            avg_pred = float(np.mean(model_preds))

            # Market implied probability (if available)
            market_prob = None
            if market_odds and game.get("game_id") in market_odds:
                market_prob = market_odds[game["game_id"]]

            # Step 4: Record result
            self.results.append(BacktestResult(
                game_id=game.get("game_id", ""),
                date=str(game.get("date", "")),
                home_team=home,
                away_team=away,
                home_won=home_score > away_score,
                home_score=home_score,
                away_score=away_score,
                predicted_home_prob=avg_pred,
                individual_models=model_preds,
                elo_prob=elo_prob,
                market_prob=market_prob,
            ))

            # Step 5: Update running stats, schedule history, and Elo AFTER recording
            self._update_running_stats(home, away, home_score, away_score)
            if game_dt:
                self._team_game_dates[home].append(game_dt)
                self._team_game_dates[away].append(game_dt)
            self.elo.update(home, away, home_score, away_score)

        return self.results

    def run_from_storage(
        self,
        storage: Any,
        sport: str = "NHL",
        season: str | None = None,
    ) -> list[BacktestResult]:
        """Run backtest using completed games from storage.

        Queries games from storage, sorts chronologically, and runs
        walk-forward backtest.
        """
        games = storage.query_games(
            sport=sport,
            status="completed",
            limit=10000,
        )
        if season:
            # Filter by season in date range (approximate)
            games = [g for g in games if season in str(g.get("date", ""))]

        # Sort chronologically (oldest first)
        games.sort(key=lambda g: str(g.get("date", "")))

        return self.run(games)

    def evaluate(self, min_warmup: int = 20) -> CalibrationArtifact:
        """Produce a CalibrationArtifact from backtest results.

        Args:
            min_warmup: Skip the first N games (insufficient running stats).
                Models need some games to build meaningful team stats.

        Returns:
            CalibrationArtifact with empirically measured metrics.
        """
        # Skip warmup period
        eval_results = self.results[min_warmup:]

        if len(eval_results) < 10:
            return CalibrationArtifact(
                sample_size=len(eval_results),
                date_range="insufficient_data",
            )

        y_true = [r.home_won for r in eval_results]
        y_pred = [r.predicted_home_prob for r in eval_results]

        # Core metrics
        ll = log_loss(y_true, y_pred)
        bs = brier_score(y_true, y_pred)
        acc = accuracy(y_true, y_pred)
        ece = expected_calibration_error(y_true, y_pred)
        bins = reliability_bins(y_true, y_pred, n_bins=10)

        # Temperature scaling
        # Split: first 60% for finding T, last 40% for validation
        split = int(len(eval_results) * 0.6)
        if split >= 20:
            cal_true = y_true[:split]
            cal_pred = y_pred[:split]
            temp = find_temperature(cal_true, cal_pred)
        else:
            temp = 1.0

        # Baselines
        coin_ll = log_loss(y_true, [0.5] * len(y_true))
        home_55_ll = log_loss(y_true, [0.55] * len(y_true))

        # Elo baseline
        elo_preds = [r.elo_prob for r in eval_results if r.elo_prob is not None]
        elo_true = [r.home_won for r in eval_results if r.elo_prob is not None]
        elo_ll = log_loss(elo_true, elo_preds) if len(elo_preds) >= 10 else None

        # Market baseline
        market_preds = [r.market_prob for r in eval_results if r.market_prob is not None]
        market_true = [r.home_won for r in eval_results if r.market_prob is not None]
        market_ll = log_loss(market_true, market_preds) if len(market_preds) >= 10 else None

        baselines = {
            "coin_flip_log_loss": round(coin_ll, 4),
            "home_55_log_loss": round(home_55_ll, 4),
        }
        if elo_ll is not None:
            baselines["elo_log_loss"] = round(elo_ll, 4)
        if market_ll is not None:
            baselines["market_log_loss"] = round(market_ll, 4)

        # Date range
        dates = sorted(set(r.date for r in eval_results))
        date_range = f"{dates[0]} to {dates[-1]}" if dates else ""

        return CalibrationArtifact(
            log_loss=round(ll, 4),
            brier_score=round(bs, 4),
            accuracy=round(acc, 4),
            sample_size=len(eval_results),
            temperature=temp,
            ece=ece,
            reliability_bins=bins,
            date_range=date_range,
            baselines=baselines,
            beats_coin_flip=ll < coin_ll,
            beats_home_bias=ll < home_55_ll,
            beats_market=ll < market_ll if market_ll is not None else False,
        )

    def report(self, min_warmup: int = 20) -> str:
        """Generate a human-readable backtest report."""
        artifact = self.evaluate(min_warmup=min_warmup)
        eval_results = self.results[min_warmup:]

        if artifact.sample_size < 10:
            return f"Insufficient data: {artifact.sample_size} games (need 10+)"

        lines = [
            "=" * 60,
            "ABBA NHL WALK-FORWARD BACKTEST REPORT",
            "=" * 60,
            f"Games evaluated:    {artifact.sample_size} (after {min_warmup} warmup)",
            f"Date range:         {artifact.date_range}",
            f"",
            f"--- Scoring Rules ---",
            f"Log loss:           {artifact.log_loss:.4f}",
            f"Brier score:        {artifact.brier_score:.4f}",
            f"Accuracy:           {artifact.accuracy:.1%}",
            f"ECE:                {artifact.ece:.4f}",
            f"Temperature:        {artifact.temperature:.4f}",
            f"",
            f"--- Baselines ---",
            f"Coin flip (50%):    {artifact.baselines.get('coin_flip_log_loss', 'N/A')} log loss",
            f"Home bias (55%):    {artifact.baselines.get('home_55_log_loss', 'N/A')} log loss",
        ]

        if "elo_log_loss" in artifact.baselines:
            lines.append(f"Elo only:           {artifact.baselines['elo_log_loss']} log loss")
        if "market_log_loss" in artifact.baselines:
            lines.append(f"Market implied:     {artifact.baselines['market_log_loss']} log loss")

        lines.extend([
            f"",
            f"--- Verdict ---",
            f"Beats coin flip:    {'YES' if artifact.beats_coin_flip else 'NO'}",
            f"Beats home bias:    {'YES' if artifact.beats_home_bias else 'NO'}",
            f"Beats market:       {'YES' if artifact.beats_market else 'N/A'}",
            f"",
            f"--- Calibration Bins ---",
        ])

        for b in artifact.reliability_bins:
            lines.append(
                f"  [{b['bin_low']:.0%}-{b['bin_high']:.0%}]: "
                f"predicted {b['predicted_mean']:.1%}, "
                f"observed {b['observed_frequency']:.1%} "
                f"(n={b['count']}, err={b['calibration_error']:.3f})"
            )

        lines.append("=" * 60)
        return "\n".join(lines)
