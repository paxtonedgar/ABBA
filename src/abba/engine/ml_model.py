"""Local ML model for NHL game prediction.

Trains a GradientBoostingClassifier on completed games at startup.
Inference is <1ms per game. No network calls, no API keys.

The model is optional — if insufficient training data exists (<30 games),
it returns None and the ensemble proceeds without it.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# Feature keys in canonical order — must match what build_nhl_features produces
_FEATURE_KEYS = [
    "home_pts_pct", "away_pts_pct",
    "home_goal_diff_pg", "away_goal_diff_pg",
    "home_recent_form", "away_recent_form",
    "home_gf_per_game", "home_ga_per_game",
    "away_gf_per_game", "away_ga_per_game",
    "home_corsi_pct", "away_corsi_pct",
    "home_xgf_pct", "away_xgf_pct",
    "goaltender_edge", "home_st_edge", "rest_edge",
]

_MIN_TRAINING_GAMES = 30


class NHLGameModel:
    """Gradient boosting model trained on local historical game data."""

    def __init__(self) -> None:
        self._model: Any = None
        self._n_train: int = 0
        self._ready: bool = False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def n_train(self) -> int:
        return self._n_train

    def train(self, games: list[dict], team_stats: dict[str, dict]) -> bool:
        """Train on completed games. Returns True if model is usable.

        Args:
            games: completed games with home_team, away_team, home_score, away_score
            team_stats: {team_id: stats_dict} for feature building
        """
        try:
            from sklearn.ensemble import GradientBoostingClassifier
        except ImportError:
            return False

        X, y = self._build_dataset(games, team_stats)
        if len(X) < _MIN_TRAINING_GAMES:
            return False

        self._model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        self._model.fit(X, y)
        self._n_train = len(X)
        self._ready = True
        return True

    def predict(self, features: dict[str, float]) -> float | None:
        """Return P(home_win) or None if model not ready."""
        if not self._ready:
            return None
        vec = np.array([[features.get(k, 0.0) for k in _FEATURE_KEYS]])
        prob = self._model.predict_proba(vec)[0]
        # Class 1 = home win
        idx = list(self._model.classes_).index(1) if 1 in self._model.classes_ else -1
        if idx < 0:
            return None
        return float(np.clip(prob[idx], 0.01, 0.99))

    def _build_dataset(
        self, games: list[dict], team_stats: dict[str, dict]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build feature matrix and labels from historical games."""
        rows: list[list[float]] = []
        labels: list[int] = []

        for g in games:
            home = g.get("home_team", "")
            away = g.get("away_team", "")
            hs = g.get("home_score")
            as_ = g.get("away_score")

            if hs is None or as_ is None:
                continue
            if home not in team_stats or away not in team_stats:
                continue

            h_stats = team_stats[home]
            a_stats = team_stats[away]

            row = self._stats_to_features(h_stats, a_stats)
            rows.append(row)
            labels.append(1 if hs > as_ else 0)

        if not rows:
            return np.empty((0, len(_FEATURE_KEYS))), np.empty(0)
        return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.int32)

    @staticmethod
    def _stats_to_features(h: dict, a: dict) -> list[float]:
        """Extract feature vector from team stats dicts."""
        def _s(stats: dict, key: str, default: float = 0.0) -> float:
            return float(stats.get(key, default))

        h_wins = _s(h, "wins", 40)
        h_losses = _s(h, "losses", 40)
        h_otl = _s(h, "overtime_losses", 0)
        h_gp = max(h_wins + h_losses + h_otl, 1)
        h_gf = _s(h, "goals_for", 0)
        h_ga = _s(h, "goals_against", 0)

        a_wins = _s(a, "wins", 40)
        a_losses = _s(a, "losses", 40)
        a_otl = _s(a, "overtime_losses", 0)
        a_gp = max(a_wins + a_losses + a_otl, 1)
        a_gf = _s(a, "goals_for", 0)
        a_ga = _s(a, "goals_against", 0)

        h_pts_pct = (h_wins * 2 + h_otl) / (h_gp * 2)
        a_pts_pct = (a_wins * 2 + a_otl) / (a_gp * 2)

        return [
            h_pts_pct,                              # home_pts_pct
            a_pts_pct,                              # away_pts_pct
            (h_gf - h_ga) / h_gp,                  # home_goal_diff_pg
            (a_gf - a_ga) / a_gp,                  # away_goal_diff_pg
            _s(h, "recent_form", 0.5),              # home_recent_form
            _s(a, "recent_form", 0.5),              # away_recent_form
            h_gf / h_gp if h_gp > 0 else 3.0,      # home_gf_per_game
            h_ga / h_gp if h_gp > 0 else 3.0,      # home_ga_per_game
            a_gf / a_gp if a_gp > 0 else 3.0,      # away_gf_per_game
            a_ga / a_gp if a_gp > 0 else 3.0,      # away_ga_per_game
            0.50,                                    # home_corsi_pct (default)
            0.50,                                    # away_corsi_pct (default)
            0.50,                                    # home_xgf_pct (default)
            0.50,                                    # away_xgf_pct (default)
            0.0,                                     # goaltender_edge (default)
            0.0,                                     # home_st_edge (default)
            0.0,                                     # rest_edge (default)
        ]
