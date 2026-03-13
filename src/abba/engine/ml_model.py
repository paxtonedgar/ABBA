"""Local ML model for NHL game prediction.

Trains a gradient boosting classifier on completed games at startup.
Supports CatBoost (preferred, if installed) with sklearn GBM fallback.
Inference is <1ms per game. No network calls, no API keys.

The model is optional — if insufficient training data exists (<30 games),
it returns None and the ensemble proceeds without it.

Feature contract: consumes the same feature keys that build_nhl_features
produces. See types.MODEL_REQUIRED_FEATURES and MODEL_OPTIONAL_FEATURES.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# Feature keys in canonical order — must match what build_nhl_features produces.
# These are ALL model-consumed features (required + optional).
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

# Neutral defaults for features where real data may not be available.
# These must match MODEL_NEUTRAL_DEFAULTS in types.py.
_DEFAULTS = {
    "home_pts_pct": 0.5, "away_pts_pct": 0.5,
    "home_goal_diff_pg": 0.0, "away_goal_diff_pg": 0.0,
    "home_recent_form": 0.5, "away_recent_form": 0.5,
    "home_gf_per_game": 3.0, "home_ga_per_game": 3.0,
    "away_gf_per_game": 3.0, "away_ga_per_game": 3.0,
    "home_corsi_pct": 0.50, "away_corsi_pct": 0.50,
    "home_xgf_pct": 0.50, "away_xgf_pct": 0.50,
    "goaltender_edge": 0.0, "home_st_edge": 0.0, "rest_edge": 0.0,
}

_MIN_TRAINING_GAMES = 30


def _try_catboost() -> Any | None:
    """Try to import CatBoost. Returns CatBooostClassifier class or None."""
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except ImportError:
        return None


def _try_sklearn_gbm() -> Any | None:
    """Try to import sklearn GBM. Returns GradientBoostingClassifier class or None."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier
    except ImportError:
        return None


class NHLGameModel:
    """Gradient boosting model trained on local historical game data.

    Prefers CatBoost (handles categoricals, ordered boosting reduces leakage)
    but falls back to sklearn GradientBoosting if CatBoost is not installed.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._n_train: int = 0
        self._ready: bool = False
        self._backend: str = "none"

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def n_train(self) -> int:
        return self._n_train

    @property
    def backend(self) -> str:
        """Which ML backend is in use: 'catboost', 'sklearn', or 'none'."""
        return self._backend

    def train(
        self,
        games: list[dict],
        team_stats: dict[str, dict],
        advanced_stats: dict[str, dict] | None = None,
        goaltender_stats: dict[str, dict] | None = None,
    ) -> bool:
        """Train on completed games. Returns True if model is usable.

        Args:
            games: completed games with home_team, away_team, home_score, away_score
            team_stats: {team_id: stats_dict} for basic feature building
            advanced_stats: {team_id: advanced_stats_dict} for Corsi/xG features
            goaltender_stats: {team_id: goalie_stats_dict} for goaltender features
        """
        X, y = self._build_dataset(games, team_stats, advanced_stats, goaltender_stats)
        if len(X) < _MIN_TRAINING_GAMES:
            return False

        # Try CatBoost first, fall back to sklearn
        CatBoost = _try_catboost()
        if CatBoost is not None:
            self._model = CatBoost(
                iterations=200,
                depth=4,
                learning_rate=0.05,
                l2_leaf_reg=3.0,
                random_seed=42,
                verbose=0,
            )
            self._model.fit(X, y)
            self._backend = "catboost"
            log.info("Trained CatBoost model on %d games", len(X))
        else:
            GBM = _try_sklearn_gbm()
            if GBM is None:
                return False
            self._model = GBM(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.08,
                subsample=0.8,
                min_samples_leaf=5,
                random_state=42,
            )
            self._model.fit(X, y)
            self._backend = "sklearn"
            log.info("Trained sklearn GBM on %d games", len(X))

        self._n_train = len(X)
        self._ready = True
        return True

    def predict(self, features: dict[str, float]) -> float | None:
        """Return P(home_win) or None if model not ready."""
        if not self._ready:
            return None
        vec = np.array([[features.get(k, _DEFAULTS.get(k, 0.0)) for k in _FEATURE_KEYS]])

        if self._backend == "catboost":
            prob = self._model.predict_proba(vec)[0]
            # CatBoost: class order matches training label order (0, 1)
            return float(np.clip(prob[1], 0.01, 0.99))
        else:
            prob = self._model.predict_proba(vec)[0]
            idx = list(self._model.classes_).index(1) if 1 in self._model.classes_ else -1
            if idx < 0:
                return None
            return float(np.clip(prob[idx], 0.01, 0.99))

    def _build_dataset(
        self,
        games: list[dict],
        team_stats: dict[str, dict],
        advanced_stats: dict[str, dict] | None = None,
        goaltender_stats: dict[str, dict] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build feature matrix and labels from historical games."""
        rows: list[list[float]] = []
        labels: list[int] = []
        adv = advanced_stats or {}
        goalies = goaltender_stats or {}

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
            h_adv = adv.get(home, {})
            a_adv = adv.get(away, {})
            h_goalie = goalies.get(home, {})
            a_goalie = goalies.get(away, {})

            row = self._stats_to_features(h_stats, a_stats, h_adv, a_adv, h_goalie, a_goalie)
            rows.append(row)
            labels.append(1 if hs > as_ else 0)

        if not rows:
            return np.empty((0, len(_FEATURE_KEYS))), np.empty(0)
        return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.int32)

    @staticmethod
    def _stats_to_features(
        h: dict, a: dict,
        h_adv: dict | None = None, a_adv: dict | None = None,
        h_goalie: dict | None = None, a_goalie: dict | None = None,
    ) -> list[float]:
        """Extract feature vector from team stats dicts.

        Now uses real advanced stats and goaltender data when available,
        instead of hardcoding neutral defaults.
        """
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

        # Advanced stats: use real values when available, else neutral defaults
        h_adv = h_adv or {}
        a_adv = a_adv or {}
        home_corsi = _s(h_adv, "corsi_pct", 50.0) / 100.0
        away_corsi = _s(a_adv, "corsi_pct", 50.0) / 100.0
        home_xgf = _s(h_adv, "xgf_pct", 50.0) / 100.0
        away_xgf = _s(a_adv, "xgf_pct", 50.0) / 100.0

        # Goaltender edge
        h_goalie = h_goalie or {}
        a_goalie = a_goalie or {}
        h_sv = _s(h_goalie, "save_pct", 0.907)
        a_sv = _s(a_goalie, "save_pct", 0.907)
        # Same calibration as HockeyAnalytics.goaltender_matchup_edge
        sv_edge = (h_sv - a_sv) / 0.01 * 0.0375
        goaltender_edge = float(np.clip(sv_edge * 0.7, -0.5, 0.5))

        # Special teams edge
        h_pp = _s(h, "power_play_percentage", 22.0)
        h_pk = _s(h, "penalty_kill_percentage", 80.0)
        a_pp = _s(a, "power_play_percentage", 22.0)
        a_pk = _s(a, "penalty_kill_percentage", 80.0)
        st_edge = ((h_pp - a_pk) + (h_pk - a_pp)) / 100.0

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
            home_corsi,                              # home_corsi_pct
            away_corsi,                              # away_corsi_pct
            home_xgf,                                # home_xgf_pct
            away_xgf,                                # away_xgf_pct
            goaltender_edge,                         # goaltender_edge
            st_edge,                                 # home_st_edge
            0.0,                                     # rest_edge (requires schedule data)
        ]
