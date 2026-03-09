"""Feature engineering for game prediction.

Takes raw game/team/player/weather data and produces feature vectors
for the ensemble models. Each feature has documented meaning and units.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class FeatureEngine:
    """Produces feature vectors from raw sports data."""

    # Feature definitions for documentation / schema discovery
    FEATURE_SCHEMA = {
        "home_win_pct": "Home team win percentage (0-1)",
        "away_win_pct": "Away team win percentage (0-1)",
        "home_run_diff_per_game": "Home team avg run differential per game",
        "away_run_diff_per_game": "Away team avg run differential per game",
        "home_recent_form": "Home team win rate last 10 games (0-1)",
        "away_recent_form": "Away team win rate last 10 games (0-1)",
        "home_advantage": "Historical home win rate for sport (constant per sport)",
        "temp_impact": "Temperature impact factor (-1 to 1, 0 = neutral 72F)",
        "wind_impact": "Wind impact factor (0 to 1, higher = more impact)",
        "precip_risk": "Precipitation probability (0-1)",
    }

    def build_features(
        self,
        home_team_stats: dict[str, Any],
        away_team_stats: dict[str, Any],
        weather: dict[str, Any] | None = None,
        sport: str = "MLB",
    ) -> dict[str, float]:
        """Build feature vector from raw data.

        Returns dict of feature_name -> value, not a numpy array,
        so agents can inspect what went into the prediction.
        """
        features: dict[str, float] = {}

        # Team performance features
        hs = home_team_stats.get("stats", home_team_stats)
        as_ = away_team_stats.get("stats", away_team_stats)

        h_wins = hs.get("wins", 40)
        h_losses = hs.get("losses", 40)
        a_wins = as_.get("wins", 40)
        a_losses = as_.get("losses", 40)

        h_games = max(h_wins + h_losses, 1)
        a_games = max(a_wins + a_losses, 1)

        features["home_win_pct"] = h_wins / h_games
        features["away_win_pct"] = a_wins / a_games

        # Run/goal differential
        if sport.upper() == "MLB":
            h_scored = hs.get("runs_scored", 0)
            h_allowed = hs.get("runs_allowed", 0)
            a_scored = as_.get("runs_scored", 0)
            a_allowed = as_.get("runs_allowed", 0)
        else:
            h_scored = hs.get("goals_for", 0)
            h_allowed = hs.get("goals_against", 0)
            a_scored = as_.get("goals_for", 0)
            a_allowed = as_.get("goals_against", 0)

        features["home_run_diff_per_game"] = (h_scored - h_allowed) / h_games
        features["away_run_diff_per_game"] = (a_scored - a_allowed) / a_games

        # Recent form (if available, else fallback to season win pct)
        features["home_recent_form"] = hs.get("recent_form", features["home_win_pct"])
        features["away_recent_form"] = as_.get("recent_form", features["away_win_pct"])

        # Home advantage constant by sport
        home_adv = {"MLB": 0.54, "NHL": 0.55, "NBA": 0.60, "NFL": 0.57}
        features["home_advantage"] = home_adv.get(sport.upper(), 0.54)

        # Weather features
        if weather:
            temp = weather.get("temperature", 72)
            # Normalized: 72F is neutral (0), extremes push toward +/-1
            features["temp_impact"] = float(np.clip((temp - 72) / 30, -1, 1))
            features["wind_impact"] = float(np.clip(weather.get("wind_speed", 0) / 25, 0, 1))
            features["precip_risk"] = float(weather.get("precipitation_chance", 0))
        else:
            features["temp_impact"] = 0.0
            features["wind_impact"] = 0.0
            features["precip_risk"] = 0.0

        return features

    def features_to_vector(self, features: dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy array in canonical order."""
        keys = sorted(self.FEATURE_SCHEMA.keys())
        return np.array([features.get(k, 0.0) for k in keys], dtype=np.float64)

    def predict_from_features(self, features: dict[str, float]) -> list[float]:
        """Generate multiple model-like predictions from features.

        Uses different weighting strategies to simulate ensemble diversity.
        These are heuristic models -- they produce reasonable predictions
        from the feature set without requiring trained sklearn models.
        Real deployment would swap these for serialized model artifacts.
        """
        hw = features.get("home_win_pct", 0.5)
        aw = features.get("away_win_pct", 0.5)
        hrd = features.get("home_run_diff_per_game", 0)
        ard = features.get("away_run_diff_per_game", 0)
        hrf = features.get("home_recent_form", 0.5)
        arf = features.get("away_recent_form", 0.5)
        ha = features.get("home_advantage", 0.54)
        temp = features.get("temp_impact", 0)
        wind = features.get("wind_impact", 0)

        # Model 1: Win percentage + home advantage (log5 method)
        # log5: P(A beats B) = (pA - pA*pB) / (pA + pB - 2*pA*pB)
        pa = hw
        pb = aw
        denom = pa + pb - 2 * pa * pb
        if abs(denom) < 1e-8:
            m1 = 0.5
        else:
            m1 = (pa - pa * pb) / denom
        # Blend with home advantage
        m1 = 0.7 * m1 + 0.3 * ha

        # Model 2: Run differential (pythagorean expectation inspired)
        # Higher exponent = more separation
        exp = 1.83 if features.get("home_advantage", 0.54) < 0.56 else 2.0  # MLB vs NHL
        h_strength = max(hrd + 5, 0.01) ** exp
        a_strength = max(ard + 5, 0.01) ** exp
        total = h_strength + a_strength
        m2 = h_strength / total if total > 0 else 0.5

        # Model 3: Recent form weighted
        m3 = 0.6 * hrf + 0.4 * (1.0 - arf)
        m3 = 0.8 * m3 + 0.2 * ha

        # Model 4: Combined with weather adjustment
        base = (m1 + m2 + m3) / 3
        weather_adj = -0.02 * wind - 0.01 * abs(temp)  # bad weather = more variance, slight home disadvantage
        m4 = float(np.clip(base + weather_adj, 0.01, 0.99))

        return [
            float(np.clip(m1, 0.01, 0.99)),
            float(np.clip(m2, 0.01, 0.99)),
            float(np.clip(m3, 0.01, 0.99)),
            m4,
        ]
