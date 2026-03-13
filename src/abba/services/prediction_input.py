"""Prediction input assembly for NHL game forecasts.

This module owns the boundary between storage records and the modeling layer.
It prepares a validated PredictionInput plus the legacy metadata the current
workflow and API surfaces still depend on.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any

from ..engine.hockey import HockeyAnalytics
from ..storage import Storage
from ..types import PredictionContext, PredictionInput, PredictionProvenance, build_prediction_input


# Required fields that must exist in goalie stats for the model to consume them.
_REQUIRED_GOALIE_FIELDS = {"save_pct", "gaa", "gsaa"}

# Current season — single source of truth for prediction input assembly.
CURRENT_SEASON = "2025-26"


@dataclass(frozen=True)
class PreparedNHLPredictionInput:
    """Fully prepared input bundle for NHL prediction orchestration."""

    prediction_input: PredictionInput
    data_provenance: dict[str, Any]
    data_warnings: list[str]
    home_goalie: dict[str, Any] | None
    away_goalie: dict[str, Any] | None
    home_player_impact: dict[str, float]
    away_player_impact: dict[str, float]
    data_source: str


def _select_starter(goalies: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Deterministic starter selection: explicit role, then max games_started."""
    if not goalies:
        return None
    for goalie in goalies:
        stats = goalie.get("stats", {})
        if stats.get("role") == "starter":
            return stats
    best = max(goalies, key=lambda goalie: goalie.get("stats", {}).get("games_started", 0))
    return best.get("stats")


def _validate_goalie_stats(
    stats: dict[str, Any] | None, team: str
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate goalie stats have required fields."""
    if stats is None:
        return None, [f"No goalie data for {team}"]
    missing = _REQUIRED_GOALIE_FIELDS - set(stats.keys())
    if missing:
        return None, [
            f"Goalie data for {team} missing required fields: {missing}. "
            f"Available keys: {set(stats.keys())}. "
            f"Goaltender matchup model will be excluded."
        ]
    return stats, []


def build_nhl_prediction_input(
    *,
    storage: Storage,
    hockey: HockeyAnalytics,
    game_id: str,
    last_refresh_ts: float | None = None,
    player_impact_fn: Any = None,
    season: str = CURRENT_SEASON,
) -> PreparedNHLPredictionInput | dict[str, Any]:
    """Build the shared NHL PredictionInput contract and legacy provenance."""
    game = storage.get_game_by_id(game_id)
    if not game:
        return {"error": f"game not found: {game_id}"}
    if game.get("sport") != "NHL":
        return {"error": "not an NHL game, use predict_game for other sports"}

    home = game.get("home_team", "")
    away = game.get("away_team", "")
    data_warnings: list[str] = []

    home_stats_list = storage.query_team_stats(team_id=home, sport="NHL", season=season)
    away_stats_list = storage.query_team_stats(team_id=away, sport="NHL", season=season)

    missing_stats = []
    if not home_stats_list:
        missing_stats.append(home)
    if not away_stats_list:
        missing_stats.append(away)
    if missing_stats:
        return {
            "error": "missing_team_stats",
            "missing_teams": missing_stats,
            "season": season,
            "recommendation": "Call refresh_data(source='nhl') first to populate team stats.",
        }

    home_stats = home_stats_list[0]
    away_stats = away_stats_list[0]
    home_season = home_stats.get("season", season)
    away_season = away_stats.get("season", season)
    if home_season != season or away_season != season:
        return {
            "error": "season_mismatch",
            "expected_season": season,
            "home_stats_season": home_season,
            "away_stats_season": away_season,
            "recommendation": "Team stats are from the wrong season. Call refresh_data() to update.",
        }

    home_adv_list = storage.query_nhl_advanced_stats(team_id=home, season=season)
    away_adv_list = storage.query_nhl_advanced_stats(team_id=away, season=season)
    home_adv = home_adv_list[0].get("stats", {}) if home_adv_list else None
    away_adv = away_adv_list[0].get("stats", {}) if away_adv_list else None
    if not home_adv or not away_adv:
        data_warnings.append("Advanced stats (Corsi/xG) absent — features will use neutral defaults")

    home_goalies = storage.query_goaltender_stats(team=home, season=season)
    away_goalies = storage.query_goaltender_stats(team=away, season=season)
    raw_home_goalie = _select_starter(home_goalies)
    raw_away_goalie = _select_starter(away_goalies)
    home_goalie, home_goalie_warnings = _validate_goalie_stats(raw_home_goalie, home)
    away_goalie, away_goalie_warnings = _validate_goalie_stats(raw_away_goalie, away)
    data_warnings.extend(home_goalie_warnings)
    data_warnings.extend(away_goalie_warnings)

    if player_impact_fn:
        home_player_impact = player_impact_fn(home)
        away_player_impact = player_impact_fn(away)
    else:
        home_player_impact = {"injury_impact": 0.0, "top_scorer_available": 1.0, "roster_completeness": 1.0}
        away_player_impact = {"injury_impact": 0.0, "top_scorer_available": 1.0, "roster_completeness": 1.0}

    game_odds = storage.query_odds(game_id=game_id, latest_only=True)
    odds_status = "present" if game_odds else "absent"
    if not game_odds:
        data_warnings.append(
            f"No odds data matched game_id={game_id}. "
            "Market blend model excluded. Possible ID mismatch between schedule and odds providers."
        )

    features = hockey.build_nhl_features(
        home_stats,
        away_stats,
        home_advanced=home_adv,
        away_advanced=away_adv,
        home_goalie=home_goalie,
        away_goalie=away_goalie,
        odds_data=game_odds,
    )
    features["home_injury_impact"] = home_player_impact["injury_impact"]
    features["away_injury_impact"] = away_player_impact["injury_impact"]
    features["home_roster_completeness"] = home_player_impact["roster_completeness"]
    features["away_roster_completeness"] = away_player_impact["roster_completeness"]

    defaulted_features: list[str] = []
    if not home_adv or not away_adv:
        defaulted_features.extend(["home_corsi_pct", "away_corsi_pct", "home_xgf_pct", "away_xgf_pct"])
    if not home_goalie or not away_goalie:
        defaulted_features.append("goaltender_edge")
    defaulted_features.append("rest_edge")
    if not game_odds:
        defaulted_features.append("market_implied_prob")
    if defaulted_features:
        data_warnings.append(
            f"Defaulted features (neutral values, not measured): {defaulted_features}"
        )

    now = _dt.datetime.now().isoformat()

    def _as_of(val: Any) -> str:
        if val is None:
            return now
        return str(val) if not isinstance(val, str) else val

    data_provenance = {
        "home_team_stats": {
            "status": "present",
            "season": home_season,
            "source": home_stats.get("source", "unknown"),
            "as_of": _as_of(home_stats.get("updated_at")),
        },
        "away_team_stats": {
            "status": "present",
            "season": away_season,
            "source": away_stats.get("source", "unknown"),
            "as_of": _as_of(away_stats.get("updated_at")),
        },
        "home_advanced_stats": {"status": "present" if home_adv else "absent", "season": season, "as_of": now},
        "away_advanced_stats": {"status": "present" if away_adv else "absent", "season": season, "as_of": now},
        "home_goaltender": {
            "status": "present" if home_goalie else "absent",
            "name": home_goalie.get("name", "unknown") if home_goalie else None,
            "selection_method": "role_tag" if (home_goalie and home_goalie.get("role")) else "max_games_started" if home_goalie else "none",
            "season": season,
            "as_of": now,
        },
        "away_goaltender": {
            "status": "present" if away_goalie else "absent",
            "name": away_goalie.get("name", "unknown") if away_goalie else None,
            "selection_method": "role_tag" if (away_goalie and away_goalie.get("role")) else "max_games_started" if away_goalie else "none",
            "season": season,
            "as_of": now,
        },
        "odds": {"status": odds_status, "season": season, "books_matched": len(game_odds), "as_of": now},
    }

    time_since_refresh = (
        round((_dt.datetime.now().timestamp() - last_refresh_ts), 1) if last_refresh_ts is not None else None
    )
    keyed_provenance: PredictionProvenance = {
        "team_stats": {
            "status": "present",
            "source": home_stats.get("source", "unknown"),
            "as_of": _as_of(home_stats.get("updated_at")),
            "season": season,
            "freshness_seconds": time_since_refresh,
            "defaulted_features": [],
            "notes": [],
        },
        "advanced_stats": {
            "status": "present" if home_adv and away_adv else "absent",
            "source": "moneypuck" if home_adv and away_adv else "default",
            "as_of": now,
            "season": season,
            "freshness_seconds": time_since_refresh,
            "defaulted_features": [
                feature
                for feature in ("home_corsi_pct", "away_corsi_pct", "home_xgf_pct", "away_xgf_pct")
                if feature in defaulted_features
            ],
            "notes": ["Optional analytics missing; neutral defaults used"] if (not home_adv or not away_adv) else [],
        },
        "goaltenders": {
            "status": "present" if home_goalie and away_goalie else "absent",
            "source": "nhl_api" if home_goalie and away_goalie else "default",
            "as_of": now,
            "season": season,
            "freshness_seconds": time_since_refresh,
            "defaulted_features": ["goaltender_edge"] if "goaltender_edge" in defaulted_features else [],
            "notes": home_goalie_warnings + away_goalie_warnings,
        },
        "odds": {
            "status": odds_status,
            "source": "odds_api" if game_odds else "default",
            "as_of": now,
            "season": season,
            "freshness_seconds": time_since_refresh,
            "defaulted_features": ["market_implied_prob"] if "market_implied_prob" in defaulted_features else [],
            "notes": ["No matching odds snapshot; market feature defaulted"] if not game_odds else [],
        },
        "roster": {
            "status": "present",
            "source": "computed" if player_impact_fn else "default",
            "as_of": now,
            "season": season,
            "freshness_seconds": time_since_refresh,
            "defaulted_features": [],
            "notes": [],
        },
        "rest": {
            "status": "defaulted",
            "source": "default",
            "as_of": now,
            "season": season,
            "freshness_seconds": None,
            "defaulted_features": ["rest_edge"],
            "notes": ["Rest edge is not yet wired into PredictionService inputs"],
        },
    }
    context_only: PredictionContext = {
        "home_goaltender": home_goalie,
        "away_goaltender": away_goalie,
        "player_impact": {"home": home_player_impact, "away": away_player_impact},
        "data_warnings": list(data_warnings),
        "context_only_features": {"rest_edge_reason": "workflow-computed, not yet model-fed"},
    }
    prediction_input = build_prediction_input(
        game_id=game_id,
        home_team=home,
        away_team=away,
        sport="NHL",
        season=season,
        features=features,
        defaulted_features=defaulted_features,
        provenance=keyed_provenance,
        context_only=context_only,
    )

    return PreparedNHLPredictionInput(
        prediction_input=prediction_input,
        data_provenance=data_provenance,
        data_warnings=data_warnings,
        home_goalie=home_goalie,
        away_goalie=away_goalie,
        home_player_impact=home_player_impact,
        away_player_impact=away_player_impact,
        data_source=home_stats.get("source", "unknown"),
    )
