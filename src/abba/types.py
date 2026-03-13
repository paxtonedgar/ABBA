"""Domain types for key data boundaries.

TypedDicts for the most-used record types crossing storage ↔ engine ↔ tool
boundaries. Annotation-only — zero runtime overhead since storage already
returns plain dicts.

Inner `stats` JSON blobs stay as dict[str, Any] because they vary across
sports and data sources.
"""

from __future__ import annotations

from typing import Any, TypedDict, cast


class Game(TypedDict, total=False):
    """A scheduled, live, or completed game."""
    game_id: str
    sport: str
    date: str
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None
    venue: str
    status: str
    metadata: dict[str, Any]
    source: str
    ingested_at: str


class TeamStatsRecord(TypedDict, total=False):
    """Team season statistics row from storage."""
    team_id: str
    sport: str
    season: str
    stats: dict[str, Any]
    source: str
    updated_at: str


class GoaltenderStatsRecord(TypedDict, total=False):
    """Goaltender season statistics row from storage."""
    goaltender_id: str
    team: str
    season: str
    stats: dict[str, Any]
    updated_at: str


class AdvancedStatsRecord(TypedDict, total=False):
    """NHL advanced stats (Corsi, xG, etc.) row from storage."""
    team_id: str
    season: str
    stats: dict[str, Any]
    updated_at: str


class OddsSnapshot(TypedDict, total=False):
    """A single odds snapshot from a sportsbook."""
    id: int
    game_id: str
    sportsbook: str
    market_type: str
    home_odds: float | None
    away_odds: float | None
    spread: float | None
    total: float | None
    over_odds: float | None
    under_odds: float | None
    captured_at: str


class RosterPlayer(TypedDict, total=False):
    """A player on a team roster."""
    player_id: str
    team: str
    season: str
    name: str
    position: str
    line_number: int | None
    stats: dict[str, Any]
    injury_status: str
    updated_at: str


# ---------------------------------------------------------------------------
# Model contract: PredictionInput → PredictionOutput
#
# This is the shared interface between Agent 1 (trust spine / modeling) and
# Agent 2 (data plumbing / workflows). Agent 2 prepares PredictionInput.
# Agent 1 consumes it for modeling and returns PredictionOutput.
# Neither should invent separate schemas.
# ---------------------------------------------------------------------------


class ModelFeatures(TypedDict, total=False):
    """Features consumed by the prediction model.

    REQUIRED features: must be present for prediction to run.
    OPTIONAL features: used if present, neutral defaults if not.
    """
    # --- REQUIRED: model will error without these ---
    home_pts_pct: float          # Home team points percentage (0-1)
    away_pts_pct: float          # Away team points percentage (0-1)
    home_goal_diff_pg: float     # Home team goal differential per game
    away_goal_diff_pg: float     # Away team goal differential per game
    home_games_played: float     # For regression to mean
    away_games_played: float     # For regression to mean
    home_gf_per_game: float      # Goals for per game
    home_ga_per_game: float      # Goals against per game
    away_gf_per_game: float      # Goals for per game
    away_ga_per_game: float      # Goals against per game

    # --- OPTIONAL: improve prediction when present ---
    home_recent_form: float      # L10 win rate (0-1)
    away_recent_form: float      # L10 win rate (0-1)
    home_corsi_pct: float        # 5v5 Corsi% (0-1, neutral=0.50)
    away_corsi_pct: float        # 5v5 Corsi% (0-1, neutral=0.50)
    home_xgf_pct: float          # Expected goals for% (0-1, neutral=0.50)
    away_xgf_pct: float          # Expected goals for% (0-1, neutral=0.50)
    goaltender_edge: float       # Goaltender matchup edge (-0.5 to 0.5)
    home_st_edge: float          # Special teams differential
    rest_edge: float             # Rest/schedule advantage
    market_implied_prob: float   # De-vigged market home win prob (0=absent)

    # --- Player-level (optional) ---
    home_injury_impact: float
    away_injury_impact: float
    home_roster_completeness: float
    away_roster_completeness: float


# Features that are REQUIRED for the model. If any are missing, prediction
# should fail-closed rather than silently defaulting.
MODEL_REQUIRED_FEATURES = frozenset({
    "home_pts_pct", "away_pts_pct",
    "home_goal_diff_pg", "away_goal_diff_pg",
    "home_games_played", "away_games_played",
    "home_gf_per_game", "home_ga_per_game",
    "away_gf_per_game", "away_ga_per_game",
})

# Features that improve the model when present but have known neutral defaults.
MODEL_OPTIONAL_FEATURES = frozenset({
    "home_recent_form", "away_recent_form",
    "home_corsi_pct", "away_corsi_pct",
    "home_xgf_pct", "away_xgf_pct",
    "goaltender_edge", "home_st_edge", "rest_edge",
    "market_implied_prob",
    "home_injury_impact", "away_injury_impact",
    "home_roster_completeness", "away_roster_completeness",
})

# Neutral defaults for optional features — the value that means "no signal."
MODEL_NEUTRAL_DEFAULTS: dict[str, float] = {
    "home_recent_form": 0.5,
    "away_recent_form": 0.5,
    "home_corsi_pct": 0.50,
    "away_corsi_pct": 0.50,
    "home_xgf_pct": 0.50,
    "away_xgf_pct": 0.50,
    "goaltender_edge": 0.0,
    "home_st_edge": 0.0,
    "rest_edge": 0.0,
    "market_implied_prob": 0.0,
    "home_injury_impact": 0.0,
    "away_injury_impact": 0.0,
    "home_roster_completeness": 1.0,
    "away_roster_completeness": 1.0,
}


class SourceProvenance(TypedDict, total=False):
    """Tracks one logical source boundary used to build a prediction."""
    status: str                 # "present" | "absent" | "defaulted"
    source: str                 # "sportradar", "moneypuck", "nhl_api", "computed", "default"
    as_of: str | None
    season: str | None
    freshness_seconds: float | None
    defaulted_features: list[str]
    notes: list[str]


class PredictionProvenance(TypedDict, total=False):
    """Keyed provenance map for stable cross-boundary access."""
    team_stats: SourceProvenance
    advanced_stats: SourceProvenance
    goaltenders: SourceProvenance
    odds: SourceProvenance
    roster: SourceProvenance
    rest: SourceProvenance


class PredictionContext(TypedDict, total=False):
    """Workflow-visible context that is not guaranteed to be model-consumed."""
    home_goaltender: dict[str, Any] | None
    away_goaltender: dict[str, Any] | None
    player_impact: dict[str, Any]
    data_warnings: list[str]
    context_only_features: dict[str, Any]


class PredictionInput(TypedDict, total=False):
    """Complete input to the prediction pipeline.

    Agent 2 (data plumbing) prepares this. Agent 1 (trust spine) consumes it.
    """
    game_id: str
    home_team: str
    away_team: str
    sport: str
    season: str
    required_features: ModelFeatures
    optional_features: ModelFeatures
    features: ModelFeatures
    defaulted_features: list[str]   # names of features that used neutral defaults
    provenance: PredictionProvenance
    context_only: PredictionContext


class PredictionOutput(TypedDict, total=False):
    """Complete output from the prediction pipeline.

    Agent 1 (trust spine) produces this. Workflows and UI consume it.
    """
    game_id: str
    home_team: str
    away_team: str
    sport: str
    season: str

    # Core prediction
    home_win_prob: float           # calibrated probability
    home_win_prob_raw: float       # pre-calibration probability
    individual_models: list[float]
    model_types: list[str]
    ensemble_method: str

    # Calibration metadata (replaces fabricated confidence)
    calibration_temperature: float
    calibration_ece: float
    calibration_status: str        # "empirically_validated" | "insufficient_data" | "uncalibrated"
    confidence_interval: dict[str, float]  # lower, upper, width
    reliability_grade: str

    # Provenance
    features: dict[str, float]
    defaulted_features: list[str]
    data_warnings: list[str]
    data_provenance: dict[str, Any]
    model_features_used: list[str]
    context_only: PredictionContext
    prediction_input: PredictionInput


def split_model_features(features: dict[str, Any]) -> tuple[ModelFeatures, ModelFeatures]:
    """Partition a flat feature dict into required and optional model inputs."""
    required = cast(ModelFeatures, {name: float(features[name]) for name in MODEL_REQUIRED_FEATURES if name in features})
    optional = cast(ModelFeatures, {name: float(features[name]) for name in MODEL_OPTIONAL_FEATURES if name in features})
    return required, optional


def missing_required_features(features: dict[str, Any]) -> list[str]:
    """Return required model features absent from a candidate input."""
    return sorted(name for name in MODEL_REQUIRED_FEATURES if name not in features)


def build_prediction_input(
    *,
    game_id: str,
    home_team: str,
    away_team: str,
    sport: str,
    season: str,
    features: dict[str, Any],
    defaulted_features: list[str] | None = None,
    provenance: PredictionProvenance | None = None,
    context_only: PredictionContext | None = None,
) -> PredictionInput:
    """Assemble and validate the shared Agent 1/Agent 2 prediction contract."""
    missing = missing_required_features(features)
    if missing:
        raise ValueError(
            f"PredictionInput missing required features: {missing}. "
            f"Available features: {sorted(features.keys())}"
        )

    required_features, optional_features = split_model_features(features)
    return {
        "game_id": game_id,
        "home_team": home_team,
        "away_team": away_team,
        "sport": sport,
        "season": season,
        "required_features": required_features,
        "optional_features": optional_features,
        "features": cast(ModelFeatures, {name: float(value) for name, value in features.items()}),
        "defaulted_features": list(defaulted_features or []),
        "provenance": provenance or {},
        "context_only": context_only or {},
    }
