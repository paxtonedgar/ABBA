"""Domain types for key data boundaries.

TypedDicts for the most-used record types crossing storage ↔ engine ↔ tool
boundaries. Annotation-only — zero runtime overhead since storage already
returns plain dicts.

Inner `stats` JSON blobs stay as dict[str, Any] because they vary across
sports and data sources.
"""

from __future__ import annotations

from typing import Any, TypedDict


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
