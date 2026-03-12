"""NHL-specific tools mixin."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...types import GoaltenderStatsRecord


# Required fields that must exist in goalie stats for the model to consume them.
# If any are missing, the goalie data is unusable and we fail closed.
_REQUIRED_GOALIE_FIELDS = {"save_pct", "gaa", "gsaa"}


def _select_starter(goalies: list[GoaltenderStatsRecord]) -> dict[str, Any] | None:
    """Deterministic starter selection: explicit role, then max games_started.

    Returns the stats dict for the selected goalie, or None if no goalies.
    Never falls back to arbitrary first-row.
    """
    if not goalies:
        return None

    # Prefer explicit role tag
    for g in goalies:
        stats = g.get("stats", {})
        if stats.get("role") == "starter":
            return stats

    # Fallback: most games_started (deterministic)
    best = max(goalies, key=lambda g: g.get("stats", {}).get("games_started", 0))
    return best.get("stats")


def _validate_goalie_stats(stats: dict[str, Any] | None, team: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate goalie stats have required fields. Returns (stats_or_None, warnings)."""
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


class NHLToolsMixin:
    """NHL prediction, goaltender, advanced stats, cap, roster, season review, and playoff tools."""

    # Current season — single source of truth for prediction-path queries
    _CURRENT_SEASON = "2025-26"

    def nhl_predict_game(
        self,
        game_id: str,
        method: str = "weighted",
    ) -> dict[str, Any]:
        """NHL-specific prediction with fail-closed guards on all data inputs.

        Delegates to PredictionService for core logic; wraps with _track for observability.
        """
        start = time.time()
        result = self.prediction.predict_nhl(
            game_id,
            method=method,
            version=self.VERSION,
            last_refresh_ts=getattr(self, '_last_refresh_ts', None),
            player_impact_fn=self._player_impact,
        )
        return self._track("nhl_predict_game", {"game_id": game_id}, result, start)

    def query_goaltender_stats(
        self,
        team: str | None = None,
        goaltender_id: str | None = None,
        season: str | None = None,
    ) -> dict[str, Any]:
        """Query NHL goaltender stats."""
        start = time.time()
        params = {k: v for k, v in locals().items() if k != "self" and v is not None}
        goalies = self.storage.query_goaltender_stats(
            goaltender_id=goaltender_id, team=team, season=season,
        )
        result = {"goaltenders": goalies, "count": len(goalies)}
        return self._track("query_goaltender_stats", params, result, start)

    def query_advanced_stats(
        self,
        team_id: str | None = None,
        season: str | None = None,
    ) -> dict[str, Any]:
        """Query NHL advanced stats."""
        start = time.time()
        params = {k: v for k, v in locals().items() if k != "self" and v is not None}
        stats = self.storage.query_nhl_advanced_stats(team_id=team_id, season=season)
        result = {"teams": stats, "count": len(stats)}
        return self._track("query_advanced_stats", params, result, start)

    def query_cap_data(
        self,
        team: str | None = None,
        season: str | None = None,
        position: str | None = None,
    ) -> dict[str, Any]:
        """Query salary cap data with cap analysis."""
        start = time.time()
        params = {k: v for k, v in locals().items() if k != "self" and v is not None}
        contracts = self.storage.query_salary_cap(team=team, season=season, position=position)

        analysis = None
        if team and contracts:
            roster_for_analysis = [
                {"name": c["name"], "position": c.get("position", ""),
                 "cap_hit": c.get("cap_hit", 0),
                 "contract_years_remaining": c.get("contract_years_remaining", 1),
                 "status": c.get("status", "active")}
                for c in contracts
            ]
            analysis = self.hockey.cap_analysis(roster_for_analysis)

        result = {"contracts": contracts, "count": len(contracts)}
        if analysis:
            result["cap_analysis"] = analysis
        return self._track("query_cap_data", params, result, start)

    def query_roster(
        self,
        team: str | None = None,
        season: str | None = None,
        position: str | None = None,
    ) -> dict[str, Any]:
        """Query team roster."""
        start = time.time()
        params = {k: v for k, v in locals().items() if k != "self" and v is not None}
        players = self.storage.query_roster(team=team, season=season, position=position)
        result = {"players": players, "count": len(players)}
        return self._track("query_roster", params, result, start)

    def season_review(
        self,
        team_id: str,
        season: str | None = None,
    ) -> dict[str, Any]:
        """Comprehensive NHL season review."""
        start = time.time()
        season = season or "2025-26"

        team_stats_list = self.storage.query_team_stats(team_id=team_id, sport="NHL", season=season)
        if not team_stats_list:
            return self._track("season_review", {"team_id": team_id},
                               {"error": f"no stats found for {team_id}"}, start)

        team_stats = team_stats_list[0].get("stats", {})

        adv_list = self.storage.query_nhl_advanced_stats(team_id=team_id, season=season)
        advanced = adv_list[0].get("stats", {}) if adv_list else None

        goalies = self.storage.query_goaltender_stats(team=team_id, season=season)
        goalie_stats = [g.get("stats", {}) for g in goalies] if goalies else None

        review = self.hockey.season_review(team_stats, advanced, goalie_stats)
        review["team_id"] = team_id
        review["season"] = season

        return self._track("season_review", {"team_id": team_id, "season": season}, review, start)

    def playoff_odds(
        self,
        team_id: str,
        season: str | None = None,
        division_cutline: int = 90,
        wildcard_cutline: int = 95,
    ) -> dict[str, Any]:
        """Estimate playoff probability from current points pace."""
        start = time.time()
        season = season or "2025-26"

        team_stats_list = self.storage.query_team_stats(team_id=team_id, sport="NHL", season=season)
        if not team_stats_list:
            return self._track("playoff_odds", {"team_id": team_id},
                               {"error": f"no stats found for {team_id}"}, start)

        stats = team_stats_list[0].get("stats", {})
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        otl = stats.get("overtime_losses", 0)
        gp = wins + losses + otl
        points = wins * 2 + otl
        remaining = 82 - gp

        result = self.hockey.playoff_probability(
            current_points=points,
            games_remaining=remaining,
            games_played=gp,
            division_cutline=division_cutline,
            wildcard_cutline=wildcard_cutline,
        )
        result["team_id"] = team_id
        result["season"] = season
        return self._track("playoff_odds", {"team_id": team_id}, result, start)
