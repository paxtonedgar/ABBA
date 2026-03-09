"""Data query tools mixin."""

from __future__ import annotations

import time
from typing import Any


class DataToolsMixin:
    """Query methods for games, odds, team stats, and schema discovery."""

    def query_games(
        self,
        sport: str | None = None,
        date: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        team: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Query games with filters. Returns structured game list."""
        start = time.time()
        params = {k: v for k, v in locals().items() if k != "self" and v is not None}
        games = self.storage.query_games(
            sport=sport, date=date, date_from=date_from, date_to=date_to,
            team=team, status=status, limit=limit,
        )
        result = {"games": games, "count": len(games)}
        return self._track("query_games", params, result, start)

    def query_odds(
        self,
        game_id: str | None = None,
        sportsbook: str | None = None,
        latest_only: bool = True,
    ) -> dict[str, Any]:
        """Query odds snapshots. Returns current lines across books."""
        start = time.time()
        params = {k: v for k, v in locals().items() if k != "self" and v is not None}
        odds = self.storage.query_odds(game_id=game_id, sportsbook=sportsbook, latest_only=latest_only)
        result = {"odds": odds, "count": len(odds)}
        return self._track("query_odds", params, result, start)

    def query_team_stats(
        self,
        team_id: str | None = None,
        sport: str | None = None,
        season: str | None = None,
    ) -> dict[str, Any]:
        """Query team statistics."""
        start = time.time()
        params = {k: v for k, v in locals().items() if k != "self" and v is not None}
        stats = self.storage.query_team_stats(team_id=team_id, sport=sport, season=season)
        result = {"teams": stats, "count": len(stats)}
        return self._track("query_team_stats", params, result, start)

    def list_sources(self) -> dict[str, Any]:
        """Schema discovery -- list available data tables and row counts."""
        start = time.time()
        tables = self.storage.list_tables()
        result = {"sources": tables}
        return self._track("list_sources", {}, result, start)

    def describe_dataset(self, table: str) -> dict[str, Any]:
        """Describe a data table's columns and types."""
        start = time.time()
        columns = self.storage.describe_table(table)
        result = {"table": table, "columns": columns}
        return self._track("describe_dataset", {"table": table}, result, start)
