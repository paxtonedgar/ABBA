"""PostgreSQL (Supabase) storage backend for ABBA.

Mirrors the DuckDB Storage API exactly -- same method signatures, same dict shapes.
Uses psycopg (v3) for JSONB support.

Usage:
    from abba.storage.postgres import PostgresStorage
    pg = PostgresStorage("postgresql://postgres:pw@host:6543/postgres")
"""

from __future__ import annotations

import json
import os
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .duckdb import (
    StorageValidationError,
    _validate_stats_keys,
    _TEAM_STATS_REQUIRED,
    _GOALIE_STATS_REQUIRED,
    _ADVANCED_STATS_REQUIRED,
)


def _json_col(val: Any) -> Any:
    """Parse a JSONB column if it came back as a string."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val
    return val


class PostgresStorage:
    """Supabase/Postgres storage backend -- API-compatible with DuckDB Storage."""

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or os.environ.get("SUPABASE_DB_URL", "")
        if not self.dsn:
            raise ValueError("No Postgres DSN. Set SUPABASE_DB_URL or pass dsn=.")
        self.conn = psycopg.connect(self.dsn, row_factory=dict_row, autocommit=True)

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _JSON_KEYS = frozenset({
        "stats", "metadata", "prediction", "input_params", "output_summary",
        "uncertainty", "data_trust", "workflow_gaps", "want_to_verify", "context_snapshot",
    })

    def _execute(self, sql: str, params: list | tuple | None = None) -> list[dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(sql, params or [])
            if cur.description:
                rows = cur.fetchall()
                for row in rows:
                    for key in self._JSON_KEYS:
                        if key in row:
                            row[key] = _json_col(row[key])
                return rows
            return []

    def _execute_one(self, sql: str, params: list | tuple | None = None) -> dict[str, Any] | None:
        rows = self._execute(sql, params)
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    # Data freshness
    # ------------------------------------------------------------------

    def record_refresh(self, table_name: str, source: str = "unknown", row_count: int = 0) -> None:
        self._execute("""
            INSERT INTO data_freshness (table_name, last_refresh_at, source, row_count)
            VALUES (%s, NOW(), %s, %s)
            ON CONFLICT (table_name) DO UPDATE SET
                last_refresh_at = NOW(), source = EXCLUDED.source, row_count = EXCLUDED.row_count
        """, [table_name, source, row_count])

    def get_last_refresh(self, table_name: str) -> float | None:
        row = self._execute_one(
            "SELECT EXTRACT(EPOCH FROM last_refresh_at) AS ts FROM data_freshness WHERE table_name = %s",
            [table_name],
        )
        return float(row["ts"]) if row else None

    # ------------------------------------------------------------------
    # Standings snapshots
    # ------------------------------------------------------------------

    def snapshot_standings(self, snapshot_date: str, team_stats: list[dict[str, Any]]) -> int:
        count = 0
        for ts in team_stats:
            self._execute("""
                INSERT INTO standings_snapshots (snapshot_date, team_id, sport, stats)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (snapshot_date, team_id) DO UPDATE SET stats = EXCLUDED.stats
            """, [snapshot_date, ts.get("team_id"), ts.get("sport", "NHL"),
                  json.dumps(ts.get("stats", {}))])
            count += 1
        return count

    # ------------------------------------------------------------------
    # Games  (list[dict] signature, matching DuckDB)
    # ------------------------------------------------------------------

    def upsert_games(self, games: list[dict[str, Any]]) -> int:
        if not games:
            return 0
        for g in games:
            self._execute("""
                INSERT INTO games (game_id, sport, date, home_team, away_team,
                    home_score, away_score, venue, status, metadata, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (game_id) DO UPDATE SET
                    home_score = EXCLUDED.home_score, away_score = EXCLUDED.away_score,
                    status = EXCLUDED.status, metadata = EXCLUDED.metadata,
                    source = EXCLUDED.source, ingested_at = NOW()
            """, [
                g["game_id"], g.get("sport", "NHL"), g["date"],
                g["home_team"], g["away_team"],
                g.get("home_score"), g.get("away_score"),
                g.get("venue"), g.get("status", "scheduled"),
                json.dumps(g.get("metadata", {})),
                g.get("source", "unknown"),
            ])
        return len(games)

    def query_games(self, sport: str | None = None, date: str | None = None,
                    date_from: str | None = None, date_to: str | None = None,
                    team: str | None = None, status: str | None = None,
                    limit: int = 100) -> list[dict[str, Any]]:
        conditions, params = [], []
        if sport:
            conditions.append("sport = %s"); params.append(sport.upper())
        if date:
            conditions.append("date = %s"); params.append(date)
        if date_from:
            conditions.append("date >= %s"); params.append(date_from)
        if date_to:
            conditions.append("date <= %s"); params.append(date_to)
        if team:
            conditions.append("(home_team ILIKE %s OR away_team ILIKE %s)")
            params.extend([f"%{team}%", f"%{team}%"])
        if status:
            conditions.append("status = %s"); params.append(status)
        where = " AND ".join(conditions) if conditions else "TRUE"
        params.append(limit)
        return self._execute(
            f"SELECT * FROM games WHERE {where} ORDER BY date DESC LIMIT %s", params
        )

    def get_game_by_id(self, game_id: str) -> dict[str, Any] | None:
        return self._execute_one("SELECT * FROM games WHERE game_id = %s", [game_id])

    # ------------------------------------------------------------------
    # Odds (list[dict] signature)
    # ------------------------------------------------------------------

    def insert_odds(self, odds: list[dict[str, Any]]) -> int:
        if not odds:
            return 0
        for o in odds:
            self._execute("""
                INSERT INTO odds_snapshots (game_id, sportsbook, market_type,
                    home_odds, away_odds, spread, total, over_odds, under_odds)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, [
                o["game_id"], o.get("sportsbook", "unknown"), o.get("market_type", "h2h"),
                o.get("home_odds"), o.get("away_odds"),
                o.get("spread"), o.get("total"),
                o.get("over_odds"), o.get("under_odds"),
            ])
        return len(odds)

    def query_odds(self, game_id: str | None = None, sportsbook: str | None = None,
                   latest_only: bool = True, hours_back: int | None = None) -> list[dict[str, Any]]:
        conditions, params = [], []
        if game_id:
            conditions.append("game_id = %s"); params.append(game_id)
        if sportsbook:
            conditions.append("sportsbook = %s"); params.append(sportsbook)
        if hours_back:
            conditions.append("captured_at >= NOW() - make_interval(hours => %s)")
            params.append(hours_back)
        where = " AND ".join(conditions) if conditions else "TRUE"
        if latest_only:
            return self._execute(f"""
                SELECT DISTINCT ON (game_id, sportsbook, market_type) *
                FROM odds_snapshots WHERE {where}
                ORDER BY game_id, sportsbook, market_type, captured_at DESC
            """, params)
        return self._execute(
            f"SELECT * FROM odds_snapshots WHERE {where} ORDER BY captured_at DESC", params
        )

    # ------------------------------------------------------------------
    # Player stats (list[dict] signature)
    # ------------------------------------------------------------------

    def upsert_player_stats(self, stats: list[dict[str, Any]]) -> int:
        if not stats:
            return 0
        for s in stats:
            self._execute("""
                INSERT INTO player_stats (player_id, sport, season, team, stats)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (player_id, sport, season) DO UPDATE SET
                    team = EXCLUDED.team, stats = EXCLUDED.stats, updated_at = NOW()
            """, [s["player_id"], s["sport"], s["season"],
                  s.get("team"), json.dumps(s.get("stats", {}))])
        return len(stats)

    # ------------------------------------------------------------------
    # Team stats (list[dict] signature)
    # ------------------------------------------------------------------

    def upsert_team_stats(self, stats: list[dict[str, Any]]) -> int:
        if not stats:
            return 0
        for s in stats:
            _validate_stats_keys(s.get("stats", {}), _TEAM_STATS_REQUIRED,
                                 f"team_stats:{s.get('team_id', '?')}")
            self._execute("""
                INSERT INTO team_stats (team_id, sport, season, stats, source, updated_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                ON CONFLICT (team_id, sport, season) DO UPDATE SET
                    stats = EXCLUDED.stats, source = EXCLUDED.source, updated_at = NOW()
            """, [s["team_id"], s["sport"], s["season"],
                  json.dumps(s.get("stats", {})), s.get("source", "unknown")])
        return len(stats)

    def query_team_stats(self, team_id: str | None = None, sport: str | None = None,
                         season: str | None = None) -> list[dict[str, Any]]:
        conditions, params = [], []
        if team_id:
            conditions.append("UPPER(team_id) = UPPER(%s)"); params.append(team_id)
        if sport:
            conditions.append("sport = %s"); params.append(sport.upper())
        if season:
            conditions.append("season = %s"); params.append(season)
        where = " AND ".join(conditions) if conditions else "TRUE"
        return self._execute(f"SELECT * FROM team_stats WHERE {where} ORDER BY season DESC", params)

    # ------------------------------------------------------------------
    # Goaltender stats (list[dict] signature)
    # ------------------------------------------------------------------

    def upsert_goaltender_stats(self, stats: list[dict[str, Any]]) -> int:
        if not stats:
            return 0
        for s in stats:
            _validate_stats_keys(s.get("stats", {}), _GOALIE_STATS_REQUIRED,
                                 f"goaltender_stats:{s.get('goaltender_id', '?')}")
            self._execute("""
                INSERT INTO goaltender_stats (goaltender_id, team, season, stats, updated_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (goaltender_id, season) DO UPDATE SET
                    team = EXCLUDED.team, stats = EXCLUDED.stats, updated_at = NOW()
            """, [s["goaltender_id"], s["team"], s["season"], json.dumps(s.get("stats", {}))])
        return len(stats)

    def query_goaltender_stats(self, goaltender_id: str | None = None, team: str | None = None,
                               season: str | None = None) -> list[dict[str, Any]]:
        conditions, params = [], []
        if goaltender_id:
            conditions.append("goaltender_id = %s"); params.append(goaltender_id)
        if team:
            conditions.append("team ILIKE %s"); params.append(f"%{team}%")
        if season:
            conditions.append("season = %s"); params.append(season)
        where = " AND ".join(conditions) if conditions else "TRUE"
        return self._execute(f"SELECT * FROM goaltender_stats WHERE {where} ORDER BY season DESC", params)

    # ------------------------------------------------------------------
    # Advanced stats (list[dict] signature)
    # ------------------------------------------------------------------

    def upsert_nhl_advanced_stats(self, stats: list[dict[str, Any]]) -> int:
        if not stats:
            return 0
        for s in stats:
            _validate_stats_keys(s.get("stats", {}), _ADVANCED_STATS_REQUIRED,
                                 f"nhl_advanced_stats:{s.get('team_id', '?')}")
            self._execute("""
                INSERT INTO nhl_advanced_stats (team_id, season, stats, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (team_id, season) DO UPDATE SET
                    stats = EXCLUDED.stats, updated_at = NOW()
            """, [s["team_id"], s["season"], json.dumps(s.get("stats", {}))])
        return len(stats)

    def query_nhl_advanced_stats(self, team_id: str | None = None,
                                 season: str | None = None) -> list[dict[str, Any]]:
        conditions, params = [], []
        if team_id:
            conditions.append("UPPER(team_id) = UPPER(%s)"); params.append(team_id)
        if season:
            conditions.append("season = %s"); params.append(season)
        where = " AND ".join(conditions) if conditions else "TRUE"
        return self._execute(f"SELECT * FROM nhl_advanced_stats WHERE {where}", params)

    # ------------------------------------------------------------------
    # Salary cap (list[dict] signature)
    # ------------------------------------------------------------------

    def upsert_salary_cap(self, contracts: list[dict[str, Any]]) -> int:
        if not contracts:
            return 0
        for c in contracts:
            self._execute("""
                INSERT INTO salary_cap (player_id, team, season, name, position,
                    cap_hit, aav, contract_years_remaining, status, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (player_id, season) DO UPDATE SET
                    team = EXCLUDED.team, name = EXCLUDED.name, position = EXCLUDED.position,
                    cap_hit = EXCLUDED.cap_hit, aav = EXCLUDED.aav,
                    contract_years_remaining = EXCLUDED.contract_years_remaining,
                    status = EXCLUDED.status, updated_at = NOW()
            """, [c["player_id"], c["team"], c["season"], c["name"],
                  c.get("position"), c.get("cap_hit", 0), c.get("aav", 0),
                  c.get("contract_years_remaining", 1), c.get("status", "active")])
        return len(contracts)

    # ------------------------------------------------------------------
    # Roster (list[dict] signature)
    # ------------------------------------------------------------------

    def upsert_roster(self, players: list[dict[str, Any]]) -> int:
        if not players:
            return 0
        for p in players:
            self._execute("""
                INSERT INTO roster (player_id, team, season, name, position,
                    line_number, stats, injury_status, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (player_id, team, season) DO UPDATE SET
                    name = EXCLUDED.name, position = EXCLUDED.position,
                    line_number = EXCLUDED.line_number, stats = EXCLUDED.stats,
                    injury_status = EXCLUDED.injury_status, updated_at = NOW()
            """, [p["player_id"], p["team"], p["season"], p["name"],
                  p.get("position"), p.get("line_number"),
                  json.dumps(p.get("stats", {})), p.get("injury_status", "healthy")])
        return len(players)

    # ------------------------------------------------------------------
    # Sessions & logging
    # ------------------------------------------------------------------

    def create_session(self, session_id: str, budget: float = 1000.0) -> dict[str, Any]:
        self._execute("""
            INSERT INTO sessions (session_id, budget_remaining, budget_total)
            VALUES (%s, %s, %s) ON CONFLICT (session_id) DO NOTHING
        """, [session_id, budget, budget])
        return self._execute_one("SELECT * FROM sessions WHERE session_id = %s", [session_id])

    def charge_session(self, session_id: str, cost: float) -> dict[str, Any] | None:
        self._execute("""
            UPDATE sessions SET budget_remaining = budget_remaining - %s,
                tool_calls = tool_calls + 1, last_activity = NOW()
            WHERE session_id = %s
        """, [cost, session_id])
        return self._execute_one("SELECT * FROM sessions WHERE session_id = %s", [session_id])

    def log_tool_call(self, session_id: str, tool_name: str,
                      input_params: dict | None = None, output_summary: dict | None = None,
                      cost: float = 0.0, latency_ms: float | None = None,
                      cache_hit: bool = False) -> None:
        self._execute("""
            INSERT INTO tool_call_log (session_id, tool_name, input_params, output_summary,
                cost, latency_ms, cache_hit)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, [session_id, tool_name,
              json.dumps(input_params) if input_params else None,
              json.dumps(output_summary) if output_summary else None,
              cost, latency_ms, cache_hit])

    def log_reasoning(self, session_id: str, phase: str, **kwargs: Any) -> None:
        self._execute("""
            INSERT INTO reasoning_log (session_id, phase, plan, uncertainty, data_trust,
                workflow_gaps, want_to_verify, raw_thought, context_snapshot)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, [
            session_id, phase, kwargs.get("plan"),
            json.dumps(kwargs.get("uncertainty")) if kwargs.get("uncertainty") else None,
            json.dumps(kwargs.get("data_trust")) if kwargs.get("data_trust") else None,
            json.dumps(kwargs.get("workflow_gaps")) if kwargs.get("workflow_gaps") else None,
            json.dumps(kwargs.get("want_to_verify")) if kwargs.get("want_to_verify") else None,
            kwargs.get("raw_thought"),
            json.dumps(kwargs.get("context_snapshot")) if kwargs.get("context_snapshot") else None,
        ])

    # ------------------------------------------------------------------
    # SR query budget tracking
    # ------------------------------------------------------------------

    def log_sr_query(self, endpoint: str, params: dict | None = None,
                     status_code: int | None = None) -> None:
        self._execute("""
            INSERT INTO sr_query_log (endpoint, params, status_code)
            VALUES (%s, %s, %s)
        """, [endpoint, json.dumps(params) if params else None, status_code])

    def get_sr_query_count(self, since_hours: int = 24) -> int:
        row = self._execute_one(
            "SELECT COUNT(*) AS cnt FROM sr_query_log WHERE called_at >= NOW() - make_interval(hours => %s)",
            [since_hours],
        )
        return row["cnt"] if row else 0

    def get_sr_query_budget_remaining(self, total_budget: int = 1000) -> dict[str, Any]:
        total_used = self._execute_one("SELECT COUNT(*) AS cnt FROM sr_query_log")
        used = total_used["cnt"] if total_used else 0
        return {
            "total_budget": total_budget,
            "total_used": used,
            "remaining": total_budget - used,
            "last_24h": self.get_sr_query_count(24),
        }
