"""DuckDB embedded analytical storage for ABBA.

Columnar store for sports data, odds, predictions, and session state.
Replaces SQLite for analytical workloads -- fast aggregations, time-series
queries, and Parquet-native for cheap historical archival.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb

from ..types import (
    AdvancedStatsRecord,
    Game,
    GoaltenderStatsRecord,
    OddsSnapshot,
    RosterPlayer,
    TeamStatsRecord,
)


class StorageValidationError(ValueError):
    """Raised when data fails schema validation on write."""


# Minimal required keys for JSON stats blobs.
# These catch the most common data corruption (missing fields, wrong schema).
_TEAM_STATS_REQUIRED = {"wins", "losses"}
_GOALIE_STATS_REQUIRED = {"save_pct", "gaa"}
_ADVANCED_STATS_REQUIRED = {"corsi_pct"}


def _validate_stats_keys(stats: dict[str, Any], required: set[str], context: str) -> None:
    """Raise StorageValidationError if required keys are missing from stats blob."""
    if not stats:
        return  # empty stats are allowed (will be filled later)
    missing = required - set(stats.keys())
    if missing:
        raise StorageValidationError(
            f"Stats blob for {context} missing required keys: {missing}. "
            f"Got keys: {set(stats.keys())}"
        )


class Storage:
    """Embedded DuckDB storage layer."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._init_schema()

    @staticmethod
    def _fetchdicts(cursor: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
        """Convert a DuckDB cursor result to a list of dicts using fetchall()."""
        desc = cursor.description
        if not desc:
            return []
        cols = [d[0] for d in desc]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def _init_schema(self) -> None:
        # Create sequences first (before tables that reference them)
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS odds_seq START 1")
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS log_seq START 1")

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id VARCHAR PRIMARY KEY,
                sport VARCHAR NOT NULL,
                date DATE NOT NULL,
                home_team VARCHAR NOT NULL,
                away_team VARCHAR NOT NULL,
                home_score INTEGER,
                away_score INTEGER,
                venue VARCHAR,
                status VARCHAR DEFAULT 'scheduled',
                metadata JSON,
                source VARCHAR DEFAULT 'unknown',
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                id INTEGER DEFAULT nextval('odds_seq'),
                game_id VARCHAR NOT NULL,
                sportsbook VARCHAR NOT NULL,
                market_type VARCHAR NOT NULL,
                home_odds DOUBLE,
                away_odds DOUBLE,
                spread DOUBLE,
                total DOUBLE,
                over_odds DOUBLE,
                under_odds DOUBLE,
                captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS player_stats (
                player_id VARCHAR NOT NULL,
                sport VARCHAR NOT NULL,
                season VARCHAR NOT NULL,
                team VARCHAR,
                stats JSON NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (player_id, sport, season)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS team_stats (
                team_id VARCHAR NOT NULL,
                sport VARCHAR NOT NULL,
                season VARCHAR NOT NULL,
                stats JSON NOT NULL,
                source VARCHAR DEFAULT 'unknown',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team_id, sport, season)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS weather (
                game_id VARCHAR NOT NULL,
                temperature DOUBLE,
                humidity DOUBLE,
                wind_speed DOUBLE,
                wind_direction VARCHAR,
                precipitation_chance DOUBLE,
                conditions VARCHAR,
                captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions_cache (
                prediction_id VARCHAR PRIMARY KEY,
                game_id VARCHAR NOT NULL,
                model_version VARCHAR NOT NULL,
                data_hash VARCHAR NOT NULL,
                prediction JSON NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id VARCHAR PRIMARY KEY,
                budget_remaining DOUBLE DEFAULT 1000.0,
                budget_total DOUBLE DEFAULT 1000.0,
                tool_calls INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_call_log (
                id INTEGER DEFAULT nextval('log_seq'),
                session_id VARCHAR NOT NULL,
                tool_name VARCHAR NOT NULL,
                input_params JSON,
                output_summary JSON,
                cost DOUBLE DEFAULT 0.0,
                latency_ms DOUBLE,
                cache_hit BOOLEAN DEFAULT FALSE,
                called_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS reasoning_seq START 1")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS reasoning_log (
                id INTEGER DEFAULT nextval('reasoning_seq'),
                session_id VARCHAR NOT NULL,
                phase VARCHAR NOT NULL,
                plan VARCHAR,
                uncertainty JSON,
                data_trust JSON,
                workflow_gaps JSON,
                want_to_verify JSON,
                raw_thought VARCHAR,
                context_snapshot JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # --- NHL-specific tables ---

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS goaltender_stats (
                goaltender_id VARCHAR NOT NULL,
                team VARCHAR NOT NULL,
                season VARCHAR NOT NULL,
                stats JSON NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (goaltender_id, season)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS nhl_advanced_stats (
                team_id VARCHAR NOT NULL,
                season VARCHAR NOT NULL,
                stats JSON NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team_id, season)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS salary_cap (
                player_id VARCHAR NOT NULL,
                team VARCHAR NOT NULL,
                season VARCHAR NOT NULL,
                name VARCHAR NOT NULL,
                position VARCHAR,
                cap_hit DOUBLE,
                aav DOUBLE,
                contract_years_remaining INTEGER,
                status VARCHAR DEFAULT 'active',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (player_id, season)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS roster (
                player_id VARCHAR NOT NULL,
                team VARCHAR NOT NULL,
                season VARCHAR NOT NULL,
                name VARCHAR NOT NULL,
                position VARCHAR,
                line_number INTEGER,
                stats JSON,
                injury_status VARCHAR DEFAULT 'healthy',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (player_id, team, season)
            )
        """)

        # --- Data freshness tracking ---

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS data_freshness (
                table_name VARCHAR PRIMARY KEY,
                last_refresh_at TIMESTAMP NOT NULL,
                source VARCHAR DEFAULT 'unknown',
                row_count INTEGER DEFAULT 0
            )
        """)

        # Standings snapshots — daily captures for leakage-free backtesting.
        # Each row is one team's stats on one date. This accumulates over time
        # so we can reconstruct "what did we know on date X" without lookahead bias.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS standings_snapshots (
                snapshot_date DATE NOT NULL,
                team_id VARCHAR NOT NULL,
                sport VARCHAR DEFAULT 'NHL',
                stats JSON NOT NULL,
                PRIMARY KEY (snapshot_date, team_id)
            )
        """)

    # --- Data freshness ---

    def record_refresh(self, table_name: str, source: str = "unknown", row_count: int = 0) -> None:
        """Record that a table was refreshed."""
        self.conn.execute("""
            INSERT OR REPLACE INTO data_freshness (table_name, last_refresh_at, source, row_count)
            VALUES (?, CURRENT_TIMESTAMP, ?, ?)
        """, [table_name, source, row_count])

    def get_last_refresh(self, table_name: str) -> float | None:
        """Return Unix timestamp of last refresh for a table, or None."""
        row = self.conn.execute(
            "SELECT epoch(last_refresh_at) FROM data_freshness WHERE table_name = ?",
            [table_name],
        ).fetchone()
        return float(row[0]) if row else None

    # --- Standings snapshots (for leakage-free backtesting) ---

    def snapshot_standings(self, snapshot_date: str, team_stats: list[dict[str, Any]]) -> int:
        """Capture today's standings as a snapshot for future backtesting.

        Call this during refresh_data to accumulate point-in-time data.
        """
        if not team_stats:
            return 0
        for s in team_stats:
            self.conn.execute("""
                INSERT OR REPLACE INTO standings_snapshots (snapshot_date, team_id, sport, stats)
                VALUES (?, ?, ?, ?)
            """, [snapshot_date, s["team_id"], s.get("sport", "NHL"), json.dumps(s.get("stats", {}))])
        return len(team_stats)

    def get_standings_snapshot(self, snapshot_date: str, team_id: str | None = None) -> list[dict[str, Any]]:
        """Retrieve standings as they were on a given date."""
        conditions = ["snapshot_date = ?"]
        params: list[Any] = [snapshot_date]
        if team_id:
            conditions.append("UPPER(team_id) = UPPER(?)")
            params.append(team_id)
        where = " AND ".join(conditions)
        rows = self._fetchdicts(self.conn.execute(
            f"SELECT * FROM standings_snapshots WHERE {where}", params
        ))
        for row in rows:
            if isinstance(row.get("stats"), str):
                row["stats"] = json.loads(row["stats"])
        return rows

    def get_snapshot_dates(self) -> list[str]:
        """Return all dates for which we have standings snapshots."""
        rows = self.conn.execute(
            "SELECT DISTINCT snapshot_date FROM standings_snapshots ORDER BY snapshot_date"
        ).fetchall()
        return [str(r[0]) for r in rows]

    # --- Games ---

    def upsert_games(self, games: list[dict[str, Any]]) -> int:
        if not games:
            return 0
        for g in games:
            self.conn.execute("""
                INSERT OR REPLACE INTO games (game_id, sport, date, home_team, away_team,
                    home_score, away_score, venue, status, metadata, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                g["game_id"], g["sport"], g["date"], g["home_team"], g["away_team"],
                g.get("home_score"), g.get("away_score"), g.get("venue"),
                g.get("status", "scheduled"),
                json.dumps(g.get("metadata", {})),
                g.get("source", "unknown"),
            ])
        return len(games)

    def query_games(
        self,
        sport: str | None = None,
        date: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        team: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[Game]:
        conditions = []
        params = []

        if sport:
            conditions.append("sport = ?")
            params.append(sport.upper())
        if date:
            conditions.append("date = ?")
            params.append(date)
        if date_from:
            conditions.append("date >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("date <= ?")
            params.append(date_to)
        if team:
            conditions.append("(home_team ILIKE ? OR away_team ILIKE ?)")
            params.extend([f"%{team}%", f"%{team}%"])
        if status:
            conditions.append("status = ?")
            params.append(status)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM games {where} ORDER BY date DESC LIMIT ?"
        params.append(limit)

        rows = self._fetchdicts(self.conn.execute(query, params))
        for row in rows:
            if isinstance(row.get("metadata"), str):
                row["metadata"] = json.loads(row["metadata"])
        return rows

    # --- Odds ---

    def insert_odds(self, odds: list[dict[str, Any]]) -> int:
        if not odds:
            return 0
        for o in odds:
            self.conn.execute("""
                INSERT INTO odds_snapshots (game_id, sportsbook, market_type,
                    home_odds, away_odds, spread, total, over_odds, under_odds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                o["game_id"], o.get("sportsbook", "unknown"), o.get("market_type", "moneyline"),
                o.get("home_odds"), o.get("away_odds"), o.get("spread"),
                o.get("total"), o.get("over_odds"), o.get("under_odds"),
            ])
        return len(odds)

    def query_odds(
        self,
        game_id: str | None = None,
        sportsbook: str | None = None,
        latest_only: bool = True,
        hours_back: int = 24,
    ) -> list[OddsSnapshot]:
        conditions = []
        params: list[Any] = []

        if game_id:
            conditions.append("game_id = ?")
            params.append(game_id)
        if sportsbook:
            conditions.append("sportsbook = ?")
            params.append(sportsbook)

        cutoff = datetime.now() - timedelta(hours=hours_back)
        conditions.append("captured_at >= ?")
        params.append(cutoff)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        if latest_only:
            query = f"""
                SELECT DISTINCT ON (game_id, sportsbook, market_type) *
                FROM odds_snapshots {where}
                ORDER BY game_id, sportsbook, market_type, captured_at DESC
            """
        else:
            query = f"SELECT * FROM odds_snapshots {where} ORDER BY captured_at DESC"

        return self._fetchdicts(self.conn.execute(query, params))

    # --- Player stats ---

    def upsert_player_stats(self, stats: list[dict[str, Any]]) -> int:
        if not stats:
            return 0
        for s in stats:
            self.conn.execute("""
                INSERT OR REPLACE INTO player_stats (player_id, sport, season, team, stats)
                VALUES (?, ?, ?, ?, ?)
            """, [
                s["player_id"], s["sport"], s["season"],
                s.get("team"), json.dumps(s.get("stats", {})),
            ])
        return len(stats)

    def query_player_stats(
        self,
        player_id: str | None = None,
        sport: str | None = None,
        season: str | None = None,
        team: str | None = None,
    ) -> list[dict[str, Any]]:
        conditions = []
        params = []

        if player_id:
            conditions.append("player_id = ?")
            params.append(player_id)
        if sport:
            conditions.append("sport = ?")
            params.append(sport.upper())
        if season:
            conditions.append("season = ?")
            params.append(season)
        if team:
            conditions.append("team ILIKE ?")
            params.append(f"%{team}%")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._fetchdicts(self.conn.execute(
            f"SELECT * FROM player_stats {where}", params
        ))
        for row in rows:
            if isinstance(row.get("stats"), str):
                row["stats"] = json.loads(row["stats"])
        return rows

    # --- Team stats ---

    def upsert_team_stats(self, stats: list[dict[str, Any]]) -> int:
        if not stats:
            return 0
        for s in stats:
            _validate_stats_keys(s.get("stats", {}), _TEAM_STATS_REQUIRED, f"team_stats:{s.get('team_id', '?')}")
            self.conn.execute("""
                INSERT OR REPLACE INTO team_stats (team_id, sport, season, stats, source)
                VALUES (?, ?, ?, ?, ?)
            """, [s["team_id"], s["sport"], s["season"], json.dumps(s.get("stats", {})), s.get("source", "unknown")])
        return len(stats)

    def query_team_stats(
        self,
        team_id: str | None = None,
        sport: str | None = None,
        season: str | None = None,
    ) -> list[TeamStatsRecord]:
        conditions = []
        params = []
        if team_id:
            conditions.append("UPPER(team_id) = UPPER(?)")
            params.append(team_id)
        if sport:
            conditions.append("sport = ?")
            params.append(sport.upper())
        if season:
            conditions.append("season = ?")
            params.append(season)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._fetchdicts(self.conn.execute(
            f"SELECT * FROM team_stats {where} ORDER BY season DESC", params
        ))
        for row in rows:
            if isinstance(row.get("stats"), str):
                row["stats"] = json.loads(row["stats"])
        return rows

    # --- Weather ---

    def insert_weather(self, weather: dict[str, Any]) -> None:
        self.conn.execute("""
            INSERT INTO weather (game_id, temperature, humidity, wind_speed,
                wind_direction, precipitation_chance, conditions)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            weather["game_id"], weather.get("temperature"),
            weather.get("humidity"), weather.get("wind_speed"),
            weather.get("wind_direction"), weather.get("precipitation_chance"),
            weather.get("conditions"),
        ])

    def get_weather(self, game_id: str) -> dict[str, Any] | None:
        rows = self._fetchdicts(self.conn.execute(
            "SELECT * FROM weather WHERE game_id = ? ORDER BY captured_at DESC LIMIT 1",
            [game_id],
        ))
        return rows[0] if rows else None

    # --- Prediction cache ---

    def cache_prediction(
        self,
        game_id: str,
        model_version: str,
        data_hash: str,
        prediction: dict[str, Any],
        ttl_minutes: int = 30,
    ) -> str:
        prediction_id = f"pred-{game_id}-{model_version}-{data_hash[:8]}"
        expires = datetime.now() + timedelta(minutes=ttl_minutes)
        self.conn.execute("""
            INSERT OR REPLACE INTO predictions_cache
                (prediction_id, game_id, model_version, data_hash, prediction, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [prediction_id, game_id, model_version, data_hash,
              json.dumps(prediction), expires])
        return prediction_id

    def get_cached_prediction(
        self, game_id: str, model_version: str, data_hash: str
    ) -> dict[str, Any] | None:
        result = self.conn.execute("""
            SELECT prediction FROM predictions_cache
            WHERE game_id = ? AND model_version = ? AND data_hash = ?
              AND expires_at > CURRENT_TIMESTAMP
        """, [game_id, model_version, data_hash]).fetchone()
        if result:
            val = result[0]
            return json.loads(val) if isinstance(val, str) else val
        return None

    # --- Sessions ---

    def create_session(self, session_id: str, budget: float = 1000.0) -> dict[str, Any]:
        self.conn.execute("""
            INSERT OR REPLACE INTO sessions (session_id, budget_remaining, budget_total)
            VALUES (?, ?, ?)
        """, [session_id, budget, budget])
        return {"session_id": session_id, "budget_remaining": budget}

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        rows = self._fetchdicts(self.conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", [session_id]
        ))
        return rows[0] if rows else None

    def charge_session(self, session_id: str, cost: float) -> float:
        self.conn.execute("""
            UPDATE sessions SET
                budget_remaining = budget_remaining - ?,
                tool_calls = tool_calls + 1,
                last_activity = CURRENT_TIMESTAMP
            WHERE session_id = ?
        """, [cost, session_id])
        session = self.get_session(session_id)
        return session["budget_remaining"] if session else 0.0

    # --- Tool call logging ---

    def log_tool_call(
        self,
        session_id: str,
        tool_name: str,
        input_params: dict[str, Any],
        output_summary: dict[str, Any],
        cost: float,
        latency_ms: float,
        cache_hit: bool = False,
    ) -> None:
        self.conn.execute("""
            INSERT INTO tool_call_log
                (session_id, tool_name, input_params, output_summary, cost, latency_ms, cache_hit)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            session_id, tool_name, json.dumps(input_params),
            json.dumps(output_summary), cost, latency_ms, cache_hit,
        ])

    # --- Reasoning log ---

    def log_reasoning(
        self,
        session_id: str,
        phase: str,
        plan: str | None = None,
        uncertainty: list[str] | None = None,
        data_trust: list[dict[str, Any]] | None = None,
        workflow_gaps: list[str] | None = None,
        want_to_verify: list[str] | None = None,
        raw_thought: str | None = None,
        context_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute("""
            INSERT INTO reasoning_log
                (session_id, phase, plan, uncertainty, data_trust,
                 workflow_gaps, want_to_verify, raw_thought, context_snapshot)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            session_id, phase, plan,
            json.dumps(uncertainty) if uncertainty else None,
            json.dumps(data_trust) if data_trust else None,
            json.dumps(workflow_gaps) if workflow_gaps else None,
            json.dumps(want_to_verify) if want_to_verify else None,
            raw_thought,
            json.dumps(context_snapshot) if context_snapshot else None,
        ])

    def query_reasoning_log(
        self, session_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        rows = self._fetchdicts(self.conn.execute("""
            SELECT * FROM reasoning_log
            WHERE session_id = ?
            ORDER BY created_at ASC
            LIMIT ?
        """, [session_id, limit]))
        for row in rows:
            for col in ("uncertainty", "data_trust", "workflow_gaps", "want_to_verify", "context_snapshot"):
                if isinstance(row.get(col), str):
                    row[col] = json.loads(row[col])
        return rows

    def query_session_replay(
        self, session_id: str, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Interleave tool calls and reasoning entries chronologically via SQL UNION ALL."""
        rows_raw = self.conn.execute("""
            SELECT id, session_id, tool_name, input_params, output_summary,
                   cost, latency_ms, cache_hit,
                   NULL AS phase, NULL AS plan, NULL AS uncertainty,
                   NULL AS data_trust, NULL AS workflow_gaps,
                   NULL AS want_to_verify, NULL AS raw_thought,
                   NULL AS context_snapshot,
                   called_at AS ts, 'tool_call' AS entry_type
            FROM tool_call_log WHERE session_id = ?
            UNION ALL
            SELECT id, session_id, NULL, NULL, NULL,
                   NULL, NULL, NULL,
                   phase, plan, uncertainty,
                   data_trust, workflow_gaps,
                   want_to_verify, raw_thought,
                   context_snapshot,
                   created_at AS ts, 'reasoning' AS entry_type
            FROM reasoning_log WHERE session_id = ?
            ORDER BY ts ASC
            LIMIT ?
        """, [session_id, session_id, limit]).fetchall()

        if not rows_raw:
            return []

        columns = [
            "id", "session_id", "tool_name", "input_params", "output_summary",
            "cost", "latency_ms", "cache_hit",
            "phase", "plan", "uncertainty", "data_trust", "workflow_gaps",
            "want_to_verify", "raw_thought", "context_snapshot",
            "ts", "entry_type",
        ]
        rows = [dict(zip(columns, row)) for row in rows_raw]

        # Parse JSON columns and clean nulls
        json_cols = ("input_params", "output_summary", "uncertainty",
                     "data_trust", "workflow_gaps", "want_to_verify", "context_snapshot")
        for row in rows:
            for col in json_cols:
                val = row.get(col)
                if isinstance(val, str):
                    row[col] = json.loads(val)
            # Remove None-valued keys from the non-applicable type
            row = {k: v for k, v in row.items() if v is not None}
        return rows

    # --- Goaltender stats ---

    def upsert_goaltender_stats(self, stats: list[dict[str, Any]]) -> int:
        if not stats:
            return 0
        for s in stats:
            _validate_stats_keys(s.get("stats", {}), _GOALIE_STATS_REQUIRED, f"goaltender_stats:{s.get('goaltender_id', '?')}")
            self.conn.execute("""
                INSERT OR REPLACE INTO goaltender_stats (goaltender_id, team, season, stats)
                VALUES (?, ?, ?, ?)
            """, [s["goaltender_id"], s["team"], s["season"], json.dumps(s.get("stats", {}))])
        return len(stats)

    def query_goaltender_stats(
        self,
        goaltender_id: str | None = None,
        team: str | None = None,
        season: str | None = None,
    ) -> list[GoaltenderStatsRecord]:
        conditions = []
        params = []
        if goaltender_id:
            conditions.append("goaltender_id = ?")
            params.append(goaltender_id)
        if team:
            conditions.append("team ILIKE ?")
            params.append(f"%{team}%")
        if season:
            conditions.append("season = ?")
            params.append(season)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._fetchdicts(self.conn.execute(
            f"SELECT * FROM goaltender_stats {where} ORDER BY season DESC", params
        ))
        for row in rows:
            if isinstance(row.get("stats"), str):
                row["stats"] = json.loads(row["stats"])
        return rows

    # --- NHL advanced stats ---

    def upsert_nhl_advanced_stats(self, stats: list[dict[str, Any]]) -> int:
        if not stats:
            return 0
        for s in stats:
            _validate_stats_keys(s.get("stats", {}), _ADVANCED_STATS_REQUIRED, f"nhl_advanced_stats:{s.get('team_id', '?')}")
            self.conn.execute("""
                INSERT OR REPLACE INTO nhl_advanced_stats (team_id, season, stats)
                VALUES (?, ?, ?)
            """, [s["team_id"], s["season"], json.dumps(s.get("stats", {}))])
        return len(stats)

    def query_nhl_advanced_stats(
        self,
        team_id: str | None = None,
        season: str | None = None,
    ) -> list[AdvancedStatsRecord]:
        conditions = []
        params = []
        if team_id:
            conditions.append("UPPER(team_id) = UPPER(?)")
            params.append(team_id)
        if season:
            conditions.append("season = ?")
            params.append(season)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._fetchdicts(self.conn.execute(
            f"SELECT * FROM nhl_advanced_stats {where}", params
        ))
        for row in rows:
            if isinstance(row.get("stats"), str):
                row["stats"] = json.loads(row["stats"])
        return rows

    # --- Salary cap ---

    def upsert_salary_cap(self, contracts: list[dict[str, Any]]) -> int:
        if not contracts:
            return 0
        for c in contracts:
            self.conn.execute("""
                INSERT OR REPLACE INTO salary_cap
                    (player_id, team, season, name, position, cap_hit, aav,
                     contract_years_remaining, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                c["player_id"], c["team"], c["season"], c["name"],
                c.get("position"), c.get("cap_hit", 0), c.get("aav", 0),
                c.get("contract_years_remaining", 1), c.get("status", "active"),
            ])
        return len(contracts)

    def query_salary_cap(
        self,
        team: str | None = None,
        season: str | None = None,
        position: str | None = None,
    ) -> list[dict[str, Any]]:
        conditions = []
        params = []
        if team:
            conditions.append("UPPER(team) = UPPER(?)")
            params.append(team)
        if season:
            conditions.append("season = ?")
            params.append(season)
        if position:
            conditions.append("position = ?")
            params.append(position)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        return self._fetchdicts(self.conn.execute(
            f"SELECT * FROM salary_cap {where} ORDER BY cap_hit DESC", params
        ))

    # --- Roster ---

    def upsert_roster(self, players: list[dict[str, Any]]) -> int:
        if not players:
            return 0
        for p in players:
            self.conn.execute("""
                INSERT OR REPLACE INTO roster
                    (player_id, team, season, name, position, line_number, stats, injury_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                p["player_id"], p["team"], p["season"], p["name"],
                p.get("position"), p.get("line_number"),
                json.dumps(p.get("stats", {})), p.get("injury_status", "healthy"),
            ])
        return len(players)

    def query_roster(
        self,
        team: str | None = None,
        season: str | None = None,
        position: str | None = None,
    ) -> list[RosterPlayer]:
        conditions = []
        params = []
        if team:
            conditions.append("UPPER(team) = UPPER(?)")
            params.append(team)
        if season:
            conditions.append("season = ?")
            params.append(season)
        if position:
            conditions.append("position = ?")
            params.append(position)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._fetchdicts(self.conn.execute(
            f"SELECT * FROM roster {where} ORDER BY line_number ASC, name", params
        ))
        for row in rows:
            if isinstance(row.get("stats"), str):
                row["stats"] = json.loads(row["stats"])
        return rows

    # --- Direct lookups ---

    def get_game_by_id(self, game_id: str) -> Game | None:
        """Fetch a single game by ID. O(1) via primary key instead of scanning all games."""
        rows = self._fetchdicts(self.conn.execute(
            "SELECT * FROM games WHERE game_id = ?", [game_id]
        ))
        if rows and isinstance(rows[0].get("metadata"), str):
            rows[0]["metadata"] = json.loads(rows[0]["metadata"])
        return rows[0] if rows else None

    # --- Schema discovery ---

    def list_tables(self) -> list[dict[str, Any]]:
        table_rows = self.conn.execute("""
            SELECT table_name
            FROM duckdb_tables()
            WHERE schema_name = 'main'
        """).fetchall()
        tables = []
        for (tname,) in table_rows:
            count = self.conn.execute(
                f"SELECT COUNT(*) FROM {tname}"
            ).fetchone()[0]
            tables.append({
                "table": tname,
                "row_count": count,
            })
        return tables

    # Known tables for SQL injection prevention
    KNOWN_TABLES = {
        "games", "odds_snapshots", "player_stats", "team_stats",
        "weather", "predictions_cache", "sessions", "tool_call_log",
        "reasoning_log", "goaltender_stats", "nhl_advanced_stats",
        "salary_cap", "roster", "data_freshness", "standings_snapshots",
    }

    def describe_table(self, table_name: str) -> list[dict[str, Any]]:
        # Validate table name against known schema to prevent SQL injection
        if table_name not in self.KNOWN_TABLES:
            return [{"error": f"unknown table: {table_name}"}]
        return self._fetchdicts(self.conn.execute(f"DESCRIBE {table_name}"))

    def close(self) -> None:
        self.conn.close()
