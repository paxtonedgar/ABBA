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
import numpy as np


class Storage:
    """Embedded DuckDB storage layer."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._init_schema()

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

        # (sequences created above, before tables)

    # --- Games ---

    def upsert_games(self, games: list[dict[str, Any]]) -> int:
        if not games:
            return 0
        for g in games:
            self.conn.execute("""
                INSERT OR REPLACE INTO games (game_id, sport, date, home_team, away_team,
                    home_score, away_score, venue, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                g["game_id"], g["sport"], g["date"], g["home_team"], g["away_team"],
                g.get("home_score"), g.get("away_score"), g.get("venue"),
                g.get("status", "scheduled"),
                json.dumps(g.get("metadata", {})),
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
    ) -> list[dict[str, Any]]:
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

        result = self.conn.execute(query, params).fetchdf()
        return result.to_dict("records") if len(result) > 0 else []

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
    ) -> list[dict[str, Any]]:
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

        result = self.conn.execute(query, params).fetchdf()
        return result.to_dict("records") if len(result) > 0 else []

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
        result = self.conn.execute(
            f"SELECT * FROM player_stats {where}", params
        ).fetchdf()
        rows = result.to_dict("records") if len(result) > 0 else []
        for row in rows:
            if isinstance(row.get("stats"), str):
                row["stats"] = json.loads(row["stats"])
        return rows

    # --- Team stats ---

    def upsert_team_stats(self, stats: list[dict[str, Any]]) -> int:
        if not stats:
            return 0
        for s in stats:
            self.conn.execute("""
                INSERT OR REPLACE INTO team_stats (team_id, sport, season, stats)
                VALUES (?, ?, ?, ?)
            """, [s["team_id"], s["sport"], s["season"], json.dumps(s.get("stats", {}))])
        return len(stats)

    def query_team_stats(
        self,
        team_id: str | None = None,
        sport: str | None = None,
        season: str | None = None,
    ) -> list[dict[str, Any]]:
        conditions = []
        params = []
        if team_id:
            conditions.append("team_id = ?")
            params.append(team_id)
        if sport:
            conditions.append("sport = ?")
            params.append(sport.upper())
        if season:
            conditions.append("season = ?")
            params.append(season)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        result = self.conn.execute(
            f"SELECT * FROM team_stats {where}", params
        ).fetchdf()
        rows = result.to_dict("records") if len(result) > 0 else []
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
        result = self.conn.execute(
            "SELECT * FROM weather WHERE game_id = ? ORDER BY captured_at DESC LIMIT 1",
            [game_id],
        ).fetchdf()
        rows = result.to_dict("records") if len(result) > 0 else []
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
            INSERT INTO sessions (session_id, budget_remaining, budget_total)
            VALUES (?, ?, ?)
        """, [session_id, budget, budget])
        return {"session_id": session_id, "budget_remaining": budget}

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        result = self.conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", [session_id]
        ).fetchdf()
        rows = result.to_dict("records") if len(result) > 0 else []
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

    # --- Schema discovery ---

    def list_tables(self) -> list[dict[str, Any]]:
        result = self.conn.execute("""
            SELECT table_name, estimated_size
            FROM duckdb_tables()
            WHERE schema_name = 'main'
        """).fetchdf()
        tables = []
        for _, row in result.iterrows():
            count = self.conn.execute(
                f"SELECT COUNT(*) FROM {row['table_name']}"
            ).fetchone()[0]
            tables.append({
                "table": row["table_name"],
                "row_count": count,
            })
        return tables

    def describe_table(self, table_name: str) -> list[dict[str, Any]]:
        result = self.conn.execute(
            f"DESCRIBE {table_name}"
        ).fetchdf()
        return result.to_dict("records") if len(result) > 0 else []

    def close(self) -> None:
        self.conn.close()
