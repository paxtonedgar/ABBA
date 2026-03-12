"""Dual-write storage: DuckDB (local/fast) + Postgres (Supabase/durable).

Writes go to both backends. Reads come from DuckDB (faster for analytics).
If Postgres is unavailable, falls back to DuckDB-only silently.

Design:
    - Inherits from Storage (DuckDB) so the entire existing API works unchanged.
    - Postgres writes are best-effort: failures are logged, never raised.
    - All method signatures match DuckDB exactly (list[dict] upserts, etc.).
    - To add a new dual-written method, just add it to _DUAL_WRITE_METHODS.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .duckdb import Storage

log = logging.getLogger(__name__)


# Methods that should dual-write to Postgres.
# Each entry is a method name that exists on both Storage and PostgresStorage
# with identical signatures.
_DUAL_WRITE_METHODS = frozenset({
    "record_refresh",
    "snapshot_standings",
    "upsert_games",
    "insert_odds",
    "upsert_player_stats",
    "upsert_team_stats",
    "upsert_goaltender_stats",
    "upsert_nhl_advanced_stats",
    "upsert_salary_cap",
    "upsert_roster",
    "cache_prediction",
    "create_session",
    "charge_session",
    "log_tool_call",
    "log_reasoning",
})


class DualStorage(Storage):
    """DuckDB primary + Postgres durable mirror.

    Inherits from Storage so the entire existing API works unchanged.
    Postgres writes are best-effort -- failures are logged, not raised.
    """

    def __init__(self, db_path: str = ":memory:", pg_dsn: str | None = None):
        super().__init__(db_path)
        self._pg = None
        dsn = pg_dsn or os.environ.get("SUPABASE_DB_URL", "")
        if dsn:
            try:
                from .postgres import PostgresStorage
                self._pg = PostgresStorage(dsn)
                log.info("DualStorage: Postgres connected")
            except Exception as e:
                log.warning("DualStorage: Postgres unavailable (%s), DuckDB-only", e)

    @property
    def has_postgres(self) -> bool:
        return self._pg is not None

    @property
    def pg(self):
        """Direct access to Postgres backend for PG-only operations."""
        return self._pg

    def _pg_mirror(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Best-effort call to Postgres mirror. Swallows exceptions."""
        if not self._pg:
            return
        try:
            getattr(self._pg, method)(*args, **kwargs)
        except Exception as e:
            log.warning("Postgres mirror %s failed: %s", method, e)

    def __getattribute__(self, name: str) -> Any:
        """Intercept dual-write methods to mirror calls to Postgres.

        For methods in _DUAL_WRITE_METHODS, we wrap the DuckDB call to also
        call the same method on PostgresStorage with the same arguments.
        """
        # Avoid infinite recursion for internal attrs
        if name.startswith("_") or name not in _DUAL_WRITE_METHODS:
            return super().__getattribute__(name)

        duck_method = super().__getattribute__(name)

        def dual_wrapper(*args: Any, **kwargs: Any) -> Any:
            result = duck_method(*args, **kwargs)
            self._pg_mirror(name, *args, **kwargs)
            return result

        return dual_wrapper

    # ------------------------------------------------------------------
    # SR query budget (Postgres-only -- DuckDB doesn't have sr_query_log)
    # ------------------------------------------------------------------

    def log_sr_query(self, endpoint: str, params: dict | None = None,
                     status_code: int | None = None) -> None:
        if self._pg:
            try:
                self._pg.log_sr_query(endpoint, params, status_code)
            except Exception as e:
                log.warning("log_sr_query failed: %s", e)

    def get_sr_query_budget_remaining(self, total_budget: int = 1000) -> dict[str, Any]:
        if self._pg:
            try:
                return self._pg.get_sr_query_budget_remaining(total_budget)
            except Exception as e:
                log.warning("get_sr_query_budget_remaining failed: %s", e)
        return {"total_budget": total_budget, "remaining": "unknown (no postgres)", "total_used": 0}

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        super().close()
        if self._pg:
            try:
                self._pg.close()
            except Exception:
                pass
