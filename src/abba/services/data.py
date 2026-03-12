"""Data service — owns data refresh orchestration.

Extracted from SessionToolsMixin.refresh_data.
"""

from __future__ import annotations

import time
from typing import Any

from ..storage import Storage


class DataService:
    """Owns data refresh from live connectors."""

    def __init__(self, storage: Storage):
        self.storage = storage
        self.last_refresh_ts: float | None = storage.get_last_refresh("team_stats")

    def refresh(
        self,
        source: str = "all",
        team: str | None = None,
    ) -> dict[str, Any]:
        """Refresh data from live sources.

        Returns status dict with refreshed_sources, details, and errors.
        """
        from ..connectors.live import NHLLiveConnector, OddsLiveConnector
        from ..connectors.moneypuck import MoneyPuckConnector
        from ..connectors.sportradar import SportsRadarConnector

        results: dict[str, Any] = {}

        # SportsRadar replaces nhl + moneypuck when key is available
        if source in ("sportradar", "all"):
            sr = SportsRadarConnector()
            if sr.api_key:
                try:
                    sr_result = sr.refresh(self.storage, team=team)
                    results["sportradar"] = sr_result
                except Exception as e:
                    results["sportradar"] = {"error": str(e)}

        # Fall back to free sources if SR not used
        if "sportradar" not in results:
            if source in ("nhl", "all"):
                nhl = NHLLiveConnector()
                try:
                    nhl_result = nhl.refresh(self.storage, team=team)
                    results["nhl"] = nhl_result
                except Exception as e:
                    results["nhl"] = {"error": str(e)}

            if source in ("moneypuck", "advanced", "all"):
                mp = MoneyPuckConnector()
                try:
                    mp_result = mp.refresh(self.storage, team=team)
                    results["moneypuck"] = mp_result
                except Exception as e:
                    results["moneypuck"] = {"error": str(e)}

        if source in ("odds", "all"):
            odds = OddsLiveConnector()
            try:
                odds_result = odds.refresh(self.storage)
                results["odds"] = odds_result
            except Exception as e:
                results["odds"] = {"error": str(e)}

        errors = {k: v["error"] for k, v in results.items() if isinstance(v, dict) and "error" in v}

        successes = {k for k, v in results.items() if isinstance(v, dict) and "error" not in v}
        if successes:
            self.last_refresh_ts = time.time()
            for src in successes:
                tables = {
                    "nhl": ["team_stats", "goaltender_stats", "games"],
                    "sportradar": ["team_stats", "goaltender_stats", "games", "nhl_advanced_stats"],
                    "moneypuck": ["nhl_advanced_stats"],
                    "odds": ["odds_snapshots"],
                }
                for tbl in tables.get(src, []):
                    self.storage.record_refresh(tbl, source=src)

            if "nhl" in successes or "sportradar" in successes:
                from datetime import date as _date
                today = _date.today().isoformat()
                all_stats = self.storage.query_team_stats(sport="NHL")
                self.storage.snapshot_standings(today, all_stats)

        result: dict[str, Any] = {
            "refreshed_sources": list(results.keys()),
            "details": results,
        }
        if errors:
            result["errors"] = errors
            result["warning"] = "Some data sources failed to refresh. Predictions may use stale data."
        return result
