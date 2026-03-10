"""Session and workflow tools mixin."""

from __future__ import annotations

import time
from typing import Any


class SessionToolsMixin:
    """Session management, data refresh, and workflow tools."""

    def refresh_data(
        self,
        source: str = "all",
        team: str | None = None,
    ) -> dict[str, Any]:
        """Refresh data from live sources."""
        start = time.time()
        from ...connectors.live import NHLLiveConnector, OddsLiveConnector

        results: dict[str, Any] = {}

        if source in ("nhl", "all"):
            nhl = NHLLiveConnector()
            try:
                nhl_result = nhl.refresh(self.storage, team=team)
                results["nhl"] = nhl_result
            except Exception as e:
                results["nhl"] = {"error": str(e)}

        if source in ("odds", "all"):
            odds = OddsLiveConnector()
            try:
                odds_result = odds.refresh(self.storage)
                results["odds"] = odds_result
            except Exception as e:
                results["odds"] = {"error": str(e)}

        errors = {k: v["error"] for k, v in results.items() if isinstance(v, dict) and "error" in v}
        result = {
            "refreshed_sources": list(results.keys()),
            "details": results,
        }
        if errors:
            result["errors"] = errors
            result["warning"] = "Some data sources failed to refresh. Predictions may use stale data."
        return self._track("refresh_data", {"source": source, "team": team}, result, start)

    def run_workflow(self, workflow: str, **kwargs: Any) -> dict[str, Any]:
        """Run a multi-step analytical workflow by name."""
        start = time.time()
        from ...workflows.engine import WorkflowEngine
        engine = WorkflowEngine(self)
        result = engine.run(workflow, **kwargs)
        return self._track("run_workflow", {"workflow": workflow, **kwargs}, result, start)

    def list_workflows(self) -> dict[str, Any]:
        """List available multi-step workflows for agent discovery."""
        start = time.time()
        from ...workflows.engine import list_workflows
        result = {"workflows": list_workflows()}
        return self._track("list_workflows", {}, result, start)

    def session_budget(self) -> dict[str, Any]:
        """Check remaining session budget and usage."""
        start = time.time()
        session = self.storage.get_session(self._session_id)
        result = {
            "session_id": self._session_id,
            "budget_remaining": round(session["budget_remaining"], 2) if session else 0,
            "budget_total": session["budget_total"] if session else 0,
            "tool_calls": session["tool_calls"] if session else 0,
        }
        return self._track("session_budget", {}, result, start)
