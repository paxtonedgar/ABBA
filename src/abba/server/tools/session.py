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
        season: str | None = None,
    ) -> dict[str, Any]:
        """Refresh data from live sources. Delegates to DataService."""
        start = time.time()
        result = self.data_service.refresh(source=source, team=team, season=season)
        # Sync refresh timestamp back to toolkit level
        self._last_refresh_ts = self.data_service.last_refresh_ts
        return self._track("refresh_data", {"source": source, "team": team, "season": season}, result, start)

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
