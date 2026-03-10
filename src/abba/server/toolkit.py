"""ABBA Toolkit -- agent-callable analytics tools.

The toolkit is the main entry point. When an agent loads the ABBA package,
it instantiates ABBAToolkit which:
1. Boots DuckDB storage (in-memory or persistent)
2. Initializes engine components (ensemble, features, kelly, value, graph)
3. Seeds sample data if storage is empty
4. Exposes tools as methods with structured input/output

Each tool method returns a plain dict that agents can reason over.
Tool discovery is via list_tools() which returns schema for every tool.

Architecture: tool methods live in mixin classes under server/tools/.
ABBAToolkit composes all mixins to present a single unified API.

Usage:
    from abba.server import ABBAToolkit
    toolkit = ABBAToolkit()          # boots with sample data
    toolkit.query_games(sport="MLB") # returns structured dict
"""

from __future__ import annotations

import time
from typing import Any

from ..connectors.seed import seed_sample_data
from ..engine.elo import EloRatings
from ..engine.ensemble import EnsembleEngine
from ..engine.features import FeatureEngine
from ..engine.graph import GraphEngine
from ..engine.hockey import HockeyAnalytics
from ..engine.kelly import KellyEngine
from ..engine.value import ValueEngine
from ..storage import Storage
from .tools import (
    AnalyticsToolsMixin,
    DataToolsMixin,
    MarketToolsMixin,
    NHLToolsMixin,
    SessionToolsMixin,
    ToolRegistryMixin,
)


class ABBAToolkit(
    DataToolsMixin,
    AnalyticsToolsMixin,
    MarketToolsMixin,
    NHLToolsMixin,
    SessionToolsMixin,
    ToolRegistryMixin,
):
    """Agent-callable sports analytics toolkit.

    Instantiate once, call tools by name. Each tool returns a dict.

    Tool methods are organized into mixins:
    - DataToolsMixin: query_games, query_odds, query_team_stats, list_sources, describe_dataset
    - AnalyticsToolsMixin: predict_game, explain_prediction, graph_analysis
    - MarketToolsMixin: find_value, compare_odds, calculate_ev, kelly_sizing
    - NHLToolsMixin: nhl_predict_game, query_goaltender_stats, query_advanced_stats,
                     query_cap_data, query_roster, season_review, playoff_odds
    - SessionToolsMixin: refresh_data, run_workflow, list_workflows, session_budget
    - ToolRegistryMixin: list_tools, call_tool
    """

    VERSION = "2.0.0"

    def __init__(
        self,
        db_path: str = ":memory:",
        auto_seed: bool = True,
        session_budget: float = 1000.0,
    ):
        self.storage = Storage(db_path)
        self.ensemble = EnsembleEngine()
        self.features = FeatureEngine()
        self.kelly = KellyEngine()
        self.value = ValueEngine()
        self.graph = GraphEngine()
        self.hockey = HockeyAnalytics()
        self.elo = EloRatings(k=4, home_advantage=50)
        self._last_refresh_ts: float | None = None

        # Create default session
        self._session_id = "default"
        self.storage.create_session(self._session_id, session_budget)

        if auto_seed:
            tables = self.storage.list_tables()
            game_count = next((t["row_count"] for t in tables if t["table"] == "games"), 0)
            if game_count == 0:
                seed_sample_data(self.storage)

        # Initialize Elo ratings from completed games
        self._init_elo()

    def _init_elo(self) -> None:
        """Initialize Elo ratings from completed NHL games in storage."""
        completed = self.storage.query_games(sport="NHL", status="final", limit=500)
        if not completed:
            return
        games = sorted(completed, key=lambda g: g.get("date", ""))
        self.elo.initialize_from_games(games)

    def _track(self, tool_name: str, params: dict, result: dict, start: float) -> dict:
        """Track tool call for observability."""
        elapsed = (time.time() - start) * 1000
        self.storage.log_tool_call(
            session_id=self._session_id,
            tool_name=tool_name,
            input_params=params,
            output_summary={"result_keys": list(result.keys())},
            cost=0.01,
            latency_ms=elapsed,
        )
        self.storage.charge_session(self._session_id, 0.01)
        result["_meta"] = {
            "tool": tool_name,
            "latency_ms": round(elapsed, 1),
            "version": self.VERSION,
        }
        return result

    def _player_impact(self, team: str) -> dict[str, float]:
        """Compute player-level impact features for a team.

        Looks at roster injuries, top scorer availability, and depth.
        Returns features that adjust prediction confidence.
        """
        roster = self.storage.query_roster(team=team)
        if not roster:
            return {"injury_impact": 0.0, "top_scorer_available": 1.0, "roster_completeness": 1.0}

        total = len(roster)
        healthy = [p for p in roster if p.get("injury_status", "healthy") == "healthy"]

        # Sort skaters by points to find top contributors
        skaters = [p for p in roster if p.get("position", "") not in ("G",)]
        skaters_by_pts = sorted(
            skaters,
            key=lambda p: (p.get("stats") or {}).get("points", 0)
            if isinstance(p.get("stats"), dict) else 0,
            reverse=True,
        )

        # Top-10 skater availability
        top_players = skaters_by_pts[:10]
        top_healthy = sum(
            1 for p in top_players if p.get("injury_status", "healthy") == "healthy"
        )
        top_scorer_available = top_healthy / max(len(top_players), 1)

        # Injury impact: top-line players cost more
        injury_impact = 0.0
        for i, p in enumerate(skaters_by_pts):
            if p.get("injury_status", "healthy") != "healthy":
                weight = max(0.015 - i * 0.001, 0.003)
                injury_impact += weight

        # Goalie injury check
        goalies = [p for p in roster if p.get("position") == "G"]
        starter_injured = any(
            g.get("injury_status", "healthy") != "healthy"
            for g in goalies[:1]
        )
        if starter_injured:
            injury_impact += 0.03

        return {
            "injury_impact": round(min(injury_impact, 0.10), 4),
            "top_scorer_available": round(top_scorer_available, 3),
            "roster_completeness": round(len(healthy) / max(total, 1), 3),
        }

    @staticmethod
    def _build_model_types(
        model_preds: list[float],
        elo_prob: float | None,
        features: dict[str, float],
    ) -> list[str]:
        """Build the list of model type labels matching predict_nhl_game output."""
        types = ["points_log5", "pythagorean", "recent_form",
                 "goal_differential", "goaltender_matchup", "combined_adjusted"]
        market = features.get("market_implied_prob", 0)
        if market > 0 and 0.15 <= market <= 0.85:
            types.append("market_implied")
        if elo_prob is not None and 0.01 <= elo_prob <= 0.99:
            types.append("elo")
        return types[:len(model_preds)]
