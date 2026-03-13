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
from ..engine.ml_model import NHLGameModel
from ..engine.value import ValueEngine
from ..services.data import DataService
from ..services.market import MarketService
from ..services.prediction import PredictionService
from ..storage import Storage, create_storage
from .tools import (
    AnalyticsToolsMixin,
    DataToolsMixin,
    MarketToolsMixin,
    NHLToolsMixin,
    ReasoningToolsMixin,
    SessionToolsMixin,
    ToolRegistryMixin,
)


class ABBAToolkit(
    DataToolsMixin,
    AnalyticsToolsMixin,
    MarketToolsMixin,
    NHLToolsMixin,
    ReasoningToolsMixin,
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
        self.storage = create_storage(db_path)
        self.ensemble = EnsembleEngine()
        self.features = FeatureEngine()
        self.kelly = KellyEngine()
        self.value = ValueEngine()
        self.graph = GraphEngine()
        self.hockey = HockeyAnalytics()
        self.elo = EloRatings(k=4, home_advantage=50)
        self.ml_model = NHLGameModel()
        # Recover persisted refresh timestamp from DB (survives restarts)
        self._last_refresh_ts: float | None = self.storage.get_last_refresh("team_stats")

        # --- Services ---
        self.prediction = PredictionService(
            storage=self.storage,
            hockey=self.hockey,
            ensemble=self.ensemble,
            features=self.features,
            elo=self.elo,
            ml_model=self.ml_model,
        )
        self.data_service = DataService(self.storage)
        # MarketService.predict_fn wired after init (needs self reference)
        self.market = MarketService(
            storage=self.storage,
            value=self.value,
            kelly=self.kelly,
        )

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

        # Train ML model on completed games (fast — <1s on startup)
        self._init_ml_model()

        # Wire cross-service dependencies now that toolkit is fully initialized
        self.market._predict_fn = lambda gid: self.predict_game(gid)
        self.market._nhl_predict_fn = lambda gid: self.nhl_predict_game(gid)

    def _init_elo(self) -> None:
        """Initialize Elo ratings from completed NHL games in storage."""
        completed = self.storage.query_games(sport="NHL", status="final", limit=500)
        if not completed:
            return
        games = sorted(completed, key=lambda g: g.get("date", ""))
        self.elo.initialize_from_games(games)

    def _init_ml_model(self) -> None:
        """Train gradient boosting model on completed NHL games in storage."""
        completed = self.storage.query_games(sport="NHL", status="final", limit=500)
        if not completed:
            return

        # Build team stats lookup from current data
        all_stats = self.storage.query_team_stats(sport="NHL")
        stats_by_team: dict[str, dict] = {}
        for row in all_stats:
            tid = row.get("team_id", "")
            stats = row.get("stats", {})
            if isinstance(stats, dict) and tid:
                stats_by_team[tid] = stats

        if not stats_by_team:
            return

        # Build advanced stats lookup (Corsi/xG — no longer ghost features)
        all_advanced = self.storage.query_nhl_advanced_stats()
        adv_by_team: dict[str, dict] = {}
        for row in all_advanced:
            tid = row.get("team_id", "")
            stats = row.get("stats", {})
            if isinstance(stats, dict) and tid:
                adv_by_team[tid] = stats

        # Build goaltender stats lookup
        all_goalies = self.storage.query_goaltender_stats()
        goalie_by_team: dict[str, dict] = {}
        for row in all_goalies:
            team = row.get("team", "")
            stats = row.get("stats", {})
            if isinstance(stats, dict) and team:
                # Keep the goalie with highest games played per team
                existing = goalie_by_team.get(team, {})
                if stats.get("games_played", 0) >= existing.get("games_played", 0):
                    goalie_by_team[team] = stats

        self.ml_model.train(
            completed, stats_by_team,
            advanced_stats=adv_by_team or None,
            goaltender_stats=goalie_by_team or None,
        )

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
        types = ["points_log5", "pythagorean_situational", "goaltender_matchup"]
        if elo_prob is not None and 0.01 <= elo_prob <= 0.99:
            types.append("elo")
        market = features.get("market_implied_prob", 0)
        if market > 0 and 0.15 <= market <= 0.85:
            types.append("market_implied")
        # Pad with gradient_boosting if ML model added an extra prediction
        while len(types) < len(model_preds):
            types.append("gradient_boosting")
        return types[:len(model_preds)]
