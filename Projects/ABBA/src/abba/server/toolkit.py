"""ABBA Toolkit -- agent-callable analytics tools.

The toolkit is the main entry point. When an agent loads the ABBA package,
it instantiates ABBAToolkit which:
1. Boots DuckDB storage (in-memory or persistent)
2. Initializes engine components (ensemble, features, kelly, value, graph)
3. Seeds sample data if storage is empty
4. Exposes tools as methods with structured input/output

Each tool method returns a plain dict that agents can reason over.
Tool discovery is via list_tools() which returns schema for every tool.

Usage:
    from abba.server import ABBAToolkit
    toolkit = ABBAToolkit()          # boots with sample data
    toolkit.query_games(sport="MLB") # returns structured dict
"""

from __future__ import annotations

import time
from typing import Any

from ..connectors.seed import seed_sample_data
from ..engine.ensemble import EnsembleEngine
from ..engine.features import FeatureEngine
from ..engine.graph import GraphEngine
from ..engine.kelly import KellyEngine
from ..engine.value import ValueEngine
from ..storage import Storage


class ABBAToolkit:
    """Agent-callable sports analytics toolkit.

    Instantiate once, call tools by name. Each tool returns a dict.
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

        # Create default session
        self._session_id = "default"
        self.storage.create_session(self._session_id, session_budget)

        if auto_seed:
            tables = self.storage.list_tables()
            game_count = next((t["row_count"] for t in tables if t["table"] == "games"), 0)
            if game_count == 0:
                seed_sample_data(self.storage)

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

    # =====================================================================
    # DATA TOOLS
    # =====================================================================

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

    # =====================================================================
    # ANALYTICS TOOLS
    # =====================================================================

    def predict_game(
        self,
        game_id: str,
        method: str = "weighted",
    ) -> dict[str, Any]:
        """Ensemble prediction for a game. Returns home win probability."""
        start = time.time()

        # Get game
        games = self.storage.query_games()
        game = next((g for g in games if g.get("game_id") == game_id), None)
        if not game:
            return self._track("predict_game", {"game_id": game_id},
                               {"error": f"game not found: {game_id}"}, start)

        sport = game.get("sport", "MLB")
        home = game.get("home_team", "")
        away = game.get("away_team", "")

        # Get team stats
        home_stats_list = self.storage.query_team_stats(team_id=home, sport=sport)
        away_stats_list = self.storage.query_team_stats(team_id=away, sport=sport)

        home_stats = home_stats_list[0] if home_stats_list else {"stats": {}}
        away_stats = away_stats_list[0] if away_stats_list else {"stats": {}}

        # Get weather
        weather = self.storage.get_weather(game_id)

        # Build features
        features = self.features.build_features(home_stats, away_stats, weather, sport)

        # Generate model predictions
        model_preds = self.features.predict_from_features(features)

        # Check cache
        data_hash = self.ensemble.data_hash(game_id, self.VERSION, features)
        cached = self.storage.get_cached_prediction(game_id, self.VERSION, data_hash)
        if cached:
            cached["_cache_hit"] = True
            return self._track("predict_game", {"game_id": game_id}, cached, start)

        # Combine
        prediction = self.ensemble.combine(model_preds, method=method)

        result = {
            "game_id": game_id,
            "home_team": home,
            "away_team": away,
            "sport": sport,
            "prediction": prediction.to_dict(),
            "features": {k: round(v, 4) for k, v in features.items()},
            "_cache_hit": False,
        }

        # Cache it
        self.storage.cache_prediction(game_id, self.VERSION, data_hash, result)

        return self._track("predict_game", {"game_id": game_id}, result, start)

    def explain_prediction(self, game_id: str) -> dict[str, Any]:
        """Explain what's driving a prediction -- feature importance."""
        start = time.time()
        pred = self.predict_game(game_id)
        if "error" in pred:
            return self._track("explain_prediction", {"game_id": game_id}, pred, start)

        features = pred.get("features", {})

        # Rank features by absolute deviation from neutral
        neutral = {
            "home_win_pct": 0.5, "away_win_pct": 0.5,
            "home_run_diff_per_game": 0.0, "away_run_diff_per_game": 0.0,
            "home_recent_form": 0.5, "away_recent_form": 0.5,
            "home_advantage": 0.54,
            "temp_impact": 0.0, "wind_impact": 0.0, "precip_risk": 0.0,
        }

        importance = []
        for feat, val in features.items():
            n = neutral.get(feat, 0.0)
            deviation = abs(val - n)
            direction = "favors_home" if val > n else "favors_away" if val < n else "neutral"
            importance.append({
                "feature": feat,
                "value": val,
                "neutral_value": n,
                "deviation": round(deviation, 4),
                "direction": direction,
                "description": self.features.FEATURE_SCHEMA.get(feat, ""),
            })

        importance.sort(key=lambda x: x["deviation"], reverse=True)

        result = {
            "game_id": game_id,
            "home_team": pred.get("home_team"),
            "away_team": pred.get("away_team"),
            "prediction": pred.get("prediction"),
            "top_factors": importance[:5],
            "all_factors": importance,
        }
        return self._track("explain_prediction", {"game_id": game_id}, result, start)

    def graph_analysis(self, team_data: dict[str, Any]) -> dict[str, Any]:
        """Team network analysis. Pass players + relationships."""
        start = time.time()
        result = self.graph.analyze_team(team_data)
        return self._track("graph_analysis", {"player_count": len(team_data.get("players", []))},
                           result, start)

    # =====================================================================
    # MARKET TOOLS
    # =====================================================================

    def find_value(
        self,
        sport: str | None = None,
        date: str | None = None,
        min_ev: float = 0.03,
    ) -> dict[str, Any]:
        """Scan for +EV betting opportunities."""
        start = time.time()

        games = self.storage.query_games(sport=sport, date=date, status="scheduled")
        if not games:
            return self._track("find_value", {"sport": sport, "date": date},
                               {"opportunities": [], "count": 0, "games_scanned": 0}, start)

        # Get predictions for all games
        predictions: dict[str, float] = {}
        for g in games:
            gid = g.get("game_id", "")
            pred = self.predict_game(gid)
            pred_data = pred.get("prediction", {})
            if isinstance(pred_data, dict) and "value" in pred_data:
                predictions[gid] = pred_data["value"]

        # Get all odds
        all_odds = self.storage.query_odds(latest_only=True)

        # Find value
        self.value.min_ev = min_ev
        opportunities = self.value.find_value(games, predictions, all_odds)

        result = {
            "opportunities": opportunities[:20],
            "count": len(opportunities),
            "games_scanned": len(games),
        }
        return self._track("find_value", {"sport": sport, "min_ev": min_ev}, result, start)

    def compare_odds(self, game_id: str) -> dict[str, Any]:
        """Compare odds across sportsbooks for a game."""
        start = time.time()
        all_odds = self.storage.query_odds(game_id=game_id, latest_only=True)
        result = self.value.compare_odds(all_odds, game_id)
        return self._track("compare_odds", {"game_id": game_id}, result, start)

    def calculate_ev(
        self,
        win_probability: float,
        decimal_odds: float,
    ) -> dict[str, Any]:
        """Calculate expected value for a specific bet."""
        start = time.time()
        ev = win_probability * (decimal_odds - 1.0) - (1.0 - win_probability)
        implied = 1.0 / decimal_odds if decimal_odds > 0 else 0
        edge = win_probability - implied

        result = {
            "win_probability": round(win_probability, 4),
            "decimal_odds": round(decimal_odds, 4),
            "implied_probability": round(implied, 4),
            "edge": round(edge, 4),
            "expected_value": round(ev, 4),
            "is_positive_ev": ev > 0,
        }
        return self._track("calculate_ev",
                           {"win_probability": win_probability, "decimal_odds": decimal_odds},
                           result, start)

    def kelly_sizing(
        self,
        win_probability: float,
        decimal_odds: float,
        bankroll: float = 10000.0,
    ) -> dict[str, Any]:
        """Calculate optimal position size using Kelly Criterion."""
        start = time.time()
        sizing = self.kelly.calculate(win_probability, decimal_odds, bankroll)
        result = sizing.to_dict()
        result["bankroll"] = bankroll
        result["decimal_odds"] = decimal_odds
        result["win_probability"] = round(win_probability, 4)
        return self._track("kelly_sizing",
                           {"win_probability": win_probability, "decimal_odds": decimal_odds,
                            "bankroll": bankroll}, result, start)

    # =====================================================================
    # SESSION / META TOOLS
    # =====================================================================

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

    # =====================================================================
    # TOOL DISCOVERY
    # =====================================================================

    def list_tools(self) -> list[dict[str, Any]]:
        """Return schema for all available tools. This is what agents
        call first to understand what capabilities are available."""
        return [
            {
                "name": "query_games",
                "category": "data",
                "description": "Query games with filters (sport, date, team, status)",
                "params": {
                    "sport": {"type": "string", "optional": True, "enum": ["MLB", "NHL"]},
                    "date": {"type": "string", "optional": True, "format": "YYYY-MM-DD"},
                    "date_from": {"type": "string", "optional": True},
                    "date_to": {"type": "string", "optional": True},
                    "team": {"type": "string", "optional": True},
                    "status": {"type": "string", "optional": True, "enum": ["scheduled", "final"]},
                    "limit": {"type": "integer", "optional": True, "default": 50},
                },
            },
            {
                "name": "query_odds",
                "category": "data",
                "description": "Query odds snapshots across sportsbooks",
                "params": {
                    "game_id": {"type": "string", "optional": True},
                    "sportsbook": {"type": "string", "optional": True},
                    "latest_only": {"type": "boolean", "optional": True, "default": True},
                },
            },
            {
                "name": "query_team_stats",
                "category": "data",
                "description": "Query team statistics (wins, losses, differentials)",
                "params": {
                    "team_id": {"type": "string", "optional": True},
                    "sport": {"type": "string", "optional": True},
                    "season": {"type": "string", "optional": True},
                },
            },
            {
                "name": "list_sources",
                "category": "data",
                "description": "List available data tables and row counts",
                "params": {},
            },
            {
                "name": "describe_dataset",
                "category": "data",
                "description": "Get column names and types for a data table",
                "params": {"table": {"type": "string", "required": True}},
            },
            {
                "name": "predict_game",
                "category": "analytics",
                "description": "Ensemble prediction for a game (home win probability)",
                "params": {
                    "game_id": {"type": "string", "required": True},
                    "method": {"type": "string", "optional": True, "enum": ["weighted", "average", "median", "voting"], "default": "weighted"},
                },
            },
            {
                "name": "explain_prediction",
                "category": "analytics",
                "description": "Feature importance breakdown for a prediction",
                "params": {"game_id": {"type": "string", "required": True}},
            },
            {
                "name": "graph_analysis",
                "category": "analytics",
                "description": "Team network analysis (centrality, cohesion, key players)",
                "params": {"team_data": {"type": "object", "required": True, "fields": ["players", "relationships"]}},
            },
            {
                "name": "find_value",
                "category": "market",
                "description": "Scan for +EV betting opportunities across all games",
                "params": {
                    "sport": {"type": "string", "optional": True},
                    "date": {"type": "string", "optional": True},
                    "min_ev": {"type": "number", "optional": True, "default": 0.03},
                },
            },
            {
                "name": "compare_odds",
                "category": "market",
                "description": "Compare odds across sportsbooks for a game",
                "params": {"game_id": {"type": "string", "required": True}},
            },
            {
                "name": "calculate_ev",
                "category": "market",
                "description": "Calculate expected value for a specific bet",
                "params": {
                    "win_probability": {"type": "number", "required": True},
                    "decimal_odds": {"type": "number", "required": True},
                },
            },
            {
                "name": "kelly_sizing",
                "category": "market",
                "description": "Calculate optimal bet size using Kelly Criterion",
                "params": {
                    "win_probability": {"type": "number", "required": True},
                    "decimal_odds": {"type": "number", "required": True},
                    "bankroll": {"type": "number", "optional": True, "default": 10000},
                },
            },
            {
                "name": "session_budget",
                "category": "meta",
                "description": "Check remaining session budget and call count",
                "params": {},
            },
        ]

    def call_tool(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        """Dynamic tool dispatch -- call any tool by name.

        This is the primary interface for agent frameworks that
        discover tools via list_tools() then dispatch dynamically.
        """
        tool_map = {
            "query_games": self.query_games,
            "query_odds": self.query_odds,
            "query_team_stats": self.query_team_stats,
            "list_sources": self.list_sources,
            "describe_dataset": self.describe_dataset,
            "predict_game": self.predict_game,
            "explain_prediction": self.explain_prediction,
            "graph_analysis": self.graph_analysis,
            "find_value": self.find_value,
            "compare_odds": self.compare_odds,
            "calculate_ev": self.calculate_ev,
            "kelly_sizing": self.kelly_sizing,
            "session_budget": self.session_budget,
        }

        fn = tool_map.get(tool_name)
        if not fn:
            return {"error": f"unknown tool: {tool_name}", "available": list(tool_map.keys())}

        return fn(**kwargs)
