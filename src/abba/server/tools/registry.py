"""Tool registry mixin -- list_tools() and call_tool() dispatch."""

from __future__ import annotations

from typing import Any


class ToolRegistryMixin:
    """Tool discovery and dynamic dispatch."""

    def list_tools(self) -> list[dict[str, Any]]:
        """Return schema for all available tools."""
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
            {
                "name": "nhl_predict_game",
                "category": "analytics",
                "description": "NHL-specific prediction using Corsi, xG, goaltender matchup, Elo, player impact, special teams, and rest",
                "params": {
                    "game_id": {"type": "string", "required": True},
                    "method": {"type": "string", "optional": True, "default": "weighted"},
                },
            },
            {
                "name": "query_goaltender_stats",
                "category": "data",
                "description": "Query NHL goaltender stats (Sv%, GAA, GSAA, xGSAA, quality starts)",
                "params": {
                    "team": {"type": "string", "optional": True},
                    "goaltender_id": {"type": "string", "optional": True},
                    "season": {"type": "string", "optional": True},
                },
            },
            {
                "name": "query_advanced_stats",
                "category": "data",
                "description": "Query NHL advanced stats (Corsi, Fenwick, xG, PDO, score-adjusted)",
                "params": {
                    "team_id": {"type": "string", "optional": True},
                    "season": {"type": "string", "optional": True},
                },
            },
            {
                "name": "query_cap_data",
                "category": "data",
                "description": "Query salary cap data with cap analysis (space, dead cap, top earners)",
                "params": {
                    "team": {"type": "string", "optional": True},
                    "season": {"type": "string", "optional": True},
                    "position": {"type": "string", "optional": True},
                },
            },
            {
                "name": "query_roster",
                "category": "data",
                "description": "Query team roster with line combinations, player stats, and injury status",
                "params": {
                    "team": {"type": "string", "optional": True},
                    "season": {"type": "string", "optional": True},
                    "position": {"type": "string", "optional": True},
                },
            },
            {
                "name": "season_review",
                "category": "analytics",
                "description": "Comprehensive NHL season review (record, analytics grades, goaltending, special teams)",
                "params": {
                    "team_id": {"type": "string", "required": True},
                    "season": {"type": "string", "optional": True},
                },
            },
            {
                "name": "playoff_odds",
                "category": "analytics",
                "description": "Estimate playoff probability from current points pace (Monte Carlo simulation)",
                "params": {
                    "team_id": {"type": "string", "required": True},
                    "season": {"type": "string", "optional": True},
                    "division_cutline": {"type": "integer", "optional": True, "default": 90},
                    "wildcard_cutline": {"type": "integer", "optional": True, "default": 95},
                },
            },
            {
                "name": "refresh_data",
                "category": "data",
                "description": "Refresh data from live sources (NHL API, Odds API). Requires API keys for some sources.",
                "params": {
                    "source": {"type": "string", "optional": True, "enum": ["sportradar", "nhl", "moneypuck", "advanced", "odds", "all"]},
                    "team": {"type": "string", "optional": True},
                },
            },
            {
                "name": "run_workflow",
                "category": "workflow",
                "description": "Run a multi-step analytical workflow (game_prediction, season_story, value_scan, betting_strategy, etc.)",
                "params": {
                    "workflow": {"type": "string", "required": True,
                                 "enum": ["game_prediction", "tonights_slate", "season_story", "value_scan",
                                          "cap_strategy", "playoff_race", "goaltender_duel", "team_comparison",
                                          "betting_strategy"]},
                },
            },
            {
                "name": "list_workflows",
                "category": "workflow",
                "description": "List available multi-step workflows with trigger phrases and parameters",
                "params": {},
            },
        ]

    def call_tool(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        """Dynamic tool dispatch -- call any tool by name."""
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
            "nhl_predict_game": self.nhl_predict_game,
            "query_goaltender_stats": self.query_goaltender_stats,
            "query_advanced_stats": self.query_advanced_stats,
            "query_cap_data": self.query_cap_data,
            "query_roster": self.query_roster,
            "season_review": self.season_review,
            "playoff_odds": self.playoff_odds,
            "refresh_data": self.refresh_data,
            "run_workflow": self.run_workflow,
            "list_workflows": self.list_workflows,
        }

        fn = tool_map.get(tool_name)
        if not fn:
            return {"error": f"unknown tool: {tool_name}", "available": list(tool_map.keys())}

        return fn(**kwargs)
