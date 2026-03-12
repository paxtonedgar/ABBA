"""Tests for MCP server interface (FastMCP-based).

Validates that:
1. All MCP tools are registered and callable
2. Tool functions return valid JSON strings
3. The direct Python SDK interface (ABBAToolkit) works
"""

import json

import pytest

from abba.server.toolkit import ABBAToolkit


@pytest.fixture
def toolkit():
    return ABBAToolkit(db_path=":memory:", auto_seed=True)


class TestFastMCPRegistration:
    """Verify all expected tools are registered in the FastMCP server."""

    def test_mcp_server_importable(self):
        """The MCP module imports without error and exposes a FastMCP instance."""
        from abba.server.mcp import mcp
        assert mcp is not None
        assert mcp.name == "ABBA Sports Analytics"

    def test_all_expected_tools_registered(self):
        """Every expected tool name is registered on the FastMCP server."""
        from abba.server.mcp import mcp

        expected_tools = {
            "query_games", "query_odds", "query_team_stats", "list_sources",
            "describe_dataset", "predict_game", "explain_prediction",
            "nhl_predict_game", "find_value", "compare_odds", "calculate_ev",
            "kelly_sizing", "query_goaltender_stats", "query_advanced_stats",
            "query_cap_data", "query_roster", "season_review", "playoff_odds",
            "refresh_data", "session_budget", "run_workflow", "list_workflows",
            "think", "session_replay",
        }
        registered = set(mcp._tool_manager._tools.keys())
        missing = expected_tools - registered
        assert not missing, f"Missing MCP tools: {missing}"


class TestMCPToolFunctions:
    """Call each MCP tool function directly to verify they return valid JSON."""

    def test_list_sources(self):
        from abba.server.mcp import list_sources
        result = json.loads(list_sources())
        assert "sources" in result

    def test_query_games(self):
        from abba.server.mcp import query_games
        result = json.loads(query_games(sport="NHL"))
        assert "games" in result

    def test_query_team_stats(self):
        from abba.server.mcp import query_team_stats
        result = json.loads(query_team_stats(sport="NHL"))
        assert "teams" in result

    def test_describe_dataset(self):
        from abba.server.mcp import describe_dataset
        result = json.loads(describe_dataset(table="games"))
        assert isinstance(result, (list, dict))

    def test_calculate_ev(self):
        from abba.server.mcp import calculate_ev
        result = json.loads(calculate_ev(win_probability=0.55, decimal_odds=2.0))
        assert "expected_value" in result

    def test_kelly_sizing(self):
        from abba.server.mcp import kelly_sizing
        result = json.loads(kelly_sizing(win_probability=0.55, decimal_odds=2.0))
        assert "recommended_stake" in result or "fraction" in result

    def test_session_budget(self):
        from abba.server.mcp import session_budget
        result = json.loads(session_budget())
        assert isinstance(result, dict)

    def test_list_workflows(self):
        from abba.server.mcp import list_workflows
        result = json.loads(list_workflows())
        assert isinstance(result, dict)

    def test_predict_game_nonexistent(self):
        from abba.server.mcp import predict_game
        result = json.loads(predict_game(game_id="nonexistent"))
        assert "error" in result

    def test_nhl_predict_game_nonexistent(self):
        from abba.server.mcp import nhl_predict_game
        result = json.loads(nhl_predict_game(game_id="nonexistent"))
        assert "error" in result


class TestDirectSDK:
    """Test the direct Python import path — the most stable interface."""

    def test_import_and_use(self):
        from abba import ABBAToolkit
        tk = ABBAToolkit()
        result = tk.call_tool("list_sources")
        assert "sources" in result

    def test_all_tools_dispatchable(self):
        """Every listed tool should be in the dispatch map."""
        from abba import ABBAToolkit
        tk = ABBAToolkit()
        tool_names = [t["name"] for t in tk.list_tools()]
        safe_args: dict[str, dict] = {
            "describe_dataset": {"table": "games"},
            "predict_game": {"game_id": "nonexistent"},
            "explain_prediction": {"game_id": "nonexistent"},
            "graph_analysis": {"team_data": {"players": ["A", "B"], "relationships": []}},
            "compare_odds": {"game_id": "nonexistent"},
            "calculate_ev": {"win_probability": 0.5, "decimal_odds": 2.0},
            "kelly_sizing": {"win_probability": 0.5, "decimal_odds": 2.0},
            "nhl_predict_game": {"game_id": "nonexistent"},
            "season_review": {"team_id": "nonexistent"},
            "playoff_odds": {"team_id": "nonexistent"},
            "run_workflow": {"workflow": "tonights_slate"},
        }
        for name in tool_names:
            result = tk.call_tool(name, **safe_args.get(name, {}))
            assert isinstance(result, dict), f"{name} returned {type(result)}"
