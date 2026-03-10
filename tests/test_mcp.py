"""Tests for MCP and HTTP server interfaces."""

import json

import pytest

from abba.server.mcp import create_mcp_server, handle_mcp_request, tools_to_mcp_schema
from abba.server.toolkit import ABBAToolkit


@pytest.fixture
def toolkit():
    return ABBAToolkit(db_path=":memory:", auto_seed=True)


class TestMCPProtocol:
    def test_initialize(self, toolkit):
        req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        resp = handle_mcp_request(toolkit, req)
        assert resp["id"] == 1
        assert "serverInfo" in resp["result"]
        assert resp["result"]["serverInfo"]["name"] == "abba"

    def test_tools_list(self, toolkit):
        req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        resp = handle_mcp_request(toolkit, req)
        tools = resp["result"]["tools"]
        assert len(tools) >= 20
        # Every tool should have MCP-compliant schema
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"

    def test_tools_call(self, toolkit):
        req = {
            "jsonrpc": "2.0", "id": 3,
            "method": "tools/call",
            "params": {"name": "list_sources", "arguments": {}},
        }
        resp = handle_mcp_request(toolkit, req)
        content = resp["result"]["content"]
        assert len(content) == 1
        data = json.loads(content[0]["text"])
        assert "sources" in data

    def test_tools_call_with_args(self, toolkit):
        req = {
            "jsonrpc": "2.0", "id": 4,
            "method": "tools/call",
            "params": {"name": "query_games", "arguments": {"sport": "MLB"}},
        }
        resp = handle_mcp_request(toolkit, req)
        data = json.loads(resp["result"]["content"][0]["text"])
        assert data["count"] > 0

    def test_resources_list(self, toolkit):
        req = {"jsonrpc": "2.0", "id": 5, "method": "resources/list", "params": {}}
        resp = handle_mcp_request(toolkit, req)
        resources = resp["result"]["resources"]
        uris = [r["uri"] for r in resources]
        assert any("games" in uri for uri in uris)

    def test_unknown_method(self, toolkit):
        req = {"jsonrpc": "2.0", "id": 6, "method": "bogus/method", "params": {}}
        resp = handle_mcp_request(toolkit, req)
        assert "error" in resp
        assert resp["error"]["code"] == -32601


class TestMCPSchema:
    def test_schema_generation(self, toolkit):
        tools = tools_to_mcp_schema(toolkit)
        predict = next(t for t in tools if t["name"] == "predict_game")
        assert "game_id" in predict["inputSchema"]["properties"]
        assert "game_id" in predict["inputSchema"]["required"]

    def test_server_info(self):
        info = create_mcp_server()
        assert info["name"] == "abba"
        assert "tools" in info["capabilities"]


class TestDirectSDK:
    """Test the direct Python import path -- the most stable interface."""

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
        # Verify each name resolves (call with safe defaults)
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
