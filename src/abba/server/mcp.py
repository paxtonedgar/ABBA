"""MCP (Model Context Protocol) server for ABBA.

This is the entry point that makes ABBA a subscribable tool service.
When an agent (Claude, GPT, or custom) connects via MCP, it discovers
all ABBA tools automatically and can call them with structured I/O.

The server solves the stale data problem:
- LLMs hallucinate sports records, rosters, and outcomes
- ABBA provides live, queryable data the agent can trust
- Every response includes freshness metadata so the agent knows
  exactly how old the data is

Run as MCP server:
    python -m abba.server.mcp

Or register in Claude Desktop / claude_desktop_config.json:
    {
      "mcpServers": {
        "abba": {
          "command": "python",
          "args": ["-m", "abba.server.mcp"],
          "env": {"ABBA_DB_PATH": "abba.duckdb"}
        }
      }
    }

For pip/artifact install:
    pip install abba
    # Then register as MCP server, or import directly:
    from abba import ABBAToolkit
    toolkit = ABBAToolkit()
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from ..server.toolkit import ABBAToolkit


def create_mcp_server() -> dict[str, Any]:
    """Create the MCP server configuration.

    Returns the server info that MCP clients use for capability discovery.
    """
    return {
        "name": "abba",
        "version": ABBAToolkit.VERSION,
        "description": (
            "Sports analytics toolkit -- live game data, ensemble ML predictions, "
            "odds comparison, expected value scanning, and Kelly Criterion sizing. "
            "Solves the stale data problem: agents get real-time sports data "
            "instead of hallucinating records and rosters."
        ),
        "capabilities": {
            "tools": True,
            "resources": True,
        },
    }


def tools_to_mcp_schema(toolkit: ABBAToolkit) -> list[dict[str, Any]]:
    """Convert ABBAToolkit tools to MCP tool schema format."""
    mcp_tools = []
    for tool in toolkit.list_tools():
        properties = {}
        required = []
        for param_name, param_info in tool["params"].items():
            prop: dict[str, Any] = {}
            ptype = param_info.get("type", "string")
            type_map = {
                "string": "string",
                "integer": "integer",
                "number": "number",
                "boolean": "boolean",
                "object": "object",
            }
            prop["type"] = type_map.get(ptype, "string")
            if "enum" in param_info:
                prop["enum"] = param_info["enum"]
            if "default" in param_info:
                prop["default"] = param_info["default"]
            if "format" in param_info:
                prop["description"] = f"Format: {param_info['format']}"
            properties[param_name] = prop
            if param_info.get("required"):
                required.append(param_name)

        mcp_tools.append({
            "name": tool["name"],
            "description": tool["description"],
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        })
    return mcp_tools


def handle_mcp_request(toolkit: ABBAToolkit, request: dict[str, Any]) -> dict[str, Any]:
    """Handle a single MCP JSON-RPC request.

    Supports:
    - initialize: capability exchange
    - tools/list: tool discovery
    - tools/call: tool execution
    - resources/list: data source listing
    """
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": create_mcp_server(),
                "capabilities": {"tools": {}, "resources": {}},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": tools_to_mcp_schema(toolkit)},
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        result = toolkit.call_tool(tool_name, **arguments)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": json.dumps(result, default=str)}],
            },
        }

    if method == "resources/list":
        sources = toolkit.list_sources()
        resources = []
        for src in sources.get("sources", []):
            resources.append({
                "uri": f"abba://data/{src['table']}",
                "name": src["table"],
                "description": f"Sports data table: {src['table']} ({src['row_count']} rows)",
                "mimeType": "application/json",
            })
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"resources": resources},
        }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def run_stdio_server() -> None:
    """Run MCP server over stdio (standard MCP transport).

    Reads JSON-RPC requests from stdin, writes responses to stdout.
    This is what Claude Desktop and other MCP clients expect.
    """
    db_path = os.environ.get("ABBA_DB_PATH", ":memory:")
    toolkit = ABBAToolkit(db_path=db_path, auto_seed=True)

    # Write to stderr so it doesn't pollute the JSON-RPC stream
    sys.stderr.write(f"ABBA MCP server v{ABBAToolkit.VERSION} started\n")
    sys.stderr.write(f"Database: {db_path}\n")
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            response = handle_mcp_request(toolkit, request)
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except json.JSONDecodeError:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
            sys.stdout.write(json.dumps(error_response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_server()
