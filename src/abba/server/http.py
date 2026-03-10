"""HTTP server for ABBA — REST API for GPT Actions, Claude MCP, and direct use.

Three integration paths:

1. **GPT Actions (Custom GPT)**:
   - Generate OpenAPI schema: python -m abba.server.openapi https://your-server.com
   - Paste into Custom GPT Actions config
   - Each tool gets its own POST /tools/{tool_name} endpoint

2. **Claude MCP**:
   - python -m abba.server.mcp (stdio transport)
   - Or configure in claude_desktop_config.json

3. **Direct REST API**:
   - POST /tools/call {"name": "predict_game", "arguments": {"game_id": "..."}}

Usage:
    python -m abba.server.http          # starts on :8420
    ABBA_PORT=9000 python -m abba.server.http
"""

from __future__ import annotations

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

from .toolkit import ABBAToolkit


class ABBAHandler(BaseHTTPRequestHandler):
    """HTTP request handler for ABBA tools."""

    toolkit: ABBAToolkit  # set by the factory

    # Build tool name set for routing
    _tool_names: set[str] = set()

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self._cors_headers()
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json_response({"status": "ok", "version": ABBAToolkit.VERSION})
        elif self.path == "/tools":
            self._json_response({"tools": self.toolkit.list_tools()})
        elif self.path == "/openapi.json":
            from .openapi import generate_openapi_spec
            host = self.headers.get("Host", "localhost:8420")
            scheme = "https" if "443" in host else "http"
            spec = generate_openapi_spec(f"{scheme}://{host}")
            self._json_response(spec)
        elif self.path == "/connectors":
            from ..connectors.live import list_connectors
            self._json_response({"connectors": list_connectors()})
        else:
            self._json_response({"error": "not found"}, 404)

    def do_POST(self) -> None:
        # Generic dispatch: POST /tools/call
        if self.path == "/tools/call":
            body = self._read_body()
            if not body:
                self._json_response({"error": "empty body"}, 400)
                return
            name = body.get("name", "")
            arguments = body.get("arguments", {})
            result = self.toolkit.call_tool(name, **arguments)
            self._json_response(result)
            return

        # Per-tool endpoints: POST /tools/{tool_name}
        if self.path.startswith("/tools/"):
            tool_name = self.path[7:]  # strip "/tools/"
            # Lazily populate tool names
            if not self._tool_names:
                ABBAHandler._tool_names = {
                    t["name"] for t in self.toolkit.list_tools()
                }
            if tool_name in self._tool_names:
                body = self._read_body() or {}
                result = self.toolkit.call_tool(tool_name, **body)
                self._json_response(result)
                return

        self._json_response({"error": "not found"}, 404)

    def _read_body(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _json_response(self, data: Any, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def log_message(self, format: str, *args: Any) -> None:
        pass


def run_http_server(port: int = 8420, db_path: str = ":memory:") -> None:
    """Start the HTTP server."""
    toolkit = ABBAToolkit(db_path=db_path, auto_seed=True)
    ABBAHandler.toolkit = toolkit

    server = HTTPServer(("0.0.0.0", port), ABBAHandler)
    print(f"ABBA HTTP server v{ABBAToolkit.VERSION} on port {port}")
    print(f"  GET  /health         - health check")
    print(f"  GET  /tools          - tool discovery")
    print(f"  GET  /openapi.json   - OpenAPI spec (for GPT Actions)")
    print(f"  GET  /connectors     - data source info")
    print(f"  POST /tools/call     - execute a tool (generic)")
    print(f"  POST /tools/{{name}} - execute a tool (per-tool endpoint)")
    print()
    print(f"GPT Actions setup:")
    print(f"  1. Copy the OpenAPI schema from http://localhost:{port}/openapi.json")
    print(f"  2. In ChatGPT → Create a GPT → Configure → Actions → Import from URL")
    print(f"  3. Or: python -m abba.server.openapi https://your-public-url")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    port = int(os.environ.get("ABBA_PORT", "8420"))
    db_path = os.environ.get("ABBA_DB_PATH", ":memory:")
    run_http_server(port, db_path)
