"""HTTP server for ABBA -- stable fallback for when MCP is flaky.

MCP over stdio is fragile: process crashes lose state, reconnection
is not standardized, and most agent frameworks have incomplete MCP
support. This HTTP server provides the same tools over a stable
REST API that any agent framework can call.

Three ways to consume ABBA, in order of reliability:

1. **Direct Python import** (most stable):
   from abba import ABBAToolkit
   toolkit = ABBAToolkit()
   toolkit.call_tool("predict_game", game_id="...")

2. **HTTP REST API** (stable, language-agnostic):
   POST /tools/call {"name": "predict_game", "arguments": {"game_id": "..."}}

3. **MCP stdio** (standard but fragile):
   python -m abba.server.mcp

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

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json_response({"status": "ok", "version": ABBAToolkit.VERSION})
        elif self.path == "/tools":
            self._json_response({"tools": self.toolkit.list_tools()})
        elif self.path == "/connectors":
            from ..connectors.live import list_connectors
            self._json_response({"connectors": list_connectors()})
        else:
            self._json_response({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if self.path == "/tools/call":
            body = self._read_body()
            if not body:
                self._json_response({"error": "empty body"}, 400)
                return
            name = body.get("name", "")
            arguments = body.get("arguments", {})
            result = self.toolkit.call_tool(name, **arguments)
            self._json_response(result)
        else:
            self._json_response({"error": "not found"}, 404)

    def _read_body(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None

    def _json_response(self, data: Any, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default logging noise
        pass


def run_http_server(port: int = 8420, db_path: str = ":memory:") -> None:
    """Start the HTTP server."""
    toolkit = ABBAToolkit(db_path=db_path, auto_seed=True)
    ABBAHandler.toolkit = toolkit

    server = HTTPServer(("0.0.0.0", port), ABBAHandler)
    print(f"ABBA HTTP server v{ABBAToolkit.VERSION} on port {port}")
    print(f"  GET  /health       - health check")
    print(f"  GET  /tools        - tool discovery")
    print(f"  GET  /connectors   - data source info")
    print(f"  POST /tools/call   - execute a tool")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    port = int(os.environ.get("ABBA_PORT", "8420"))
    db_path = os.environ.get("ABBA_DB_PATH", ":memory:")
    run_http_server(port, db_path)
