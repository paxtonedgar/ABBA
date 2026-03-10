"""OpenAPI schema generator for ABBA — powers GPT Actions and Swagger docs.

Generates an OpenAPI 3.1 spec from the toolkit's tool registry.
Paste this into a Custom GPT's Actions config to give ChatGPT
direct access to ABBA's NHL analytics tools.

Usage:
    python -m abba.server.openapi          # prints JSON schema to stdout
    python -m abba.server.openapi > openapi.json
"""

from __future__ import annotations

import json
from typing import Any

from .toolkit import ABBAToolkit


def generate_openapi_spec(server_url: str = "http://localhost:8420") -> dict[str, Any]:
    """Generate OpenAPI 3.1 spec from ABBA's tool registry."""
    toolkit = ABBAToolkit(auto_seed=False)
    tools = toolkit.list_tools()

    paths: dict[str, Any] = {}

    # Health endpoint
    paths["/health"] = {
        "get": {
            "operationId": "healthCheck",
            "summary": "Check if ABBA server is running",
            "responses": {
                "200": {
                    "description": "Server status",
                    "content": {"application/json": {"schema": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string"},
                            "version": {"type": "string"},
                        },
                    }}},
                }
            },
        }
    }

    # Tool discovery endpoint
    paths["/tools"] = {
        "get": {
            "operationId": "listTools",
            "summary": "List all available ABBA tools and their parameters",
            "responses": {
                "200": {
                    "description": "Tool schemas",
                    "content": {"application/json": {"schema": {
                        "type": "object",
                        "properties": {
                            "tools": {"type": "array", "items": {"type": "object"}},
                        },
                    }}},
                }
            },
        }
    }

    # Generate a dedicated endpoint for each tool
    for tool in tools:
        tool_name = tool["name"]
        operation_id = tool_name.replace("_", "")
        params = tool.get("params", {})

        # Build request body schema from tool params
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, param_info in params.items():
            ptype = param_info.get("type", "string")
            type_map = {"string": "string", "integer": "integer",
                        "number": "number", "boolean": "boolean", "object": "object"}
            prop: dict[str, Any] = {"type": type_map.get(ptype, "string")}
            if "enum" in param_info:
                prop["enum"] = param_info["enum"]
            if "default" in param_info:
                prop["default"] = param_info["default"]
            properties[param_name] = prop
            if param_info.get("required"):
                required.append(param_name)

        # Each tool gets its own POST endpoint for cleaner GPT Actions
        path_spec: dict[str, Any] = {
            "post": {
                "operationId": operation_id,
                "summary": tool["description"],
                "tags": [tool.get("category", "general")],
                "responses": {
                    "200": {
                        "description": "Tool result",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    }
                },
            }
        }

        if properties:
            path_spec["post"]["requestBody"] = {
                "required": bool(required),
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": properties,
                            "required": required if required else [],
                        }
                    }
                },
            }

        paths[f"/tools/{tool_name}"] = path_spec

    # Also keep the generic call endpoint
    paths["/tools/call"] = {
        "post": {
            "operationId": "callTool",
            "summary": "Call any ABBA tool by name (generic dispatch)",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Tool name"},
                                "arguments": {"type": "object", "description": "Tool parameters"},
                            },
                            "required": ["name"],
                        }
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Tool result",
                    "content": {"application/json": {"schema": {"type": "object"}}},
                }
            },
        }
    }

    return {
        "openapi": "3.1.0",
        "info": {
            "title": "ABBA Sports Analytics API",
            "version": ABBAToolkit.VERSION,
            "description": (
                "NHL analytics toolkit — live game data, ensemble ML predictions, "
                "odds comparison, expected value scanning, Kelly Criterion sizing, "
                "goaltender matchups, advanced stats, salary cap analysis, and "
                "playoff probability. Every response includes confidence metadata "
                "so you know how much to trust the numbers."
            ),
        },
        "servers": [{"url": server_url}],
        "paths": paths,
    }


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8420"
    spec = generate_openapi_spec(url)
    print(json.dumps(spec, indent=2))
