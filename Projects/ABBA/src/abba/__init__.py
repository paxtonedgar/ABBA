"""
ABBA - Sports Analytics Toolkit for AI Agents

Agent-callable analytics tools for sports betting analysis.
DuckDB-backed, ensemble ML predictions, Kelly Criterion sizing.

Usage:
    from abba import ABBAToolkit
    toolkit = ABBAToolkit()
    tools = toolkit.list_tools()
    result = toolkit.call_tool("predict_game", game_id="...")
"""

__version__ = "2.0.0"

from .server.toolkit import ABBAToolkit

__all__ = ["ABBAToolkit"]
