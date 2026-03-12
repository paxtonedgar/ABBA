"""MCP server for ABBA — uses the official MCP Python SDK.

This makes ABBA available as a tool server in Claude Desktop,
Claude Code, and any MCP-compatible client.

Run as MCP server (stdio transport — what Claude Desktop uses):
    python -m abba.server.mcp

Register in Claude Desktop (claude_desktop_config.json):
    {
      "mcpServers": {
        "abba": {
          "command": "/path/to/abba-nhl/.venv/bin/python",
          "args": ["-m", "abba.server.mcp"],
          "env": {"ABBA_DB_PATH": "abba.duckdb"}
        }
      }
    }
"""

from __future__ import annotations

import json
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from ..server.toolkit import ABBAToolkit

# Initialize toolkit at module level so it's ready when tools are called
_db_path = os.environ.get("ABBA_DB_PATH", ":memory:")
_toolkit = ABBAToolkit(db_path=_db_path, auto_seed=True)

mcp = FastMCP(
    "ABBA Sports Analytics",
    instructions=(
        "ABBA provides live NHL data, ensemble ML predictions, odds comparison, "
        "expected value scanning, and Kelly Criterion sizing. Every response includes "
        "confidence metadata — always mention the reliability grade and caveats to the user.\n\n"
        "WORKFLOW PROTOCOL:\n"
        "1. Call refresh_data first to pull live NHL standings and schedule.\n"
        "2. Call think(phase='planning') BEFORE starting any analysis — articulate what "
        "you plan to do, what data you expect to need, and what might go wrong.\n"
        "3. After getting prediction results, call think(phase='post_analysis') to reflect "
        "on whether you trust the numbers, what the confidence interval really means, and "
        "what you can't verify.\n"
        "4. Before giving a final recommendation, call think(phase='caveat_check') to audit "
        "your own honesty — are you presenting uncertainty faithfully?\n\n"
        "The think tool is NOT optional. It forces epistemic discipline. Users can review "
        "your full reasoning chain via session_replay."
    ),
)


# ---------------------------------------------------------------------------
# Data tools
# ---------------------------------------------------------------------------

@mcp.tool()
def query_games(
    sport: str | None = None,
    date: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    team: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> str:
    """Query games with filters. Sport: NHL or MLB. Status: scheduled, final. Date format: YYYY-MM-DD."""
    result = _toolkit.query_games(
        sport=sport, date=date, date_from=date_from, date_to=date_to,
        team=team, status=status, limit=limit,
    )
    return json.dumps(result, default=str)


@mcp.tool()
def query_odds(
    game_id: str | None = None,
    sportsbook: str | None = None,
    latest_only: bool = True,
) -> str:
    """Query odds snapshots across sportsbooks for a game."""
    result = _toolkit.query_odds(game_id=game_id, sportsbook=sportsbook, latest_only=latest_only)
    return json.dumps(result, default=str)


@mcp.tool()
def query_team_stats(
    team_id: str | None = None,
    sport: str | None = None,
    season: str | None = None,
) -> str:
    """Query team statistics — wins, losses, goal/run differentials."""
    result = _toolkit.query_team_stats(team_id=team_id, sport=sport, season=season)
    return json.dumps(result, default=str)


@mcp.tool()
def list_sources() -> str:
    """List available data tables and row counts."""
    result = _toolkit.list_sources()
    return json.dumps(result, default=str)


@mcp.tool()
def describe_dataset(table: str) -> str:
    """Get column names and types for a data table."""
    result = _toolkit.describe_dataset(table=table)
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Analytics tools
# ---------------------------------------------------------------------------

@mcp.tool()
def predict_game(game_id: str, method: str = "weighted") -> str:
    """Ensemble prediction for a game. NHL games delegate to the NHL-specific predictor. Returns home win probability with confidence metadata."""
    result = _toolkit.predict_game(game_id=game_id, method=method)
    return json.dumps(result, default=str)


@mcp.tool()
def explain_prediction(game_id: str) -> str:
    """Feature importance breakdown for a prediction — what's driving the numbers."""
    result = _toolkit.explain_prediction(game_id=game_id)
    return json.dumps(result, default=str)


@mcp.tool()
def nhl_predict_game(game_id: str, method: str = "weighted") -> str:
    """NHL-specific multi-model prediction (up to 9 models when ML is trained). Includes data provenance, defaulted-feature tracking, and fail-closed guards on missing data."""
    result = _toolkit.nhl_predict_game(game_id=game_id, method=method)
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Market tools
# ---------------------------------------------------------------------------

@mcp.tool()
def find_value(
    sport: str | None = None,
    date: str | None = None,
    min_ev: float = 0.03,
) -> str:
    """Scan for +EV betting opportunities across scheduled games. Requires odds data (ODDS_API_KEY) to function."""
    result = _toolkit.find_value(sport=sport, date=date, min_ev=min_ev)
    return json.dumps(result, default=str)


@mcp.tool()
def compare_odds(game_id: str) -> str:
    """Compare odds across sportsbooks for a game — find the best line."""
    result = _toolkit.compare_odds(game_id=game_id)
    return json.dumps(result, default=str)


@mcp.tool()
def calculate_ev(win_probability: float, decimal_odds: float) -> str:
    """Calculate expected value for a specific bet. Returns EV, edge, and implied probability."""
    result = _toolkit.calculate_ev(win_probability=win_probability, decimal_odds=decimal_odds)
    return json.dumps(result, default=str)


@mcp.tool()
def kelly_sizing(
    win_probability: float,
    decimal_odds: float,
    bankroll: float = 10000.0,
) -> str:
    """Calculate optimal bet size using Kelly Criterion. Returns full, half, and quarter Kelly."""
    result = _toolkit.kelly_sizing(
        win_probability=win_probability, decimal_odds=decimal_odds, bankroll=bankroll,
    )
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# NHL-specific tools
# ---------------------------------------------------------------------------

@mcp.tool()
def query_goaltender_stats(
    team: str | None = None,
    goaltender_id: str | None = None,
    season: str | None = None,
) -> str:
    """Query NHL goaltender stats — Sv%, GAA, GSAA, xGSAA, quality starts."""
    result = _toolkit.query_goaltender_stats(team=team, goaltender_id=goaltender_id, season=season)
    return json.dumps(result, default=str)


@mcp.tool()
def query_advanced_stats(
    team_id: str | None = None,
    season: str | None = None,
) -> str:
    """Query NHL advanced stats — Corsi, Fenwick, xG, PDO, score-adjusted metrics."""
    result = _toolkit.query_advanced_stats(team_id=team_id, season=season)
    return json.dumps(result, default=str)


@mcp.tool()
def query_cap_data(
    team: str | None = None,
    season: str | None = None,
    position: str | None = None,
) -> str:
    """Query salary cap data with cap analysis — space, dead cap, top earners."""
    result = _toolkit.query_cap_data(team=team, season=season, position=position)
    return json.dumps(result, default=str)


@mcp.tool()
def query_roster(
    team: str | None = None,
    season: str | None = None,
    position: str | None = None,
) -> str:
    """Query team roster with player stats and injury status."""
    result = _toolkit.query_roster(team=team, season=season, position=position)
    return json.dumps(result, default=str)


@mcp.tool()
def season_review(team_id: str, season: str | None = None) -> str:
    """Comprehensive NHL season review — record, analytics grades, goaltending, special teams."""
    result = _toolkit.season_review(team_id=team_id, season=season)
    return json.dumps(result, default=str)


@mcp.tool()
def playoff_odds(
    team_id: str,
    season: str | None = None,
    division_cutline: int = 90,
    wildcard_cutline: int = 95,
) -> str:
    """Estimate playoff probability from current points pace using Monte Carlo simulation. Uses seed data or live stats if refreshed."""
    result = _toolkit.playoff_odds(
        team_id=team_id, season=season,
        division_cutline=division_cutline, wildcard_cutline=wildcard_cutline,
    )
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Session tools
# ---------------------------------------------------------------------------

@mcp.tool()
def refresh_data(source: str = "all", team: str | None = None) -> str:
    """Refresh data from live sources — NHL API (free, no key) and Odds API (needs ODDS_API_KEY). Call this first!"""
    result = _toolkit.refresh_data(source=source, team=team)
    return json.dumps(result, default=str)


@mcp.tool()
def session_budget() -> str:
    """Check remaining session budget and tool call count."""
    result = _toolkit.session_budget()
    return json.dumps(result, default=str)


@mcp.tool()
def run_workflow(workflow: str) -> str:
    """Run a multi-step analytical workflow. Available: game_prediction, tonights_slate, season_story, value_scan, cap_strategy, playoff_race, goaltender_duel, team_comparison, betting_strategy."""
    result = _toolkit.run_workflow(workflow=workflow)
    return json.dumps(result, default=str)


@mcp.tool()
def list_workflows() -> str:
    """List available multi-step workflows with descriptions and trigger phrases."""
    result = _toolkit.list_workflows()
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Reasoning tools
# ---------------------------------------------------------------------------

@mcp.tool()
def think(
    phase: str,
    plan: str | None = None,
    uncertainty: str | None = None,
    data_trust: str | None = None,
    workflow_gaps: str | None = None,
    want_to_verify: str | None = None,
    raw_thought: str | None = None,
) -> str:
    """Externalize your reasoning before, during, and after analysis.

    REQUIRED before analytics calls. Forces epistemic honesty.

    phase: planning | pre_analysis | post_analysis | synthesis | caveat_check | recommendation
    plan: What you intend to do next and why.
    uncertainty: JSON array of strings — what gives you uncertainty despite the numbers.
    data_trust: JSON array of objects [{source, trust_level (high/medium/low/unknown), reason}] — do you trust ABBA's data?
    workflow_gaps: JSON array of strings — shortcomings you see in the analytical workflow.
    want_to_verify: JSON array of strings — what you'd cross-reference if you could.
    raw_thought: Free-form reasoning for anything else.
    """
    # Parse JSON string arrays from MCP (Claude sends strings, not lists)
    parsed_uncertainty = json.loads(uncertainty) if uncertainty else None
    parsed_data_trust = json.loads(data_trust) if data_trust else None
    parsed_workflow_gaps = json.loads(workflow_gaps) if workflow_gaps else None
    parsed_want_to_verify = json.loads(want_to_verify) if want_to_verify else None

    result = _toolkit.think(
        phase=phase,
        plan=plan,
        uncertainty=parsed_uncertainty,
        data_trust=parsed_data_trust,
        workflow_gaps=parsed_workflow_gaps,
        want_to_verify=parsed_want_to_verify,
        raw_thought=raw_thought,
    )
    return json.dumps(result, default=str)


@mcp.tool()
def session_replay(limit: int = 200) -> str:
    """Get the full reasoning + tool call timeline for this session.

    Returns chronological entries showing every think() reflection and
    every tool call, so users can see the agent's complete decision chain
    including uncertainties, trust assessments, and workflow gap observations.
    """
    result = _toolkit.session_replay(limit=limit)
    return json.dumps(result, default=str)


def run_stdio_server() -> None:
    """Run the MCP server over stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_stdio_server()
