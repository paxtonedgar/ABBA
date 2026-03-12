# ABBA

ABBA is a DuckDB-backed sports analytics toolkit for agent workflows. It exposes query, prediction, market, and workflow tools through three interfaces:

- direct Python import
- a small HTTP server
- an MCP server

The active runtime path lives under `src/abba/server`, `src/abba/engine`, `src/abba/storage`, and `src/abba/workflows`.

## What It Does

- Queries games, odds, team stats, rosters, and NHL-specific datasets
- Produces general game predictions and NHL-specific predictions
- Calculates EV and Kelly sizing
- Runs multi-step workflows such as `game_prediction`, `tonights_slate`, and `betting_strategy`
- Stores local data in DuckDB for repeatable development and testing

Current toolkit surface: `23` callable tools.

## Quick Start

### Install for local development

```bash
git clone https://github.com/paxtonedgar/ABBA.git
cd abba-nhl
pip install -e ".[dev]"
```

### Use from Python

```python
from abba import ABBAToolkit

toolkit = ABBAToolkit(db_path=":memory:", auto_seed=True)

tools = toolkit.list_tools()
games = toolkit.query_games(sport="NHL", status="scheduled")

if games["count"]:
    game_id = games["games"][0]["game_id"]
    prediction = toolkit.nhl_predict_game(game_id)
    odds = toolkit.compare_odds(game_id)
```

### Run the HTTP server

```bash
abba-server
```

Then call:

```bash
curl http://localhost:8420/tools
curl -X POST http://localhost:8420/tools/call \
  -H "Content-Type: application/json" \
  -d '{"name":"list_sources","arguments":{}}'
```

### Run the MCP server

```bash
abba-mcp
```

Example MCP config:

```json
{
  "mcpServers": {
    "abba": {
      "command": "abba-mcp",
      "env": {
        "ABBA_DB_PATH": "abba.duckdb"
      }
    }
  }
}
```

## Tool Categories

### Data

- `query_games`
- `query_odds`
- `query_team_stats`
- `query_goaltender_stats`
- `query_advanced_stats`
- `query_cap_data`
- `query_roster`
- `list_sources`
- `describe_dataset`
- `refresh_data`

### Prediction and analysis

- `predict_game`
- `nhl_predict_game`
- `explain_prediction`
- `graph_analysis`
- `season_review`
- `playoff_odds`

### Market and session

- `find_value`
- `compare_odds`
- `calculate_ev`
- `kelly_sizing`
- `session_budget`

### Workflows

- `run_workflow`
- `list_workflows`

## Current Scope

The repo is strongest today as:

- a local agent toolkit
- an NHL-oriented demo and experimentation surface
- a testbed for prediction, workflow, and trust-contract checks

The repository also contains experimental and legacy modules outside the main toolkit path. If you are new to the codebase, start with:

- `src/abba/server/toolkit.py`
- `src/abba/server/tools/`
- `src/abba/engine/`
- `src/abba/storage/duckdb.py`
- `src/abba/workflows/engine.py`

## Live Data Notes

The active live-refresh path currently covers:

- NHL standings and schedule
- NHL rosters and goalie stats
- optional odds ingestion when `ODDS_API_KEY` is set

Development and tests also rely on deterministic seed data.

## Development

### Run tests

From the repo root:

```bash
pytest
```

For a focused local pass:

```bash
pytest tests/test_storage.py tests/test_engine.py tests/test_toolkit.py
```

### Repo layout

```text
src/abba/server/      Public toolkit interfaces
src/abba/engine/      Prediction, EV, Kelly, graph, and NHL logic
src/abba/storage/     DuckDB-backed persistence
src/abba/workflows/   Multi-step workflow orchestration
src/abba/connectors/  Seed and live data ingestion
tests/                Unit, integration, and contract-style checks
docs/                 Project notes, plans, and audit material
```

## Documentation

Start with:

- `docs/README.md`
- `docs/architecture-dossier.md`

Not every document in `docs/` is canonical. Some are historical planning notes or audit artifacts.

## License

MIT
