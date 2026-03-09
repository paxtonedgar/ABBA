# ABBA

Sports analytics toolkit that AI agents call natively. DuckDB-backed data store, ensemble ML predictions, real-time odds scanning, and Kelly Criterion position sizing -- exposed as callable tools that any agent framework can discover and use.

Solves the stale data problem: LLMs hallucinate sports records, get rosters wrong, and have no idea about recent outcomes. ABBA provides a live, queryable data layer the agent can trust, with freshness metadata on every response.

## Three ways to use it

**1. Direct Python import** (most stable)
```python
from abba import ABBAToolkit

toolkit = ABBAToolkit()
tools = toolkit.list_tools()  # discover available tools

games = toolkit.query_games(sport="MLB", status="scheduled")
prediction = toolkit.predict_game(game_id="mlb-2026-04-12-NYY-BOS")
value = toolkit.find_value(sport="MLB", min_ev=0.03)
sizing = toolkit.kelly_sizing(win_probability=0.62, decimal_odds=2.10)
```

**2. HTTP REST API** (language-agnostic)
```bash
pip install abba
abba-server  # starts on :8420

# Discovery
curl localhost:8420/tools

# Call a tool
curl -X POST localhost:8420/tools/call \
  -d '{"name": "predict_game", "arguments": {"game_id": "mlb-2026-04-12-NYY-BOS"}}'
```

**3. MCP server** (for Claude Desktop / MCP-compatible agents)
```json
{
  "mcpServers": {
    "abba": {
      "command": "python",
      "args": ["-m", "abba.server.mcp"],
      "env": {"ABBA_DB_PATH": "abba.duckdb"}
    }
  }
}
```

MCP is the standard but it's fragile in practice -- stdio transport loses state on crash, reconnection isn't standardized, and most agent frameworks have incomplete support. The Python import and HTTP interfaces are production-stable fallbacks that provide the exact same tools.

## Tools

| Tool | Category | Description |
|------|----------|-------------|
| `query_games` | data | Games with filters (sport, date, team, status) |
| `query_odds` | data | Current odds across sportsbooks |
| `query_team_stats` | data | Team performance stats |
| `list_sources` | data | Available data tables and row counts |
| `describe_dataset` | data | Column schema for a table |
| `predict_game` | analytics | Ensemble prediction (home win probability) |
| `explain_prediction` | analytics | Feature importance breakdown |
| `graph_analysis` | analytics | Team network metrics (centrality, cohesion) |
| `find_value` | market | Scan for +EV opportunities |
| `compare_odds` | market | Cross-sportsbook line comparison |
| `calculate_ev` | market | Expected value for a specific bet |
| `kelly_sizing` | market | Optimal position size (half-Kelly, capped) |
| `session_budget` | meta | Remaining compute budget |

## What the agent actually does

```
Agent: "Find me the best MLB bets for tonight"

1. calls list_sources()                    -> sees games, odds, team_stats tables
2. calls query_games(sport="MLB")          -> 5 scheduled games
3. calls find_value(sport="MLB")           -> 2 opportunities with edge > 3%
4. calls explain_prediction(game_id=...)   -> pitcher matchup driving the edge
5. calls kelly_sizing(prob=0.62, odds=2.1) -> recommends $420 stake (half-Kelly)
```

Six tool calls. Each returns structured JSON the agent reasons over. No raw data dumps, no context window waste.

## Architecture

```
Agent (Claude / GPT / LangGraph / custom)
    |
    |-- Python SDK ---- ABBAToolkit ----+
    |-- HTTP REST ----- /tools/call ----+
    +-- MCP stdio ----- JSON-RPC -------+
                                        |
                              +---------v---------+
                              |   Tool Dispatch    |
                              |   (13 tools)       |
                              +--------+-----------+
                                       |
                    +------------------+------------------+
                    v                  v                  v
              +----------+     +------------+     +----------+
              |  Engine   |     |  Storage   |     | Connectors|
              |          |     |  (DuckDB)  |     | (live data)|
              | Ensemble  |     |  games     |     | MLB API   |
              | Features  |     |  odds      |     | NHL API   |
              | Kelly     |     |  stats     |     | Odds API  |
              | Value     |     |  cache     |     | Weather   |
              | Graph     |     |  sessions  |     |           |
              +----------+     +------------+     +----------+
```

## Engine math

**Ensemble predictions**: 4 models combined via inverse-variance weighting. Model 1 uses log5 (the baseball standard for head-to-head probability). Model 2 uses Pythagorean expectation. Model 3 uses recent form weighting. Model 4 adds weather adjustment. Confidence intervals via t-distribution.

**Kelly Criterion**: Half-Kelly by default with 5% bankroll cap. Full Kelly formula: `f* = (bp - q) / b`. Won't bet below 2% edge or 3% EV threshold.

**Graph analysis**: scipy shortest_path for closeness centrality, Brandes' algorithm for betweenness, scipy.linalg.eigh for eigenvector centrality. Matrix method (`A^3` diagonal) for clustering coefficient.

**Expected value**: `EV = P(win) * (odds - 1) - P(loss)`. Scans all sportsbooks against model probability to find positive EV with minimum edge threshold.

## Data layer

DuckDB embedded columnar database. Fast analytical queries over millions of rows without a server process. Parquet-native for cheap historical archival.

Auto-seeds with realistic sample data on first boot (deterministic random seed, reproducible). In production, live connectors refresh from:
- MLB Stats API (free, no auth, real-time games + standings)
- NHL Stats API (free, no auth)
- The Odds API (API key, 500 req/mo free tier)
- OpenWeather (API key, 1000 calls/day free)

Every query response includes freshness metadata so agents know how stale the data is.

## Local development

```bash
git clone https://github.com/paxtonedgar/ABBA.git
cd ABBA/Projects/ABBA
pip install -e ".[dev]"
pytest  # 77 tests
```

```bash
# Start HTTP server
python -m abba.server.http

# Or use directly
python -c "
from abba import ABBAToolkit
tk = ABBAToolkit()
print(tk.find_value(sport='MLB'))
"
```

## Project structure

```
Projects/ABBA/
  src/abba/
    server/           # tool interfaces
      toolkit.py      # ABBAToolkit (main entry point, 13 tools)
      mcp.py          # MCP stdio server
      http.py         # HTTP REST server
    engine/           # analytics compute
      ensemble.py     # inverse-variance weighted model combining
      features.py     # feature engineering (log5, pythagorean, weather)
      kelly.py        # Kelly Criterion position sizing
      value.py        # expected value scanning
      graph.py        # team network analysis (scipy)
    storage/
      duckdb.py       # embedded columnar store
    connectors/
      seed.py         # deterministic sample data
      live.py         # live data source adapters
  tests/
    test_engine.py    # math correctness (33 tests)
    test_storage.py   # DuckDB operations (11 tests)
    test_toolkit.py   # integration + agent workflow (23 tests)
    test_mcp.py       # protocol + SDK (10 tests)
  pyproject.toml
```

## License

MIT
