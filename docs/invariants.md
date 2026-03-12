# ABBA Non-Negotiable Invariants

> These are hard constraints. If any invariant is violated, the system's output cannot be trusted.
> Every invariant has a specific fail condition, the current violation evidence, and a machine-checkable test.
> Nothing ships to users until all five pass.

---

## Invariant A — Predictor Family Consistency

**Rule**: If a workflow is labeled NHL and is used for staking/value decisions, it must use the NHL predictor family or explicitly and provably declare why it does not.

### Fail Condition

`find_value(sport="NHL")` routes through the generic predictor without an explicit design contract.

### Current Violation: YES

**Evidence chain**:

1. `src/abba/server/tools/market.py:29` — `find_value()` calls `self.predict_game(gid)` for EVERY game, regardless of sport.
2. `src/abba/server/tools/analytics.py:12-58` — `predict_game()` is the generic predictor. It uses `FeatureEngine.build_features()` + `FeatureEngine.predict_from_features()` — the MLB/generic model path.
3. `predict_game()` never checks `if sport == "NHL": use nhl_predict_game()`. It always routes through `self.features` (the generic `FeatureEngine`), not `self.hockey` (the `HockeyAnalytics` engine).
4. Result: **An NHL game going through `find_value()` gets its probability from the generic pseudo-model, NOT from the 6-model NHL ensemble with goalie matchups, Elo, etc.** The generic model doesn't know about goaltenders, Corsi, special teams, or any NHL-specific signal.
5. Meanwhile, `nhl_predict_game()` — the NHL-specific predictor with all the hockey models — is only called directly by the `nhl_predict_game` MCP tool and by workflows that explicitly invoke it.

**Severity**: CRITICAL. Every `find_value(sport="NHL")` call produces EV calculations from the wrong model. Users scanning for value bets on NHL games are getting probabilities that ignore all hockey-specific signal.

### Required Test

```python
def test_find_value_nhl_uses_nhl_predictor():
    """find_value with NHL games must use nhl_predict_game, not generic predict_game."""
    toolkit = ABBAToolkit()
    # Seed an NHL game + odds
    # Call find_value(sport="NHL")
    # Assert the prediction used matches nhl_predict_game output
    # NOT the generic predict_game output
```

### Required Fix

In `market.py:find_value()`, check `game.get("sport")`. If NHL, call `self.nhl_predict_game(gid)` instead of `self.predict_game(gid)`. Extract prediction value from the NHL-specific response format.

---

## Invariant B — Schema Contract Integrity

**Rule**: Every field produced by live ingestion that is consumed by the NHL prediction path must exist under the exact keys and semantics the model expects.

### Fail Condition

Ingestion writes one key name, consumer reads a different key name. Consumer expects a field that ingestion never produces.

### Current Violations: 4 CONFIRMED

**Violation B.1 — `save_percentage` vs `save_pct`**

| Layer | Key Written | File:Line |
|-------|-------------|-----------|
| Live connector | `"save_percentage"` | `src/abba/connectors/live.py:274` |
| Seed data | `"save_pct"` | `src/abba/connectors/seed.py:261` |
| Model consumer | `goalie.get("save_pct", 0.907)` | `src/abba/engine/hockey.py:830` |

Live-ingested goalies have `save_percentage`. The model reads `save_pct`. **Key mismatch → model gets default 0.907 for every live goalie → goaltender matchup edge is always near zero.**

**Violation B.2 — `goals_against_average` vs `gaa`**

| Layer | Key Written | File:Line |
|-------|-------------|-----------|
| Live connector | `"goals_against_average"` | `src/abba/connectors/live.py:275` |
| Seed data | `"gaa"` | `src/abba/connectors/seed.py:262` |
| Model consumer | `goalie.get("gaa", 0)` | `src/abba/engine/hockey.py:613` |

Same pattern. Live GAA stored under wrong key → model gets 0.

**Violation B.3 — `gsaa` never produced by live connector**

| Layer | Key Written | File:Line |
|-------|-------------|-----------|
| Live connector | *(not produced)* | `src/abba/connectors/live.py:263-281` |
| Seed data | `"gsaa"` | `src/abba/connectors/seed.py:267` |
| Model consumer | `goalie.get("gsaa", 0)` | `src/abba/engine/hockey.py:832` |

The live connector fetches `goals_against` and `saves` but never computes GSAA. The model reads `gsaa` and gets default 0 for every live goalie. **Goaltender matchup `gsaa_edge` is always zero on live data.**

**Violation B.4 — `role` never produced by live connector**

| Layer | Key Written | File:Line |
|-------|-------------|-----------|
| Live connector | *(not produced)* | `src/abba/connectors/live.py:263-281` |
| Seed data | `"role": "starter"` | `src/abba/connectors/seed.py:259` |
| Model consumer | `g.get("stats", {}).get("role") == "starter"` | `src/abba/server/tools/nhl.py:49` |

Starter lookup filter finds nothing → falls back to `home_goalies[0]["stats"]` → first row is arbitrary (DB insertion order) → could be the backup.

### Combined Impact

On live data, the goaltender matchup model is **completely inert**. It receives default values for every input:
- `save_pct` = 0.907 (league average default) for both goalies
- `gsaa` = 0 for both goalies
- `role` = missing → wrong goalie may be selected

Result: `goaltender_matchup_edge ≈ 0.0` always. The model effectively has 5 models, not 6. Users see "goaltender matchup" in the output but the number is meaningless.

### Required Test

```python
def test_schema_contract_goalie_keys():
    """Every key the model reads must match what live ingestion writes."""
    # The canonical schema contract
    MODEL_READS = {"save_pct", "gaa", "gsaa", "role", "name", "games_played", "games_started"}

    # Simulate live ingestion
    connector = NHLLiveConnector()
    # ... fetch or mock goalie data
    stored_keys = set(goalie_record["stats"].keys())

    missing = MODEL_READS - stored_keys
    assert not missing, f"Model expects keys not produced by ingestion: {missing}"
```

### Required Fix

In `live.py:_fetch_goaltender_stats()`:
1. Rename `"save_percentage"` → `"save_pct"`
2. Rename `"goals_against_average"` → `"gaa"`
3. Compute and store `"gsaa"`: `round(shots_against * 0.907 - goals_against, 2)`
4. Compute and store `"role"`: sort by `games_started` desc, first = `"starter"`, rest = `"backup"`

---

## Invariant C — Cross-Provider Identity Resolution

**Rule**: Odds rows must be joinable to schedule rows through a proven mapping rule. Exact ID equality is not assumed unless explicitly guaranteed by source contract.

### Fail Condition

Schedule uses provider A game IDs, odds uses provider B event IDs, no mapping table or resolver exists, system silently treats unmatched odds as absent market data.

### Current Violation: YES

**Evidence chain**:

1. **NHL schedule game IDs**: `src/abba/connectors/live.py:173` — `game_id = f"nhl-{game.get('id', '')}"` where `id` is the NHL API's internal game ID (e.g., `nhl-2025020456`).

2. **Odds game IDs**: `src/abba/connectors/live.py:329` — `game_id = f"nhl-{event.get('id', '')}"` where `id` is The Odds API's event ID (e.g., `nhl-a1b2c3d4e5f6`).

3. **These are different ID systems.** The NHL API uses sequential numeric IDs. The Odds API uses its own alphanumeric event IDs. The `nhl-` prefix creates a false appearance of compatibility.

4. **Join point**: `src/abba/server/tools/nhl.py:62` — `game_odds = self.storage.query_odds(game_id=game_id)` uses the NHL-sourced `game_id` to look up odds. Since odds were stored with Odds-API-sourced IDs, **the join returns empty**.

5. **Silent failure**: `src/abba/engine/hockey.py:build_nhl_features()` receives `odds_data=[]`, computes `market_implied_prob = 0`, and the market blend model is silently excluded from the ensemble. No error, no warning.

6. **No mapping table exists.** No resolver exists. No code attempts to match by team names + date.

### Required Test

```python
def test_odds_join_to_schedule():
    """Odds records must be joinable to schedule records by game_id."""
    toolkit = ABBAToolkit()
    toolkit.refresh_data(source="all")  # Needs ODDS_API_KEY

    scheduled = toolkit.storage.query_games(sport="NHL", status="scheduled")
    for game in scheduled:
        odds = toolkit.storage.query_odds(game_id=game["game_id"])
        # At minimum, if odds exist for this matchup, they must be findable
        # This test documents the current failure rate
```

### Required Fix

Option 1 (simple): In `OddsLiveConnector.refresh()`, match odds events to existing schedule rows by `(home_team, away_team, date)` instead of by event ID. Use the NHL schedule's `game_id` for the stored odds record.

Option 2 (robust): Create a `game_id_mapping` table that resolves `(provider, provider_id) → canonical_game_id`. Populate during odds ingestion by matching on team names + date.

---

## Invariant D — Snapshot Coherence

**Rule**: Every prediction must carry coherent `as_of`, `season`, `source`, and `default/missing` metadata for every dataset used.

### Fail Condition

Prediction mixes datasets from different seasons or freshness windows without surfacing it.

### Current Violation: YES

**Evidence chain**:

1. **No per-dataset timestamps exist.** The only freshness signal is `_last_refresh_ts` — a single in-memory float for the entire system. `src/abba/server/toolkit.py:86` — `self._last_refresh_ts: float | None = None`.

2. **No per-dataset season tracking.** `nhl_predict_game()` queries team stats, advanced stats, goalies, odds, and rosters independently. Each query may return data from different seasons:
   - `src/abba/server/tools/nhl.py:35` — `query_team_stats(team_id=home, sport="NHL")` — no season filter.
   - `src/abba/server/tools/nhl.py:40` — `query_nhl_advanced_stats(team_id=home)` — no season filter.
   - `src/abba/server/tools/nhl.py:46` — `query_goaltender_stats(team=home)` — no season filter.

3. **Concrete scenario**: After a live refresh at the start of a new season, `team_stats` has current-season data (0 wins, 0 losses). But `nhl_advanced_stats` still has last season's seed data (full season Corsi/xG). The prediction mixes "just started playing" standings with "full prior season" analytics.

4. **No provenance in output.** The prediction result (`nhl.py:113-135`) contains no `as_of`, no per-source timestamps, no `"data_freshness"` field. The only metadata is `confidence_meta` which is itself fabricated (TG-011).

5. **Default/missing not surfaced.** When advanced stats are absent (`home_adv = None`), `build_nhl_features()` silently uses `0.0` for Corsi/xG features. The output includes `"home_corsi_pct": 0.0` with no flag indicating this is a missing-data default, not a measured value.

### Required Test

```python
def test_prediction_carries_provenance():
    """Every prediction must declare data sources with as_of and season per dataset."""
    toolkit = ABBAToolkit()
    result = toolkit.nhl_predict_game("nhl-2025020001")

    provenance = result.get("data_provenance")
    assert provenance is not None, "Prediction must carry data_provenance"

    required_sources = ["team_stats", "goaltender_stats"]
    for source in required_sources:
        assert source in provenance, f"Missing provenance for {source}"
        assert "as_of" in provenance[source], f"Missing as_of for {source}"
        assert "season" in provenance[source], f"Missing season for {source}"
        assert "status" in provenance[source], f"Missing status for {source}"
        # status: "present" | "absent" | "stale" | "default"

def test_prediction_rejects_cross_season_data():
    """Prediction must not silently mix seasons."""
    # Insert team_stats for 2025-26, advanced_stats for 2024-25
    # Call nhl_predict_game
    # Assert either: error, or provenance flags the season mismatch
```

### Required Fix

1. Add `captured_at TIMESTAMP` column to `team_stats`, `goaltender_stats`, `nhl_advanced_stats`, `roster` tables. Populated at ingestion time.
2. Add `season` parameter to all prediction-path queries in `nhl.py` (currently missing on lines 35, 40, 46).
3. Build provenance dict in `nhl_predict_game()` recording status of each data source: `{"team_stats": {"status": "present", "season": "2025-26", "as_of": "2026-03-10T14:22:00"}, "advanced_stats": {"status": "absent", "reason": "no_live_connector"}, ...}`.
4. Include provenance in prediction output.

---

## Invariant E — Starter Determinism

**Rule**: If goalie matchup is part of the model, starter identity must be explicit and deterministic.

### Fail Condition

First-row wins. Missing role falls through. Starter chosen implicitly from sort order or incidental DB ordering.

### Current Violation: YES

**Evidence chain**:

1. **Starter selection code** (`src/abba/server/tools/nhl.py:48-54`):
   ```python
   home_goalie = next(
       (g["stats"] for g in home_goalies if g.get("stats", {}).get("role") == "starter"),
       home_goalies[0]["stats"] if home_goalies else None,
   )
   ```

2. **The `next()` generator** tries to find a goalie with `role == "starter"`. If none found, falls back to `home_goalies[0]["stats"]` — **first row wins**.

3. **Live data never sets `role`** (Invariant B, Violation B.4). So the `next()` generator always exhausts without a match. The fallback `home_goalies[0]` executes every time on live data.

4. **What determines `home_goalies[0]`?** `storage.query_goaltender_stats()` at `duckdb.py:610-633` — the SQL has no `ORDER BY`. DuckDB returns rows in insertion order, which is iteration order of the API response. The NHL API's `club-stats/{team}/now` endpoint returns goalies in an undocumented order that may change between requests.

5. **Result**: Starter identity is determined by:
   - Which goalie the NHL API lists first in its response (undocumented, not guaranteed)
   - DuckDB insertion order (implementation detail, not a contract)
   - No explicit determinism at any layer

6. **Workflow path is worse.** `src/abba/workflows/engine.py:138-145` does the same pattern but also lacks season filtering, compounding the problem.

### Required Test

```python
def test_starter_selection_is_deterministic():
    """Starter must be chosen by explicit criteria, not row order."""
    storage = Storage(":memory:")
    # Insert two goalies for same team:
    #   Goalie A: 50 games started (but inserted second)
    #   Goalie B: 10 games started (but inserted first)
    # Query goalies
    # Assert starter is Goalie A (most games started), NOT Goalie B (first row)

def test_starter_selection_with_no_role_field():
    """When role field is absent, starter must be inferred deterministically."""
    # Insert goalies without role field
    # Call nhl_predict_game
    # Assert the goalie used is the one with most games_started
    # Assert this is stable across multiple calls
```

### Required Fix

1. In `live.py:_fetch_goaltender_stats()`: Sort goalies by `games_started` descending. Tag first as `"starter"`, rest as `"backup"`. Store `role` in stats dict.

2. In `nhl.py:48-54`: Replace first-row fallback with explicit selection:
   ```python
   # Deterministic starter selection: most games started
   home_goalie = max(
       (g["stats"] for g in home_goalies if g.get("stats")),
       key=lambda s: s.get("games_started", 0),
       default=None,
   )
   ```

3. Add `ORDER BY` to `query_goaltender_stats()`: `ORDER BY (stats->>'games_started')::INT DESC`.

---

## Enforcement

These invariants are not suggestions. They are preconditions for trusting output.

### Invariant Test Suite

All five invariants must be tested in a dedicated test file: `tests/test_invariants.py`.

This file runs in CI. If any invariant test fails, the build is red. No exceptions, no skips.

### Violation ↔ Trust Gap Cross-Reference

| Invariant | Trust Gap IDs |
|-----------|--------------|
| A — Predictor family | TG-007, TG-009, TG-035 |
| B — Schema contract | TG-002, TG-004, TG-025, TG-029 |
| C — Identity resolution | TG-014, TG-029 |
| D — Snapshot coherence | TG-003, TG-006, TG-008, TG-014 |
| E — Starter determinism | TG-002, TG-006 |

### Priority

Fix order must be: **B → E → A → D → C**

- **B first**: Without correct field names, every downstream test is meaningless.
- **E second**: Depends on B (role field). Small change, high impact.
- **A third**: Routing fix in one file. Unlocks correct value scanning.
- **D fourth**: Requires storage schema changes. Larger scope.
- **C last**: Requires either team-name matching or mapping table. Most complex.
