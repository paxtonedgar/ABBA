# Architecture Dossier

Date: 2026-03-10

Scope: actual runtime architecture for the toolkit path under `src/abba/server`, `src/abba/workflows`, `src/abba/connectors`, `src/abba/storage`, and `src/abba/engine`, plus the parallel `src/abba/analytics` stack that remains in the repo and in tests.

## Severity Rubric

- `Critical`: architecture debt that can silently produce wrong mathematical output or false confidence.
- `High`: architecture debt that makes correctness hard to validate, swap, or reason about.
- `Medium`: architecture debt that duplicates logic, obscures boundaries, or increases regression risk.

## Executive Finding

This repo does not have one architecture. It has two:

1. The active toolkit stack:
   `HTTP/MCP/direct import -> ABBAToolkit -> tool mixins -> Storage/Connectors/Engines -> dict outputs`

2. A parallel analytics stack:
   `src/abba/analytics/*`, `src/abba/agents/*`, `src/abba/core/dependency_injection.py`, and `tests/integration/test_analytics_zero_mock.py`

These stacks do not share a stable domain model, persistence contract, or algorithm boundary. A future engineer cannot safely tell which abstractions are canonical, which are vestigial, and which outputs are mathematically authoritative.

## Concise Architecture Map

### End-to-End Data Flow

#### Direct tool path

Call chain:

`ABBAHandler.do_POST` -> `ABBAToolkit.call_tool()` -> one of `AnalyticsToolsMixin`, `NHLToolsMixin`, `MarketToolsMixin`, `SessionToolsMixin` -> `Storage` queries + `FeatureEngine` / `HockeyAnalytics` / `EnsembleEngine` / `ValueEngine` / `KellyEngine` -> `Storage.cache_prediction()` + `_track()` -> JSON response

Primary files:

- `src/abba/server/http.py`
- `src/abba/server/toolkit.py`
- `src/abba/server/tools/*.py`
- `src/abba/storage/duckdb.py`
- `src/abba/engine/*.py`

#### Live ingestion path

Call chain:

`SessionToolsMixin.refresh_data()` -> `NHLLiveConnector.refresh()` / `OddsLiveConnector.refresh()` -> connector-specific `_fetch_*` methods -> `Storage.upsert_*()` / `Storage.insert_odds()` -> later prediction tools read from `Storage`

Primary files:

- `src/abba/server/tools/session.py`
- `src/abba/connectors/live.py`
- `src/abba/storage/duckdb.py`

#### Workflow path

Call chain:

`SessionToolsMixin.run_workflow()` -> `WorkflowEngine.run()` -> workflow methods such as `game_prediction()`, `tonights_slate()`, `season_story()` -> repeated toolkit calls -> controller-level narrative assembly and ad hoc heuristics -> workflow response

Primary files:

- `src/abba/server/tools/session.py`
- `src/abba/workflows/engine.py`

#### Parallel analytics path

Call chain in tests:

`tests/integration/test_analytics_zero_mock.py` -> `src.abba.analytics.advanced_analytics.AdvancedAnalyticsManager`

This is not the same runtime architecture as the toolkit stack.

Primary files:

- `src/abba/analytics/advanced_analytics.py`
- `src/abba/analytics/manager.py`
- `src/abba/analytics/interfaces.py`
- `src/abba/core/dependency_injection.py`
- `tests/integration/test_analytics_zero_mock.py`

## Component Map

| Component | Responsibility | Depends on | Mixes concerns | Leaks implementation details |
|---|---|---|---|---|
| `ABBAHandler` in `src/abba/server/http.py` | Transport and route dispatch | `ABBAToolkit` | Mildly | Exposes tool names directly as HTTP surface |
| `ABBAToolkit` in `src/abba/server/toolkit.py` | Dependency container, session state, seeding, tracking, orchestration root | `Storage`, all engines, seed connector | Yes | Yes, all downstream details are reachable from one mutable object |
| `DataToolsMixin` | Thin read access to storage | `Storage` | No | Returns storage-shaped dicts |
| `AnalyticsToolsMixin.predict_game()` | Read game/team/weather, build features, run pseudo-models, cache, format response | `Storage`, `FeatureEngine`, `EnsembleEngine` | Yes | Yes, response shape is built from raw feature dicts and storage rows |
| `NHLToolsMixin.nhl_predict_game()` | Read team/goalie/advanced/odds data, compute injury heuristics, run NHL model, cache, compute confidence, format output | `Storage`, `HockeyAnalytics`, `EloRatings`, `EnsembleEngine`, confidence module | Yes, heavily | Yes, tool knows too much about storage schema and model internals |
| `MarketToolsMixin` | Value scan, odds compare, EV, Kelly sizing | `Storage`, `ValueEngine`, `KellyEngine`, `predict_game()` | Yes | Yes, duplicates EV math outside `ValueEngine` |
| `SessionToolsMixin.refresh_data()` | Ingestion entrypoint | concrete connectors imported inside the method | Yes | Yes, concrete connector classes are hard-coded at tool boundary |
| `WorkflowEngine` | Multi-step orchestration and report building | entire toolkit surface | Yes, severely | Yes, workflows know prediction structure, odds structure, and narrative thresholds |
| `Storage` | Persistence, cache, session budgets, reasoning logs, direct query API | DuckDB, pandas-style `fetchdf()` conversion | Yes, severely | Yes, JSON blobs and dict rows are leaked everywhere |
| `FeatureEngine` | MLB/general feature engineering and heuristic pseudo-model generation | raw dict stats and weather | Yes | Yes, algorithm and features are fused |
| `HockeyAnalytics` | NHL formulas, feature construction, pseudo-model generation, grading, playoff simulation, cap analysis | raw dict stats | Yes | Yes, one class holds unrelated math domains |
| `EnsembleEngine` | Combine model outputs and produce pseudo-confidence | numpy, raw float lists | Yes | Claims statistical meaning it does not actually encode |
| `src/abba/analytics/*` parallel stack | separate analytics/ML/DI architecture | unrelated interfaces, direct imports, external DB manager | Yes | Entire subsystem leaks a different worldview than the toolkit stack |

## Boundary Violations

### Critical

1. `NHLToolsMixin.nhl_predict_game()` is simultaneously a controller, repository client, feature assembler, model runner, cache manager, and confidence formatter.
   Call chain:
   `nhl_predict_game()` -> `Storage.query_team_stats()` / `query_nhl_advanced_stats()` / `query_goaltender_stats()` / `query_odds()` -> `HockeyAnalytics.build_nhl_features()` -> `EloRatings.predict()` -> `HockeyAnalytics.predict_nhl_game()` -> `EnsembleEngine.combine()` -> `build_prediction_meta()` -> `Storage.cache_prediction()`
   Files:
   `src/abba/server/tools/nhl.py:12-138`
   `src/abba/engine/hockey.py:743-989`

2. `WorkflowEngine` contains domain decision logic that should live in deterministic mathematical services or reporting adapters, not orchestration.
   Examples:
   - confidence manufactured as `abs(pred_val - 0.5) * 200`
   - risk tolerance remaps Kelly results with ad hoc multipliers
   - narrative thresholds such as `pred_value > 0.55`
   Files:
   `src/abba/workflows/engine.py:235-313`
   `src/abba/workflows/engine.py:743-820`

3. `Storage` is not just persistence. It is also cache, session accounting, observability log, reasoning log, and ad hoc repository layer returning mutable dicts with JSON-decoded subdocuments.
   Files:
   `src/abba/storage/duckdb.py:19-213`
   `src/abba/storage/duckdb.py:437-596`

### High

1. `FeatureEngine` and `HockeyAnalytics` both mix feature construction with algorithm execution.
   Files:
   `src/abba/engine/features.py:31-154`
   `src/abba/engine/hockey.py:743-989`

2. `SessionToolsMixin.refresh_data()` hard-codes concrete connector imports at the tool boundary and swallows failures into a dict response. There is no acquisition contract, freshness ledger per dataset, or partial-refresh transaction boundary.
   File:
   `src/abba/server/tools/session.py:12-53`

3. Workflow methods consume toolkit outputs directly and rely on storage-shaped dicts rather than typed snapshots. That locks orchestration to current response formatting.
   File:
   `src/abba/workflows/engine.py:100-820`

### Medium

1. EV math exists in two places:
   - `MarketToolsMixin.calculate_ev()`
   - `ValueEngine._calculate_ev()`
   Files:
   `src/abba/server/tools/market.py:53-74`
   `src/abba/engine/value.py:138-145`

2. Tool registry metadata is manual and detached from implementation.
   File:
   `src/abba/server/tools/registry.py`

3. The repo keeps a separate protocol-heavy analytics architecture that does not govern the active toolkit path.
   Files:
   `src/abba/analytics/interfaces.py`
   `src/abba/core/dependency_injection.py`

## Architecture Debt That Will Corrupt Correctness

### Critical

#### 1. Live refresh does not populate the datasets the NHL model claims to use

`SessionToolsMixin.refresh_data()` calls `NHLLiveConnector.refresh()` and `OddsLiveConnector.refresh()` when asked, but `NHLLiveConnector.refresh()` only fetches standings, schedule, and optionally roster.

Evidence:

- `src/abba/server/tools/session.py:12-53`
- `src/abba/connectors/live.py:75-95`
- `src/abba/connectors/live.py:97-224`

The prediction path then reads:

- team stats
- advanced stats
- goaltender stats
- odds
- roster injury proxies

Evidence:

- `src/abba/server/tools/nhl.py:34-70`

But no live connector writes `nhl_advanced_stats`, `goaltender_stats`, or `salary_cap`, and workflows frequently refresh only `source="nhl"` before consuming odds.

Result: "live" NHL predictions are assembled from a hybrid of fresh standings/schedule, stale seed rows, absent rows that become defaults, and sometimes stale or missing odds.

This is not a style problem. It means the mathematical result is based on an input state the architecture cannot describe honestly.

#### 2. The NHL model advertises Corsi, xG, player impact, and rest, but the active prediction function does not use them as claimed

`HockeyAnalytics.build_nhl_features()` constructs:

- `home_corsi_pct`, `away_corsi_pct`
- `home_xgf_pct`, `away_xgf_pct`
- `rest_edge`
- market-implied probability

Evidence:

- `src/abba/engine/hockey.py:815-872`

`NHLToolsMixin.nhl_predict_game()` also injects:

- `home_injury_impact`
- `away_injury_impact`
- `home_roster_completeness`
- `away_roster_completeness`

Evidence:

- `src/abba/server/tools/nhl.py:72-76`

But `HockeyAnalytics.predict_nhl_game()` only consumes:

- points percentage
- goal differential
- recent form
- goaltender edge
- special teams edge
- rest edge
- market probability
- Elo

It does not read the Corsi features, xG features, or injury features at all.

Evidence:

- `src/abba/engine/hockey.py:905-989`

This means the architecture claims model dimensions that are not active in the actual scoring path. That will mislead validation, backtesting, and user trust.

#### 3. Rest is computed in workflows and narrated to the user, but not fed into the active NHL prediction path

`WorkflowEngine.game_prediction()` computes `rest_info = self._compute_rest(home, away)` and includes it in the response and narrative.

Evidence:

- `src/abba/workflows/engine.py:141-145`
- `src/abba/workflows/engine.py:189-226`
- `src/abba/workflows/engine.py:858-886`

But it then calls `self.toolkit.nhl_predict_game(gid)` without passing `rest_info`.

Evidence:

- `src/abba/workflows/engine.py:147-148`

Inside the tool, `HockeyAnalytics.build_nhl_features()` is invoked without `rest_info`, so `rest_edge` defaults to `0.0`.

Evidence:

- `src/abba/server/tools/nhl.py:64-70`
- `src/abba/engine/hockey.py:847-851`

The system tells the user fatigue matters while the model ignores it in the default path.

#### 4. Workflows often compare or size against odds they did not refresh

`WorkflowEngine.game_prediction()`, `tonights_slate()`, `value_scan()`, and `betting_strategy()` call `refresh_data(source="nhl")`, not `source="all"` or `source="odds"`, then immediately use `compare_odds()`, `find_value()`, or Kelly sizing.

Evidence:

- `src/abba/workflows/engine.py:113-159`
- `src/abba/workflows/engine.py:237-313`
- `src/abba/workflows/engine.py:452-490` if expanded in source
- `src/abba/workflows/engine.py:752-820`

The architecture has no guarantee that market data and model data are from the same as-of time.

#### 5. Season selection is implicit and unstable in the hottest prediction paths

`NHLToolsMixin.nhl_predict_game()` and `AnalyticsToolsMixin.predict_game()` query team data without specifying season and then take the first row:

- `home_stats_list[0]`
- `away_stats_list[0]`

Evidence:

- `src/abba/server/tools/analytics.py:29-33`
- `src/abba/server/tools/nhl.py:35-43`

`Storage.query_team_stats()` has no `ORDER BY`, so multi-season rows are returned in unspecified order.

Evidence:

- `src/abba/storage/duckdb.py:385-411`

That means a future backfill or multi-season ingest can silently change which season the model scores against.

### High

#### 6. Confidence metadata is disconnected from actual provenance and model lineage

`build_prediction_meta()` uses hard-coded backtest baselines and derives freshness from a single `_last_refresh_ts` on the toolkit object.

Evidence:

- `src/abba/engine/confidence.py:61-77`
- `src/abba/engine/confidence.py:261-353`

`NHLToolsMixin.nhl_predict_game()` passes `data_source = home_stats.get("source", "unknown")`, which reflects only one row, not the full mixture of team stats, advanced stats, goalie stats, roster, and odds actually used.

Evidence:

- `src/abba/server/tools/nhl.py:95-111`

The system can therefore issue a reliability grade for a blended input state it cannot represent accurately.

#### 7. Workflow confidence means a different thing than prediction confidence

`WorkflowEngine.run()` extracts `confidence_from_pred = result.get("prediction", {}).get("confidence")`.

Evidence:

- `src/abba/workflows/engine.py:53-74`

For prediction responses, that field is the ensemble disagreement-derived scalar from `PredictionResult.to_dict()`, not the provenance-aware metadata stored separately in `result["confidence"]`.

Evidence:

- `src/abba/engine/ensemble.py:22-42`
- `src/abba/server/tools/nhl.py:113-135`

This is semantic corruption. One layer treats "confidence" as model agreement; another layer documents confidence as calibrated trust metadata. They are not the same quantity.

#### 8. Storage schema uses JSON blobs where correctness-heavy systems need explicit typed columns

`games.metadata`, `team_stats.stats`, `goaltender_stats.stats`, `nhl_advanced_stats.stats`, `roster.stats`, `predictions_cache.prediction`, and multiple logs are stored as JSON.

Evidence:

- `src/abba/storage/duckdb.py:32-213`

Consequences:

- no schema-level validation for required mathematical inputs
- no query-level guarantees about units or completeness
- no easy backfill validation
- algorithm changes require string-key threading across the system

This will make correctness regressions invisible until late.

## Abstraction Failures Ranked by Severity

### Critical

1. `ABBAToolkit` is not a toolkit abstraction. It is a mutable god-object.
   - Responsibilities: dependency container, session owner, seeding bootstrapper, observability wrapper, cache client, orchestrator root.
   - File: `src/abba/server/toolkit.py`

2. `HockeyAnalytics` is not a single mathematical subsystem. It is a grab bag of unrelated domains:
   - shot metrics
   - xG
   - goalie evaluation
   - special teams
   - schedule effects
   - cap analysis
   - season review
   - playoff Monte Carlo
   - feature building
   - prediction synthesis
   File: `src/abba/engine/hockey.py`

3. `WorkflowEngine` is not an orchestration layer. It is partially a reporting layer, partially a risk policy layer, partially a math post-processor.
   File: `src/abba/workflows/engine.py`

### High

4. `EnsembleEngine` is a misleading abstraction.
   It claims "inverse-variance weighting" but the default weighted path is "inverse distance from group mean."
   File:
   `src/abba/engine/ensemble.py:95-117`

5. `FeatureEngine` is a misleading abstraction.
   It does not stop at feature engineering; it also contains pseudo-model generation in `predict_from_features()`.
   File:
   `src/abba/engine/features.py:102-154`

6. `Storage` is too generic and too coupled to current implementation details.
   It is a DB wrapper, repository surface, cache store, session ledger, and observability sink.
   File:
   `src/abba/storage/duckdb.py`

7. The protocol-heavy `src/abba/analytics/interfaces.py` stack is unnecessary in the current repo state.
   It defines abstractions that do not govern the toolkit path and are not the contracts most users or tests exercise.
   Files:
   `src/abba/analytics/interfaces.py`
   `src/abba/core/dependency_injection.py`

### Medium

8. `ToolRegistryMixin.list_tools()` is a hard-coded metadata mirror of runtime behavior.
   Any evolution requires updating both implementation and registry by hand.
   File:
   `src/abba/server/tools/registry.py`

9. `MarketToolsMixin.calculate_ev()` is an unnecessary abstraction leak from `ValueEngine`.
   Same formula, two authorities.
   Files:
   `src/abba/server/tools/market.py:53-74`
   `src/abba/engine/value.py:138-145`

10. The parallel analytics stack is too coupled to missing external modules and a different persistence model.
   Example:
   `src/abba/analytics/advanced_analytics.py` imports `database` and `models` from an unrelated namespace.
   Files:
   `src/abba/analytics/advanced_analytics.py:14-21`
   `tests/integration/test_analytics_zero_mock.py:16-18`

## Where Algorithm Swaps or Correctness Validation Will Break Down

1. Swapping the NHL model requires edits across:
   - `src/abba/server/tools/nhl.py`
   - `src/abba/engine/hockey.py`
   - `src/abba/engine/confidence.py`
   - `src/abba/workflows/engine.py`
   - `src/abba/server/tools/registry.py`

2. Changing data inputs requires threading stringly typed dict keys through:
   connectors -> storage JSON blobs -> tool mixins -> feature builders -> workflow renderers.

3. Validating model correctness against a historical dataset is blocked by:
   - no canonical snapshot object
   - no explicit feature artifact
   - no model-versioned calibration artifact
   - no guaranteed as-of timestamp alignment across sources

4. Validating confidence/calibration is blocked by:
   - hard-coded baseline metrics in `src/abba/engine/confidence.py`
   - no artifact linkage from prediction output to calibration run
   - semantic drift between ensemble confidence and reliability metadata

5. Introducing multi-season or backfilled data will destabilize current predictions because repository methods often omit season and order.

## Recommended Target Architecture For Correctness-Heavy Mathematical Systems

### 1. Separate acquisition snapshots from model inputs

Create explicit snapshot services:

- `IngestionService`
- `SnapshotRepository`
- `Snapshot` / `GameContext` domain objects

Requirements:

- every dataset has `source`, `as_of`, `season`, and `completeness`
- no prediction reads raw storage rows directly
- workflows request one immutable snapshot per analysis

### 2. Split feature construction from model execution

Create:

- `MlbFeatureBuilder`
- `NhlFeatureBuilder`
- `FeatureSet` objects with explicit typed fields

Do not let feature builders score games.

### 3. Split individual algorithms from ensemble combination

Create per-algorithm model runners:

- `PointsPctModel`
- `PythagoreanModel`
- `GoalieModel`
- `SpecialTeamsModel`
- `RestModel`
- `MarketBlendModel`
- `EloModel`

Each returns:

- point estimate
- required inputs
- missing input flags
- version

Then let `EnsembleCombiner` combine model outputs only.

### 4. Make calibration a first-class artifact

Prediction output should carry:

- model version
- ensemble version
- calibration artifact version
- dataset timestamp bundle
- missing-input bundle

Remove hard-coded backtest stats from runtime code.

### 5. Reduce orchestration to coordination only

`WorkflowEngine` should:

- request a snapshot
- invoke domain services
- call a report assembler

It should not:

- invent confidence scores
- apply narrative thresholds
- alter Kelly sizing math
- decide risk policy inline

### 6. Split persistence responsibilities

Break `Storage` into:

- `CanonicalDataRepository`
- `PredictionCacheRepository`
- `SessionRepository`
- `AuditLogRepository`

Use explicit schemas, not JSON blobs, for model-critical fields.

### 7. Delete or quarantine the parallel analytics stack

Either:

- remove `src/abba/analytics/*`, `src/abba/agents/*`, and DI modules from the active repo

or:

- move them into an isolated experimental package with separate tests and entrypoints

The current state guarantees architectural confusion.

## Immediate Remediation Order

1. Define a canonical runtime path and deprecate the parallel analytics stack.
2. Build a single `NhlGameContext` snapshot that includes all required inputs and provenance.
3. Make `refresh_data()` populate every dataset the NHL model actually consumes, or fail closed.
4. Remove fake model dimensions from docs and outputs until they are truly active.
5. Move rest, injury, Corsi, and xG into explicit model components with tests proving they affect outputs.
6. Replace storage JSON blobs for model-critical stats with typed columns or validated typed payloads.
7. Replace workflow-invented confidence/risk heuristics with domain services or report adapters.

## Mathematical Claim Inventory

### Active Runtime Path

| Claim | File / symbol | Exact inputs -> outputs | Formula / method | Matches known standard? | Required assumptions / invariants | Assumptions status | Numerical risk | Silent garbage triggers |
|---|---|---|---|---|---|---|---|---|
| Player injury impact heuristic | `src/abba/server/toolkit.py` `ABBAToolkit._player_impact` | roster rows with `position`, `injury_status`, `stats.points` -> `injury_impact`, `top_scorer_available`, `roster_completeness` | top-10 availability ratio; descending point-rank penalty `max(0.015 - i*0.001, 0.003)`; goalie injury adds `0.03`; cap at `0.10` | No | points must exist and rank player importance; injury states must be current; roster must be season-aligned | documented in docstring, not enforced, not separately validated | low | missing `points`, stale roster, wrong season, all players defaulting healthy |
| General feature normalization | `src/abba/engine/features.py` `FeatureEngine.build_features` | team stat dicts + optional weather -> 10-feature dict | win pct, run/goal diff per game, recent form fallback, fixed home-advantage constant, linear weather scaling with clipping | Partial | wins/losses nonnegative; weather in Fahrenheit / mph; recent form in `[0,1]`; same season for both teams | documented, partially unit-tested, not enforced | medium | mixed seasons, wrong weather units, missing stats falling back to arbitrary defaults |
| General game pseudo-model ensemble | `src/abba/engine/features.py` `FeatureEngine.predict_from_features` | feature dict -> 4 probabilities | log5 blend, Pythagorean-inspired strength on shifted differentials, recent-form weighted model, weather-adjusted average | Partial at best; mostly heuristic | probabilities in `[0,1]`; differentials on comparable scales; weather effect small; no missing-feature bias | documented as heuristic, unit-tested for shape/range, not validated empirically | medium | differentials near extreme values, default features dominating, users assuming trained models exist |
| Neutral-deviation feature importance | `src/abba/server/tools/analytics.py` `explain_prediction` | prediction feature dict -> ranked factor list | `deviation = abs(value - neutral_value)`; sign by comparison to neutral baseline | No | neutral baselines must be meaningful; features independent enough to rank individually | documented, not validated, no theoretical support | low | correlated features, arbitrary neutral constants, omission of interaction effects |
| Ensemble summary statistics | `src/abba/engine/ensemble.py` `EnsembleEngine.combine` | list of probabilities, optional weights -> point estimate, confidence, CI margin | mean/median/vote or weighted; `std`; `margin = 1.96*std/sqrt(n)`; `confidence = clip(1 - 2*std, 0, 1)` | Partial for average/median; confidence and CI semantics are ad hoc | model outputs exchangeable; std of model disagreement proxies uncertainty; predictions bounded | documented, unit-tested, not calibrated | medium | correlated models, tiny `n`, disagreement interpreted as probabilistic confidence |
| "Inverse-variance" weighting | `src/abba/engine/ensemble.py` `EnsembleEngine._weighted_combine` | prediction vector -> weighted mean | weights `1 / abs(pred - group_mean)` normalized | No, this is not inverse-variance weighting | proximity to consensus must imply reliability | documented incorrectly, unit-tested for behavior, theoretical basis absent | low | one outlier dominates denominator behavior; clustered but biased models get rewarded |
| Expected value per wager | `src/abba/engine/value.py` `ValueEngine._calculate_ev` | probability, decimal odds -> EV per dollar | `EV = p*(odds-1) - (1-p)` with `p` clipped to `[0.001,0.999]` | Yes | odds are decimal odds > 1; probability is calibrated and on same event definition | documented, indirectly tested, not enforced on caller | low | American/fractional odds passed in, miscalibrated probabilities |
| Market scan decision rule | `src/abba/engine/value.py` `ValueEngine.find_value` | scheduled games, home win probs, odds rows -> value opportunities | implied prob `1/odds`; edge threshold and EV threshold on both sides | Partial | market rows correspond to same event and side; no vig normalization except raw implied prob | documented, unit-tested, not robustly validated | medium | stale odds, wrong game mapping, mismatched market types |
| Tool-level EV duplicate | `src/abba/server/tools/market.py` `MarketToolsMixin.calculate_ev` | probability, decimal odds -> EV summary dict | same EV formula plus implied prob and edge | Yes, but duplicate authority | same as above | documented, tested through tool path, duplicated instead of centralized | low | one copy updated without the other |
| Kelly sizing | `src/abba/engine/kelly.py` `KellyEngine.calculate` | win probability, decimal odds, bankroll -> fraction, stake, EV, edge | `full_kelly = (b*p - q)/b`; fractional Kelly, cap, min-edge/min-EV gates | Yes | binary payoff, known calibrated `p`, decimal odds > 1, bankroll meaningful, edge and EV thresholds appropriate | documented, unit-tested, partially enforced | medium | bad probability calibration, wrong odds format, correlated bets treated independently |
| Elo win probability | `src/abba/engine/elo.py` `EloRatings._win_probability` / `predict` | home/away Elo ratings -> home/away win probs | logistic Elo formula `1 / (1 + 10^((Rb-Ra)/400))` with home bonus | Yes | Elo scale comparable over teams and seasons; home advantage fixed | documented, unit-tested, not empirically recalibrated here | low | rating drift, wrong home advantage, replaying inconsistent seasons |
| Elo update and MOV multiplier | `src/abba/engine/elo.py` `update` / `_margin_of_victory_multiplier` | ratings + scores -> updated ratings | `shift = K * MOV * (actual - expected)`; MOV uses `ln(goal_diff+1)*(2.2/(elo_diff*0.001+2.2))` clamped to `>=1` | Partial, recognizable FiveThirtyEight-style variant | score differential informative; games independent; no ties in NHL unless encoded consistently | documented, not obviously backtested in repo, some unit coverage | medium | season boundaries missing, overtime/shootout treated like regulation, repeated replays |
| Degree centrality | `src/abba/engine/graph.py` `_degree_centrality` | adjacency matrix -> normalized centrality vector | weighted degree normalized by max degree | Yes | nonnegative symmetric adjacency; weights comparable | documented, unit-tested, not enforced | low | negative weights, disconnected graph with all-zero rows |
| Closeness centrality | `src/abba/engine/graph.py` `_closeness_centrality` | adjacency matrix -> normalized closeness vector | unweighted shortest paths on binarized adjacency | Yes, but not for weighted graph | edges represent connectivity, not weighted distances | documented in comments, not enforced; weight semantics explicitly discarded | low | users assume edge weights matter when they do not |
| Betweenness centrality | `src/abba/engine/graph.py` `_betweenness_centrality` | adjacency matrix -> normalized betweenness vector | Brandes-style BFS on binarized graph | Yes for unweighted graphs | graph small enough for O(n^3)-ish traversal; edges unweighted | documented, not validated against reference library | medium | large graphs, weighted relationships treated as binary |
| Eigenvector centrality | `src/abba/engine/graph.py` `_eigenvector_centrality` | adjacency matrix -> normalized principal eigenvector | largest-eigenvalue eigenvector magnitude via `eigh` | Yes | adjacency symmetric; principal eigenvector meaningful; weights nonnegative | documented, not cross-checked | medium | disconnected or sign-indefinite graphs, degenerate spectra |
| Network density / clustering / cohesion / key-player threshold | `src/abba/engine/graph.py` `analyze_team`, `_density`, `_clustering_coefficient` | relationships -> density, clustering, cohesion, key players | density `actual/max`; clustering via `diag(A^3)/(k*(k-1))`; cohesion `(density+clustering)/2`; key players at 70th percentile of ad hoc combined score | Partial for density/clustering; cohesion and threshold are heuristic | simple undirected graph; triangle count formula on binarized adjacency; percentile split meaningful | partially documented, lightly tested, heuristic parts unvalidated | low | sparse or tiny graphs, weight information discarded, arbitrary 70th percentile |
| Logistic xG shot model | `src/abba/engine/hockey.py` `HockeyAnalytics.expected_goals` | list of shots with distance, angle, type, rebound/rush/strength flags -> shot xG list and totals | additive log-odds with hard-coded coefficients, logistic transform `1/(1+exp(-z))` | Yes in form; coefficients are frozen constants | distances in feet, angle in degrees, shot taxonomy consistent, coefficients calibrated to same source data | documented, tested in hockey tests, not re-estimated in repo | medium | wrong units, unseen shot types defaulting to zero effect, coefficient drift |
| Goalie rate stats | `src/abba/engine/hockey.py` `goaltender_metrics` | saves, shots against, goals against, xG against, games, minutes -> Sv%, GAA, GSAA, xGSAA, QS% | standard rate formulas; GSAA uses league-average Sv% `0.907`; xGSAA = `xGA - GA` | Partial/standard | minutes > 0, shots/goals consistent, league average current enough | documented, likely unit-tested, league average hard-coded | low | stale league average, mixed game states, wrong minutes |
| Goalie matchup edge | `src/abba/engine/hockey.py` `goaltender_matchup_edge` | starter/opponent Sv% and GSAA -> edge terms and combined edge | scaled differences: `sv_diff/0.01*0.0375`, `gsaa_diff/10*0.05`, then `0.7*sv_edge + 0.3*gsaa_edge`, clipped | No, heuristic calibration | linear mapping from season-long Sv% and GSAA to single-game win probability | documented as calibrated, not validated in code, assumptions ignored | medium | small-sample goalie stats, season GSAA treated as game-strength proxy |
| Special teams index | `src/abba/engine/hockey.py` `special_teams_rating` | PP/PK goals/opportunities and shot-quality inputs -> rates and `special_teams_index` | PP%, PK%, above-average deltas from 22/80; combined index `0.45*PP + 0.55*PK` | Partial; index is heuristic | league averages stable; PP and PK additive on comparable scale | documented, not validated against outcomes | low | league-average drift, missing shot-quality data defaulting to zero |
| Rest/fatigue model | `src/abba/engine/hockey.py` `rest_advantage` | rest days, B2B flags, travel km, games in last 7 -> rest-edge dict | additive penalties/bonuses: B2B `0.045`, rest bonus `0.01/day`, travel up to `0.02`, density `0.008/game`, edge = away_total - home_total | No, heuristic | travel/rest effects approximately additive; scales chosen correctly; units are days/km/games | documented, not enforced, not backtested in code | low | wrong travel units, stale schedules, double-counting fatigue factors |
| Score-adjusted Corsi | `src/abba/engine/hockey.py` `score_adjusted_corsi` | state-specific CF/CA and minutes -> adjusted Corsi stats | weighted CF/CA using simplified state multipliers, then recompute percentages/per60 | Partial | score-state buckets representative; chosen factors approximate published estimates | documented as simplified, not cross-validated here | low | biased state buckets, multipliers not matching actual score effects |
| Cap and LTIR model | `src/abba/engine/hockey.py` `cap_analysis` | roster contracts -> cap space, LTIR relief, top-heavy metrics, grades | aggregation; LTIR approximation `max(ltir_total - max(cap_space,0), 0)`; categorical `cap_health` thresholds | Partial; LTIR part explicitly approximate | contract statuses correct, cap numbers same season and league rules, LTIR timing irrelevant | documented as simplified, not enforced, not audited | low | stale CBA rules, incomplete roster, cap statuses wrong |
| Season review and grades | `src/abba/engine/hockey.py` `season_review` | team stats, optional advanced stats, optional goalie stats -> review dict | Pythagorean wins with exponent `2.05`; luck factor `wins - pyth_wins`; threshold grades for analytics/special teams/goalies | Partial | goals for/against representative; thresholds meaningful; goalie list ordered by importance | documented, tested in hockey tests, grade thresholds heuristic | low | mixed seasons, goalie ordering wrong, thresholds becoming stale |
| Playoff Monte Carlo | `src/abba/engine/hockey.py` `playoff_probability` | current points, games remaining, games played, optional opponent win probs -> projected points, probabilities, status | regress points pace toward `.500`; multinomial simulation with win/OTL/reg-loss probabilities; 50,000 sims | Partial | OTL fixed at 25% of non-wins; team talent stationary; cutlines fixed and independent | documented, likely unit-tested, assumptions not enforced | medium | wrong cutlines, unstable RNG, extreme small samples, schedule effects omitted |
| NHL feature builder | `src/abba/engine/hockey.py` `build_nhl_features` | team stats, advanced stats, goalies, optional rest and odds -> NHL feature dict | points%, goal diff, home/road splits, goalie edge, special-teams edge, devigged market probability | Partial | all inputs same season; advanced stats and odds correspond to same teams/game; decimal odds > 1 | documented, not enforced, partially tested | medium | missing advanced/goalie data defaults to neutral, wrong odds mapping, season mismatch |
| Empirical Bayes regression to mean | `src/abba/engine/hockey.py` `regress_to_mean` | observed stat, league avg, games played, `k` -> shrunk stat | `league_avg + (observed - league_avg) * n/(n+k)` | Yes | `k` appropriate for target stat; observed on same scale as league average | documented, not validated per feature type | low | using same `k` for incompatible stats |
| NHL pseudo-model ensemble | `src/abba/engine/hockey.py` `predict_nhl_game` | NHL feature dict + optional Elo -> list of model probabilities | log5 on points%; Pythagorean log5; recent-form linear rule; goal-diff linear rule; goalie linear rule; composite with ST/rest; optional market blend and Elo append | Partial to heuristic | features calibrated, independent enough to ensemble, home boost `0.04` valid, linear coefficients correct | documented, unit-tested for shape, not validated against backtests in code | medium | inactive features assumed active, rest often zero, market and Elo correlated with base models |
| Prediction confidence metadata | `src/abba/engine/confidence.py` `build_prediction_meta` | feature dict, prediction, source flags, staleness -> metadata dict | hard-coded reliability grade rules; CI widening by calibration error and sample factor `sqrt(50/min_gp)`; 80% interval via `1.28` | Partial; structure recognizable, calibration not grounded in runtime artifacts | backtest metrics apply to current model version; `data_source` accurate; sample size proxies quality | documented, not enforced, not artifact-linked | medium | stale calibration baseline, mixed live/seed inputs collapsed to one source, missing refresh ts |
| Workflow confidence metadata | `src/abba/engine/confidence.py` `build_workflow_meta` | workflow name, data sources, steps, staleness, min GP -> metadata dict | weakest-link source reduction + same grade/caveat machinery as prediction metadata | No strong standard | data_sources list complete and accurate; min GP known; steps meaningful | documented, not enforced, weakly tested | low | workflows pass inferred or incorrect provenance and GP values |
| Tool-budget accounting | `src/abba/server/toolkit.py` `_track` | latency, fixed cost `0.01` -> budget decrement and metadata | constant cost model and latency recording | No | each tool call should have equal economic cost | undocumented as model, not validated | low | users assuming cost signal reflects real compute or API cost |
| Slate confidence scale | `src/abba/workflows/engine.py` `tonights_slate` | predicted win prob -> `confidence` 0-100 | `abs(pred_val - 0.5) * 200` | No | probability distance from 0.5 is a valid confidence proxy | undocumented as theory, not validated | low | users read it as calibrated confidence; correlated model errors ignored |
| Betting-strategy risk scaling | `src/abba/workflows/engine.py` `betting_strategy` | risk tolerance, bankroll, Kelly recommendation -> stake sizes and expected profit | heuristic mapping of risk tolerance to Kelly multiplier/min EV/max daily risk; expected profit `sum(stake*EV)` | No | Kelly output scales linearly with subjective risk tolerance; bets independent | partly documented in prose, not validated | low | correlated bets, bankroll constraints, users assuming optimization or portfolio theory |

### Parallel Analytics Stack

| Claim | File / symbol | Exact inputs -> outputs | Formula / method | Matches known standard? | Required assumptions / invariants | Assumptions status | Numerical risk | Silent garbage triggers |
|---|---|---|---|---|---|---|---|---|
| Ensemble model training pipeline | `src/abba/analytics/advanced_analytics.py` `train_ensemble` | feature matrix `X`, labels `y`, model ensemble -> fitted models and train/test scores | random `train_test_split`, `StandardScaler`, sklearn `.fit()`, `.score()`, overfitting = train-test gap | Yes in broad pipeline shape | IID samples, scaling appropriate, classification accuracy adequate metric, class balance acceptable | not documented rigorously, not validated, no CV/stratification checks | medium | leakage, class imbalance, tiny samples, nonstationary data |
| Biometric trend estimator | `src/abba/analytics/advanced_analytics.py` `BiometricsProcessor._calculate_trend` | numeric series -> slope | first-order polynomial fit via `np.polyfit` | Yes | approximately linear trend, enough points, equally spaced samples | undocumented, not validated | medium | short/noisy series, irregular sampling, outliers |
| Heart-rate fatigue proxy | `src/abba/analytics/advanced_analytics.py` `BiometricsProcessor._calculate_hr_fatigue` | HR time series -> fatigue score | std of successive differences, normalize by `10`, fatigue = `1 - normalized_hrv` | No | HRV proxy from raw diffs is meaningful and scale `10` is valid | undocumented, not validated | low | sample-rate changes, wrong units, healthy variation read as fatigue |
| Weighted fatigue score | `src/abba/analytics/advanced_analytics.py` `BiometricsProcessor._calculate_fatigue` / `_normalize_fatigue_metric` | fatigue metrics dict -> fatigue score in `[0,1]` | weighted sum of normalized HRV/sleep/recovery/stress metrics with fixed weights | No | all metrics present on comparable scales; weights sum to domain reality | partially commented, not validated | low | missing metrics, wrong metric scales, heuristic weights dominating |
| Recovery score | `src/abba/analytics/advanced_analytics.py` `BiometricsProcessor._calculate_recovery` | processed biometrics -> recovery score | additive rule on HR trend, inverse fatigue, avg-speed presence; cap at 1 | No | HR trend negative implies recovery; movement implies readiness | undocumented, not validated | low | movement/noise confounded with recovery |
| Personalized risk tolerance | `src/abba/analytics/personalization.py` `_calculate_risk_tolerance` | bet amounts -> risk score `[0,1]` | mean stake and variance normalized by `100` and `1000`, average, clip | No | amount and variance represent psychological risk tolerance; scaling constants meaningful | commented, not validated | low | currencies differ, one outlier bet inflates variance, sparse history |
| Personalized model hyperparameters | `src/abba/analytics/personalization.py` `create_model` | `patterns.risk_tolerance` -> RandomForest hyperparameters | `n_estimators=max(50,int(100*risk))`; `max_depth=max(5,int(10*risk))` | No | higher risk tolerance should imply more trees/depth | undocumented, not validated | low | meaningless parameter-personality linkage |
| Personalized training features | `src/abba/analytics/personalization.py` `_prepare_training_data` | bet history objects -> `X`, `y` | raw amount, odds, confidence, normalized hour/day-of-week, binary outcome label | Partial as feature extraction | labels are well-defined and not post-treatment; time normalizations informative | documented in comments only, not validated | low | timestamp defaults to now, confidence self-referential, tiny sample |
| Parallel ensemble combiner | `src/abba/analytics/advanced_analytics.py` `EnsembleManager._weighted_combination` / `calculate_error_bars` | prediction list -> combined prediction, confidence, margin | claims inverse variance; confidence `1 - std`; margin `std/sqrt(n)` | Partial to misleading | prediction variance captures model uncertainty; models exchangeable and independent | not validated; comments overclaim | medium | identical correlated models, tiny `n`, confidence interpreted as probability |

## Claims With No Theoretical Grounding In Code

- `ABBAToolkit._player_impact` injury penalties and goalie injury bonus in `src/abba/server/toolkit.py`
- `FeatureEngine.predict_from_features` pseudo-model coefficients in `src/abba/engine/features.py`
- `HockeyAnalytics.goaltender_matchup_edge` linear conversion from Sv% / GSAA differences to win-probability edge in `src/abba/engine/hockey.py`
- `HockeyAnalytics.special_teams_rating` combined special-teams index in `src/abba/engine/hockey.py`
- `HockeyAnalytics.rest_advantage` additive fatigue constants in `src/abba/engine/hockey.py`
- workflow confidence scale `abs(pred_val - 0.5) * 200` in `src/abba/workflows/engine.py`
- workflow bankroll/risk-tolerance stake scaling in `src/abba/workflows/engine.py`
- biometric fatigue, recovery, and personalization heuristics in `src/abba/analytics/advanced_analytics.py` and `src/abba/analytics/personalization.py`

## Claims With Insufficient Validation

- `build_prediction_meta` and `build_workflow_meta` in `src/abba/engine/confidence.py`
  - hard-coded backtest baselines exist, but no artifact lineage ties them to the current model surface
- `HockeyAnalytics.predict_nhl_game` in `src/abba/engine/hockey.py`
  - unit tests cover shape/range, not calibration or out-of-sample predictive quality
- `EnsembleEngine.combine` in `src/abba/engine/ensemble.py`
  - no validation that disagreement-derived confidence or CI margins correspond to empirical error
- `playoff_probability` in `src/abba/engine/hockey.py`
  - no visible validation against historical playoff races or opponent-specific schedules in normal use
- `GraphEngine` metrics in `src/abba/engine/graph.py`
  - no reference-check against a graph library baseline for weighted graphs
- `train_ensemble` in `src/abba/analytics/advanced_analytics.py`
  - no cross-validation, calibration assessment, class-balance handling, or leakage detection

## Claims Likely To Be Wrong Or Fragile

- The NHL model claims Corsi, xG, player-impact, and rest-aware reasoning, but the active scoring function does not consume several of those features at all.
- Workflow-level confidence is semantically inconsistent with prediction-level confidence.
- Live refresh provenance is too weak to support the reliability grades emitted by `build_prediction_meta`.
- `EnsembleEngine._weighted_combine` is mislabeled as inverse-variance weighting.
- `FeatureEngine.predict_from_features` presents heuristics in the shape of an ensemble model, which invites false trust.
- `playoff_probability` bakes in a fixed `25%` overtime-loss share for non-wins and fixed cutlines; this will fail silently if league conditions or interpretation change.
- The parallel analytics stack imports external modules (`database`, `models`) and contains numerous implicit scaling constants without evidence they were ever estimated from data.

## Algorithm Verification Report

Severity scale used here:

- `P0` correctness-corrupting; output is mathematically untrusted in normal use
- `P1` major deviation from claimed method; usable only as a heuristic with explicit caveats
- `P2` approximate but defensible if assumptions are documented and validated
- `P3` broadly faithful implementation with bounded residual risk

### P0 - NHL Composite Predictor Is Not Faithful To Its Claimed Feature Set

Files and symbols:

- `src/abba/engine/hockey.py` `HockeyAnalytics.build_nhl_features`
- `src/abba/engine/hockey.py` `HockeyAnalytics.predict_nhl_game`
- `src/abba/server/tools/nhl.py` `NHLToolsMixin.nhl_predict_game`
- `src/abba/workflows/engine.py` `WorkflowEngine.game_prediction`

Intended algorithm:

- A feature-based NHL win-probability ensemble using points percentage, Pythagorean strength, recent form, goal differential, goalie quality, special teams, rest, market, Elo, and advanced analytics such as Corsi/xG.

Governing equations as implied by the code:

- `m1 = log5(regressed home_pts_pct, regressed away_pts_pct) + 0.04`
- `m2 = log5(regressed pyth(home_gf_pg, home_ga_pg), regressed pyth(away_gf_pg, away_ga_pg)) + 0.04`
- `m3 = 0.5 + 0.25 * (home_recent_form - away_recent_form) + 0.04`
- `m4 = 0.5 + 0.16 * (home_goal_diff_pg - away_goal_diff_pg) + 0.04`
- `m5 = 0.5 + goaltender_edge + 0.04`
- `m6 = mean(m1..m5) + 0.3 * home_st_edge + rest_edge`
- optional market blend: `0.7 * mean(models) + 0.3 * market_implied_prob`
- optional Elo append: raw `elo_prob`

Step-by-step verification:

- `build_nhl_features` computes `home_corsi_pct`, `away_corsi_pct`, `home_xgf_pct`, `away_xgf_pct`, but `predict_nhl_game` never reads them.
- `NHLToolsMixin.nhl_predict_game` injects player-injury proxies into `features`, but `predict_nhl_game` never reads those either.
- `WorkflowEngine.game_prediction` computes `rest_info` and narrates rest effects, but passes only `gid` into `nhl_predict_game`, so the active model path usually scores with `rest_edge = 0.0`.
- The additive `home_boost = 0.04` is applied directly in probability space to every submodel. That is not equivalent to a home-ice term in logit, Elo, or latent-strength space and distorts tails.
- The market blend is described as inverse-variance weighting in comments, but the implementation is fixed `70/30`.
- Correlated submodels are averaged as if they were distinct evidence sources. No decorrelation, calibration, or covariance handling exists.

Omitted terms and invalid simplifications:

- Corsi/xG terms are omitted entirely from scoring despite being described as important predictive inputs.
- Injury features are omitted entirely from scoring.
- Rest is omitted on the common workflow path.
- No interaction terms exist for goalie x team defense, market x model disagreement, or home/road split uncertainty.
- No calibration layer maps raw ensemble outputs to observed win frequencies.

Assumptions that must hold but are not enforced:

- all features refer to the same season and game context
- the fixed coefficients `0.25`, `0.16`, `0.3`, and `0.04` are stable across eras
- averaging highly correlated pseudo-models yields a better estimate
- neutral defaults such as `0.50`, `0.0`, and league-average goalie values do not bias outputs materially

Invariance and boundedness checks:

- Boundedness is enforced only by clipping to `[0.01, 0.99]`.
- Symmetry is broken by hard-coded home boost and by market/home odds handling.
- Monotonicity mostly holds locally, but clipping and additive probability offsets can flatten or distort response near the boundaries.

Verdict:

- `likely incorrect`
- The implementation is not faithful to the algorithm the surrounding code and docs claim to be using.

Required fixes / proofs / tests:

- Either remove all unused feature claims from the API and docs, or wire those features into the score path with explicit coefficients.
- Move the model into a single latent-space formulation, such as logistic regression or calibrated stacking, instead of additive probability heuristics.
- Add calibration tests against held-out games: reliability curve, Brier score, log loss, and coverage of any reported intervals.
- Add an execution-path test proving that workflow rest data, advanced stats, and injury proxies actually affect the final probability.

### P0 - Ensemble Weighting And Confidence Are Misstated As Statistical Inference

Files and symbols:

- `src/abba/engine/ensemble.py` `EnsembleEngine.combine`
- `src/abba/engine/ensemble.py` `EnsembleEngine._weighted_combine`
- `src/abba/analytics/advanced_analytics.py` `EnsembleManager._weighted_combination`
- `src/abba/analytics/advanced_analytics.py` `EnsembleManager.calculate_error_bars`

Intended algorithm:

- Inverse-variance ensemble aggregation with confidence intervals derived from model disagreement.

Expected formulation:

- If model-specific uncertainties exist, inverse-variance weighting is `w_i = 1 / sigma_i^2` and `p_hat = sum(w_i p_i) / sum(w_i)`.
- A valid uncertainty estimate must account for model dependence, calibration error, and the fact that model spread is not the same thing as forecast error.

Actual code:

- Active path weighting is `w_i proportional to 1 / (abs(p_i - mean(p)) + 1e-8)`.
- Active path interval is `1.96 * std(predictions) / sqrt(n)`.
- Active path confidence is `clip(1 - 2 * std(predictions), 0, 1)`.
- Parallel path weighting is `w_i proportional to 1 / (1 + abs(p_i - 0.5))`, which rewards predictions close to `0.5`, not low-variance models.
- Parallel path margin is `std / sqrt(n)` and confidence is `1 - std`.

Step-by-step verification:

- Neither path has per-model variance estimates.
- Neither path estimates covariance between models.
- The active path weights predictions close to the ensemble mean more heavily; that is consensus weighting, not inverse-variance weighting.
- The parallel path weights low-confidence probabilities more heavily because values near `0.5` get larger weights.
- The reported `95% CI` assumes IID draws from a population of unbiased model predictions. That assumption is false for a fixed small set of correlated deterministic models.

Omitted terms and invalid simplifications:

- omitted model error variances
- omitted model correlation structure
- omitted calibration mapping from disagreement to empirical error
- hidden epsilon `1e-8` prevents division by zero but also makes exact consensus singularly dominant

Numerical and theoretical risks:

- With nearly identical predictions, the active weighting becomes numerically dominated by arbitrary tiny differences.
- Confidence can be high even when all models are jointly wrong in the same direction.
- Error margins shrink with `sqrt(n)` even when added models are redundant.

Verdict:

- `likely incorrect`

Required fixes / proofs / tests:

- Rename the methods honestly if they remain heuristic: `consensus_weighted_average`, not inverse-variance.
- If statistical intervals are desired, estimate them from historical residuals or conformal calibration, not model spread alone.
- Add a regression test showing monotonic behavior when one model moves while others stay fixed.
- Add empirical coverage tests: reported 95% intervals must contain outcomes at approximately 95% frequency over a backtest.

### P1 - Playoff Probability Simulation Is A Fragile Approximation, Not A League Model

Files and symbols:

- `src/abba/engine/hockey.py` `HockeyAnalytics.playoff_probability`

Intended algorithm:

- Monte Carlo simulation of the remaining season to estimate playoff qualification probability.

Governing equations in code:

- `pts_per_game = current_points / games_played`
- `true_talent = 0.5 + ((pts_per_game / 2) - 0.5) * games_played / (games_played + 55)`
- if schedule-aware:
  - each game uses multinomial outcome `[win, otl, reg_loss] = [p_win, 0.25 * (1 - p_win), 0.75 * (1 - p_win)]`
- else:
  - same multinomial repeated `games_remaining` times with constant `true_talent`
- playoff probability is `mean(sim_final_points >= cutoff)`

Step-by-step verification:

- The regression-to-mean step is mathematically coherent as an empirical-Bayes shrinkage on a rate parameter.
- The conversion from points-per-game to `win_rate = pts_per_game / 2` is not a true win probability when overtime losses contribute one point. It mixes event types into one latent Bernoulli.
- The overtime-loss share is hard-coded to `25%` of non-wins with no evidence from current-season data.
- Qualification is modeled against fixed cutlines, not against competing teams' simulated point totals. That is not how playoff races work.
- Division and wildcard probabilities are treated as simple threshold exceedance events, ignoring tie-breaks and the rest of the conference distribution.

Broken assumptions:

- stationarity of team talent
- fixed league cutlines
- independence of remaining games
- fixed overtime-loss rate across teams and schedules

Edge-case failures:

- early season: shrinkage dominates and the latent win interpretation is least valid
- late season bubble races: threshold-only logic misses correlated opponent outcomes
- changing league environment: fixed cutlines silently drift out of date

Verdict:

- `approximate but defensible` only as a coarse point-threshold heuristic
- `likely incorrect` if presented as actual playoff probability

Required fixes / proofs / tests:

- Rename outputs to `cutline_exceedance_probability` unless the league table is simulated jointly.
- Replace the fixed OTL split with historical team- or league-estimated outcome rates.
- Simulate the relevant competitors and apply actual tiebreak logic.
- Backtest the method on prior seasons and report calibration by month of season.

### P1 - Graph Metrics Claim Weighted Team-Network Analysis But Mostly Use Unweighted Topology

Files and symbols:

- `src/abba/engine/graph.py` `GraphEngine.analyze_team`
- `src/abba/engine/graph.py` `_degree_centrality`
- `src/abba/engine/graph.py` `_closeness_centrality`
- `src/abba/engine/graph.py` `_betweenness_centrality`
- `src/abba/engine/graph.py` `_eigenvector_centrality`
- `src/abba/engine/graph.py` `_density`
- `src/abba/engine/graph.py` `_clustering_coefficient`

Intended algorithm:

- Weighted undirected graph analysis for player relationships and team cohesion.

Expected formulation:

- If weights encode tie strength, weighted closeness and weighted betweenness should use distances derived from weights, typically `distance_ij = 1 / weight_ij` or another documented transform.

Actual code:

- Degree uses weighted sums.
- Eigenvector centrality uses the weighted adjacency matrix.
- Closeness, betweenness, density, and clustering all binarize the graph with `(adj > 0)`.
- Team cohesion is defined as `(density + clustering) / 2`.
- Key players are the top `30%` by an arbitrary weighted combination of centrality scores.

Step-by-step verification:

- The code explicitly discards edge magnitudes for closeness and betweenness, contradicting the module-level weighted-graph claim.
- The binary transformation changes units and meaning across metrics inside the same combined score.
- The cohesion scalar is not a recognized invariant of weighted team networks; it is a convenience heuristic.
- Normalizing each metric by its own maximum hides magnitude differences and makes cross-team comparisons unstable.

Verdict:

- `underspecified` for cohesion and key-player semantics
- `likely incorrect` if interpreted as weighted centrality analysis

Required fixes / proofs / tests:

- Decide whether this engine is weighted or unweighted and implement consistently.
- Validate against `networkx` or another graph-library baseline on known toy graphs.
- Add property tests for symmetry, permutation invariance of node ordering, and expected behavior on complete/path/star graphs.
- Remove or clearly label `team_cohesion` and `is_key_player` as heuristics unless domain validation exists.

### P1 - FeatureEngine Pseudo-Models Are Heuristics Wearing Model-Shaped Interfaces

Files and symbols:

- `src/abba/engine/features.py` `FeatureEngine.build_features`
- `src/abba/engine/features.py` `FeatureEngine.predict_from_features`

Intended algorithm:

- Generic sports feature engineering plus ensemble-like prediction generation.

Governing equations in code:

- `m1 = 0.7 * log5(home_win_pct, away_win_pct) + 0.3 * home_advantage`
- `m2 = (max(home_run_diff_per_game + 5, 0.01)^exp) / total`
- `m3 = 0.8 * (0.6 * home_recent_form + 0.4 * (1 - away_recent_form)) + 0.2 * home_advantage`
- `m4 = clip(mean(m1, m2, m3) - 0.02 * wind_impact - 0.01 * abs(temp_impact), 0.01, 0.99)`

Step-by-step verification:

- `m1` is a recognizable log5 variant, but the linear blend with `home_advantage` is not a standard derivation.
- `m2` uses `run_diff + 5` to force positivity before exponentiation. That hidden offset dominates interpretation and is not documented as a model assumption.
- `m3` is a weighted average of season form and the complement of opponent form; it is purely heuristic.
- `m4` applies weather as a direct additive probability penalty, not through team-specific run/goal expectation or variance.

Broken assumptions:

- run differential can be translated into pseudo-strength by adding `5`
- temperature and wind effects are symmetric across teams
- the same formula family is suitable across MLB and NHL with only an exponent switch

Verdict:

- `underspecified`
- usable only if explicitly labeled as a fallback heuristic, not as a mathematically grounded model

Required fixes / proofs / tests:

- Rename outputs from `predict_from_features` to `heuristic_scores` unless trained/calibrated.
- Externalize all constants (`5`, `0.7`, `0.3`, `0.6`, `0.4`, `0.02`, `0.01`) into documented configuration with provenance.
- Add monotonicity tests proving that stronger home inputs never reduce home probability absent weather changes.

### P2 - Expected Goals Is A Recognizable Logistic Shot Model, But The Coefficient Contract Is Weak

Files and symbols:

- `src/abba/engine/hockey.py` `HockeyAnalytics.expected_goals`

Intended algorithm:

- Logistic-regression xG model over shot features.

Governing equations:

- For each shot `i`, `z_i = beta_0 + beta_d * distance_i + beta_a * angle_i + beta_type[shot_type_i] + beta_r * rebound_i + beta_u * rush_i + beta_pp * 1(pp) + beta_sh * 1(sh)`
- `xg_i = 1 / (1 + exp(-z_i))`
- `xg_total = sum_i xg_i`

Step-by-step verification:

- The implementation matches the additive logistic form.
- `angle` is transformed with `abs(angle)`, which assumes left/right symmetry. That is defensible.
- No interaction terms exist for distance-angle coupling or pre-shot movement, which many modern xG models include.
- Coefficients are loaded from class config; the code does not prove which training set, units, or feature encoding they came from.

Assumptions and invariants:

- distance in feet
- angle in degrees with sign irrelevant
- categorical shot types match the coefficient dictionary exactly
- coefficients were fit on the same feature definitions

Numerical risks:

- Large `|z|` can overflow `exp`, though likely not with current coefficient scales.

Verdict:

- `approximate but defensible`

Required fixes / proofs / tests:

- Pin coefficient provenance in code or an adjacent artifact with training date, source, and feature units.
- Use a numerically stable sigmoid such as `scipy.special.expit`.
- Add golden tests using a few canonical shots with expected xG values.

### P2 - Elo Core Is Broadly Faithful, But The Margin Multiplier And Draw Handling Need Justification

Files and symbols:

- `src/abba/engine/elo.py` `EloRatings.predict`
- `src/abba/engine/elo.py` `EloRatings.update`
- `src/abba/engine/elo.py` `EloRatings._win_probability`
- `src/abba/engine/elo.py` `EloRatings._margin_of_victory_multiplier`

Intended algorithm:

- FiveThirtyEight-style NHL Elo with home advantage and margin-of-victory scaling.

Governing equations:

- `P(home win) = 1 / (1 + 10^((R_away - (R_home + H)) / 400))`
- `shift = K * MOV(goal_diff, elo_diff) * (actual - expected)`
- seasonal reset: `R <- R + (1500 - R) / 3`

Step-by-step verification:

- The probability equation is the standard Elo logistic form.
- Rating updates are zero-sum, which preserves total rating mass.
- The MOV multiplier uses `log(goal_diff + 1) * (2.2 / (0.001 * elo_diff + 2.2))` and clamps the minimum to `1.0`.
- Clamping to `1.0` means one-goal games are never downweighted, which is a design choice rather than a theorem.
- The update path allows `actual_home = 0.5` for tied scores, but NHL games do not end tied in the modern period. If this function receives regulation-only scores by mistake, the semantics become ambiguous.

Verdict:

- `faithful` for the core Elo equations
- `approximate but defensible` for the hockey-specific MOV treatment

Required fixes / proofs / tests:

- Document whether inputs are final NHL results, regulation-only results, or historical tie-era games.
- Add invariance tests: equal and opposite rating shifts, monotonicity of win probability in rating difference, and season reset toward the mean.
- Add calibration/backtest evidence for the chosen `K=4`, `H=50`, and MOV constants.

### P2 - Kelly Sizing Is Correct For A Binary Bet, But The Policy Layer Must Not Be Confused With Theory

Files and symbols:

- `src/abba/engine/kelly.py` `KellyEngine.calculate`

Intended algorithm:

- Fractional Kelly bet sizing for a binary wager.

Governing equations:

- `b = decimal_odds - 1`
- `q = 1 - p`
- `EV = p * b - q`
- `f_star = (b * p - q) / b`
- actual stake fraction = `clip(kelly_fraction * f_star, 0, max_bet_pct)` and then zeroed if `edge < min_edge` or `EV < min_ev`

Step-by-step verification:

- The full Kelly formula is implemented correctly.
- Input `p` is clipped to `[0.001, 0.999]`; that is a numerical guard, not a theoretical change of method.
- The `edge` uses `1 / decimal_odds`, which is only the raw implied probability for a single binary price and does not remove bookmaker margin in multi-outcome markets.
- The engine mixes theory with risk policy: fractional Kelly, hard cap, minimum edge, and minimum EV are bankroll controls, not part of Kelly’s derivation.

Verdict:

- `faithful` for binary full Kelly
- `approximate but defensible` for the deployed policy-wrapped version

Required fixes / proofs / tests:

- Separate `full_kelly_fraction` from deployment policy in the returned structure.
- Add tests showing monotonicity in `p` and `odds`, and zero stake when `EV <= 0`.
- Document that this is only valid for calibrated probabilities and repeated independent opportunities.

### P1 - Biometric Fatigue And Recovery Logic Is Untrusted Heuristic Physiology

Files and symbols:

- `src/abba/analytics/advanced_analytics.py` `BiometricsProcessor._calculate_trend`
- `src/abba/analytics/advanced_analytics.py` `BiometricsProcessor._calculate_hr_fatigue`
- `src/abba/analytics/advanced_analytics.py` `BiometricsProcessor._calculate_fatigue`
- `src/abba/analytics/advanced_analytics.py` `BiometricsProcessor._calculate_recovery`

Intended algorithm:

- Convert wearable biometrics into fatigue and recovery scores suitable for downstream prediction.

Actual governing rules:

- trend = slope of `polyfit(range(n), data, 1)`
- HR fatigue = `1 - min(std(diff(hr_data)) / 10, 1)`
- overall fatigue = weighted sum of normalized HRV, sleep, recovery-time, and stress metrics
- recovery = `0.3 * 1(negative_hr_trend) + 0.4 * (1 - fatigue) + 0.3 * 1(avg_speed > 0)`

Step-by-step verification:

- The trend estimator is mathematically fine as a line fit, but only if samples are equally spaced and linear trend is meaningful.
- The HRV proxy is not standard HRV. It uses the standard deviation of raw heart-rate differences, not RR-interval measures.
- The normalization constant `10` is hidden and unsupported.
- Recovery gives a full `0.3` bonus for any positive `avg_speed`, regardless of magnitude or context.

Verdict:

- `likely incorrect` if interpreted as validated physiology
- `underspecified` if intended only as a UI heuristic

Required fixes / proofs / tests:

- Quarantine these outputs from any correctness-critical prediction path until there is domain validation.
- Rename them to `heuristic_fatigue_score` and `heuristic_recovery_score` unless they are fit to labeled outcomes.
- Add unit tests for sampling-frequency sensitivity and outlier robustness.

### P1 - Personalization Logic Turns User Behavior Counts Into Model Parameters Without Theory

Files and symbols:

- `src/abba/analytics/personalization.py` `PersonalizationEngine._calculate_risk_tolerance`
- `src/abba/analytics/personalization.py` `PersonalizationEngine.create_model`
- `src/abba/analytics/personalization.py` `PersonalizationEngine._prepare_training_data`

Intended algorithm:

- Infer a user's risk tolerance from betting behavior and build a personalized classifier.

Actual governing rules:

- `risk_tolerance = clip((min(mean(amount)/100, 1) + min(var(amount)/1000, 1)) / 2, 0, 1)`
- `n_estimators = max(50, int(100 * risk_tolerance))`
- `max_depth = max(5, int(10 * risk_tolerance))`
- training features are raw amount, odds, confidence, hour/24, weekday/7

Step-by-step verification:

- The risk score is dimensionful: it changes if the currency unit changes or if stakes are rescaled.
- Variance in bet size is treated as psychological risk tolerance without support.
- Random-forest hyperparameters are then tied to that score with no learning objective connecting them.
- Training features include prior `confidence`, which can leak a previous model’s opinion back into the target.

Verdict:

- `likely incorrect`

Required fixes / proofs / tests:

- Remove the hyperparameter mapping unless there is empirical evidence it improves out-of-sample performance.
- Normalize monetary inputs by bankroll or median stake at minimum.
- Add ablation tests showing that personalization improves calibration or return versus a non-personalized baseline.

### P2 - Supervised Ensemble Training Uses Standard Library Calls But Violates Key Validation Assumptions

Files and symbols:

- `src/abba/analytics/advanced_analytics.py` `AdvancedAnalyticsManager.train_ensemble`

Intended algorithm:

- Standard supervised train/test pipeline for an ensemble of sklearn models.

Actual method:

- random `train_test_split(test_size=0.2, random_state=42)`
- `StandardScaler` fit on train, transform train/test
- model `.fit(X_train_scaled, y_train)`
- performance by `.score()` on train/test
- overfitting proxy = `train_score - test_score`

Step-by-step verification:

- This is mechanically a valid sklearn workflow.
- It assumes IID samples and a random split is appropriate; that is unsafe for time-series, season-based, or grouped sports data.
- `.score()` is model-dependent and often plain accuracy, which is a poor objective for imbalanced probabilistic forecasting.
- The scaler is kept in memory but `_save_ensemble` persists only the models and results JSON, not the scaler artifact; that can corrupt future inference even if the math were sound.

Verdict:

- `approximate but defensible` for a toy IID classification setting
- `underspecified` for production forecasting

Required fixes / proofs / tests:

- Use temporal or grouped validation if samples are time-ordered or team-correlated.
- Track proper scoring rules: log loss, Brier score, calibration error.
- Persist the fitted scaler alongside the ensemble and add a load/save round-trip test.

## Abstraction-Risk Report

This section is about abstraction pressure in a correctness-heavy mathematical system. The main failure mode in this codebase is not “insufficient abstraction.” It is abstraction that erases domain semantics, so the math becomes harder to inspect, harder to validate, and easier to reuse incorrectly.

Severity scale:

- `P0` abstraction directly hides correctness risk in normal operation
- `P1` abstraction materially obstructs testing, substitution, or inspection
- `P2` abstraction is weak or noisy but not yet the primary source of incorrect results

### Ranked Abstraction Risks

| Severity | Issue | Files / symbols / call chain | Why the abstraction is bad | Correctness risk | Invalid reuse risk | Why math gets harder to inspect/test | Better boundary |
|---|---|---|---|---|---|---|---|
| `P0` | Mutable god-object toolkit | `src/abba/server/toolkit.py` `ABBAToolkit`; call chain `WorkflowEngine.* -> ABBAToolkit tool mixins -> Storage/engines/connectors` | One object owns storage, caches, session budget, tracking, Elo state, seeding, and every tool surface. That is framework-shaped composition, not domain composition. | Hidden state and side effects change mathematical results by call order, refresh timing, and cache state. | Very high; any workflow or agent method can reuse the toolkit as if it were stateless when it is not. | You cannot test a mathematical routine without dragging in storage, cache, session, and tracking behavior. | Replace with explicit application services whose dependencies are passed in as narrow typed ports. |
| `P0` | Dict-and-JSON storage boundary erases domain semantics | `src/abba/storage/duckdb.py` `query_team_stats`, `query_goaltender_stats`, `query_nhl_advanced_stats`, `query_roster`, `cache_prediction` | Core mathematical inputs cross boundaries as `dict[str, Any]` and JSON blobs instead of typed domain records with units and invariants. | Mixed seasons, missing fields, wrong scales, and neutral defaults can silently flow into models. | Very high; any caller can reinterpret the same blob differently. | No compiler/type layer can tell whether `recent_form`, `save_pct`, or `points` are present, normalized, or season-aligned. | Use explicit immutable domain models: `TeamSeasonStats`, `GoalieSeasonStats`, `OddsSnapshot`, `GameContext`, `PredictionFeatures`. |
| `P0` | Workflow abstraction entangles orchestration with numerical meaning | `src/abba/workflows/engine.py` `game_prediction`, `tonights_slate`, `betting_strategy`; call chain `WorkflowEngine -> toolkit methods -> narrative/confidence/risk heuristics` | Workflows are presented as orchestration but they also assign confidence, rank bets, convert probabilities to “confidence 0-100,” and narrate causes. | Numerical semantics drift between workflow outputs and underlying models. | High; future workflows will copy these heuristics as if they were mathematically sanctioned. | Tests must validate both orchestration and embedded numerical policy in the same method. | Separate orchestration from decision policy and from presentation. Workflows should assemble typed intermediate results, not invent math. |
| `P1` | Mixin-based tool surface hides dependencies and encourages broad coupling | `src/abba/server/toolkit.py` `ABBAToolkit(DataToolsMixin, AnalyticsToolsMixin, MarketToolsMixin, NHLToolsMixin, ...)` | Mixins make every tool method implicitly depend on the full toolkit state instead of explicit constructor-injected services. | Easy to call math with the wrong side data, stale state, or missing refresh provenance. | High; new mixins can reach any field on `self` and smuggle in unrelated logic. | Unit tests need a half-real toolkit object or extensive monkeypatching. | Replace mixins with small service objects: `PredictionService`, `OddsService`, `NHLAnalysisService`, `WorkflowRunner`. |
| `P1` | Utility methods smuggle domain logic outside the math layer | `src/abba/server/toolkit.py` `_player_impact`; `src/abba/server/tools/analytics.py` `explain_prediction`; `src/abba/server/tools/market.py` `calculate_ev` | Domain rules live in helper/utility/tooling methods instead of a visible mathematical model layer. | Injury penalties, “neutral values,” and EV semantics drift independently of the main algorithms. | High; these helpers will be reused as if they were canonical domain logic. | Reviewers looking at `engine/` do not see the real math because some of it lives in toolkit helpers and tool mixins. | Put every numerical transformation in pure domain functions or algorithm strategy objects with explicit tests. |
| `P1` | Generic protocol layer erases domain semantics | `src/abba/analytics/interfaces.py` `DataProcessor`, `PredictionModel`, `AgentInterface`; `src/abba/core/dependency_injection.py` | The protocols are generic to the point of meaninglessness: `process(dict) -> dict`, `predict(np.ndarray) -> np.ndarray`. | No place to enforce domain invariants, units, sample shape, or output semantics. | Very high; almost any object can satisfy the interface while violating mathematical assumptions. | Reference implementations cannot be swapped in cleanly because there is no contract for feature schema, calibration artifacts, or uncertainty semantics. | Use domain-specific ports: `WinProbabilityModel.predict(GameFeatures) -> ProbabilisticForecast`, `BiometricFatigueModel.predict(BiometricSeries) -> FatigueScore`. |
| `P1` | Parallel DI/framework stack competes with the active runtime and duplicates logic | `src/abba/core/dependency_injection.py`; `src/abba/analytics/advanced_analytics.py`; `src/abba/analytics/personalization.py` | There is a framework-shaped secondary architecture with DI, processors, managers, and model factories that is not the real runtime path. | Engineers can fix or validate math in the wrong stack while the live path remains untouched. | High; abstractions invite future code to plug into a dead or partial architecture. | Testing effort fragments across two incompatible seams. | Delete or quarantine the unused stack, or make it the real typed domain core. Not both. |
| `P1` | Storage doubles as repository, cache, telemetry sink, and session ledger | `src/abba/storage/duckdb.py` `Storage` | One class manages persistence, prediction cache, session budget, tool-call logs, and reasoning logs. | Operational metadata and mathematical data share the same abstraction, which invites accidental coupling and stale-cache errors. | Medium-high; callers treat storage as a universal service. | Correctness tests must instantiate and reason about unrelated operational tables. | Split into repositories: `SportsRepository`, `PredictionCache`, `SessionLedger`, `TelemetryStore`. |
| `P1` | Algorithms are classes when they should be pure functions with explicit parameter objects | `src/abba/engine/hockey.py` `HockeyAnalytics`; `src/abba/engine/features.py` `FeatureEngine`; `src/abba/engine/confidence.py` builder functions plus workflow usage | The class shell suggests stateful extensibility, but the real content is collections of formulas and heuristics. | Configuration and provenance are hidden in defaults, comments, or class-level constants. | Medium; method calls on a large class encourage broad reuse without understanding local assumptions. | It is harder to snapshot intermediate values and compare against a reference implementation. | Use pure functions over immutable inputs plus small strategy objects where true algorithm substitution is needed. |
| `P2` | “Explain” abstraction implies interpretability that does not exist | `src/abba/server/tools/analytics.py` `explain_prediction` | The tool reports factor importance using distance from hard-coded neutral values, not model attribution. | Users and engineers can mistake a synthetic narrative for an explanation of the actual predictive mechanism. | Medium; this pattern can spread to NHL workflows and confidence outputs. | It is impossible to validate against the model because it is not derived from the model. | Rename as heuristic rationale or replace with attribution tied to the actual model equations. |
| `P2` | Duplicate market math across layers | `src/abba/server/tools/market.py` `calculate_ev`; `src/abba/engine/value.py` `_calculate_ev` | The same math exists in two layers separated by an abstraction boundary. | Silent drift in EV definitions or rounding policy. | Medium. | Reviewers must compare multiple layers to know which EV is canonical. | One pure EV function in a domain math module, reused everywhere. |

### Detailed Findings

#### P0 - `ABBAToolkit` Is The Wrong Abstraction Root

Files and symbols:

- `src/abba/server/toolkit.py` `ABBAToolkit`
- `src/abba/workflows/engine.py` `WorkflowEngine.__init__`, `WorkflowEngine.run`

Why it is bad:

- The toolkit is simultaneously an API surface, dependency container, session owner, cache client, data seeder, Elo lifecycle manager, and numerical service hub.
- Mixins make tool methods look modular, but the real dependency graph is hidden on `self`.

Correctness risks:

- Refresh state, cache state, and Elo state are mutable and shared across calls.
- Mathematical outputs depend on operational sequencing rather than only on explicit inputs.

Invalid reuse risk:

- Very high. A future engineer can call a toolkit method in a new context and unknowingly inherit stale caches, auto-seeding, or session effects.

Why it makes the math harder to inspect or test:

- There is no clean way to instantiate “just the NHL predictor” with explicit dependencies and no side effects.
- Tests tend to become integration tests by accident.

Better boundary:

- `AppContext` or `RuntimeContext` for storage/cache/session concerns.
- Domain services with explicit dependencies and typed inputs for prediction, pricing, and narrative assembly.

#### P0 - The Storage Abstraction Prevents Unit-Aware Validation

Files and symbols:

- `src/abba/storage/duckdb.py` `query_team_stats`
- `src/abba/storage/duckdb.py` `query_goaltender_stats`
- `src/abba/storage/duckdb.py` `query_nhl_advanced_stats`
- `src/abba/storage/duckdb.py` `query_roster`

Why it is bad:

- JSON `stats` blobs flatten all domain meaning into untyped bags.
- The repository layer cannot enforce season identity, unit conventions, or required fields.

Correctness risks:

- A caller can combine per-game rates, percentages, and cumulative totals without any boundary object objecting.
- Missing data silently becomes neutral defaults in downstream code.

Invalid reuse risk:

- Extreme. Every new algorithm will reinterpret the same dictionaries differently.

Why it makes the math harder to inspect or test:

- There is no canonical fixture type. Tests have to build ad hoc dicts with implicit contracts.

Better boundary:

- Parse database rows immediately into typed immutable records.
- Validate required fields, ranges, and season consistency at repository exit.

#### P1 - Workflow Methods Are Controllers Masquerading As Analytical Pipelines

Files and symbols:

- `src/abba/workflows/engine.py` `game_prediction`
- `src/abba/workflows/engine.py` `tonights_slate`
- `src/abba/workflows/engine.py` `run`

Why it is bad:

- These methods refresh data, query repositories, compute or import predictions, create confidence scales, perform pricing logic, and build user-facing narrative in one place.

Correctness risks:

- Orchestration code can silently override or reinterpret numerical outputs.
- Narrative claims drift away from what the model actually consumed.

Invalid reuse risk:

- High. A new workflow will likely copy an existing one and inherit hidden numerical policy.

Why it makes the math harder to inspect or test:

- There is no single object representing “computed game analysis” prior to formatting.

Better boundary:

- `GameAnalysisWorkflow` should return a typed immutable result composed of `GameContext`, `Forecast`, `MarketAssessment`, and `DataProvenance`.
- Narrative rendering should be a separate adapter.

#### P1 - The Generic Protocol/DI Layer Is Abstraction Theater

Files and symbols:

- `src/abba/analytics/interfaces.py`
- `src/abba/core/dependency_injection.py`

Why it is bad:

- `process(dict) -> dict` is not a meaningful correctness boundary in a mathematical system.
- Generic service-provider and processor abstractions describe software plumbing, not domain semantics.

Correctness risks:

- No contract exists for feature ordering, uncertainty semantics, units, or calibration state.
- Fake implementations can satisfy the interface while violating the math completely.

Invalid reuse risk:

- Very high. The abstractions invite reuse of mathematically incompatible components.

Why it makes the math harder to inspect or test:

- You cannot plug in a reference implementation and compare outputs unless both sides happen to agree on undocumented dict conventions.

Better boundary:

- Replace generic processors with strategy interfaces that name the domain object and expected output.

#### P1 - `HockeyAnalytics` Is A Kitchen-Sink Class

Files and symbols:

- `src/abba/engine/hockey.py` `HockeyAnalytics`

Why it is bad:

- One class owns shot-quality xG, goalie metrics, special teams, schedule fatigue, cap analysis, season review, playoff simulation, feature building, regression to mean, and final prediction synthesis.

Correctness risks:

- Shared defaults and convenience methods create false cohesion between unrelated models.
- It becomes too easy to mix descriptive analytics, forecasting, and front-office heuristics inside one “analytics” bucket.

Invalid reuse risk:

- High. Engineers will reuse methods from this class simply because they are nearby, not because they share assumptions.

Why it makes the math harder to inspect or test:

- There is no clear decomposition by mathematical model family.

Better boundary:

- Split into `ShotQualityModel`, `GoalieModel`, `ScheduleModel`, `TeamStrengthModel`, `PlayoffSimulator`, and `CapAnalysis` modules.

#### P1 - Tool Mixins Hide Cross-Layer Leakage

Files and symbols:

- `src/abba/server/tools/analytics.py`
- `src/abba/server/tools/nhl.py`
- `src/abba/server/tools/market.py`

Why it is bad:

- Tool methods are nominally thin adapters, but they perform caching, feature assembly, data-source inference, confidence construction, and sometimes numerical logic directly.

Correctness risks:

- Business and mathematical rules are split between tool adapters and engine modules.
- Reviewers cannot trust that the engine layer contains the whole algorithm.

Invalid reuse risk:

- Medium-high. New adapters will likely continue smuggling domain rules into “thin” transport code.

Why it makes the math harder to inspect or test:

- Cross-checking an algorithm requires reading both the engine and its tool wrapper.

Better boundary:

- Tool layer should only validate inputs, call a domain service, and serialize outputs.

### Concrete Refactor Proposals

1. Introduce explicit immutable domain models at repository boundaries.
   Examples: `ScheduledGame`, `TeamSeasonStats`, `AdvancedTeamStats`, `GoalieSeasonStats`, `RestProfile`, `OddsBookSnapshot`, `WinProbabilityForecast`, `PredictionProvenance`.

2. Replace mixin-based toolkit composition with explicit services.
   Suggested services: `GameDataService`, `NHLForecastService`, `MarketPricingService`, `WorkflowRunner`, `TelemetryRecorder`.

3. Move all mathematical transforms into pure functions or strategy objects.
   Pure functions for EV, log5, regression-to-mean, xG shot scoring, special-teams scoring, goalie-edge scoring.
   Strategy objects only where algorithm substitution is real: `TeamStrengthModel`, `EnsembleCombiner`, `PlayoffSimulationStrategy`.

4. Separate orchestration from presentation.
   A workflow should return typed intermediate values first.
   Narrative strings, headlines, and “key factors” should be rendered from those values in a separate adapter.

5. Create a provenance-carrying `PredictionContext`.
   It should contain season, data source, refresh timestamp, odds timestamp, and completeness flags.
   Every forecast and confidence object should carry this context.

6. Split `Storage` into narrower repositories plus operational stores.
   `SportsRepository`, `OddsRepository`, `PredictionCache`, `SessionStore`, `ReasoningLogStore`.

7. Add reference implementations for the mathematically significant algorithms.
   Use pure functions or trusted libraries for Elo, graph metrics, and Monte Carlo checks so production implementations can be cross-validated.

8. Make validation explicit at every boundary.
   Validate units, ranges, season alignment, and completeness before constructing feature vectors.

### Do Not Abstract This

- Feature vectors for mathematically significant models should not be represented as anonymous `dict[str, float]` forever. The semantics are too important.
- Probability/confidence/provenance should not be collapsed into one generic `prediction` dict.
- Rest, injury, goalie, and market inputs should not be hidden behind generic “context” blobs.
- Calibration artifacts and model coefficients should not be hidden behind factory/registry indirection unless there is a concrete need for runtime swapping.
- Explanations should not be abstracted into generic “importance” utilities unless they are derived from the actual model.

### Must Isolate This Math Immediately

- NHL game forecast construction.
  Extract `build_nhl_features` and `predict_nhl_game` into a typed, pure forecast module with explicit consumed fields and no toolkit coupling.

- Ensemble aggregation and uncertainty reporting.
  The combiner and interval logic need a dedicated, testable strategy boundary because they are currently mislabeled and mathematically overclaimed.

- Playoff simulation.
  This must be isolated behind a `PlayoffSimulationStrategy` so a reference league-table simulator can replace the threshold heuristic.

- Injury/rest/goalie adjustment logic.
  These adjustments are currently scattered across toolkit helpers, feature builders, and workflows. They need one domain module with explicit assumptions and tests.

- Confidence metadata.
  Prediction confidence, workflow confidence, and presentation confidence are currently mixed. They need separate typed objects or one rigorously defined uncertainty model.

## Correctness and Invariant Audit

The system currently relies on a mix of explicit guards, clipping, defaults, and test conventions. That is not the same thing as invariant enforcement. The key distinction in this section is:

- an invariant the system should satisfy by mathematical or structural necessity
- whether the code encodes it
- whether tests verify it

### Pipeline-Wide Invariants

| Invariant | Why it should hold | Where it can be violated | Enforced in code? | Checked in tests? | Exact assertion / test to add |
|---|---|---|---|---|---|
| Every published numeric result must be finite: no `NaN`, no `inf`. | Agents and downstream math cannot reason safely over non-finite numbers. | `src/abba/engine/ensemble.py` explicit weights can divide by zero; `src/abba/engine/confidence.py` trusts caller-supplied `prediction_value`; dict-based inputs everywhere can carry bad values. | No global enforcement. | No direct test. | For every public tool result, recursively assert `math.isfinite(x)` for each numeric leaf. |
| A single prediction must be computed from one coherent data snapshot: same season, same teams, same game context, contemporaneous odds/provenance. | Otherwise the forecast is mathematically a splice of incompatible states. | `src/abba/server/tools/nhl.py` `nhl_predict_game`; `src/abba/workflows/engine.py` `game_prediction`; `src/abba/storage/duckdb.py` query methods return untyped rows with no snapshot object. | No. | No. | Add a typed `PredictionContext`; assert `home.season == away.season == advanced.season == goalie.season` and odds timestamp freshness before forecasting. |
| If a workflow emits both home and away probabilities, they must sum to `1` up to rounding tolerance. | A two-outcome game forecast is a Bernoulli complement pair. | `src/abba/workflows/engine.py` `tonights_slate`; any future adapters that round independently. | Partial; computed as `1 - pred_val` in some places. | Partial: `home_win_prob` bound is tested in [`tests/test_workflows.py`](/Users/PaxtonEdgar/Projects/abba-nhl/tests/test_workflows.py). | `assert abs(game["home_win_prob"] + game["away_win_prob"] - 1.0) < 1e-9` for every workflow entry. |
| Re-running a pure prediction with the same explicit inputs and the same data snapshot should produce the same result. | Determinism is required for debugging, regression testing, and cross-checking. | `src/abba/workflows/engine.py` always refreshes live data; `src/abba/engine/hockey.py` `playoff_probability` uses fresh RNG every call. | No for workflows and playoff simulation. | Cache behavior is partly tested in [`tests/test_toolkit.py`](/Users/PaxtonEdgar/Projects/abba-nhl/tests/test_toolkit.py). | Expose a seed or RNG parameter for stochastic stages; assert repeated calls with fixed seed and fixed snapshot are exactly equal. |
| The schema of confidence metadata must be consistent across pipeline layers. | Otherwise downstream code cannot interpret uncertainty reliably. | `src/abba/workflows/engine.py` `run` may assign `_confidence` to a scalar `prediction["confidence"]` instead of the metadata dict. | No. | No. | `assert isinstance(result["_confidence"], dict)` and `assert "reliability_grade" in result["_confidence"]` for every workflow. |
| Query methods used as mathematical inputs must resolve ambiguity deterministically. | “Take the first row” is not a mathematically stable selection rule. | `src/abba/storage/duckdb.py` `query_team_stats`, `query_goaltender_stats`, `query_nhl_advanced_stats`; callers index `[0]`. | No `ORDER BY` when season omitted. | No. | If season is omitted and multiple rows exist, raise; otherwise assert exactly one row selected. |

### Stage-Specific Invariants

| Stage / invariant | Why it should hold | Where it can be violated | Enforced in code? | Checked in tests? | Exact assertion / test to add |
|---|---|---|---|---|---|
| `expected_goals`: each per-shot `xg` must satisfy `0 < xg < 1`. | Logistic shot probability is a Bernoulli parameter. | `src/abba/engine/hockey.py` `expected_goals` if coefficients or inputs drift. | Yes, implicitly via logistic transform. | Yes in [`tests/test_hockey.py`](/Users/PaxtonEdgar/Projects/abba-nhl/tests/test_hockey.py). | Keep existing bound test and add randomized finite-input fuzz test. |
| `expected_goals`: `xg_total` must equal the sum of per-shot `xg` values up to rounding tolerance and be order-invariant. | Total expected goals is additive over independent shot events as implemented. | `src/abba/engine/hockey.py` `expected_goals`. | Partial; computed additively, but only by convention. | Sum is tested; order-invariance is not. | Shuffle the same shot list 100 times and assert identical `xg_total`. |
| `expected_goals`: with all else equal, increasing distance or absolute angle must not increase shot probability. | The implemented coefficients are negative in both terms. | `src/abba/engine/hockey.py` `expected_goals` if coefficient config changes. | No explicit assertion. | Partial: angle monotonicity tested, distance calibration sampled. | Parameterized test over monotone distance/angle ladders asserting non-increasing xG. |
| `goaltender_metrics`: `shots_against = saves + goals_against`. | Save percentage and GSAA formulas only make sense under this identity. | `src/abba/engine/hockey.py` `goaltender_metrics` accepts inconsistent inputs silently. | No. | No. | `assert shots_against == saves + goals_against` in code, plus negative test expecting `ValueError` on violation. |
| `goaltender_metrics`: `0 <= save_pct <= 1`, `gaa >= 0`, `0 <= quality_start_pct <= 1`. | These are probability/rate outputs. | `src/abba/engine/hockey.py` `goaltender_metrics` with invalid raw counts. | Partial; formulas produce these only if inputs are valid. | Partly, via happy-path tests. | Add invalid-input tests with negative counts and impossible totals. |
| `goaltender_matchup_edge` should be antisymmetric under role swap: `edge(a,b,g1,g2) = -edge(b,a,g2,g1)`. | The method is based only on pairwise differences. | `src/abba/engine/hockey.py` `goaltender_matchup_edge`. | Implicit by subtraction, not asserted. | No. | Compute both directions and assert `pytest.approx(edge1, abs=1e-9) == -edge2`. |
| `rest_advantage`: identical home/away schedule inputs must yield `rest_edge = 0`, and swapping sides must negate the edge. | The implementation defines a net edge as `away_total - home_total`. | `src/abba/engine/hockey.py` `rest_advantage`. | Partial by formula. | Neutral case tested; antisymmetry not tested in [`tests/test_hockey.py`](/Users/PaxtonEdgar/Projects/abba-nhl/tests/test_hockey.py). | Add swap test: compute `r1` and `r2` with home/away arguments exchanged; assert `r1["rest_edge"] == pytest.approx(-r2["rest_edge"])`. |
| `score_adjusted_corsi`: `0 <= adj_corsi_pct <= 100` and swapping CF/CA should complement to `100` up to rounding. | It is a share statistic derived from adjusted for/against totals. | `src/abba/engine/hockey.py` `score_adjusted_corsi`. | Partial via formula. | Directional tests exist; complement and bounds do not. | Build mirrored state inputs and assert percentages sum to `100 ± 0.1`. |
| `build_nhl_features`: percentage features must stay in `[0,1]`; `market_implied_prob` must be in `[0,1]`; games-played fields must be positive integers. | These are probabilities and counts consumed by downstream models. | `src/abba/engine/hockey.py` `build_nhl_features`. | Partial; defaults and formulas keep some features bounded, but raw input validity is not checked. | Partial in [`tests/test_hockey.py`](/Users/PaxtonEdgar/Projects/abba-nhl/tests/test_hockey.py). | Assert bounds for every feature key; add malformed odds inputs and impossible records as negative tests. |
| `predict_nhl_game`: every returned probability must be in `[0.01, 0.99]` and finite. | The forecast exposes Bernoulli probabilities. | `src/abba/engine/hockey.py` `predict_nhl_game`. | Yes via clipping. | Yes in [`tests/test_hockey.py`](/Users/PaxtonEdgar/Projects/abba-nhl/tests/test_hockey.py). | Add randomized fuzz test over valid feature ranges, asserting all outputs finite and bounded. |
| `predict_nhl_game`: stronger home point percentage, with other features fixed, should not reduce the points-based submodel. | `m1` is intended as a monotone log5 strength function. | `src/abba/engine/hockey.py` `predict_nhl_game`. | No explicit assertion. | No. | Compare outputs under `home_pts_pct = 0.55` vs `0.65` with other inputs fixed; assert `m1_new >= m1_old`. |
| `predict_nhl_game`: if advanced stats are claimed as predictive inputs, perturbing them should change at least one active model output. | Otherwise the feature contract is false. | `src/abba/engine/hockey.py` `build_nhl_features`, `predict_nhl_game`. | No; currently violated because Corsi/xG are unused. | No. | Build two identical feature dicts differing only in `home_corsi_pct/home_xgf_pct`; assert output list differs. This test should fail today. |
| `EnsembleEngine.combine`: result value must stay inside the convex hull of input predictions for average, median, and nonnegative weighted methods. | Convex combination should not overshoot inputs. | `src/abba/engine/ensemble.py` `combine`, `_weighted_combine`. | Yes for current formulas if weights normalize properly. | Not explicitly. | `assert min(preds) <= result.value <= max(preds)` across methods and randomized inputs. |
| `EnsembleEngine.combine`: confidence must be in `[0,1]`, error margin must be `>= 0`, and output must be permutation-invariant when no explicit weights are supplied. | Ensemble statistics should not depend on input ordering. | `src/abba/engine/ensemble.py` `combine`. | Partial. | Bounds partly tested in [`tests/test_engine.py`](/Users/PaxtonEdgar/Projects/abba-nhl/tests/test_engine.py); permutation invariance not tested. | Shuffle predictions repeatedly and assert identical `value`, `confidence`, and `error_margin`. |
| Explicit ensemble weights must sum to a positive finite value. | Normalization by zero produces `NaN`. | `src/abba/engine/ensemble.py` `_weighted_combine`. | No. | No. | `with pytest.raises(ValueError): engine.combine([0.4, 0.6], method="weighted", weights=[0, 0])`. |
| `KellyEngine.calculate`: for valid inputs, `0 <= fraction <= max_bet_pct`, `recommended_stake = bankroll * fraction`, and `fraction = 0` whenever `EV <= 0` or thresholds are not met. | Kelly sizing is a constrained stake policy. | `src/abba/engine/kelly.py` `calculate`. | Yes for happy path. | Mostly yes in [`tests/test_engine.py`](/Users/PaxtonEdgar/Projects/abba-nhl/tests/test_engine.py). | Add parameterized monotonic test over bankroll values and threshold-edge cases. |
| `KellyEngine.calculate`: bankroll must be nonnegative and odds must exceed `1` for a meaningful decimal-odds bet. | Negative bankroll or nonpositive net odds are outside model domain. | `src/abba/engine/kelly.py` `calculate`. | Partial; `b <= 0` returns zeros, bankroll is not validated. | No negative-bankroll test. | Raise `ValueError` on negative bankroll and test it. |
| `EloRatings.predict`: `home_win_prob + away_win_prob = 1`, both in `[0,1]`, and home probability must increase monotonically with home rating. | Standard Elo probability algebra. | `src/abba/engine/elo.py` `predict`, `_win_probability`. | Yes by formula. | Yes in [`tests/test_elo.py`](/Users/PaxtonEdgar/Projects/abba-nhl/tests/test_elo.py). | Add monotonicity property test over a sweep of home ratings. |
| `EloRatings.update`: rating updates must be zero-sum and deterministic for fixed inputs. | Elo is a closed two-team rating transfer. | `src/abba/engine/elo.py` `update`. | Yes by formula. | Yes in [`tests/test_elo.py`](/Users/PaxtonEdgar/Projects/abba-nhl/tests/test_elo.py). | Add repeated-call same-input fixture test to prove determinism exactly. |
| `playoff_probability`: `division_probability` and `wildcard_probability` must be in `[0,1]`, and if `division_cutline <= wildcard_cutline`, then `division_probability >= wildcard_probability`. | They are threshold exceedance probabilities over the same simulated final points. | `src/abba/engine/hockey.py` `playoff_probability`. | Partial by construction; cutline ordering is not asserted. | Probabilities are range-checked indirectly, not by invariant. | Add direct test for ordered cutlines and bound checks. |
| `playoff_probability`: with fixed inputs and fixed RNG seed, results should be reproducible. | Monte Carlo must be reproducible to be audited. | `src/abba/engine/hockey.py` `playoff_probability` uses `default_rng()` without seed. | No. | No. | Refactor signature to accept `rng` or `seed`; assert exact equality under fixed seed. |
| `build_prediction_meta` / `_compute_confidence_interval`: `lower <= point <= upper`, `width = upper - lower`, bounds clipped to `[0,1]`, and worse data quality must not narrow the interval. | The CI is the published uncertainty contract. | `src/abba/engine/confidence.py`. | Yes for most of these by formula. | Yes extensively in [`tests/test_confidence.py`](/Users/PaxtonEdgar/Projects/abba-nhl/tests/test_confidence.py). | Add one test that verifies interval width monotonicity across multiple degradations composed together. |
| Workflow slate ordering: `game_count == len(games)`, `value_pick_count == len(value_picks)`, and output list must be sorted by declared confidence. | Public workflow counts and rankings are part of the contract. | `src/abba/workflows/engine.py` `tonights_slate`. | Partial. | Sorting checked; count equalities not explicitly checked in [`tests/test_workflows.py`](/Users/PaxtonEdgar/Projects/abba-nhl/tests/test_workflows.py). | Add exact count assertions and `best_bet in value_picks` when picks exist. |

### Invariants Missing From Code

- No global finite-number guard exists for published outputs.
- No typed boundary enforces season alignment, unit consistency, or data completeness before feature construction.
- No invariant enforces that all advertised predictive features actually affect the score path.
- No invariant enforces deterministic selection when repository queries return multiple seasons and callers take `[0]`.
- No invariant enforces antisymmetry/complement properties for pairwise comparison functions such as goalie edge and score-share transforms.
- No invariant protects ensemble weighting from zero-sum or non-finite explicit weights.
- No invariant guarantees reproducibility for Monte Carlo playoff simulation.
- No invariant guarantees consistent `_confidence` schema across workflow outputs.

### Invariants Missing From Tests

- No permutation-invariance tests for ensemble combination.
- No anti-symmetry tests for `goaltender_matchup_edge` or `rest_advantage`.
- No complement test for `score_adjusted_corsi`.
- No tests that malformed raw data is rejected at math boundaries.
- No test that advanced NHL features or injury features materially change `predict_nhl_game`.
- No deterministic replay test for stochastic simulation under a fixed seed.
- No test that all public numeric outputs are finite.
- No test that ambiguous multi-season repository queries are rejected or ordered explicitly.
- No test that workflow-level confidence has a stable schema and meaning across workflows.

### Invariants Likely Already Violated By Current Design

- Snapshot coherence is likely violated in the live NHL path because refresh populates only part of the datasets the forecast claims to use.
- Feature-to-model correspondence is already violated: Corsi/xG and injury features are built and surfaced but do not affect `predict_nhl_game`.
- Confidence-schema consistency is already violated in `WorkflowEngine.run`, which may place a scalar model-agreement value into `_confidence` instead of a metadata dict.
- Deterministic behavior is already violated for playoff simulation because each call uses a fresh unseeded RNG.
- Unambiguous season selection is likely violated whenever prediction paths query stats without a season and then consume `[0]` from an unordered result set.
- “Narrative only references active model inputs” is likely violated in workflows that describe rest or other factors not actually consumed by the forecast path.

## Numerical Stability Audit

Scope:

- active engine path under `src/abba/engine/*`, toolkit adapters that participate in the math path, and the parallel analytics stack where numerically relevant code exists
- I did not find direct matrix inversion in the active path; the main issues are unstable reciprocals, ill-conditioned weighting, unsafe exponentials, and stochastic non-reproducibility

### Numerical Risk Register

| Severity | Location | Failure mode | When it appears | Recommended fix | Torture test |
|---|---|---|---|---|---|
| `Critical` | `src/abba/engine/ensemble.py:104-107` | Explicit weights are normalized by `w.sum()` with no check. Zero-sum or near-zero-sum weights produce `NaN`, `inf`, or enormous amplification. | Any caller passes `[0, 0]`, `[1, -1]`, or numerically tiny cancelling weights. | Validate `np.isfinite(w).all()` and `abs(w.sum()) > tol`; reject otherwise. Normalize only nonnegative weights if convex combination is intended. | `engine.combine([0.4, 0.6], method=\"weighted\", weights=[1.0, -1.0])` should raise, not return non-finite output. |
| `Critical` | `src/abba/engine/hockey.py:184-186` | `math.exp(-z)` in the logistic transform can overflow for large negative `z`. Python raises `OverflowError`; if caught upstream, the forecast path can fail hard. | Extremely large distances/angles, bad coefficient configuration, or future feature additions that push `z` below about `-709`. | Replace with a stable sigmoid such as `scipy.special.expit(z)` or a branch-stable implementation using `if z >= 0`. Clamp or validate raw shot inputs. | Feed `distance=1e9`, `angle=1e9` with negative coefficients and assert function returns `0.0 < xg < 1.0` without exception. |
| `High` | `src/abba/engine/features.py:124-128`; `src/abba/engine/hockey.py:930-947` | log5 denominator `pA + pB - 2 pA pB` becomes tiny near extreme or degenerate probabilities. The current `1e-8` fallback creates a discontinuous jump to `0.5`. | Inputs near `0` or `1`, or future code stops clipping upstream probabilities. | Compute in a numerically safer parameterization or clip inputs to `[eps, 1-eps]` before log5. At minimum, use a documented tolerance and a continuous fallback. | Sweep `pA,pB` near `{1e-12, 1-1e-12}` and assert no discontinuous jumps larger than a tolerance between neighboring inputs. |
| `High` | `src/abba/engine/ensemble.py:112-117` | Inverse-distance consensus weights are ill-conditioned near exact agreement. Tiny floating-point differences around the mean produce arbitrarily dominant weights. | Predictions nearly identical, especially around machine epsilon. | If spread `< tol`, return plain mean. Otherwise use a smoother weighting scheme with bounded condition number. | Compare outputs for `[0.6, 0.6, 0.6 + 1e-12]` and `[0.6, 0.6, 0.6 + 1e-15]`; current outputs should not swing materially. |
| `High` | `src/abba/engine/graph.py:171-181` | Eigenvector centrality via dense `eigh(adj)` is numerically unstable when the top eigenvalue is repeated or nearly repeated. Taking `abs()` hides sign flips but not eigenspace non-uniqueness. | Symmetric graphs, disconnected graphs, or nearly degenerate weighted graphs. | Use a Perron-Frobenius-aware power iteration for connected nonnegative graphs, or detect near-degenerate spectra and avoid overinterpreting centrality rankings. | Build two adjacency matrices differing by `1e-12` on one edge in a symmetric graph and assert rankings do not flip wildly without a warning. |
| `High` | `src/abba/engine/elo.py:200-202` | `math.pow(10.0, diff/400.0)` can overflow for absurd rating gaps; for large negative gaps it can underflow to zero, flattening probability. | Corrupted ratings, bad initialization, or reuse with another domain that has larger magnitudes. | Rewrite using a stable logistic form with exponent clipping, for example via `expit((rating_a-rating_b) * ln(10)/400)`. | Call `_win_probability(1e9, -1e9)` and `_win_probability(-1e9, 1e9)` and assert finite values in `[0,1]` without exception. |
| `Medium` | `src/abba/engine/hockey.py:665-697` | Monte Carlo playoff simulation uses a fresh unseeded RNG each call. Output jitter obscures regressions and makes numerical checks irreproducible. | Any repeated call to `playoff_probability` with the same inputs. | Accept `seed` or `rng` as a parameter and record it in outputs/tests. | Run `playoff_probability(...)` 100 times with the same inputs; current variance will expose nondeterminism. After fix, fixed-seed calls must match exactly. |
| `Medium` | `src/abba/engine/confidence.py:221-247` | Confidence interval code trusts `prediction_value` and `calibration_error` blindly. `NaN`, `inf`, or negative widths propagate through clipping and rounding into meaningless metadata. | Upstream non-finite prediction, custom override of `calibration_error`, or malformed external caller. | Validate all numeric inputs with `math.isfinite` and require `0 <= prediction_value <= 1`, `calibration_error >= 0`. | Pass `prediction_value=float(\"nan\")` and assert the builder raises instead of returning `NaN` fields. |
| `Medium` | `src/abba/analytics/advanced_analytics.py:599-600` | `np.polyfit` on short, constant, or badly scaled data can be poorly conditioned and produce noisy slope estimates without any warning handling. | Flat biometric series, outliers, long sequences with large magnitude offsets. | Center and scale inputs before fitting, or use a simpler stable slope estimator on equally spaced data. Catch `RankWarning` if retained. | Trend on `[1e12 + i*1e-6 for i in range(1000)]` should not return numerically nonsensical slopes. |
| `Medium` | `src/abba/analytics/advanced_analytics.py:609-615` | HR fatigue uses `np.std(np.diff(hr_data))` directly. Successive differencing amplifies sensor noise, and the hard normalization by `10.0` makes the score extremely scale-sensitive. | High-frequency noisy sensor feeds or unit changes. | Smooth first or operate on validated RR-interval inputs; estimate scale from data instead of a fixed `10.0`. | Compare the same underlying heart-rate trace sampled at different rates; fatigue should not change drastically. |
| `Low` | `src/abba/engine/hockey.py:196-198` | `xg_total` is accumulated with ordinary float addition. For large shot lists, rounding error can accumulate. | Very large synthetic shot batches or batch-analysis reuse beyond single games. | Use `math.fsum` for the total. | Sum 100,000 tiny xG shots and compare naive accumulation to `math.fsum`; assert bounded relative error after fix. |
| `Low` | `src/abba/engine/hockey.py:868-872`; `src/abba/server/tools/market.py:60-61`; `src/abba/engine/value.py:67-69,86-88,144-145` | Reciprocal odds computations are guarded only by `> 1.0` checks. Extremely near-1 decimal odds produce huge implied probabilities/edges that can magnify noise. | Malformed bookmaker data, parsing bugs, or synthetic tests with `1.0000001` odds. | Enforce a realistic minimum odds floor and reject values too close to `1.0`. | Feed odds `{1.0000001, 1.0000002}` and assert the code rejects or flags them instead of emitting extreme edge metrics. |
| `Low` | `src/abba/analytics/advanced_analytics.py:972-979`; `src/abba/engine/ensemble.py:72-83` | Standard deviation and `std/sqrt(n)` are used as if model predictions were IID samples. This is more a statistical invalidity than floating-point instability, but it produces numerically overconfident margins as `n` grows. | Correlated or duplicated model lists. | Replace with empirically calibrated uncertainty or at least cap the effective sample size. | Duplicate the same prediction 100 times; current margin collapses toward zero. A corrected implementation should not treat clones as 100 independent models. |

### Detailed Notes By Risk

#### `Critical` - Explicit Ensemble Weights Can Produce Non-Finite Output

Location:

- `src/abba/engine/ensemble.py:104-107`

Failure mode:

- `w = w / w.sum()` assumes the sum is finite and not near zero.
- Signed or zero weights make the normalization unstable or undefined.

Suggested fix:

- Require `weights` to be finite and either strictly positive or at least to have `abs(sum) > 1e-12`.
- If negative weights are intentionally allowed, state that explicitly and use a safer normalization/optimization path.

Suggested torture test:

- zero vector
- cancelling vector
- mixed very large and very small weights

#### `Critical` - Unsafe Logistic Transform In xG

Location:

- `src/abba/engine/hockey.py:184-186`

Failure mode:

- `math.exp(-z)` overflows when `-z` is too large.
- The current code assumes shot inputs and coefficient magnitudes will remain benign forever.

Suggested fix:

- Implement:
  - if `z >= 0`: `1 / (1 + exp(-z))`
  - else: `exp(z) / (1 + exp(z))`
- Or use `expit`.

Suggested torture test:

- extreme positive and negative `z`
- coefficient overrides with large magnitudes
- bad shot payloads from external data sources

#### `High` - log5 Uses A Tiny-Denominator Guard That Hides Conditioning Problems

Locations:

- `src/abba/engine/features.py:124-128`
- `src/abba/engine/hockey.py:930-947`

Failure mode:

- Near the singular cases of log5, tiny input perturbations can move the computation from a large finite ratio to the hard fallback `0.5`.
- That is a conditioning problem disguised as a branch guard.

Suggested fix:

- Clip inputs away from exact `0` and `1`.
- Add property tests around singular neighborhoods.
- Consider a logit-domain derivation if these extremes are expected.

#### `High` - Near-Consensus Weighting Is Ill-Conditioned

Location:

- `src/abba/engine/ensemble.py:112-117`

Failure mode:

- The `1e-8` additive floor prevents division by zero but creates a huge condition number when all predictions are almost equal.

Suggested fix:

- If `std(preds) < tol`, return `mean(preds)` exactly.
- Otherwise use bounded robust weights.

#### `High` - Eigenvector Centrality Is Numerically Unstable On Degenerate Graphs

Location:

- `src/abba/engine/graph.py:171-181`

Failure mode:

- For repeated leading eigenvalues, the chosen eigenvector basis is not unique.
- Tiny perturbations can radically change the normalized component vector.

Suggested fix:

- Detect small spectral gaps and downgrade confidence in centrality rankings.
- Prefer power iteration only when graph assumptions justify Perron behavior.

#### `Medium` - Monte Carlo Is Not Numerically Reproducible

Location:

- `src/abba/engine/hockey.py:665-697`

Failure mode:

- The routine is numerically noisy by design, but the absence of a seed prevents reproducible regression testing.

Suggested fix:

- Add `seed: int | None = None` or `rng: np.random.Generator | None = None`.

#### `Medium` - Confidence Interval Builder Accepts Non-Finite Inputs

Location:

- `src/abba/engine/confidence.py:221-247`

Failure mode:

- Non-finite values can leak into user-facing uncertainty metadata and break downstream tooling.

Suggested fix:

- Validate finite inputs at function entry and raise on invalid values.

#### `Medium` - `polyfit` Trend Estimation Is Poorly Conditioned For Some Biometric Series

Location:

- `src/abba/analytics/advanced_analytics.py:599-600`

Failure mode:

- Linear fit on large-offset, low-slope data can lose precision.
- The code also ignores warning signals about rank deficiency.

Suggested fix:

- Center `x` and `data`, or use a stable covariance-based slope formula.

### Suggested Fixes

1. Add a shared numeric validator for all public math entry points.
   Validate finiteness, domain bounds, and tolerance-sensitive denominators.

2. Replace unsafe exponentials with stable logistic/log-sum-exp style implementations.

3. Remove “magic epsilon” weighting where possible.
   If a branch is needed, branch on spread explicitly and return the symmetric mean in the near-consensus case.

4. Make stochastic algorithms reproducible.
   Accept explicit seeds or RNG objects.

5. Add a numerical torture-test suite separate from semantic unit tests.
   Include extreme magnitudes, near-singular inputs, degenerate graphs, duplicate models, and malformed odds.

### Suggested Torture Tests

- `test_xg_sigmoid_extreme_inputs_do_not_overflow`
- `test_log5_near_singularity_is_continuous`
- `test_weighted_ensemble_rejects_zero_sum_weights`
- `test_weighted_ensemble_near_consensus_returns_mean`
- `test_elo_probability_handles_extreme_rating_gaps`
- `test_playoff_probability_fixed_seed_is_reproducible`
- `test_confidence_interval_rejects_nan_inputs`
- `test_eigenvector_centrality_degenerate_graph_warns_or_stabilizes`
- `test_market_probabilities_reject_odds_too_close_to_one`
- `test_large_shot_batch_xg_accumulation_matches_fsum`

## Statistical Rigor Audit

This section evaluates the statistical procedures as statistics, not as code. Several parts of the repo are mathematically tidy but statistically untrustworthy because assumptions are implied, data requirements are not enforced, and validation is either biased or too weak to support the interpretation placed on the outputs.

### Statistical Methods Reviewed

| Procedure | File / symbol | What it is using | Assumptions stated vs implied | Data requirements checked? | Sample size treatment | Missing data handling | Leakage / lookahead risk | Uncertainty quantified? | Interpretation risk |
|---|---|---|---|---|---|---|---|---|---|
| xG shot model | `src/abba/engine/hockey.py` `expected_goals` | fixed-coefficient logistic regression | implied: feature definitions and coefficient provenance match | no | none | defaults for missing shot fields | low in isolation | no fitted uncertainty | output presented as calibrated xG without coefficient artifact provenance |
| Regression to mean | `src/abba/engine/hockey.py` `regress_to_mean` | empirical-Bayes shrinkage with fixed `k=55` | partly stated | only `games_played` numeric use | yes, via shrinkage | none beyond caller defaults | low | no uncertainty on shrunk value | same `k` reused across contexts without estimation |
| NHL prediction ensemble | `src/abba/engine/hockey.py` `predict_nhl_game` | heuristic mixture of log5, pythagorean, linear adjustments, optional market/Elo append | mostly implied | no | only GP shrinkage on some terms | neutral defaults silently substituted | medium; stale/mixed inputs common | no valid statistical uncertainty on final forecast | output treated like a probabilistic model, but several components are unvalidated heuristics |
| Ensemble confidence / interval | `src/abba/engine/ensemble.py` `combine`; `src/abba/engine/confidence.py` | model spread as confidence; hard-coded calibration error as CI width | implied and overstated | no | `sqrt(n)` shrinkage and GP heuristics | none | high; cloned/correlated models treated as independent | yes, but not credibly | confidence is interpreted beyond what disagreement and hard-coded baselines support |
| Playoff probability | `src/abba/engine/hockey.py` `playoff_probability` | Monte Carlo with regressed team talent and fixed OTL split | partly stated | only trivial input presence | yes, via shrinkage | no | medium; time dependence and league competition mishandled | yes, via simulated frequencies | presented as playoff probability despite threshold-only simplification |
| Confidence metadata | `src/abba/engine/confidence.py` `build_prediction_meta`, `build_workflow_meta` | hard-coded historical accuracy and calibration baselines | stated as placeholders, but attached as live metadata | no artifact linkage | GP and freshness heuristics only | caveats, not imputation | high; baseline may not match current model/version | yes, but synthetic | gives a veneer of statistical calibration without versioned empirical support |
| Ensemble training | `src/abba/analytics/advanced_analytics.py` `train_ensemble` | random train/test split + scaler + sklearn `.score()` | mostly implied | no domain checks | no stratification/CV/min-N checks | none | high; random split can leak time/team structure | no proper uncertainty | performance metrics too weak for forecasting claims |
| Personalized model training | `src/abba/analytics/personalization.py` `train_model`, `_prepare_training_data` | supervised classification on user history | implied | no | none | timestamp defaults to `now`, confidence defaults to `0.5` | high; feature leakage via prior confidence and post-hoc user history | none | any personalization benefit claim is unsupported |
| Real-data backtest | `tests/test_backtest.py` | short-window retrospective evaluation with live standings | lookahead bias explicitly stated | partially | minimum `10` games only | missing stats skip games silently | severe; current standings include target outcomes | some metrics reported | report is still too optimistic and overinterpreted |
| Calibration bins | `tests/test_backtest.py` `_calibration_bins` | equal-width binning, weighted avg absolute error | implied | no | weak; bins may be tiny | none | inherits backtest bias | coarse only | “well calibrated” threshold unsupported at this sample size |
| Simulated ROI | `tests/test_backtest.py` `test_simulated_roi` | flat-bet ROI against synthetic implied probabilities | assumptions admitted partially | no real odds | no power analysis | no | severe; synthetic price baseline and selection threshold | no CI in this test | ROI inference is not trustworthy without real closing lines |

### Invalid Or Unsupported Statistical Claims

- `tests/test_backtest.py:18-24` describes the backtest as “honest” and claims it measures calibration, log loss, AUC, and edge over baselines. In reality:
  - AUC is mentioned in the docstring at [`tests/test_backtest.py:22`](/Users/PaxtonEdgar/Projects/abba-nhl/tests/test_backtest.py#L22) but is not computed anywhere in the file.
  - The evaluation uses current standings to predict recent past games, which is explicit lookahead bias at [`tests/test_backtest.py:7-16`](/Users/PaxtonEdgar/Projects/abba-nhl/tests/test_backtest.py#L7) and [`tests/test_backtest.py:94-100`](/Users/PaxtonEdgar/Projects/abba-nhl/tests/test_backtest.py#L94).
  - The report says “Models used: 6 (log5, pyth, corsi, xG, goalie, combined)” at [`tests/test_backtest.py:382`](/Users/PaxtonEdgar/Projects/abba-nhl/tests/test_backtest.py#L382), but the live scoring path does not actually consume Corsi or xG.

- `src/abba/engine/confidence.py:61-77` attaches hard-coded historical calibration and accuracy baselines to live outputs as if they are current model evidence. There is no model-version linkage, no dataset lineage, and no proof the baseline belongs to the runtime path now serving forecasts.

- `src/abba/engine/ensemble.py:75-83` and `src/abba/analytics/advanced_analytics.py:965-979` present disagreement-based margins/confidence as if they were uncertainty estimates. They are not statistically justified because the “samples” are correlated model outputs, not independent draws.

- `src/abba/analytics/advanced_analytics.py:356-385` treats one random train/test split and `.score()` as a sufficient statement about overfitting and model performance. For temporally ordered sports data, that is an unsupported claim.

- `tests/test_backtest.py:404-456` reports a simulated ROI using synthetic implied probabilities (`55%` home baseline) instead of real odds. That does not support any claim about betting profitability.

- `tests/test_backtest.py:314-322` uses a weighted average calibration error threshold `< 0.15` over a tiny biased sample and implies that crossing it means acceptable calibration. That threshold is arbitrary and statistically underpowered.

- `src/abba/engine/hockey.py:638-649` and `src/abba/server/tools/nhl.py:242` present the playoff simulation as playoff probability, but it does not jointly simulate competitor teams or tie-breaks. Statistically, it is a cutline exceedance heuristic, not a league probability model.

- `src/abba/analytics/personalization.py:173-179` maps inferred user risk tolerance into random-forest hyperparameters. There is no statistical argument or validation supporting that mapping.

### Methods That Need Baseline Comparisons

- `HockeyAnalytics.predict_nhl_game`
  - compare against coin flip, home-ice-only, market implied probability, Elo-only, and simple logistic regression on the same features

- `EnsembleEngine.combine`
  - compare against plain mean, median, market-only, and calibrated stacking

- `build_prediction_meta` / `_compute_confidence_interval`
  - compare reported intervals against empirical residual quantiles and conformal-style intervals

- `playoff_probability`
  - compare against a trivial points-pace extrapolation and a proper joint standings simulator

- `AdvancedAnalyticsManager.train_ensemble`
  - compare random split results against time-split CV, grouped CV, and naive baselines

- `PersonalizationEngine`
  - compare personalized model vs non-personalized baseline on calibration, accuracy, and utility metrics

- `test_backtest` evaluation outputs
  - compare against real closing-line probabilities if the repo wants to make any “edge” or betting claims

### Tests / Simulations Needed To Justify Trust In The Statistics

1. A leakage-free historical backtest for NHL predictions.
   Use date-stamped historical standings/features prior to each game, not current standings.
   Report log loss, Brier, calibration curve, and reliability by month.

2. Baseline benchmarking on the same held-out dataset.
   Include coin flip, home-ice prior, Elo, market implied probability, and a simple regularized logistic baseline.

3. Calibration validation with uncertainty.
   Use more than five bins, report bin counts, bootstrap confidence bands, and expected calibration error with uncertainty.

4. Proper temporal validation for any trained model.
   Replace random `train_test_split` with rolling-origin or season-blocked evaluation.

5. Effective-sample-size and correlation checks for ensemble uncertainty.
   Demonstrate that reported margins still cover outcomes when models are highly correlated, or stop reporting them.

6. Missing-data stress tests.
   Quantify forecast degradation when goalie, advanced stats, odds, or roster inputs are absent.
   Right now missingness is mostly converted into neutral defaults without measured bias.

7. Selection-effect tests for value/ROI claims.
   Any threshold-based “edge” scan must be evaluated out of sample with real line data, closing-line comparison, and post-selection uncertainty.

8. Versioned calibration artifacts.
   The confidence module needs a reproducible artifact showing which model version, training window, and holdout set produced `_DEFAULT_ACCURACY_HISTORY` and `_BASE_CALIBRATION_ERROR`.

9. Ablation study for claimed NHL features.
   If Corsi, xG, goalie, rest, injuries, and market data are claimed as predictive inputs, each should be ablated on the same test set to show marginal value.

10. Power-aware evaluation thresholds.
   Replace rules like “need at least 10 games” and “ROI > -30%” with thresholds derived from uncertainty intervals or minimum detectable effect logic.

### Bottom Line

The statistical pipeline is not trustworthy as a calibrated forecasting system.

Reasons:

- the main “real-data” backtest is knowingly contaminated by lookahead bias
- uncertainty metadata is hard-coded and not linked to versioned empirical validation
- several outputs are interpreted as probabilities, intervals, or edges beyond what the implemented methods justify
- the training/evaluation path in the parallel analytics stack is not designed to control leakage, time dependence, or calibration error
- missing data is mostly neutral-filled rather than modeled and measured

Clean code does not rescue this. The repo currently has statistical instrumentation, not statistical evidence.

## Oracle And Reference-Validation Strategy

This section identifies the five computations where the repo most needs a trustworthy oracle. The goal is not speed. The goal is an independent anchor that can tell you whether the production implementation is even in the right neighborhood.

### 1. NHL Win-Probability Forecast

Production implementation:

- `src/abba/engine/hockey.py` `HockeyAnalytics.build_nhl_features`
- `src/abba/engine/hockey.py` `HockeyAnalytics.predict_nhl_game`
- `src/abba/server/tools/nhl.py` `NHLToolsMixin.nhl_predict_game`

Why it is critical:

- This is the central forecasting computation in the repo.
- Downstream EV, Kelly sizing, workflow confidence, and narrative claims all depend on it.

Simplest trustworthy oracle/reference:

- A regularized logistic regression benchmark on a frozen feature set and a leakage-free historical dataset.
- Features should be only those truly available before puck drop and only those actually intended to be used: points%, goal rates, goalie metrics, rest, market implied probability, and optionally Elo.

Why this is the right oracle:

- It is simpler and more statistically defensible than the production heuristic ensemble.
- It produces a calibrated probability from an explicit objective.
- It is easy to cross-check with sklearn/statsmodels and inspect coefficient signs.

How to compare production vs reference:

- Build the same pregame feature matrix for a held-out historical test set.
- Compute:
  - log loss
  - Brier score
  - calibration error
  - decision correlation between production and oracle
- Compare both absolute performance and per-game disagreement.

Acceptable tolerances and why:

- There is no per-game tolerance that proves correctness here; this is not a closed-form computation.
- Use aggregate tolerances:
  - production log loss should be within `0.01` to `0.02` of the logistic baseline or better
  - Brier score should be within `0.005` to `0.01`
  - mean absolute probability difference should be explainable by documented extra features
- If production is materially worse than the simpler baseline, it loses trust.

Input regimes most likely to diverge:

- early season with heavy shrinkage
- games with missing goalie/advanced data
- games where market implied probability disagrees sharply with team-strength heuristics
- extreme favorites and underdogs where additive probability adjustments distort tails

Candidate tests:

- `test_nhl_forecast_matches_logistic_baseline_out_of_sample`
- `test_nhl_forecast_disagreement_is_explained_by_missing_features`
- `test_nhl_forecast_feature_ablation_vs_baseline`

Concrete draft:

```python
def test_nhl_forecast_not_worse_than_logistic_baseline(historical_games):
    X_train, y_train, X_test, y_test = build_pregame_dataset(historical_games)
    ref = LogisticRegression(C=1.0, penalty="l2", max_iter=500)
    ref.fit(X_train, y_train)
    ref_prob = ref.predict_proba(X_test)[:, 1]

    prod_prob = np.array([run_production_nhl_forecast(row) for row in X_test_rows])

    assert log_loss(y_test, prod_prob) <= log_loss(y_test, ref_prob) + 0.02
```

Risk level if no oracle is established:

- `Critical`
- Without an external forecast baseline, the repo cannot tell whether the main predictor is signal or numerically polished folklore.

### 2. Expected Goals Shot Model

Production implementation:

- `src/abba/engine/hockey.py` `HockeyAnalytics.expected_goals`

Why it is critical:

- It is one of the few places claiming a recognizable published statistical model.
- It should be one of the easiest places to establish trust because the formula is explicit.

Simplest trustworthy oracle/reference:

- Closed-form hand calculation for toy shots plus a reference implementation using `scipy.special.expit`.

Why this is the right oracle:

- The model is additive logistic regression at the shot level.
- There is no reason not to test it against direct mathematical evaluation.

How to compare production vs reference:

- For a fixed coefficient set and shot payload, compute `z` manually and compare:
  - per-shot `xg`
  - summed `xg_total`
- Cross-check order invariance and additive accumulation.

Acceptable tolerances and why:

- Per-shot probability tolerance: `1e-12` against a stable sigmoid on unrounded internal values.
- Returned rounded output tolerance: `5e-4` for `shots[i]["xg"]`, `5e-4` to `1e-3` for totals because the production code rounds.

Input regimes most likely to diverge:

- extreme `z` where naive `exp` can overflow
- unknown shot types
- negative or absurd physical inputs from upstream APIs

Candidate tests:

- `test_xg_matches_closed_form_toy_examples`
- `test_xg_matches_expit_reference_randomized`
- `test_xg_is_order_invariant`
- `test_xg_extreme_inputs_match_stable_sigmoid`

Concrete draft:

```python
def test_xg_matches_expit_reference():
    shots = random_valid_shots(1000, seed=42)
    prod = hockey.expected_goals(shots)
    ref_total = 0.0
    for shot in shots:
        z = reference_linear_predictor(shot, hockey.XG_COEFFICIENTS)
        ref_total += expit(z)
    assert abs(prod["xg_total"] - round(ref_total, 3)) <= 1e-3
```

Risk level if no oracle is established:

- `High`
- This is precisely the kind of model that should be easy to validate; failing to do so forfeits confidence in the coefficient contract.

### 3. Elo Win Probability And Rating Update

Production implementation:

- `src/abba/engine/elo.py` `EloRatings._win_probability`
- `src/abba/engine/elo.py` `EloRatings.update`
- `src/abba/engine/elo.py` `EloRatings._margin_of_victory_multiplier`

Why it is critical:

- Elo is the only method in the repo that is close to a recognized dynamic rating model.
- It is also appended directly into the NHL prediction stack.

Simplest trustworthy oracle/reference:

- Closed-form oracle for the vanilla Elo equations.
- Hand-worked toy examples for zero-sum updates and season reversion.
- Optional independent reference implementation in a short test helper.

Why this is the right oracle:

- The core formula is simple enough to compute exactly for toy cases.
- The only ambiguous part is the MOV multiplier, which can still be validated against its written formula.

How to compare production vs reference:

- Check:
  - exact probability from the closed-form logistic
  - exact update shift from `K * MOV * (actual - expected)`
  - exact zero-sum conservation
  - exact season reset toward `1500`

Acceptable tolerances and why:

- `1e-12` for probability and shift on toy inputs because the formulas are closed form and low dimensional.

Input regimes most likely to diverge:

- extreme rating differences
- one-goal and tie-era edge cases in MOV logic
- season transitions

Candidate tests:

- `test_elo_probability_matches_closed_form`
- `test_elo_update_matches_hand_worked_example`
- `test_elo_update_zero_sum_exactly`
- `test_elo_season_reset_matches_formula`

Concrete draft:

```python
def test_elo_update_matches_reference():
    elo = EloRatings(k=4, home_advantage=50, initial_rating=1500)
    out = elo.update("A", "B", 4, 2)

    expected_home = 1 / (1 + 10 ** ((1500 - 1550) / 400))
    mov = max(math.log(3) * (2.2 / (0.0 * 0.001 + 2.2)), 1.0)
    shift = 4 * mov * (1.0 - expected_home)

    assert abs(out["shift"] - shift) < 1e-12
    assert abs((out["home_post"] - 1500) + (out["away_post"] - 1500)) < 1e-12
```

Risk level if no oracle is established:

- `Medium-High`
- The method is simple enough that not having a reference check would be an avoidable trust failure.

### 4. Ensemble Aggregation And Reported Uncertainty

Production implementation:

- `src/abba/engine/ensemble.py` `EnsembleEngine.combine`
- `src/abba/engine/ensemble.py` `EnsembleEngine._weighted_combine`
- `src/abba/engine/confidence.py` `_compute_confidence_interval`

Why it is critical:

- This layer converts several model outputs into the single probability and uncertainty that users actually see.
- It is currently mislabeled and statistically overclaimed.

Simplest trustworthy oracle/reference:

- Two references are needed:
  - deterministic oracle for pure aggregation behavior: mean/median/explicit-weight convex combination
  - simulation benchmark for uncertainty: historical residual calibration or conformal-style empirical coverage

Why this is the right oracle:

- The aggregation itself can be checked exactly.
- The uncertainty cannot be justified analytically from model disagreement alone; it needs empirical calibration.

How to compare production vs reference:

- For aggregation:
  - compare weighted combine to exact convex combination when explicit weights are provided
  - compare near-consensus cases to the arithmetic mean
- For uncertainty:
  - on held-out historical games, compare reported intervals against actual empirical coverage

Acceptable tolerances and why:

- Aggregation tolerance: `1e-12` for explicit weights.
- Coverage tolerance: if claiming a nominal `80%` interval, empirical coverage should land within roughly `75%-85%` over a sufficiently large holdout. Exact tolerance depends on sample size; use binomial confidence bands.

Input regimes most likely to diverge:

- nearly identical predictions
- duplicated or highly correlated model outputs
- explicit bad weights
- small model counts where `std/sqrt(n)` is most misleading

Candidate tests:

- `test_explicit_weighted_combine_matches_closed_form`
- `test_near_consensus_weighted_combine_matches_mean`
- `test_reported_confidence_interval_has_nominal_empirical_coverage`
- `test_duplicate_models_do_not_artificially_shrink_uncertainty`

Concrete draft:

```python
def test_reported_80_interval_has_empirical_coverage(backtest_rows):
    covered = 0
    total = 0
    for row in backtest_rows:
        pred = run_prediction_with_meta(row)
        ci = pred["confidence"]["confidence_interval"]
        y = 1.0 if row.home_won else 0.0
        covered += ci["lower"] <= y <= ci["upper"]
        total += 1
    coverage = covered / total
    assert 0.75 <= coverage <= 0.85
```

Risk level if no oracle is established:

- `Critical`
- This is where the repo currently laundered disagreement into “confidence.” Without empirical reference checks, the uncertainty layer should not be trusted.

### 5. Playoff Probability Simulation

Production implementation:

- `src/abba/engine/hockey.py` `HockeyAnalytics.playoff_probability`

Why it is critical:

- It is the repo’s only stochastic forecasting component and it outputs decision-shaping probabilities.
- It is also structurally prone to false trust because Monte Carlo output looks authoritative.

Simplest trustworthy oracle/reference:

- Two-stage reference:
  - hand-worked toy cases with one or two games remaining where exact enumeration is possible
  - a brute-force or higher-fidelity joint league-table simulator for small synthetic standings scenarios

Why this is the right oracle:

- For small horizons, exact enumeration gives a clean ground truth.
- For realistic horizons, a joint simulator is the right structural benchmark, not the current threshold-only shortcut.

How to compare production vs reference:

- For toy cases:
  - compute exact probabilities by enumerating all remaining game outcomes
  - compare the production output to the exact threshold exceedance probability
- For league realism:
  - compare current threshold heuristic to a joint competitor simulator on synthetic standings tables

Acceptable tolerances and why:

- Exact-enumeration toy cases: Monte Carlo with fixed seed and enough draws should be within `0.01` absolute error.
- Large-sample simulation benchmark: `0.02` to `0.03` absolute error on probabilities may be acceptable if the same event definition is used.

Input regimes most likely to diverge:

- bubble teams near the cutline
- small games-remaining scenarios where exact structure matters most
- cases with asymmetric opponent schedules
- seasons where overtime-loss behavior differs from the fixed `25%` assumption

Candidate tests:

- `test_playoff_probability_matches_exact_enumeration_small_horizon`
- `test_playoff_probability_matches_schedule_aware_exact_case`
- `test_playoff_probability_differs_from_joint_standings_oracle_on_bubble_cases`

Concrete draft:

```python
def test_playoff_probability_matches_exact_two_game_case():
    # Exact enumeration for 2 remaining games, threshold event on final points
    exact = exact_cutline_probability(
        current_points=92,
        game_win_probs=[0.6, 0.4],
        otl_share=0.25,
        cutoff=95,
    )
    out = hockey.playoff_probability(
        current_points=92,
        games_remaining=2,
        games_played=80,
        wildcard_cutline=95,
        opponent_win_probs=[0.6, 0.4],
    )
    assert abs(out["wildcard_probability"] - exact) <= 0.01
```

Risk level if no oracle is established:

- `High`
- Without exact toy checks and a joint-simulation benchmark, this Monte Carlo output will keep looking more authoritative than it is.

## End-to-End Trust Audit

Question answered:

> Under what conditions should a rational engineer trust the outputs of this tool?

Short answer:

- Only under narrow, explicitly constrained conditions.
- In its current state, the system is fit for exploratory, human-supervised analysis.
- It is not fit for unattended or correctness-critical decision support where users might interpret outputs as empirically validated probabilities, calibrated uncertainty, or trustworthy betting edges.

### Trustworthiness Scorecard

| Pipeline area | Score | Judgment | Why |
|---|---:|---|---|
| Data provenance | `3/10` | weak | live vs seed vs stale inputs are mixed; provenance is surfaced inconsistently and often inferred rather than carried explicitly |
| Parsing correctness | `6/10` | moderate | most raw computations are straightforward, but dict/JSON boundaries make schema drift and silent field mismatch easy |
| Unit handling | `3/10` | weak | units and scales are mostly implicit; no typed enforcement for rates, percentages, cumulative totals, or season alignment |
| Algorithm selection | `4/10` | weak | some recognizable methods exist, but the main NHL path is dominated by heuristics presented as model logic |
| Mathematical validity | `4/10` | weak | several formulas are defensible in isolation, but the assembled system overclaims what the math supports |
| Numerical robustness | `5/10` | mixed | not catastrophically fragile in normal ranges, but there are real risks around unsafe exponentials, ill-conditioned weighting, and non-finite outputs |
| Test quality | `5/10` | mixed | there are many tests, but they over-index on shape/range/happy-path behavior and under-index on leakage, invariants, and empirical validation |
| Determinism / reproducibility | `4/10` | weak | caching helps some paths, but workflows refresh live data and playoff simulation is unseeded |
| Logging / traceability | `6/10` | moderate | tool-call logging exists, but mathematical traceability is weak because inputs/provenance/versions are not captured as first-class forecast artifacts |
| Error handling | `5/10` | mixed | obvious missing-data cases often return errors, but many degraded states silently fall back to neutral defaults |
| Silent degradation risk | `2/10` | severe | this is one of the biggest problems: missing features, stale data, mixed seasons, and inactive feature claims often degrade outputs without stopping the pipeline |
| Overall end-to-end trustworthiness | `4/10` | not trustworthy for decision-critical use | the system can produce plausible structured output, but plausibility currently exceeds evidence |

### Conditions Under Which Outputs Are Trustworthy

A rational engineer should trust outputs only when all of the following are true:

1. The output is used as exploratory analysis, not as authoritative truth.
   The user must understand that several headline quantities are heuristics, not validated estimators.

2. The computation is one of the simpler, locally checkable methods.
   Examples:
   - EV calculation
   - Kelly sizing for a supplied probability and odds
   - Elo probability/update
   - xG shot scoring with known coefficients

3. Inputs are manually verified to be coherent.
   That means:
   - same season
   - same teams/game
   - no seed data unless explicitly intended
   - no silent fallback to missing goalie/advanced/odds inputs

4. The result is interpreted qualitatively, not as calibrated uncertainty.
   “leans home” may be acceptable.
   “57% with trustworthy 80% confidence interval” is not, unless the interval machinery is independently revalidated.

5. A human can inspect the contributing features and detect obvious nonsense.
   The system is more trustworthy when used as a structured calculator than when used as an autonomous recommender.

6. For stochastic outputs, a fixed seed or reproducible snapshot is used.
   Without that, debugging and comparison are unreliable.

7. For any serious use, production is cross-checked against a simpler reference implementation or baseline.

### Conditions Under Which Outputs Are Untrustworthy

Outputs should be treated as untrustworthy when any of the following holds:

1. The user interprets the NHL prediction as a fully validated probabilistic model.
   It is not.

2. The output depends on the live NHL workflow path and the user assumes all claimed features were actually ingested and used.
   They often are not.

3. Confidence intervals, reliability grades, or workflow confidence scores are used as statistical evidence.
   They are not empirically anchored to the runtime model surface.

4. The system is used for betting or financial decisions without an external baseline and leakage-free backtest.

5. The input data may span multiple seasons or stale/live/seed mixtures.

6. Missing goalie, advanced, roster, or odds data is present and the user assumes the model degrades gracefully in a measured way.
   It usually degrades silently via neutral defaults.

7. The playoff-probability output is interpreted as a real league-table probability rather than a threshold heuristic.

8. The engineer needs reproducibility across runs and the path includes live refreshes or unseeded simulation.

9. The engineer expects workflow narratives to reflect the actual mathematical drivers of the prediction.
   Workflows often narrate factors that are not active in the score path.

10. The system is being trusted because it “looks structured” or “passed tests.”
   The test suite does not establish end-to-end correctness.

### Top 10 Reasons The Current System Could Return Plausible But Wrong Results

1. The live NHL prediction path can combine fresh standings with stale or absent advanced stats, goalie stats, odds, and roster effects.

2. The system claims to use Corsi, xG, injury, and rest-aware reasoning, but several of those features do not actually affect the active scoring function.

3. Repository queries return unordered rows when season is omitted, and callers often consume the first row as if it were canonical.

4. Confidence metadata is derived from hard-coded historical baselines, not from versioned validation of the current runtime predictor.

5. Workflow-level confidence is semantically inconsistent with prediction-level confidence and may collapse to a model-agreement scalar.

6. Ensemble disagreement is treated like uncertainty, which can make a set of correlated wrong models look falsely trustworthy.

7. The real-data backtest is contaminated by lookahead bias, so its apparent success can be optimistic in exactly the cases users care about.

8. Missing inputs are frequently neutral-filled rather than rejected, so the output remains plausible even when the intended model is not actually running.

9. Playoff simulation outputs look precise but are built on fixed cutlines, fixed overtime-loss assumptions, and no joint competitor simulation.

10. The toolkit/workflow abstraction hides enough state and glue logic that a mathematically important change in provenance, cache state, or refresh behavior may not be visible in the final output.

### Minimal Changes Required Before Results Should Be Relied Upon

These are the minimum changes, not the ideal architecture:

1. Make data provenance explicit and enforceable.
   Every forecast must carry season, source, refresh timestamp, missing-feature flags, and odds timestamp as first-class fields.

2. Remove or fix false feature claims.
   Either wire Corsi/xG/injury/rest into the active forecast path or stop claiming they are used.

3. Enforce coherent snapshot selection.
   No unordered multi-season queries for mathematical inputs.
   If season is ambiguous, fail fast.

4. Establish one leakage-free historical backtest for the active NHL predictor.
   Use pregame historical data only.
   Report log loss, Brier, calibration, and baseline comparisons.

5. Replace hard-coded confidence metadata with versioned empirical calibration artifacts.
   If that cannot be done, stop presenting reliability grades and intervals as if validated.

6. Add oracle/reference tests for the top critical computations.
   At minimum:
   - NHL forecast baseline
   - xG
   - Elo
   - ensemble aggregation/coverage
   - playoff simulation toy cases

7. Make stochastic paths reproducible.
   Seed playoff simulation and capture the seed in outputs/tests.

8. Add invariant tests that fail when surfaced features do not affect model outputs.

9. Split orchestration from numerical logic enough that workflows cannot silently reinterpret model outputs.

10. Add a hard failure mode for materially incomplete inputs instead of silently substituting neutral defaults.

### Final Trust Answer

A rational engineer should trust this system only as a supervised exploratory analytics tool, and only when:

- the input snapshot is verified coherent
- the specific computation is locally understandable
- uncertainty claims are ignored unless independently revalidated
- outputs are cross-checked against a simpler baseline or oracle

A rational engineer should not trust it for autonomous, correctness-critical, or money-moving decisions in its current state.

The system often produces outputs that are structurally plausible, numerically tidy, and narratively confident. That is not the same thing as being trustworthy.

## Library-vs-Custom-Implementation Pass

### Wheel-Reinvention Report

| Rank | Severity | File / symbol | Problem being solved | Likely replacement candidates | Classification | Why this is reinvention or justified |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Critical | `src/abba/analytics/graph.py::GraphAnalyzer` | Graph centrality, clustering, cohesion, key-player ranking | `networkx`, `igraph`, or at minimum `scipy` + `networkx` oracle tests | Dangerous reinvention | The file explicitly substitutes degree-based proxies for closeness, betweenness, and eigenvector centrality, while still naming them as standard graph metrics. Correctness risk is extreme, maintenance burden is high, performance is not good enough to justify the approximation, and dependency cost is low relative to the epistemic damage. |
| 2 | High | `src/abba/utils/validation_utils.py::ValidationUtils` | Input validation for probabilities, odds, dates, configs, predictions, and tabular data | `pydantic` models, `pandera` for DataFrame contracts, stdlib `datetime` validators | Probably should use mature third-party library | This repo already uses `pydantic` in `src/abba/core/config.py::Config`, but core domain validation is still ad hoc booleans and dicts. The current helper erases field semantics, cannot enforce typed invariants, and makes incorrect reuse easy. |
| 3 | High | `src/abba/server/http.py::{ABBAHandler, run_http_server}` | HTTP routing, request parsing, response serialization, CORS, endpoint dispatch | `FastAPI`, `Starlette`, `pydantic` request/response models | Probably should use mature third-party library | The code hand-rolls routing and JSON/body parsing on top of `BaseHTTPRequestHandler`. Correctness risk is moderate to high because invalid payloads collapse into weakly typed dicts, error semantics are thin, and interface contracts are not machine-checked. Maintenance burden is higher than the dependency cost. |
| 4 | High | `src/abba/connectors/live.py::{NHLLiveConnector._fetch_json, OddsLiveConnector.refresh, MLBLiveConnector._fetch_json}` | HTTP client behavior, timeout handling, JSON fetch, transient failure handling | `httpx` or `requests`, `tenacity`/`backoff`, typed response models | Probably should use mature third-party library | This is a solved problem. The current code repeats manual `urllib` request setup, timeout handling, and JSON decoding, with no retry policy, no structured error taxonomy, and no response-schema validation. Performance gains from connection reuse alone would likely offset the dependency. |
| 5 | High | `src/abba/utils/data_utils.py::DataUtils` | Cleaning, imputation, normalization, categorical encoding, train/test split | `sklearn` `Pipeline`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`; direct `pandas` ops for explicit transforms | Probably should use mature third-party library | The file wraps standard preprocessing steps in generic helpers that silently choose means, `"Unknown"`, z-score scaling, and random train/test splitting. That increases correctness risk more than it reduces complexity. The abstraction hides assumptions that `sklearn` pipelines would make explicit and testable. |
| 6 | High | `src/abba/storage/duckdb.py::{upsert_*, query_*, cache_prediction, log_tool_call, log_reasoning}` | Serialization of domain objects, cache persistence, replay/log records | `pydantic` or `dataclasses` for schema, typed JSON encoders, `cachetools` or stdlib caching only where process-local cache is intended | Dangerous reinvention | Using DuckDB directly is fine. Using raw dicts plus repeated `json.dumps`/`json.loads` as the main domain boundary is not. Units, season identity, optionality, and schema evolution are all implicit, which creates silent garbage states that typed serializers would catch. |
| 7 | Medium | `src/abba/connectors/live.py` and `src/abba/utils/validation_utils.py::validate_date_range` | Date/time handling, freshness windows, season identity | stdlib `datetime` with timezone-aware values, `zoneinfo`, `pydantic` datetime parsing | Probably should use standard library | The code mixes `datetime.now()`, `date.today()`, pandas parsing, and hard-coded season strings. This is not a library-sized problem; it is a standard-library problem currently handled poorly. Timezone-naive timestamps and fixed season strings are a direct provenance risk. |
| 8 | Medium | `src/abba/engine/graph.py::GraphEngine` | Graph metrics in the active engine path | Keep `scipy`, cross-check against `networkx` reference implementations | Justified custom code, but only with oracle cross-checks | This file is materially better than `analytics/graph.py`: it already uses `scipy.sparse.csgraph.shortest_path` and `scipy.linalg.eigh`. The remaining custom code still carries correctness risk for betweenness and clustering, but this is a legitimate case for slim custom implementation plus reference tests, not an automatic rewrite. |

### Ranked Replacement Opportunities

#### 1. Remove or quarantine `src/abba/analytics/graph.py::GraphAnalyzer`

Call chain:
`src/abba/analytics/advanced_analytics.py::AdvancedAnalyticsManager.graph_network_analysis`
-> `src/abba/analytics/graph.py::GraphAnalyzer.build_graph`
-> `GraphAnalyzer._calculate_centrality`
-> `_calculate_closeness_centrality` / `_calculate_betweenness_centrality` / `_calculate_eigenvector_centrality`

This is the clearest dangerous reinvention in the repo. The code says "closeness centrality" and then computes normalized degree. It says "betweenness centrality" and then computes `degree * (1 - clustering)`. It says "eigenvector centrality" and then reuses degree again. Mature libraries already solve this, document edge-case behavior, and provide reference semantics. Keeping this implementation invites false scientific claims because the names imply standard methods while the code does not implement them.

Replacement guidance:
- Replace with `networkx` if this analytics path remains user-facing.
- If performance later matters, keep a faster engine only after proving equivalence against `networkx` on representative graphs.
- If this path is dead or experimental, remove it rather than preserving mathematically mislabeled code.

#### 2. Replace ad hoc validation with typed domain models

Relevant symbols:
- `src/abba/utils/validation_utils.py::ValidationUtils`
- `src/abba/storage/duckdb.py::{upsert_games, upsert_team_stats, upsert_goaltender_stats, upsert_roster, cache_prediction}`
- `src/abba/server/http.py::ABBAHandler._read_body`

The repo already accepted the dependency cost of `pydantic` in `src/abba/core/config.py::Config` but stopped at configuration. The important boundaries still move `dict[str, Any]` payloads with booleans like `validate_probability()` and post hoc JSON parsing. That is the worst of both worlds: dependency cost is already paid, but correctness benefits are not captured where the math enters the system.

Replacement guidance:
- Use explicit `pydantic` models for game snapshots, team stats, goalie stats, roster records, odds rows, and prediction outputs.
- Add `pandera` only if DataFrame-heavy validation remains in scope; otherwise keep DataFrame use out of correctness-critical boundaries.
- Reject malformed or unit-incoherent inputs before storage, not after retrieval.

#### 3. Replace the hand-rolled HTTP surface

Call chain:
`src/abba/server/http.py::ABBAHandler.do_POST`
-> `ABBAHandler._read_body`
-> `src/abba/server/toolkit.py::ABBAToolkit.call_tool`

This layer is framework-shaped code without framework guarantees. It manually parses JSON, does dynamic route dispatch with string slicing, and emits whatever `json.dumps(default=str)` happens to serialize. A mature ASGI stack would reduce custom code, improve error reporting, and let request/response models carry actual domain constraints.

Replacement guidance:
- Migrate to `FastAPI` or `Starlette`.
- Define request and response models with `pydantic`.
- Let the server layer stay thin: parse, validate, dispatch, serialize. Nothing more.

#### 4. Replace `urllib` fetch glue with a real HTTP client stack

Relevant symbols:
- `src/abba/connectors/live.py::NHLLiveConnector._fetch_json`
- `src/abba/connectors/live.py::OddsLiveConnector.refresh`
- `src/abba/connectors/live.py::MLBLiveConnector._fetch_json`

The current connector layer solves retries, transport errors, headers, timeouts, and JSON decoding manually and inconsistently. That adds maintenance burden without buying transparency. Worse, there is no boundary validation after parsing, so malformed upstream payloads degrade into default-filled dicts.

Replacement guidance:
- Use `httpx.Client` or `httpx.AsyncClient` with per-host timeout policy.
- Add retry/backoff via `tenacity` for transient network and 5xx failures.
- Parse responses into typed models before persistence.
- Capture timezone-aware fetch timestamps in UTC.

#### 5. Replace generic preprocessing helpers with explicit pipeline components

Relevant symbols:
- `src/abba/utils/data_utils.py::clean_dataframe`
- `src/abba/utils/data_utils.py::normalize_features`
- `src/abba/utils/data_utils.py::encode_categorical`
- `src/abba/utils/data_utils.py::split_train_test`

This file repackages common `pandas` and `sklearn` steps into utility methods that hide assumptions instead of naming them. Mean-imputing every numeric column, filling every categorical with `"Unknown"`, and doing random train/test splits are not neutral defaults in a time-dependent forecasting system.

Replacement guidance:
- Use `sklearn` `Pipeline` and `ColumnTransformer` so preprocessing is versioned and testable.
- Use time-aware splitters instead of random `train_test_split` where forecasting is involved.
- Keep one-off `pandas` transforms inline when they are simple and domain-specific; do not hide them behind a generic utility class.

#### 6. Replace dict-and-JSON persistence boundaries with typed records

Relevant symbols:
- `src/abba/storage/duckdb.py::Storage`
- repeated `json.dumps(...)` on insert paths
- repeated `json.loads(...)` on query paths

The DuckDB wrapper itself is justified. The typed-domain erasure is not. The code stores rich mathematical inputs such as team stats, goalie stats, uncertainty metadata, and reasoning context as untyped JSON blobs, then reconstructs them piecemeal on read. That makes schema drift invisible and prevents unit-aware validation.

Replacement guidance:
- Keep DuckDB.
- Replace dict blobs with typed records at the application boundary.
- If JSON columns remain, centralize serialization through one schema layer instead of open-coded `json.dumps`/`json.loads` per method.

### Cases Where Custom Code Is Actually Correct To Keep

| File / symbol | Why keeping custom code is justified | Conditions |
| --- | --- | --- |
| `src/abba/engine/elo.py::EloRatings` | Elo is a compact domain formula, the implementation is readable, and a heavy library would add little value over a well-tested custom implementation. | Keep only with oracle tests against hand-computed games and known Elo examples. |
| `src/abba/engine/kelly.py::KellyEngine` | Kelly sizing is a small closed-form computation. Library replacement would not materially improve transparency. | Keep only if all clipping, caps, and house rules are documented as policy rather than mathematics. |
| `src/abba/engine/value.py::ValueEngine` | EV and implied-probability arithmetic are simple enough that direct code is clearer than another dependency. | Keep only if probability bounds and odds conventions are validated at the boundary. |
| `src/abba/storage/duckdb.py::Storage` as a DuckDB wrapper | A repository layer over DuckDB is application-specific glue, not wheel reinvention by itself. | Keep the wrapper, but stop using raw dict/JSON payloads as the domain model. |
| `src/abba/engine/graph.py::GraphEngine` | It already delegates the hardest numerical pieces to `scipy`, which is the right direction. | Keep only if `networkx` or equivalent remains the oracle for correctness tests. |

### Highest-Risk Places Where Hand-Rolled Math Should Be Removed Or Cross-Checked Against A Library

1. `src/abba/analytics/graph.py::GraphAnalyzer`
   Remove or replace. The metric names are standard; the math is not.

2. `src/abba/engine/graph.py::{_betweenness_centrality, _clustering_coefficient}`
   Cross-check against `networkx.betweenness_centrality` and `networkx.clustering` on toy graphs, disconnected graphs, dense graphs, and weighted graphs.

3. `src/abba/utils/data_utils.py::{normalize_features, split_train_test}`
   Cross-check against `sklearn.preprocessing.StandardScaler` and time-aware splitting utilities. The danger is not numerical failure alone; it is silently invalid statistical procedure.

4. `src/abba/utils/validation_utils.py::validate_model_prediction`
   Replace with typed prediction schemas plus numeric validators. The current helper checks shape-level facts but has no notion of class ordering, units, or semantic completeness.

5. `src/abba/connectors/live.py::{_fetch_json, refresh}`
   Cross-check parsed payloads against typed schemas and HTTP-client retry semantics. The problem is not that `urllib` cannot fetch bytes; the problem is that transport, parsing, and validation are collapsed into one fragile custom layer.

### Bottom Line

This repo is not suffering from "too many dependencies." It is suffering from selective refusal to use mature libraries in exactly the places where mature libraries exist to protect correctness:

- graph algorithms
- schema validation
- HTTP client/server boundaries
- preprocessing pipelines
- time handling

The right posture is not "replace all custom code." The right posture is:

- keep compact domain formulas that are easy to inspect
- remove custom infrastructure and mislabeled mathematical approximations
- force every mathematically sensitive boundary through typed validation or a reference implementation

## Python Usage Quality Pass

### Python Construct Audit

| Rank | Severity | File / symbol | Exact construct | Problem type | What is wrong with the current tradeoff | More appropriate construct |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | High | `src/abba/engine/graph.py::_betweenness_centrality` | `queue = [s]` then `queue.pop(0)` inside BFS | Performance, clarity | This is list syntax used as a FIFO queue. `pop(0)` is O(n), so the Brandes traversal quietly pays repeated shifting costs inside an already expensive graph algorithm. It is Pythonic in appearance and wrong for the workload. | `collections.deque` with `popleft()` |
| 2 | High | `src/abba/storage/duckdb.py::query_session_replay` | `fetchdf()` twice, `pd.concat([tools, reasoning])`, `sort_values("ts").head(limit)` | Performance, maintainability | The function eagerly loads two full tables into pandas, concatenates them, sorts them in Python, then truncates. The limit is applied after materialization, so long sessions pay the full memory cost anyway. | Push the merge/sort/limit into SQL with `UNION ALL ... ORDER BY ts LIMIT ?`; only materialize final rows |
| 3 | High | `src/abba/storage/duckdb.py::{query_games, query_odds, query_reasoning_log, query_team_stats, ...}` | `fetchdf().to_dict("records")` on nearly every read path | Performance, clarity | The code repeatedly materializes a DataFrame only to convert it back to Python dicts. That adds an unnecessary columnar->DataFrame->dict conversion step and obscures the real boundary shape. | Use `fetchall()`/row mappings when Python records are the real output; reserve pandas for actual tabular computation |
| 4 | High | `src/abba/utils/data_utils.py::encode_categorical` | Looping columns and doing `pd.concat([df_encoded, dummies], axis=1)` each pass | Performance, maintainability | Repeated DataFrame concatenation inside a loop causes repeated whole-frame copies. That is a hidden quadratic pattern in the number of encoded columns. | `pd.get_dummies(df, columns=columns)` once, or build one list of frames and concatenate once |
| 5 | High | `src/abba/utils/data_utils.py::{clean_dataframe, normalize_features, create_time_features, calculate_rolling_stats, encode_categorical, split_train_test}` | Broad `except Exception` followed by `return df` or `(None, None, None, None)` | Correctness, maintainability | These helpers preserve outward progress by silently returning original or empty values after a transformation failure. That makes data/math errors look like successful no-op processing. | Narrow exceptions; fail fast for invariant violations; return explicit result types only when degradation is intentional and surfaced |
| 6 | High | `src/abba/utils/validation_utils.py::*` | Broad `except Exception` returning `False` or a generic invalid dict | Correctness, clarity | Validation failures and validator bugs are collapsed into the same output. The caller cannot tell whether data is invalid or the validation logic crashed. | Raise structured validation errors or return typed error objects with explicit failure origin |
| 7 | Medium | `src/abba/analytics/graph.py::{build_graph, analyze_connections}` | `adjacency_matrix.tolist()` then later `np.array(graph.get("adjacency_matrix", []))` | Performance, composability | The graph path serializes a numeric matrix to nested Python lists and immediately rehydrates it into NumPy. That is a full copy in both directions with no mathematical benefit. | Keep the matrix as `np.ndarray` internally; serialize only at the outer API boundary |
| 8 | Medium | `src/abba/workflows/engine.py::run` | `len([k for k in result if not k.startswith("_")])` computed twice | Clarity, minor performance | Two identical list comprehensions allocate the same list solely to count it. The code is short but needlessly eager and duplicates logic. | Compute once with `steps = sum(1 for k in result if not k.startswith("_"))` |
| 9 | Medium | `src/abba/server/toolkit.py::_player_impact` | `healthy = [p for p in roster if p.get("injury_status", "healthy") == "healthy"]`; `goalies = [p for p in roster if p.get("position") == "G"]`; `any(... for g in goalies[:1])` | Performance, clarity | The function materializes lists that it only counts or peeks into once. The goalie path is especially awkward: a full filtered list is built only to slice the first element and feed it to `any()`. | Use `sum(...)` for counts and `next((p for p in roster if ...), None)` for the first goalie |
| 10 | Medium | `src/abba/utils/data_utils.py::calculate_rolling_stats` | Two separate groupby/rolling passes with `reset_index()` and assignment | Performance, maintainability | Mean and std are computed in separate passes and index-reset glue is used to stitch them back together. This is more memory-heavy and easier to misalign than a single grouped aggregation. | Single grouped rolling `agg(["mean", "std"])` and join once |
| 11 | Medium | `src/abba/server/tools/reasoning.py::session_replay` | `low_trust = [d for d in e["data_trust"] if ...]` before `extend` | Minor performance, clarity | This intermediate list exists only to be extended into another list once. Small cost, but it is exactly the kind of unnecessary materialization repeated across the repo. | `all_trust_issues.extend(d for d in e["data_trust"] if ...)` |
| 12 | Medium | `src/abba/workflows/engine.py::tonights_slate` | building both `slate` and `value_picks` imperatively while also mutating `entry["value_bet"]` twice | Clarity, maintainability | The code interleaves orchestration, derived metrics, and output mutation. It is explicit, but the home/away branches can overwrite each other and the list-building logic is harder to reason about than it needs to be. | Split per-game evaluation into a helper that returns an immutable per-game result plus zero-or-more value picks |
| 13 | Medium | `src/abba/server/toolkit.py::_player_impact` | complex `sorted(..., key=lambda p: (p.get("stats") or {}).get("points", 0) if isinstance(...) else 0)` | Clarity, maintainability | The lambda crams type-checking, defaulting, and domain extraction into one expression. It looks concise and is harder to inspect than a small named helper. | Extract `_points_for_player(p)` or normalize roster records before sorting |
| 14 | Medium | `src/abba/workflows/engine.py`, `src/abba/storage/duckdb.py`, `src/abba/server/http.py` | pervasive `dict[str, Any]` payloads instead of stable models | Correctness, maintainability | Plain dicts are being used for parsing, validation, transformation, and final reporting all at once. Python’s flexibility is hiding domain shape rather than expressing it. | Use `BaseModel`, `TypedDict`, or `dataclass` based on boundary stability and validation needs |

### Top 10 High-Value Simplifications

1. Replace `queue.pop(0)` with `deque.popleft()` in `src/abba/engine/graph.py::_betweenness_centrality`. This is the cleanest single performance win in active math code.

2. Push `query_session_replay` ordering and limiting into SQL in `src/abba/storage/duckdb.py` instead of fetching two DataFrames and sorting them in pandas.

3. Stop using `fetchdf().to_dict("records")` as the default storage read path when the caller only wants Python records.

4. Collapse the duplicate step-count comprehensions in `src/abba/workflows/engine.py::run` into one generator count stored in a local variable.

5. Rewrite `src/abba/server/toolkit.py::_player_impact` to use `healthy_count = sum(...)` and `starter = next(...)` instead of list materialization plus slicing tricks.

6. Replace repeated `pd.concat` inside `src/abba/utils/data_utils.py::encode_categorical` with one-shot `get_dummies`.

7. Replace the dual rolling passes in `src/abba/utils/data_utils.py::calculate_rolling_stats` with a single aggregation and join.

8. Extract the complicated `sorted(..., key=lambda ...)` scoring logic in `src/abba/server/toolkit.py::_player_impact` into a named helper. This is a readability win, not just style.

9. Replace broad `except Exception` + silent fallback in `src/abba/utils/data_utils.py` and `src/abba/utils/validation_utils.py` with explicit failure modes. This is a correctness simplification because it removes fake success paths.

10. Stop serializing NumPy adjacency matrices to lists mid-pipeline in `src/abba/analytics/graph.py`; only serialize at the output edge.

### Top 10 Performance / Clarity Hazards

1. `src/abba/engine/graph.py::_betweenness_centrality`
   List-as-queue logic introduces avoidable O(n) shifts inside BFS.

2. `src/abba/storage/duckdb.py::query_session_replay`
   Full eager materialization of logs before truncation.

3. `src/abba/storage/duckdb.py::query_games` and sibling query methods
   DataFrame materialization when plain rows are needed.

4. `src/abba/utils/data_utils.py::encode_categorical`
   Repeated concatenation creates hidden copying costs.

5. `src/abba/utils/data_utils.py::calculate_rolling_stats`
   Recomputes grouped rolling state twice and resets indexes to reattach it.

6. `src/abba/analytics/graph.py::{build_graph, analyze_connections}`
   Converts dense numeric data to Python lists and back again.

7. `src/abba/utils/data_utils.py::*`
   Broad exception handling returns apparently valid outputs after transformation failure.

8. `src/abba/utils/validation_utils.py::*`
   Validation logic hides internal validator errors behind the same booleans used for bad input.

9. `src/abba/server/toolkit.py::_player_impact`
   Builds filtered lists purely to count or inspect one element.

10. `src/abba/workflows/engine.py::tonights_slate`
   Imperative mutation of `entry` plus parallel accumulation into `value_picks` makes it too easy for home/away branches to overwrite each other or drift.

### Materialization And Iterator Judgments

#### Where generators should replace list construction

- `src/abba/server/toolkit.py::_player_impact`
  `healthy = [p for p in roster if ...]` should be `healthy_count = sum(1 for p in roster if ...)` because only the count is used.

- `src/abba/server/toolkit.py::_player_impact`
  `goalies = [p for p in roster if p.get("position") == "G"]` followed by `goalies[:1]` should be `starter = next((p for p in roster if p.get("position") == "G"), None)`.

- `src/abba/server/tools/reasoning.py::session_replay`
  `low_trust = [d for d in e["data_trust"] if ...]` should be a generator passed directly to `extend`.

- `src/abba/workflows/engine.py::run`
  Counting keys should use a generator expression, not a list comprehension, because only the cardinality matters.

#### Where list materialization is correct and should remain

- `src/abba/engine/features.py::features_to_vector`
  The list comprehension passed to `np.array(...)` is fine. The vector must be materialized in a deterministic order, and the schema is small.

- `src/abba/engine/graph.py::analyze_team`
  `player_metrics` should remain a list because it is part of the returned API payload and is iterated/serialized later.

- `src/abba/server/tools/reasoning.py::session_replay`
  `entries` should remain materialized because the function counts it, iterates it multiple times, and returns the full timeline. Converting it to an iterator would create one-shot bugs and worse debuggability.

- `src/abba/workflows/engine.py::tonights_slate`
  `value_picks` should remain a list because the function returns it and indexes the first element as `best_bet`.

#### Where iterators would create debugging pain or one-shot bugs

- Any attempt to make `entries` in `session_replay` lazy would break the repeated passes for counts, summaries, and final return.

- Any attempt to make `individual_predictions` in `src/abba/engine/ensemble.py::PredictionResult` lazy would make serialization and debugging worse; that payload should stay concrete.

- Any attempt to lazily stream games into `src/abba/engine/elo.py::initialize_from_games` after sorting would be artificial; the algorithm needs a stable ordered collection.

### Cases Where “Clever Python” Should Be Made More Explicit

1. `src/abba/server/toolkit.py::_player_impact`
   The sort-key lambda is too clever for how semantically important it is. A named helper would make the ranking rule inspectable.

2. `src/abba/workflows/engine.py::run`
   The repeated comprehension for step counts is compact but obscures that the same concept is being recomputed twice.

3. `src/abba/workflows/engine.py::season_story`
   The nested inline conditional that constructs `luck_narrative` is readable only because it is short today. This should be a small helper before more cases are added.

4. `src/abba/server/tools/reasoning.py::session_replay`
   The current explicit loops are better than a deeply nested flattening comprehension. This is one area where keeping the code boring is correct.

5. `src/abba/analytics/graph.py`
   The code is “simple” in Python terms only because it replaces mathematically meaningful algorithms with simplified arrays and dicts. Explicitness should be restored at the algorithm level, not via more concise syntax.

### Bottom Line

The Python problems in this repo are not mainly about taste. They fall into four concrete buckets:

- Python containers are being used with the wrong asymptotics for graph and replay workloads.
- pandas is being materialized where plain rows would be cheaper and clearer.
- broad exception handling converts failures into fake successful outputs.
- plain dicts are carrying too many responsibilities across parse, validate, transform, and compute phases.

The repo does contain places where simple Python is used correctly. The problem is that the high-risk paths often look idiomatic while still being the wrong construct for the workload. In a correctness-heavy system, that is worse than obviously ugly code because it is easier to trust by accident.

## Typing And Validation Architecture Pass

### Typing/Validation Architecture Report

#### Boundary Map

| Boundary | What comes in | What gets validated now | What should be validated but is not | Current modeling choice | Verdict |
| --- | --- | --- | --- | --- | --- |
| HTTP ingress | JSON request bodies in `src/abba/server/http.py::ABBAHandler._read_body` and `do_POST` | Only JSON syntax; body is assumed to be an object; tool args are passed straight through | request shape, required fields, enum/literal constraints, numeric bounds, date formats, discriminated tool payloads | raw `dict[str, Any]` | Inappropriate |
| MCP ingress | typed-looking function params in `src/abba/server/mcp.py`, plus JSON strings for `think()` arrays/objects | Mostly SDK-level primitive parsing; `think()` manually `json.loads()` strings | schema shape for nested reasoning payloads, allowed phase literals, tool-specific payload contracts | primitive params + ad hoc JSON strings | Inappropriate |
| Live connector ingress | third-party API JSON from `src/abba/connectors/live.py::_fetch_json` | JSON decode only | response shape, required keys, season/date coherence, units, enum values, missing-field policy | `dict[str, Any] | list[Any] | None` | Dangerous |
| Storage write boundary | dict records passed into `src/abba/storage/duckdb.py::upsert_*`, `insert_odds`, `cache_prediction`, `log_reasoning` | none beyond DB column types and occasional defaults | per-record schema, key presence, nested JSON schema, season/source invariants | raw dicts + JSON blobs | Dangerous |
| Storage read boundary | rows returned from `Storage.query_*` and parsed with `json.loads()` | only JSON syntax on stored blobs | shape of `stats`, `prediction`, `data_trust`, `context_snapshot`, entry-type-specific row schemas | raw dicts + `Any` | Dangerous |
| Toolkit/domain boundary | `src/abba/server/tools/*.py` mixins assembling game/team/odds/goalie dicts and passing them into engines | almost none | season disambiguation, required feature completeness, role discriminators, result schema | raw dict composition | Inappropriate |
| Computational boundary | `src/abba/engine/*.py` feature dicts and stat dicts | local arithmetic defaults only | domain invariants should already have been guaranteed before entering math | mostly `dict[str, Any]` / `dict[str, float]`, some dataclasses | Mixed; under-modeled upstream |
| Config boundary | env/config values parsed by `src/abba/core/config.py::Config` | Pydantic field typing and coercion | strict parsing, semantic validation, allowed log levels/sports values, nonnegative money/risk bounds | `BaseModel` | Reasonable choice, weak configuration |

#### Actual Architecture

The codebase does not have a typing architecture. It has:

- one Pydantic model for configuration
- a few dataclasses for result containers
- protocol/interfaces in the parallel analytics stack
- pervasive `dict[str, Any]` at every real data boundary

That means invalid states are representable almost everywhere that matters:

- wire payloads
- connector responses
- stored records
- tool outputs
- intermediate domain objects

The dominant pattern is not "validation then lightweight typed domain values." The dominant pattern is "accept dicts, hope the keys exist, patch with defaults, keep going."

### Misuse Of TypedDict

There are effectively no `TypedDict` definitions in the codebase.

That absence is itself a design failure. There are many places where runtime validation may not be required on every hop, but structural typing absolutely is:

- tool response payloads in `src/abba/server/tools/data.py`
- reasoning payloads in `src/abba/server/tools/reasoning.py`
- storage row shapes in `src/abba/storage/duckdb.py`
- odds/game/team-stat records exchanged between storage and engines
- confidence metadata payloads in `src/abba/engine/confidence.py`

These are currently plain dicts with ad hoc key assumptions. A `TypedDict` layer would not solve ingress validation, but it would at least formalize the expected shapes after validation.

Highest-severity missing `TypedDict` candidates:

1. `GameRecord`
   Current call chain:
   `connectors/live.py` -> `storage.upsert_games` -> `storage.query_games` -> `server/tools/data.py::query_games` -> workflows/toolkit

2. `OddsSnapshot`
   Current call chain:
   `connectors/live.py::OddsLiveConnector.refresh` -> `storage.insert_odds` -> `storage.query_odds` -> `server/tools/market.py` / `server/tools/nhl.py`

3. `ReasoningEntry`, `ToolCallEntry`, `SessionReplayEntry`
   Current call chain:
   `server/tools/reasoning.py::think` -> `storage.log_reasoning` -> `storage.query_session_replay` -> `server/tools/reasoning.py::session_replay`

4. `PredictionResponse` / `PredictionConfidencePayload`
   Current call chain:
   `server/tools/nhl.py::nhl_predict_game` -> `engine/confidence.py::build_prediction_meta`

Without these, the code has no stable post-validation contract between layers.

### Misuse Of Pydantic

#### 1. Pydantic is used only for config, not for real ingress boundaries

Relevant file:
`src/abba/core/config.py::Config`

This is underuse, not overuse. The repo pays the dependency cost for Pydantic, then declines to use it for:

- HTTP request bodies
- MCP tool inputs
- live API responses
- storage row schemas
- prediction response payloads

That is the wrong place to save complexity. The code validates the least important boundary and leaves the correctness-critical boundaries untyped.

#### 2. `Config` allows coercion where strictness should exist

Relevant file:
`src/abba/core/config.py::Config`

Problems:

- `max_bet_amount: float`
- `risk_tolerance: float`
- `supported_sports: list[str]`
- `log_level: str`

None of these are strict or semantically constrained. Pydantic will happily coerce strings into floats and may interpret env-provided structures in surprising ways. This is dangerous for configuration that affects bankroll/risk behavior.

Missing safeguards:

- strict numeric types for financial/risk settings
- bounded ranges such as `risk_tolerance in [0, 1]`
- `Literal` or enum-style restriction for log level
- explicit parsing strategy for `supported_sports`

#### 3. `Config.__init__` has side effects

Relevant file:
`src/abba/core/config.py::Config.__init__`

`self.model_cache_dir.mkdir(exist_ok=True)` runs inside model construction.

That is the wrong semantic boundary for a validation model. Validation objects should describe and validate data, not mutate the environment. If this must happen, it belongs in `model_post_init` at minimum, and preferably outside the config model entirely.

#### 4. There are no Pydantic validators where domain rules actually exist

The prompt asked to review all Pydantic validators. There are none.

That absence matters because real domain rules are currently expressed nowhere centrally:

- legal sport identifiers
- legal workflow phases
- valid season formats
- nonnegative odds/points/cap values
- mutually exclusive / required parameter sets
- nested reasoning/trust-entry schemas

Instead, the code relies on string comparisons and dict lookups downstream.

### Weak Or Dangerous Unions

#### 1. `src/abba/connectors/live.py::_fetch_json`
Type:
`dict[str, Any] | list[Any] | None`

This is not a meaningful sum type. It encodes "the upstream returned some JSON." There is no discriminator and no narrowing contract. Every caller must already know the endpoint-specific shape, which defeats the point of the annotation.

Verdict:
underspecified and dangerous

Better boundary:
- endpoint-specific Pydantic response models at ingress
- or at least separate methods with endpoint-specific return types

#### 2. `src/abba/server/http.py::_read_body`
Type:
`dict[str, Any] | None`

Reality:
`json.loads()` may produce a list, string, number, boolean, or null.

The annotation is lying. The code then immediately does `body.get(...)`, so malformed-but-valid JSON can become runtime failure or silent bad dispatch.

Verdict:
likely incorrect annotation plus missing validation

#### 3. `src/abba/engine/confidence.py::PredictionConfidence.data_freshness`
Type:
`str | float`

This is a semantically weak union. A field is being used to hold either:

- a categorical state like `"unknown"` or `"seed"`
- or a numeric freshness quantity

That makes every consumer responsible for ad hoc type testing and encourages inconsistent interpretation.

Verdict:
dangerous union

Better boundary:
- separate fields like `data_source_kind: Literal["live", "seed", "none", "unknown"]`
- `staleness_seconds: float | None`

#### 4. pervasive `dict[str, Any] | None`, `list[dict[str, Any]] | None`, `Any | None`

Examples:
- `src/abba/engine/hockey.py`
- `src/abba/storage/duckdb.py`
- `src/abba/workflows/engine.py::_find_starter`
- `src/abba/analytics/manager.py`

These are not unions in the algebraic-data-type sense. They are admissions that the code does not know the shape. In a correctness-heavy system, that is architectural debt.

#### 5. MCP `think()` wire contract is a disguised union problem

Relevant file:
`src/abba/server/mcp.py::think`

Current wire types:
- `uncertainty: str | None`
- `data_trust: str | None`
- `workflow_gaps: str | None`
- `want_to_verify: str | None`

Then:
- `json.loads(...)` converts those strings into arrays/objects

This is effectively an untagged union between:

- absent
- JSON string representing array/object
- malformed JSON string

No discriminator, no schema validation, and no error reporting exist.

Verdict:
badly implemented wire/domain conversion

### Union/Dispatch Review

#### Dispatch is stringly typed, not discriminated

Relevant files:
- `src/abba/server/tools/registry.py::call_tool`
- `src/abba/server/http.py::do_POST`
- `src/abba/workflows/engine.py::run`

All routing is done by string lookup:

- tool name strings
- workflow name strings
- market type strings
- phase strings

This should be modeled with `Literal` at minimum, and in several places with discriminated unions:

- tool invocation payloads
- reasoning entry types (`tool_call` vs `reasoning`)
- odds market variants (`h2h`, `spreads`, `totals`)
- connector result statuses (`ok`, `failed`, `no_api_key`)

The current code has informal discriminators in string fields, but no type system support and almost no validation.

#### `market_type` should be a discriminated union and is not

Current shape in `src/abba/connectors/live.py::OddsLiveConnector.refresh`:

- same dict is reused for all market variants
- keys appear conditionally:
  - `home_odds`, `away_odds`
  - `spread`
  - `total`
  - `over_odds`, `under_odds`

That means invalid combinations are representable:

- spread market without `spread`
- totals market with `home_odds`
- h2h market carrying stale spread fields

This is exactly where a discriminated union should exist.

#### session replay entries should be a discriminated union and are not

Current call chain:
`storage.query_session_replay` -> `server/tools/reasoning.py::session_replay`

Rows are combined from different tables and tagged with `entry_type`, but the code still exposes them as plain dicts with optional unrelated fields.

This should be:

- `ToolCallReplayEntry`
- `ReasoningReplayEntry`
- union discriminated by `entry_type`

Instead, every consumer gets a dict with partially populated keys and must guess which keys are legal in which branch.

### Dataclass / Plain Class / Protocol Review

#### Dataclasses used correctly

- `src/abba/engine/kelly.py::KellyResult`
- `src/abba/engine/ensemble.py::PredictionResult`

These are lightweight post-validation computational/result containers. That is a good use of dataclasses.

#### Dataclasses used too weakly

- `src/abba/engine/confidence.py::PredictionConfidence`
- `src/abba/analytics/models.py::{Prediction, BiometricData, UserPatterns, ModelPerformance}`

These dataclasses mostly wrap fields that are themselves untyped dicts or unconstrained lists. They improve naming but do not prevent invalid states.

Example:
`BiometricData.heart_rate: dict[str, float]`

This still permits arbitrary keys, missing required metrics, and inconsistent units. If the structure is boundary-facing, it should be validated. If it is internal, it should at least be a `TypedDict` or smaller dataclasses.

#### Protocols are too generic to protect semantics

Relevant file:
`src/abba/analytics/interfaces.py`

Problems:

- `DataProcessor.process(self, data: dict) -> dict`
- `AgentInterface.execute(self, task: dict) -> dict`
- `DatabaseInterface.save_bet(self, bet: dict) -> bool`

These protocols erase domain structure instead of expressing it. A protocol with `dict` in and `dict` out gives almost no useful static guarantee. This is abstraction without semantic value.

### Validators Review

There are no Pydantic field validators or model validators to audit.

What exists instead is hand-written validation in:

- `src/abba/utils/validation_utils.py`

This is a poor substitute because it:

- duplicates type checks that boundary models should own
- returns booleans instead of structured errors
- conflates validator crashes with invalid data
- performs no schema transformation into safe domain objects

This is exactly the pattern the prompt warned about: validators duplicating type information without enforcing actual domain rules.

### Conversion Points Between Wire Models, Domain Models, And Computational Models

#### 1. HTTP wire -> tool kwargs

Call chain:
`server/http.py::ABBAHandler._read_body`
-> `server/http.py::do_POST`
-> `server/tools/registry.py::call_tool`

Current state:
- parse JSON
- assume dict
- unpack kwargs into arbitrary tool function

No formal wire model exists. No validated domain model exists. This is a direct wire-to-method jump.

#### 2. MCP wire -> toolkit call

Call chain:
`server/mcp.py::<tool function>`
-> `_toolkit.<tool>(...)`

Current state:
- primitive params are accepted from MCP
- nested structures for `think()` are tunneled as JSON strings and manually loaded

This is a malformed boundary: string transport details leak into application types.

#### 3. connector JSON -> storage dict

Call chain:
`connectors/live.py::_fetch_json`
-> endpoint-specific parsing logic
-> `storage.upsert_*`

Current state:
- endpoint responses are accessed by nested `.get(...)`
- missing fields are defaulted inline
- resulting dicts are stored without typed validation

This is where most invalid-state admission happens.

#### 4. storage rows -> computational models

Call chain:
`storage.query_*`
-> `server/tools/nhl.py`
-> `engine/hockey.py` / `engine/ensemble.py` / `engine/confidence.py`

Current state:
- raw dicts leave storage
- tool layer picks first row, extracts nested `stats`, and passes dicts onward
- math layer receives partially normalized dicts

No explicit conversion from persistence model to domain model exists.

### Recommended Boundary Model Strategy

#### 1. Wire boundary: strict Pydantic

Use strict `BaseModel` or `RootModel` for:

- HTTP request payloads
- MCP nested payloads
- live connector response records after parsing

Rules:
- no implicit coercion for numeric/risk fields
- `Literal` for tool names, workflow names, phase values, market types, source statuses
- model validators only for cross-field domain rules
- no side effects in validation models

#### 2. Post-validation application boundary: `TypedDict` or frozen dataclass

After ingress validation, convert into lightweight typed structures for intra-app movement:

- `GameRecord`
- `OddsSnapshot`
- `TeamStatsRecord`
- `AdvancedStatsRecord`
- `GoaltenderStatsRecord`
- `ReasoningEntry`

Use:
- `TypedDict` for serialized dict-shaped records that stay map-like
- frozen `dataclass` for semantically cohesive domain entities consumed by algorithms

#### 3. Computational boundary: dataclasses / NumPy arrays / plain numeric types

Math engines should not receive `dict[str, Any]`.

They should receive:

- small dataclasses with already-validated fields
- explicit feature vectors
- explicit optionality where truly necessary

At that point Pydantic should be gone from the hot path.

#### 4. Persistence boundary: typed repository API

`Storage` should not accept arbitrary dicts for every upsert/query pair.

Instead:
- `upsert_games(games: list[GameRecord])`
- `query_games(...) -> list[GameRecord]`
- `cache_prediction(...) -> PredictionCacheEntry`

If JSON columns remain, serialization should be centralized and schema-owned.

### Misuse Summary

#### Misuse of `TypedDict`

- There is no meaningful `TypedDict` layer where one is badly needed.
- Post-validation dict-shaped contracts are undocumented and unenforced.

#### Misuse of Pydantic

- Severe underuse at real boundaries.
- Weak config model with coercion and no semantic validators.
- Side effects inside `Config.__init__`.

#### Weak or dangerous unions

- `dict | list | None` endpoint results
- `str | float` semantic overloading in confidence metadata
- `dict[str, Any] | None` everywhere shape is unknown
- JSON-string-to-list/object tunneling in MCP `think()`

### Bottom Line

The typing system in this repo is mostly decorative. The code advertises type information, but the real contracts are still:

- string dispatch
- raw dicts
- `Any`
- default-filled missing fields

That combination is especially dangerous in a mathematically sensitive system because invalid states stay representable all the way into the computational core.

The correct target is not “more typing everywhere.” The correct target is:

- strict validated models at ingress
- lightweight typed domain records after validation
- no `Any`-shaped dict pipelines into math
- explicit discriminated unions where variant payloads really exist

## Union And Dispatch Pattern Pass

### Union/Dispatch Audit

#### 1. Tool dispatch in `src/abba/server/tools/registry.py::call_tool`

Implementation:
- static `tool_map` dict from tool-name string to bound method
- lookup by `tool_name`
- fallback error payload on miss

Why it exists:
- dynamic invocation from HTTP/MCP/generic tool calls

Assessment:
- explicit: yes
- safe: only partially
- testable: yes
- extensible: moderately

Problems:
- dispatch key is untyped `str`, not `Literal`
- request payloads are not discriminated by tool name
- argument validation happens nowhere before `fn(**kwargs)`
- schema lives separately in `list_tools()` as hand-written metadata, so dispatcher and schema can drift

Verdict:
salvageable dispatcher table, but not a typed union architecture

What it should be:
- a dispatcher table is correct here
- but it should dispatch validated command objects, not raw kwargs

#### 2. Workflow dispatch in `src/abba/workflows/engine.py::run`

Implementation:
- `workflows = {name: method}`
- lookup by `workflow_name`

Assessment:
- explicit: yes
- safe: partial
- extensible: moderate

Problems:
- workflow name is an unvalidated string
- workflow-specific parameter schemas do not exist
- adding a new workflow requires updating:
  - the dict in `run()`
  - the registry enum in `server/tools/registry.py`
  - likely MCP/HTTP docs
- no tagged request object means bad params fail only inside the chosen workflow

Verdict:
reasonable manual dispatcher table, but brittle because the schema and dispatch are not unified

What it should be:
- dispatcher table plus `Literal` workflow names
- or a discriminated union of workflow request models if the external API remains generic

#### 3. Ensemble method selection in `src/abba/engine/ensemble.py::combine`

Implementation:
- `if method == "weighted" ... elif method == "median" ... elif method == "voting" ... else average`

Assessment:
- explicit: yes
- safe: no
- extensible: brittle

Problems:
- unknown method silently falls back to average
- `method: str` is effectively a weak union without validation
- behavior is order-dependent because the catch-all `else` absorbs invalid input
- debugging is harder because typos do not fail loudly

Verdict:
fragile cleverness

What it should be:
- `Literal["weighted", "average", "median", "voting"]`
- fail-fast on unknown methods
- a small dispatcher table or `match/case` is fine here

#### 4. Odds market variant dispatch in `src/abba/connectors/live.py::OddsLiveConnector.refresh`

Implementation:
- `market_key` drives `if market_key == "h2h" / "spreads" / "totals"`
- one mutable `record` dict is populated differently per branch

Assessment:
- explicit: yes
- safe: no
- extensible: brittle

Problems:
- this is a textbook discriminated union that is not modeled as one
- all variants share one dict shape, so invalid field combinations are representable
- adding a new market type requires manual conditional surgery
- no validation ensures required fields per variant

Verdict:
should be a discriminated union and currently is not

What it should be:
- `Literal` discriminator on `market_type`
- separate models:
  - `MoneylineOdds`
  - `SpreadOdds`
  - `TotalsOdds`

#### 5. Session replay entry dispatch in `src/abba/storage/duckdb.py::query_session_replay` and `src/abba/server/tools/reasoning.py::session_replay`

Implementation:
- SQL adds `'tool_call' AS entry_type` and `'reasoning' AS entry_type`
- consumer branches on `e.get("entry_type")`

Assessment:
- explicit: yes
- safe: somewhat
- extensible: moderate

Problems:
- this is an implicit discriminated union, but the payload is still an untyped merged dict
- irrelevant fields remain present as null/NaN/absent depending on source row
- adding new replay entry kinds would spread schema assumptions across storage and presentation

Verdict:
robust pattern idea, weak implementation

What it should be:
- a proper discriminated union is appropriate here
- this is one of the few places where tagged-union architecture would genuinely improve clarity

#### 6. MCP `think()` nested payload parsing in `src/abba/server/mcp.py::think`

Implementation:
- several parameters are declared as `str | None`
- each is `json.loads(...)` into arrays/objects before toolkit call

Assessment:
- explicit: barely
- safe: no
- testable: awkward
- extensible: poor

Problems:
- subtype selection is hidden inside parsing rather than encoded in types
- valid/invalid distinction depends on runtime JSON parsing of strings
- debugging is worse because the wire type lies about the domain type
- this is effectively validator-driven subtype selection without validators

Verdict:
clever garbage

What it should be:
- a single structured request model
- nested objects/lists parsed as structured data by the boundary layer, not as strings containing JSON

#### 7. Data processor type routing in `src/abba/data/data_processor.py::process`

Implementation:
- inspect `data.get("type", "unknown")`
- `if/elif` into `_process_mlb_data`, `_process_nhl_data`, `_process_odds_data`, `_process_weather_data`

Assessment:
- explicit: yes
- safe: no
- extensible: brittle

Problems:
- another missing discriminated union
- payload schema depends on `type` but is never validated against it
- unknown types quietly return original data
- each subtype handler uses raw dicts and defaults

Verdict:
should be a discriminated union if this component remains

#### 8. Real-time connector feed/webhook dispatch in `src/abba/api/real_time_connector.py`

Implementations:
- `_get_auth_headers`: branch on `auth_type`
- `_run_data_feed`: branch on `feed_type`
- `_process_data_by_type`: branch on `data_type`
- `_process_webhook_event`: branch on `event.event_type`

Assessment:
- explicit: yes
- safe: mostly no
- extensible: poor to moderate

Problems:
- four separate string-dispatch systems exist in one module
- none are backed by `Literal`, enum, or discriminated payload models
- payload shapes overlap and remain ambiguous
- adding a new variant requires touching multiple ad hoc branches

Verdict:
dispatch logic spread across the codebase; not elegant

What it should be:
- a small number of tagged domain objects with central dispatch
- or no abstraction at all if this stack is experimental and not on the critical path

#### 9. Analytics model factory in `src/abba/analytics/model_factory.py`

Implementation:
- `ModelRegistry` holds model-name -> constructor
- `ModelFactory` registers defaults
- `EnsembleFactory` builds configured groups

Assessment:
- explicit: yes
- safe: mostly
- testable: yes
- extensible: relatively good

Problems:
- config payloads are untyped dicts
- model identifiers are raw strings
- defaults/config merging is not validated

Verdict:
one of the few dispatch patterns that is architecturally reasonable

What it should be:
- keep the registry/factory idea if this stack survives
- add typed config objects or `Literal` model identifiers

### Should This Be A Discriminated Union?

Yes:

1. Odds market records in `connectors/live.py`
2. Session replay entries in `storage/duckdb.py` + `server/tools/reasoning.py`
3. Data processor input variants in `data/data_processor.py`
4. Possibly generic workflow/tool request envelopes if the public API remains "call anything by name"

No:

1. Tool registry dispatch in `server/tools/registry.py`
   This is better as a dispatcher table over validated command objects.

2. Workflow dispatch in `workflows/engine.py`
   Also better as a dispatcher table, unless you insist on a single generic external workflow endpoint.

3. Ensemble combination method selection in `engine/ensemble.py`
   This is too small for elaborate union modeling. Use `Literal` + fail-fast dispatch.

### Robust Patterns Worth Keeping

#### 1. Central registry/factory in `src/abba/analytics/model_factory.py`

Why it is worth keeping:
- dispatch is centralized
- new variants can be registered in one place
- creation logic is not spread across the application

What must change:
- typed model identifiers
- typed config payloads
- no raw `dict` configs with `"type"` magic strings

#### 2. Tagged replay entries via `entry_type`

Why it is worth keeping:
- the tag is explicit
- the two variants are conceptually real

What must change:
- make it a real discriminated union instead of a merged dict soup

#### 3. Simple dispatcher tables for top-level tool/workflow routing

Why they are worth keeping:
- clearer than validator-driven subtype selection
- easier to test than implicit magic

What must change:
- unify schema and dispatcher
- stop unpacking raw dicts directly into functions

### Fragile Cleverness That Should Be Removed

1. `src/abba/server/mcp.py::think`
   JSON-string tunneling for nested payloads is not a robust union pattern. It is a transport leak.

2. `src/abba/engine/ensemble.py::combine`
   Silent fallback from unknown method to average is not elegant dispatch. It is typo laundering.

3. `src/abba/connectors/live.py::OddsLiveConnector.refresh`
   One mutable dict pretending to represent three market variants is fake abstraction.

4. `src/abba/data/data_processor.py::process`
   `"type"`-based routing over raw dicts with silent passthrough of unknown variants is not safe subtype handling.

5. `src/abba/api/real_time_connector.py`
   Repeated `*_type` string branching across feed, auth, event, and data handling spreads dispatch logic beyond what anyone can reason about locally.

### Patterns That Make Debugging Harder Than Necessary

1. Unknown ensemble method silently becoming average in `engine/ensemble.py`
2. Unknown data types silently returning original input in `data/data_processor.py`
3. MCP nested fields that must be mentally decoded from JSON strings in `server/mcp.py`
4. Shared odds record dict whose legal fields depend on `market_type`
5. Schema metadata in `list_tools()` that can diverge from actual dispatcher behavior in `call_tool()`

### Recommended Canonical Dispatch Pattern For This Codebase

Use one pattern per layer, not one pattern per file.

#### 1. At external boundaries: discriminated unions only when payload variants are real

Use strict validated tagged models for:

- odds market variants
- replay entry variants
- any generic "invoke tool/workflow" envelope if it remains a single endpoint

This is where discriminated unions earn their keep.

#### 2. At top-level application routing: dispatcher table over validated command objects

For tools and workflows:

- parse request into a typed command/request model
- dispatch via a centralized mapping
- do not `**kwargs` raw dicts into business functions

This is simpler and more debuggable than polymorphic model parsing.

#### 3. For tiny algorithm options: `Literal` + explicit branch

For things like ensemble combination method:

- `Literal` type
- explicit branch or small local table
- unknown value should raise immediately

Do not build elaborate abstract hierarchies for four strategies.

#### 4. For extensible algorithm families: registry/factory plus typed config

The analytics model factory is the right shape if this stack matters.
Keep the registry. Replace string-and-dict configs with typed descriptors.

### Bottom Line

This codebase does not mostly use sophisticated union-based design. It mostly uses:

- strings as tags
- dicts as payloads
- manual `if/elif` dispatch
- ad hoc factory tables

Some of that is fine. The problem is that the code repeatedly chooses the worst hybrid:

- enough abstraction to hide where variant selection happens
- not enough typing to make variant selection safe

The canonical rule for this repo should be:

- real tagged variants get discriminated unions
- top-level commands get dispatcher tables over validated models
- small option sets get `Literal` plus fail-fast branching
- raw dict/string dispatch should not cross correctness-critical boundaries

## Validation And Coercion Danger Pass

### Coercion-Risk Report

| Rank | Severity | File / symbol | Exact code pattern | Failure mode | How bad input becomes acceptable | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Critical | `src/abba/engine/hockey.py::build_nhl_features` | pervasive `.get(..., default)` on team/advanced/goalie/odds inputs | Missing mathematical inputs are normalized into league-average or neutral values | absent advanced stats become `0.50` Corsi/xG; absent goalie stats become `0.907`; absent special teams become `22.0/80.0`; absent rest becomes `0.0`; absent odds become `0.0` | Fail closed on materially incomplete feature sets, or surface an explicit degraded-model mode with a separate type and confidence penalty |
| 2 | Critical | `src/abba/server/mcp.py::think` | `uncertainty: str | None` then `json.loads(uncertainty)` | malformed nested payloads are treated as a transport concern, not a schema concern | callers provide JSON strings instead of structured arrays/objects; any parsing mismatch happens outside a validated model and can silently collapse semantics | replace with a strict request model containing nested lists/objects directly |
| 3 | High | `src/abba/utils/data_utils.py::*` | broad `except Exception` and `return df` / `return None` fallbacks | transformation failure masquerades as successful pass-through | invalid dates, bad rolling windows, broken encodings, or copy issues result in original data continuing downstream | raise explicit transformation errors; if degraded execution is allowed, annotate the output as degraded and force callers to branch |
| 4 | High | `src/abba/utils/validation_utils.py::*` | broad `except Exception` returning `False` or invalid dict | validator crashes are indistinguishable from invalid input | a bug in validation logic turns valid input into generic failure, and callers lose causal information | use structured validation errors; do not conflate parser/validator exceptions with rejected input |
| 5 | High | `src/abba/core/config.py::Config` | permissive `BaseModel` fields without strict mode or validators | env strings and loose values may be coerced into acceptable-looking config | `"0.1"` becomes float risk tolerance, malformed sports lists may be coerced, invalid log levels remain strings, negative values are not blocked | use strict numeric fields, bounded constraints, `Literal`/enum values, and explicit parsing for list env vars |
| 6 | High | `src/abba/server/http.py::_read_body` | `json.loads(raw)` typed as `dict[str, Any] | None` | non-object JSON can enter the system as if it were a valid request body | arrays, strings, booleans, and numbers are valid JSON but not valid command payloads; the code assumes `.get()` works later | validate that the top-level body is an object and then validate against a request model |
| 7 | High | `src/abba/storage/duckdb.py::{upsert_*, query_*, cache_prediction, log_reasoning}` | JSON blob storage and `json.loads()` on read with no schema check | corrupted or shape-drifted stored JSON becomes untrusted domain data | any JSON object with the right rough keys is accepted after deserialization, even if nested structures are wrong or units changed | introduce typed serializers and validate on both write and read boundaries |
| 8 | Medium | `src/abba/connectors/live.py` | inline `.get(..., default)` extraction from third-party JSON | upstream schema changes silently become "valid" local data with defaults | missing API fields degrade to zeros, empty strings, or current hard-coded season values, producing plausible but wrong records | validate connector responses against strict response models before persistence |
| 9 | Medium | `src/abba/data/data_processor.py::process` and subtype handlers | unknown `type` returns original data; subtype handlers default missing fields to zeroes | malformed or partial raw payloads become processed-looking domain dicts | weather, odds, NHL, MLB data all accept large missing-field sets and compute derived values anyway | use discriminated input models and reject unknown or incomplete variants |
| 10 | Medium | `src/abba/engine/confidence.py::PredictionConfidence.data_freshness` | semantic overloading `str | float` | consumers can misread categorical freshness as numeric freshness | `"seed"` / `"unknown"` and numeric age share one field, inviting ad hoc coercion or display bugs | split into categorical source/freshness status plus numeric staleness field |

### Fail-Open Behaviors

#### 1. Missing stats become neutral or league-average features

Relevant files:
- `src/abba/engine/hockey.py::build_nhl_features`
- `src/abba/engine/features.py::build_features`

Examples:
- `wins`, `losses`, `overtime_losses` default to synthetic records
- missing advanced stats become 50/50 possession/chance shares
- missing weather becomes neutral
- missing odds become no market signal

Failure mode:
the mathematical core still produces clean probabilities even when the intended model inputs are absent

Why this is dangerous:
it converts missingness into apparent knowledge instead of a hard boundary decision

#### 2. Unknown methods silently downgrade to fallback behavior

Relevant file:
- `src/abba/engine/ensemble.py::combine`

Current behavior:
- unknown `method` hits the catch-all `else` and becomes arithmetic mean

Failure mode:
a typo in dispatch or bad caller input produces a plausible output rather than an explicit error

#### 3. Unknown data variants are passed through

Relevant file:
- `src/abba/data/data_processor.py::process`

Current behavior:
- unknown `data_type` logs a warning and returns original `data`

Failure mode:
callers can mistake unprocessed raw input for processed domain output

#### 4. Reasoning payloads parse permissively from strings

Relevant file:
- `src/abba/server/mcp.py::think`

Failure mode:
the system accepts wrong transport shape rather than defining the right domain shape

#### 5. Storage defaults mask provenance gaps

Relevant file:
- `src/abba/storage/duckdb.py`

Examples:
- `source VARCHAR DEFAULT 'unknown'`
- market type default `"moneyline"`
- injury status default `"healthy"`
- contract years default `1`

Failure mode:
missing provenance and missing semantics are rewritten into plausible categorical values

### Places That Should Use Strict Validation

1. `src/abba/core/config.py::Config`
   Use strict numeric parsing and bounded constraints for bankroll/risk settings.

2. `src/abba/server/http.py`
   Require top-level object payloads and validate tool-specific request schemas before dispatch.

3. `src/abba/server/mcp.py::think`
   Replace JSON-string nested fields with structured validated inputs.

4. `src/abba/connectors/live.py`
   Validate third-party API payloads and reject partial/malformed records before `Storage.upsert_*`.

5. `src/abba/storage/duckdb.py`
   Validate JSON blob schemas on write and read, or stop using blob-shaped domain records.

6. `src/abba/server/tools/nhl.py::nhl_predict_game`
   Require coherent season-aligned team/advanced/goalie inputs rather than defaulting through missing branches.

7. `src/abba/engine/hockey.py::build_nhl_features`
   Separate mandatory from optional inputs and fail closed when mandatory predictive inputs are absent.

8. `src/abba/data/data_processor.py`
   Disallow unknown `type` values and incomplete subtype payloads.

### Test Cases That Should Exist To Catch Bad Inputs

1. Config strictness
   - `MAX_BET_AMOUNT="abc"` should fail, not coerce or fall back.
   - `RISK_TOLERANCE="-0.2"` should fail.
   - `SUPPORTED_SPORTS="MLB,NHL"` should parse only via an explicit supported format, not accidental coercion.

2. HTTP body validation
   - POST body `[]`, `"hello"`, `123`, and `null` should all fail as invalid request bodies.
   - unknown tool fields should fail if extra keys are not allowed.

3. MCP reasoning payload validation
   - malformed JSON strings for `uncertainty`/`data_trust` should produce a structured input error.
   - `phase="planing"` typo should fail, not record an arbitrary phase.

4. Connector schema validation
   - NHL standings response missing `standings`
   - odds response with malformed market entries
   - roster response missing `id` or `positionCode`
   should all fail before persistence.

5. Feature-builder fail-closed behavior
   - `build_nhl_features()` with missing advanced stats should either return an explicit degraded result type or raise.
   - missing goalie fields should not silently turn into league-average goalie performance if the path claims goalie-aware prediction.

6. Storage schema drift
   - cached prediction JSON missing `prediction.value`
   - reasoning log `data_trust` containing non-object list items
   - roster `stats` field with wrong nested shape
   should fail on read validation.

7. Ensemble method validation
   - `method="weigthed"` should raise or return explicit input error, never average silently.

8. Unknown processor subtype
   - `DataProcessor.process({"type": "soccer", ...})` should fail explicitly, not return raw input.

### Bottom Line

The most dangerous validation behavior in this repo is not classic Pydantic coercion. It is broader and worse:

- missing data is normalized into neutral values
- unknown variants fall through to defaults
- validator failures collapse into booleans
- raw dicts are accepted deep into the math path

Pydantic itself is barely used, which means the main fail-open mechanism is hand-rolled permissiveness rather than library coercion. The corrective action is still the same:

- strict validated ingress models
- explicit degraded-mode types instead of silent neutral defaults
- fail-closed handling for unknown variants and malformed nested payloads

## Elegance-Vs-Cleverness Design Pass

### Elegance-Vs-Cleverness Report

| Pattern | Problem it is trying to solve | Cognitive load | Data-flow traceability | Testability | Correctness impact | Classification |
| --- | --- | --- | --- | --- | --- | --- |
| `ABBAToolkit` mixin composition in `src/abba/server/toolkit.py` | present one callable surface for agents across many tool categories | increases | harder | harder | mostly aesthetics, not correctness | Architecture debt |
| tool registry in `src/abba/server/tools/registry.py` | centralized tool discovery and dispatch | mixed | mixed | mixed | can help if schemas are unified, currently they are not | Acceptable but not ideal |
| workflow engine in `src/abba/workflows/engine.py` | package common multi-step tasks into reusable flows | mixed | harder than explicit scripts, but still readable | mixed | hurts correctness when it narrates and reinterprets math | Acceptable but not ideal |
| `ModelFactory` / `EnsembleFactory` in `src/abba/analytics/model_factory.py` | centralize model instantiation for an experimental analytics stack | moderate | moderate | easier | neutral to mildly positive | Acceptable but not ideal |
| protocol/ABC stack in `src/abba/analytics/interfaces.py` and `src/abba/core/dependency_injection.py` | decouple analytics services and support interchangeable implementations | increases sharply | much harder | harder in practice | mostly aesthetics | Architecture debt |
| dependency injection container in `src/abba/core/dependency_injection.py` | late-bind services and hide concrete implementations | increases sharply | much harder | not meaningfully easier | no correctness gain | Overly clever |
| small dataclasses in `src/abba/engine/kelly.py`, `src/abba/engine/ensemble.py`, `src/abba/engine/confidence.py` | represent result values explicitly | reduces | easier | easier | positive | Strong and worth standardizing |
| utility layers in `src/abba/utils/data_utils.py` and `src/abba/utils/validation_utils.py` | avoid repeated preprocessing/validation code | increases | harder | harder | negative because semantics are hidden | Architecture debt |
| manual string dispatch tables in `registry.py` and `workflows/engine.py` | route commands explicitly | low | easier than implicit magic | easy | neutral to positive if typed | Strong pattern shape, weak implementation |
| generic `dict[str, Any]` pipelines across storage/tools/engines | maximize flexibility and reduce model boilerplate | lowers local friction, raises global load dramatically | much harder | harder | strongly negative | Architecture debt |

### Pattern Findings

#### 1. `ABBAToolkit` mixin composition is locally tidy and globally harmful

Relevant file:
`src/abba/server/toolkit.py`

Pattern:
- compose `DataToolsMixin`
- `AnalyticsToolsMixin`
- `MarketToolsMixin`
- `NHLToolsMixin`
- `ReasoningToolsMixin`
- `SessionToolsMixin`
- `ToolRegistryMixin`

Problem it is trying to solve:
- one unified API for agents

Why it looks elegant:
- categories are separated into mixins
- toolkit surface reads like a clean facade

Why it makes the system worse:
- the object is still a mutable god-object with storage, engines, session state, refresh state, Elo state, and tracking logic all attached
- a new engineer cannot tell which tool methods are pure, which mutate session state, which rely on cache, and which depend on prior refresh calls
- mixins reduce local file size while increasing global trace cost

Verdict:
architecture debt

Better boundary:
- keep one top-level facade if needed for agent UX
- but compose explicit services instead of inheriting dozens of methods into one class
- make tool handlers thin wrappers over named service objects

#### 2. Tool registry plus hand-authored schemas is elegant-looking duplication

Relevant file:
`src/abba/server/tools/registry.py`

Problem it is trying to solve:
- discover tools and dispatch them dynamically

Why it looks elegant:
- one file advertises the entire tool surface
- metadata and dispatcher are colocated

Why it is worse than it looks:
- schema and implementation are still duplicated manually
- `list_tools()` and `call_tool()` can drift independently
- adding a new tool requires editing metadata, dispatcher, and the mixin implementation

Verdict:
acceptable but not ideal

Better boundary:
- standardize on one command model per tool and derive metadata from that
- keep the dispatcher table; remove handwritten schema duplication

#### 3. Workflow orchestration is readable enough, but over-claims via composition

Relevant file:
`src/abba/workflows/engine.py`

Problem it is trying to solve:
- bundle refresh, lookup, prediction, odds, narrative, and strategy into reusable flows

Why it helps:
- explicit sequential code is easier to read than a generic pipeline framework

Why it hurts:
- the workflow layer mixes orchestration, heuristics, confidence generation, and narrative claims
- data flow is only traceable by following many toolkit calls into shared mutable services
- tests become broad integration exercises rather than narrow unit checks

Verdict:
acceptable but not ideal

Keep:
- explicit step-by-step code

Refactor:
- extract numerical decisions out of workflow functions
- leave workflows to sequencing and presentation only

#### 4. The analytics protocol + DI stack is over-composed nonsense

Relevant files:
- `src/abba/analytics/interfaces.py`
- `src/abba/core/dependency_injection.py`

Problem it is trying to solve:
- interchangeability of processors, agents, databases, and graph analyzers

Why it looks elegant:
- protocols, ABCs, service providers, and containers signal “clean architecture”

Why it makes the system worse:
- the interfaces are too generic (`dict` in, `dict` out), so they do not encode useful semantics
- the DI container is string-keyed and manually configured, so it buys indirection without type safety
- concrete implementations still live in the same repo and are not truly swappable in any correctness-critical sense
- new engineers have to learn an abstract service lattice before they can understand basic data flow

Verdict:
overly clever bordering on architecture debt

Best refactor:
- delete the DI container from paths that do not need runtime composition
- instantiate concrete services explicitly
- keep only narrow interfaces where a real second implementation exists or will exist

#### 5. `ModelFactory` / registry is one of the few abstractions that earns some of its keep

Relevant file:
`src/abba/analytics/model_factory.py`

Problem it is trying to solve:
- centralized creation of model variants and ensembles

Why it helps:
- model creation logic is not spread across the codebase
- extensibility is localized
- testing model selection is easier than with repeated `if/elif` ladders

Why it is still not ideal:
- config shapes are raw dicts
- string identifiers remain untyped
- this factory stack serves an experimental parallel analytics subsystem, not the active critical path

Verdict:
acceptable but not ideal

Standardize only if:
- the analytics stack becomes a supported subsystem

#### 6. Utility layers hide semantics rather than reusing true common behavior

Relevant files:
- `src/abba/utils/data_utils.py`
- `src/abba/utils/validation_utils.py`

Problem they are trying to solve:
- reduce duplication of dataframe cleaning, encoding, normalization, and validation

Why they look elegant:
- a static utility class gives the impression of shared infrastructure

Why they are harmful:
- each helper smuggles policy decisions such as mean imputation, `"Unknown"` fill, generic sport lists, and permissive validation
- the abstraction boundary is too generic to preserve domain semantics
- tests end up checking generic utility behavior rather than the real domain contracts

Verdict:
architecture debt

Better boundary:
- explicit preprocessing objects or inline domain-specific transformations
- schema-based validators at ingress

#### 7. Decorator use in MCP is acceptable; it is not the problem

Relevant file:
`src/abba/server/mcp.py`

Pattern:
- `@mcp.tool()` on each exported tool

Assessment:
- low cognitive load
- explicit registration
- easy to trace

The problem in this module is not decorator usage. The problem is the raw string/dict payload handling underneath.

Verdict:
strong and worth standardizing where framework integration requires it

#### 8. Explicit numerical helper methods are better than fake strategy objects here

Relevant files:
- `src/abba/engine/hockey.py`
- `src/abba/engine/elo.py`
- `src/abba/engine/kelly.py`

These engines mostly use plain methods and dataclasses rather than over-abstracted algorithm objects.

That is good. For this codebase, explicit methods are easier to inspect than a strategy hierarchy would be.

Verdict:
strong and worth standardizing

### Standardize / Avoid Recommendations

#### Standardize On

1. Small explicit service objects with named responsibilities
   Example shape:
   - ingestion service
   - storage repository
   - feature builder
   - prediction engine
   - reporting/workflow layer

2. Dataclasses for post-validation domain or result objects
   Keep using them where the values are already validated and computationally meaningful.

3. Dispatcher tables for top-level command routing
   Keep them simple and colocate them with validated request models.

4. Explicit procedural orchestration for workflows
   Prefer readable step-by-step code over generic pipeline frameworks.

5. Narrow interfaces only when there is a real substitution use case
   No speculative protocols.

#### Ban Or Avoid

1. String-keyed dependency injection containers for in-repo services

2. Generic `dict`-in/`dict`-out protocols and ABCs that erase domain meaning

3. Utility classes that hide domain policy behind generic helpers

4. Mixin-heavy god-objects for core domain behavior

5. Registry metadata duplicated manually from executable behavior

6. “Refactored” parallel subsystems that replicate active logic without a clear migration boundary

### Places Where Explicit Code Beats Abstraction

1. Prediction math in `src/abba/engine/hockey.py`
   Explicit formulas are better than strategy-object hierarchies here.

2. Kelly, EV, and Elo computations in `src/abba/engine/kelly.py`, `src/abba/engine/value.py`, and `src/abba/engine/elo.py`
   These should stay straightforward and inspectable.

3. Workflow sequencing in `src/abba/workflows/engine.py`
   Explicit multi-step code is preferable to generic pipeline combinators.

4. Storage operations in `src/abba/storage/duckdb.py`
   A direct repository is better than adding DI and interface indirection on top of it.

### Concrete Refactors For The Worst Offenders

1. Break `ABBAToolkit` apart without changing the external API immediately
   - keep the facade class
   - replace mixin inheritance with composed handler/service objects
   - move `_track`, session state, and refresh state into explicit collaborators

2. Remove `DependencyContainer` from production-facing paths
   - instantiate concrete services directly
   - keep only genuinely reusable computational services
   - delete string-keyed service lookups where no real substitution exists

3. Collapse `analytics/interfaces.py` to the minimum viable abstraction set
   - keep only interfaces with real alternate implementations
   - replace `dict`-based protocols with typed request/result objects or remove them

4. Replace `data_utils` and `validation_utils` with boundary-specific code
   - preprocessing belongs in explicit pipeline components
   - validation belongs at ingress and repository boundaries

5. Unify tool metadata and dispatch
   - define one typed request model per tool
   - derive registry metadata from those models
   - stop maintaining parallel schema dictionaries by hand

### Bottom Line

This codebase’s biggest design risk is not lack of abstraction. It is decorative abstraction:

- mixins that hide a god-object
- protocols that erase semantics
- DI that adds indirection without real substitution
- utility layers that hide policy

The code is strongest where it is most direct:

- compact engines
- explicit procedural workflows
- small dataclasses
- plain dispatcher tables

The standard for this repo should be ruthless:

- if an abstraction does not reduce trace cost for a new engineer, it is probably making the system worse
- if an abstraction does not strengthen correctness, it should not survive on aesthetics alone

## Standard Library Underuse Pass

### Standard-Library Replacement Opportunities

| Rank | Severity | File / symbol | Current pattern | Better stdlib feature | Gain | Why it is a better fit |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | High | `src/abba/engine/graph.py::_betweenness_centrality` | list used as FIFO queue via `queue = [s]` and `queue.pop(0)` | `collections.deque` | Performance, readability | This is the canonical stdlib queue for BFS. It removes O(n) front-pop churn and makes intent explicit. |
| 2 | Medium | `src/abba/analytics/personalization.py::analyze_patterns` and `_analyze_time_patterns`; `src/abba/agents_modules/dynamic_orchestrator.py` summary counting | repeated `counts[key] = counts.get(key, 0) + 1` and manual sorting of count maps | `collections.Counter` | Readability, maintainability | These are counting problems, not generic dict problems. `Counter` plus `most_common()` reduces custom counting machinery and makes the code’s intent clearer. |
| 3 | Medium | `src/abba/workflows/engine.py::cap_strategy`; `src/abba/agents_modules/dynamic_orchestrator.py` debate summary | `sorted(..., reverse=True)[:5]` to get top N | `heapq.nlargest` | Performance, readability | When only the top few items are needed, `nlargest` states that directly and avoids sorting the full collection. |
| 4 | Medium | `src/abba/analytics/advanced_analytics.py` and `src/abba/analytics/refactored_analytics_manager.py` model-dir setup | string paths plus `os.makedirs(...)` | `pathlib.Path` and `Path.mkdir()` | Readability, maintainability | The repo already uses `Path` elsewhere. These modules regress to stringly path handling for no benefit. |
| 5 | Medium | `src/abba/engine/ensemble.py`; `src/abba/server/tools/registry.py`; `src/abba/workflows/engine.py`; `src/abba/server/mcp.py`; `src/abba/connectors/live.py` | string tags for methods, sources, phases, workflows, market types | `enum.StrEnum` or `Enum` | Correctness, readability, maintainability | These are finite vocabularies. A stdlib enum would centralize legal values and reduce typo-driven fallback behavior. |
| 6 | Low | `src/abba/analytics/personalization.py::_calculate_risk_tolerance` | `np.mean`/`np.var` over small Python lists | `statistics.fmean`, `statistics.pvariance` | Readability, dependency hygiene | For small pure-Python collections, `statistics` expresses intent well. This is optional, not a must-change, because NumPy is already in use. |

### Worthwhile Changes

#### 1. Replace list-queue BFS with `collections.deque`

Relevant file:
`src/abba/engine/graph.py::_betweenness_centrality`

Current pattern:
- `queue = [s]`
- `v = queue.pop(0)`

Why stdlib fits better:
- `deque` is the standard queue primitive
- the change is mechanically simple
- the intent becomes obvious to any engineer reading the traversal

This is a worthwhile change, not a style preference.

#### 2. Use `Counter` where the code is literally counting

Relevant files:
- `src/abba/analytics/personalization.py`
- `src/abba/agents_modules/dynamic_orchestrator.py`

Current patterns:
- `sports_counts[sport] = sports_counts.get(sport, 0) + 1`
- `participant_counts[participant] = participant_counts.get(participant, 0) + 1`

Why stdlib fits better:
- these are textbook `Counter` cases
- `most_common()` removes separate sort logic
- reduces custom counting boilerplate scattered across the repo

This is a readability/maintainability win.

#### 3. Use `heapq.nlargest` when only the top few entries matter

Relevant files:
- `src/abba/workflows/engine.py::cap_strategy`
- `src/abba/agents_modules/dynamic_orchestrator.py` debate summary

Current patterns:
- `sorted(..., reverse=True)[:5]`

Why stdlib fits better:
- selecting a small top-N is exactly what `heapq.nlargest` communicates
- avoids sorting an entire collection when only 5 items are needed

This is a modest but clean improvement.

#### 4. Standardize on `pathlib.Path` for model/cache directories

Relevant files:
- `src/abba/analytics/advanced_analytics.py`
- `src/abba/analytics/refactored_analytics_manager.py`

Current pattern:
- `self.models_dir = "models/advanced_analytics"`
- `os.makedirs(self.models_dir, exist_ok=True)`

Why stdlib fits better:
- the repo already uses `Path` in `src/abba/core/config.py`
- `Path` composes better with file operations and reduces string path juggling

This is mainly a maintainability/readability improvement.

#### 5. Use `StrEnum` for real finite vocabularies

Relevant files:
- `src/abba/engine/ensemble.py` method selection
- `src/abba/server/tools/registry.py` tool/workflow/source enums in metadata only
- `src/abba/server/mcp.py` `think()` phase values
- `src/abba/connectors/live.py` `market_type`

Why stdlib fits better:
- these are domain tags, not arbitrary strings
- using `StrEnum` would reduce typo risk and centralize allowed values
- aligns with earlier findings about weak stringly dispatch

This is primarily a correctness/maintainability gain.

### Cases Where Existing Code Is Already Better Than The “Pythonic” Alternative

#### 1. `defaultdict` in Elo is correct and should stay

Relevant file:
`src/abba/engine/elo.py`

Current pattern:
- `self._ratings: dict[str, float] = defaultdict(lambda: self.initial_rating)`

Why it should stay:
- this is clearer than repeated `setdefault` calls
- the default behavior is intrinsic to the model

#### 2. `math` is already the right tool in the numerical core

Relevant files:
- `src/abba/engine/elo.py`
- `src/abba/engine/confidence.py`
- `src/abba/engine/hockey.py`

Why it should stay:
- scalar numeric formulas belong on `math`
- replacing these with `Decimal` or `Fraction` would not improve correctness for this probabilistic/Numpy-heavy workload
- it would complicate interop with NumPy and scientific calculations

#### 3. `list(dict.fromkeys(...))` for stable de-duplication is acceptable

Relevant file:
`src/abba/server/tools/reasoning.py::session_replay`

Current pattern:
- `list(dict.fromkeys(all_uncertainties))`
- `list(dict.fromkeys(all_gaps))`

Why it should stay:
- stable first-seen deduplication is the actual intent
- more “clever” itertools recipes would be harder to read

#### 4. Explicit loops are better than forcing itertools into workflows

Relevant file:
`src/abba/workflows/engine.py`

Why it should stay:
- the workflows are already hard enough to reason about
- replacing explicit sequential steps with iterator-heavy pipelines would worsen traceability

#### 5. Dataclasses are already the right standard-library choice in result containers

Relevant files:
- `src/abba/engine/kelly.py::KellyResult`
- `src/abba/engine/ensemble.py::PredictionResult`
- `src/abba/engine/confidence.py::PredictionConfidence`

Why it should stay:
- these are lightweight named value containers
- a more “dynamic” dict style would be worse

### Bottom Line

The strongest standard-library opportunities in this repo are not exotic. They are the obvious ones:

- `deque` for BFS queues
- `Counter` for counting
- `heapq.nlargest` for top-N
- `Path` for paths
- `StrEnum` for real tag vocabularies

Those are worthwhile because they reduce bespoke machinery and make intent clearer.

The repo does not need a crusade to replace every NumPy call or every explicit loop with a stdlib trick. In several places, the current direct code is already better than a more performatively “Pythonic” alternative.

## Adversarial "Find Bullshit" Pass

### Likely Bullshit

#### [Critical] `src/abba/engine/confidence.py::_compute_confidence_interval`, `build_prediction_meta`, `build_workflow_meta`

What it pretends to be:
- confidence intervals and reliability metadata attached to predictions and workflows

What it actually is:
- a hand-built narrative layer that multiplies a fixed `_BASE_CALIBRATION_ERROR` by a few heuristics (`games_played`, goalie availability, `seed` vs `live`) and then emits a bounded interval with grades and caveats

Why this is bullshit:
- nothing in this path estimates forecast error from versioned empirical residuals
- nothing conditions on model family, sport, season, or calibration dataset
- workflow confidence is not even forecast confidence; it is weakest-link storytelling over result structure
- the output shape looks like risk quantification even when it is only policy fiction

Why a stakeholder would be fooled:
- fields like `confidence_interval`, `reliability_grade`, and `accuracy_history` look like validated forecast diagnostics rather than handcrafted metadata

Call chain:
- `src/abba/server/tools/nhl.py::nhl_predict_game`
- `src/abba/engine/confidence.py::build_prediction_meta`
- `src/abba/workflows/engine.py::WorkflowEngine.run`
- `src/abba/engine/confidence.py::build_workflow_meta`

Required fix:
- stop calling this a confidence interval until it is backed by empirical calibration
- rename it to `heuristic_uncertainty` or remove it from user-facing outputs

#### [Critical] `src/abba/engine/ensemble.py::EnsembleEngine`

What it pretends to be:
- “statistically sound methods”
- inverse-variance weighting
- 95% confidence intervals

What it actually is:
- mean, median, vote fraction, or inverse-distance-from-group-mean weighting over a small list of model outputs
- spread-based confidence and margin formulas with no model-error justification

Why this is bullshit:
- inverse distance from the ensemble mean is not inverse-variance weighting
- `1.96 * std / sqrt(n)` over correlated model predictions is not a justified predictive interval
- disagreement between models is being sold as uncertainty about the world

Why a stakeholder would be fooled:
- the docstring and method names imply standard ensemble statistics
- the output contains `confidence` and `error_margin`, so it looks calibrated

Required fix:
- rewrite the documentation to match the implementation
- either fit real stacking/calibration or present this as a heuristic aggregator only

#### [High] `src/abba/analytics/graph.py::GraphAnalyzer`

What it pretends to be:
- graph-theoretic centrality analysis

What it actually is:
- degree plus homemade proxies labeled as `closeness`, `betweenness`, and `eigenvector`

Why this is bullshit:
- the code explicitly says “simplified” while still exporting canonical metric names
- `_calculate_closeness_centrality` is normalized degree, not closeness
- `_calculate_betweenness_centrality` is an approximation based on degree/clustering
- `_calculate_eigenvector_centrality` is not presented with convergence guarantees or oracle validation

Why a stakeholder would be fooled:
- reports can cite familiar network-analysis terms that are not what the code computes

Required fix:
- either replace with `networkx`/validated implementations or rename every metric to `proxy_*`

#### [High] `src/abba/server/tools/nhl.py::nhl_predict_game`, `src/abba/server/mcp.py::nhl_predict_game`

What it pretends to be:
- a comprehensive NHL predictor using Corsi, xG, goaltending, Elo, player impact, special teams, rest, and market intelligence

What it actually is:
- a six-model heuristic core with optional market blend and optional Elo append, fed by a feature builder that silently substitutes league-average or neutral defaults for missing advanced stats, goalies, and rest

Why this is bullshit:
- the surface language is stronger than the evidence path
- player impact is bolted on as extra feature fields and caveats, not a clearly isolated validated model
- missing inputs degrade to plausible neutral values instead of a hard failure

Why a stakeholder would be fooled:
- the API description sounds richer than the active evidence in the prediction path
- returned feature dictionaries make the model look fully informed even when multiple features are defaults

Required fix:
- expose which feature groups were real vs imputed
- fail closed when core NHL inputs are absent

#### [High] `src/abba/engine/hockey.py::playoff_probability`

What it pretends to be:
- playoff probability from Monte Carlo simulation

What it actually is:
- a cutline-hitting simulator over one team’s regressed points process, with no league-table coupling and no seeded reproducibility

Why this is bullshit:
- it does not simulate the competitive environment required for real playoff odds
- division and wildcard probabilities are threshold exceedance heuristics, not standings probabilities
- fresh RNG each call means identical inputs can produce different answers

Why a stakeholder would be fooled:
- “Monte Carlo” and `50000` simulations sound rigorous

Required fix:
- rename to `cutline_probability_heuristic` or replace with a standings simulation over all competitors

### Probable Bugs

#### [Critical] `tests/test_mcp.py` is testing an API that no longer exists

Evidence:
- imports `create_mcp_server`, `handle_mcp_request`, and `tools_to_mcp_schema`
- current `src/abba/server/mcp.py` exposes FastMCP tool functions and `run_stdio_server()`

Why this matters:
- this is not just stale coverage; it is proof that part of the test suite is detached from the live surface
- a passing or ignored suite here would create false confidence about the transport boundary

#### [High] `src/abba/engine/ensemble.py::combine` silently maps unknown methods to arithmetic mean

Evidence:
- any `method` other than `"weighted"`, `"median"`, or `"voting"` falls through the final `else` branch

Why this is a bug:
- typoed or unsupported methods quietly change algorithm instead of failing
- consumers can believe they ran one ensemble method while receiving another

#### [High] `src/abba/engine/ensemble.py::_weighted_combine` can emit garbage for explicit zero-sum or near-zero-sum weights

Evidence:
- explicit `weights` are normalized by `w / w.sum()` without a zero-sum guard

Why this is a bug:
- zero or near-canceling weights can produce `NaN`, `inf`, or absurd sensitivity

#### [Medium] `src/abba/workflows/engine.py::_infer_data_sources` defaults to `"seed"` from shape, not provenance

Evidence:
- if the workflow result lacks recognizable nested metadata, it appends `"seed"` by default

Why this is a bug:
- confidence labeling can become detached from real upstream data source
- metadata consumers may think a workflow used seed data solely because the result schema was thin

### Dangerous Assumptions

#### [Critical] Missing core NHL inputs are harmless enough to replace with neutral defaults

Where:
- `src/abba/engine/hockey.py::build_nhl_features`

Examples:
- missing advanced stats become `0.50` share metrics
- missing goalie data becomes `goaltender_edge = 0.0`
- missing rest data becomes `rest_edge = 0.0`
- missing record fields fall back to fabricated season records like `40-30-10`

Why this is dangerous:
- incomplete or broken ingestion produces outputs that still look model-derived
- the system degrades into fiction without telling the caller it has stopped using real evidence

#### [Critical] Tests assume seeded randomness is evidence of functional validity

Where:
- `tests/test_toolkit.py::test_find_value`
- `tests/test_toolkit.py::test_nhl_predict_game`
- `tests/test_toolkit.py::test_playoff_odds`

Why this is dangerous:
- these tests mostly assert that something is returned, not that the math is right
- “there should be some opportunities” is not a correctness criterion; it is synthetic-data theater

#### [High] Result structure is treated as evidence quality

Where:
- `src/abba/workflows/engine.py::WorkflowEngine.run`
- `src/abba/workflows/engine.py::_infer_data_sources`

Why this is dangerous:
- the workflow layer infers provenance and confidence from presence of fields like `home_goaltender` or nested confidence objects
- that means output formatting can change trust metadata without any change in underlying data quality

#### [High] Utility-layer failures can be normalized into continuation

Where:
- `src/abba/utils/data_utils.py`
- `src/abba/utils/validation_utils.py`

Why this is dangerous:
- many helpers catch exceptions and return original data, default booleans, or permissive result dicts
- transformation bugs and bad inputs can therefore survive as “valid enough” data

### Misleading Abstractions

#### [Critical] `src/abba/server/toolkit.py::ABBAToolkit`

Why it is misleading:
- it packages storage, prediction, caching, workflows, confidence decoration, and transport-facing tool semantics behind one friendly object
- the mixin shape looks modular, but the runtime object is a god-object with hidden state and mutable dependencies

Correctness risk:
- callers cannot tell which outputs depend on cached state, seeded data, live refreshes, or heuristic post-processing
- cross-cutting concerns become impossible to isolate for oracle testing

#### [High] `src/abba/core/dependency_injection.py`

Why it is misleading:
- it presents an enterprise DI surface with abstract services, but the implementations are placeholders and generic processors

Correctness risk:
- it makes the system look architecturally disciplined while adding almost no semantic protection around mathematical code
- future engineers can spend time extending abstraction theater instead of isolating domain invariants

#### [High] `src/abba/utils/data_utils.py` and `src/abba/utils/validation_utils.py`

Why they are misleading:
- generic “data” and “validation” utilities imply reusable safety infrastructure
- in practice they mix imputation, parsing, normalization, warnings, and error swallowing

Correctness risk:
- domain rules disappear into generic helper calls
- bad preprocessing becomes hard to audit because the abstraction boundary is anti-semantic

#### [Medium] `src/abba/workflows/engine.py::WorkflowEngine`

Why it is misleading:
- it sells itself as orchestration, but it also manufactures narratives, value claims, and workflow-level trust metadata

Correctness risk:
- engineers cannot swap numerical logic without also rewriting presentation logic and meta-judgment code

### Highest-Value Fixes

#### [1] Delete fake confidence before adding more math

Do this:
- remove or relabel `confidence_interval`, `reliability_grade`, and workflow confidence as heuristic metadata
- require empirical backtesting artifacts before reintroducing calibrated uncertainty

Why first:
- this is the most dangerous lie in the system because it makes weak outputs look decision-ready

#### [2] Fail closed on missing mathematical inputs

Do this:
- in `src/abba/engine/hockey.py::build_nhl_features`, stop substituting neutral defaults for missing advanced stats, goalie data, and season context
- return explicit missing-feature errors or degraded-mode flags that block “full model” claims

Why second:
- silent imputation is the main path from bad data to plausible nonsense

#### [3] Rewrite the tests so they can actually catch wrong math

Do this:
- replace key-presence tests with oracle comparisons, property tests, and controlled toy cases
- delete or quarantine `tests/test_mcp.py` until it targets the current interface

Why third:
- the current suite is too good at proving the code runs and too bad at proving the answers mean anything

#### [4] Stop overclaiming algorithm names

Do this:
- rename fake graph metrics to proxies
- remove “statistically sound” and “inverse-variance” wording from the ensemble code unless implemented
- rename playoff odds if the model is only a one-team cutline simulator

Why fourth:
- false names are not cosmetic; they directly mislead maintainers and users about epistemic status

#### [5] Break up the god-object boundary

Do this:
- isolate pure mathematical kernels from storage, caching, and transport mixins
- require explicit typed inputs and outputs for the prediction core

Why fifth:
- until the math is independently callable and inspectable, every other trust repair stays expensive

### Bottom Line

The codebase has several places where “looks quantitative” is being used as a substitute for “is justified.”

The worst offenders are:
- heuristic uncertainty sold as confidence
- model descriptions that overstate what the live path really uses
- tests that prove liveness and shape rather than correctness
- neutral defaults that turn missing evidence into respectable-looking predictions

That is the dangerous combination. It does not fail loudly. It fails with plausible numbers.
