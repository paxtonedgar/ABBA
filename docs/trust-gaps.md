# ABBA Trust Gap Ledger

> Generated 2026-03-10 from architecture dossier audit, claim scrutinizer review, and data pipeline investigation.
> Current system trust score: **4/10**. Target: **7-8/10**.

---

## How to use this ledger

Each entry follows a fixed structure. Status tracks: `open` | `in-progress` | `fixed` | `wont-fix`.
Money-path = code that directly affects a prediction, EV calculation, or bet-sizing recommendation shown to the user.

---

## DATA PIPELINE — Empty Results

### TG-001: Cold-start refresh fetches nothing

- **Issue**: `refresh()` only fetches rosters/goalies for teams with scheduled games already in storage. On cold start (empty DB), `scheduled_games = []` so `teams_to_fetch` is empty. No rosters or goalies are ever fetched.
- **Evidence**: `src/abba/connectors/live.py:97-100` — `storage.query_games(sport="NHL", status="scheduled")` returns `[]` when DB is fresh. The subsequent loop `for g in scheduled_games` never executes.
- **Impact**: All goalie, roster, and player-impact queries return empty on first use. Predictions fall back to neutral defaults silently.
- **Money path?**: Yes. Goalie matchup is a model input; empty goalie data degrades prediction without warning.
- **Missing invariant**: `refresh()` must populate at least team_stats and schedule before attempting roster/goalie fetch. Or: fetch all 32 teams unconditionally on first refresh.
- **Test**: `test_refresh_cold_start()` — call `refresh()` on empty storage, assert `query_goaltender_stats()` returns >0 records for at least 20 teams.
- **Code change**: In `live.py:refresh()`, if `scheduled_games` is empty after fetching schedule, re-query. If still empty, fetch rosters for all 32 NHL teams as fallback.
- **Status**: open
- **Owner**: unassigned

---

### TG-002: Goalie `role` field missing from live data

- **Issue**: `nhl.py:49` filters goalies by `role == "starter"`. Seed data sets `role: "starter"` but `_fetch_goaltender_stats()` never sets a `role` field on live-fetched goalies. Starter lookup silently returns nothing.
- **Evidence**: `src/abba/connectors/live.py:243-285` — goalie dict includes `games_played`, `games_started`, `save_pct`, `gaa` but no `role`. Compare `src/abba/connectors/seed.py:370-400` which hardcodes `"role": "starter"`.
- **Impact**: After live refresh replaces seed data, goalie matchup model gets `None` for both starters. Falls back to neutral `goaltender_edge = 0.0`.
- **Money path?**: Yes. Goalie matchup directly adjusts home win probability.
- **Missing invariant**: Every goalie record in storage must have a `role` field. The goalie with max `games_started` should be tagged `"starter"`.
- **Test**: `test_live_goalies_have_role()` — refresh live data, query goalies, assert every team has exactly one goalie with `role == "starter"`.
- **Code change**: In `_fetch_goaltender_stats()`, after fetching all goalies for a team, sort by `games_started` descending, tag `[0]` as `"starter"`, rest as `"backup"`.
- **Status**: open
- **Owner**: unassigned

---

### TG-003: Missing season parameter in workflow queries

- **Issue**: `engine.py:138-139` calls `query_goaltender_stats(team=home)` without `season`. If storage has multi-season data, result is unfiltered. `next()` picks first matching starter which could be from a prior season.
- **Evidence**: `src/abba/workflows/engine.py:138` — `home_goalies = self.toolkit.query_goaltender_stats(team=home)`. Also at line 226. Contrast with line 333-335 which correctly passes `season="2025-26"`.
- **Impact**: Stale season goalie stats used for current predictions. Could show a traded goalie as starter.
- **Money path?**: Yes. Wrong goalie stats → wrong matchup edge → wrong probability.
- **Missing invariant**: All prediction-path queries must include an explicit season parameter.
- **Test**: `test_workflow_queries_include_season()` — mock storage, run workflow, assert every `query_goaltender_stats` call includes `season` kwarg.
- **Code change**: Pass `season` parameter to all `query_goaltender_stats()` and `query_roster()` calls in `engine.py`. Derive season from game date or current season constant.
- **Status**: open
- **Owner**: unassigned

---

### TG-004: Case-sensitive team ID queries

- **Issue**: `query_roster()` and `query_salary_cap()` use `team = ?` (exact, case-sensitive). `query_goaltender_stats()` uses `team ILIKE ?` (case-insensitive). Inconsistent — lowercase input returns empty from some queries but not others.
- **Evidence**: `src/abba/storage/duckdb.py:695` — `conditions.append("team = ?")`. Line 735 — same. Line 622 — `conditions.append("team ILIKE ?")`.
- **Impact**: Any caller passing lowercase team IDs (e.g., from NHL API normalization) gets empty results from roster/cap queries.
- **Money path?**: Indirect. Roster feeds player impact; cap data feeds context narratives.
- **Missing invariant**: Team IDs must be normalized to uppercase at storage write boundary. All queries should be case-insensitive or inputs normalized before query.
- **Test**: `test_case_insensitive_team_queries()` — insert data with "NYR", query with "nyr", "Nyr", "NYR" — all must return same results.
- **Code change**: Add `.upper()` normalization in storage `upsert_*` methods and query methods, or switch all queries to `ILIKE`.
- **Status**: open
- **Owner**: unassigned

---

### TG-005: `_fetch_goaltender_stats` returns success on empty data

- **Issue**: When NHL API returns no goalie data for a team, the method returns `{"status": "ok", "goalies_stored": 0}`. Caller treats this as success. No data is stored, but no error is raised.
- **Evidence**: `src/abba/connectors/live.py:249-251` — `if not goalies: return {"status": "ok", "goalies_stored": 0, "note": "no goalie data in response"}`.
- **Impact**: Teams with API issues silently have no goalie data. Downstream queries return empty.
- **Money path?**: Yes. Silent missing goalie data → neutral matchup → wrong probability.
- **Missing invariant**: If a team exists in the NHL, they have goalies. Zero goalies returned should be flagged as a data quality issue, not silently accepted.
- **Test**: `test_empty_goalie_response_flagged()` — mock API returning empty goalies array, assert result has `status: "warning"` or `"degraded"`.
- **Code change**: Return `{"status": "warning", "goalies_stored": 0, "reason": "api_returned_empty"}`. Aggregate warnings in refresh manifest.
- **Status**: open
- **Owner**: unassigned

---

### TG-006: No ORDER BY on multi-season team stats queries

- **Issue**: `query_team_stats()` returns results without `ORDER BY`. When multiple seasons exist, `home_stats_list[0]` picks an arbitrary row. Could be last season's stats.
- **Evidence**: `src/abba/storage/duckdb.py:query_team_stats()` — SQL has no ORDER BY clause. `src/abba/server/tools/nhl.py:37-38` — `home_stats = home_stats_list[0] if home_stats_list else {"stats": {}}`.
- **Impact**: Prediction built on wrong season's stats. Plausible but incorrect output.
- **Money path?**: Yes. Team stats are the foundation of every model.
- **Missing invariant**: Prediction-path queries must return current season data deterministically. Either filter by season or ORDER BY season DESC.
- **Test**: `test_multi_season_returns_latest()` — insert stats for 2024-25 and 2025-26, query without season, assert first result is 2025-26.
- **Code change**: Add `ORDER BY season DESC` to `query_team_stats()`. Better: require season parameter in prediction paths.
- **Status**: open
- **Owner**: unassigned

---

### TG-007: Empty stats fallback produces zero-value predictions

- **Issue**: When `query_team_stats()` returns empty, code falls back to `{"stats": {}}`. All derived features become 0. Prediction runs on zeros — produces a result that looks real but means nothing.
- **Evidence**: `src/abba/server/tools/nhl.py:37-38` — `home_stats = home_stats_list[0] if home_stats_list else {"stats": {}}`. Then `home_games_played = 0`, all rates = 0.
- **Impact**: User sees a prediction with a probability and confidence interval for a game where we have no data. This is the core "silent degradation" problem.
- **Money path?**: Yes. This is the #1 correctness threat identified by the dossier.
- **Missing invariant**: `nhl_predict_game()` must fail closed when team stats are missing. Return an error, not a fake prediction.
- **Test**: `test_predict_fails_on_missing_stats()` — delete team stats for one team, call `nhl_predict_game()`, assert result contains `"error"` key, not a probability.
- **Code change**: After fetching `home_stats_list` and `away_stats_list`, if either is empty, return `{"error": "missing_team_stats", "missing": [...], "recommendation": "call refresh_data first"}`.
- **Status**: open
- **Owner**: unassigned

---

### TG-008: `_last_refresh_ts` always None

- **Issue**: `toolkit.py:86` initializes `_last_refresh_ts = None`. It's only set in `session.py:44` after successful refresh. But if the MCP server is restarted (new process), it resets to None even if DB has fresh data.
- **Evidence**: `src/abba/server/toolkit.py:86` — `self._last_refresh_ts: float | None = None`. `src/abba/engine/confidence.py` uses this to compute freshness grade.
- **Impact**: Confidence reliability can never reach A or B grade. Always reports stale data even after successful refresh + server restart.
- **Money path?**: Indirect. Confidence metadata shown to user is always pessimistic/wrong.
- **Missing invariant**: Freshness should be tracked per-table in storage (last successful write timestamp), not as an in-memory scalar.
- **Test**: `test_freshness_survives_restart()` — refresh data, create new toolkit instance pointing to same DB, assert `_last_refresh_ts` is not None.
- **Code change**: Store refresh timestamps per-table in DuckDB. Query them on toolkit init. Replace in-memory scalar with storage-backed freshness.
- **Status**: open
- **Owner**: unassigned

---

## SILENT DEGRADATION / CORRECTNESS

### TG-009: Ghost features — Corsi, xG advertised but not consumed

- **Issue**: `build_nhl_features()` outputs `home_corsi_pct`, `away_corsi_pct`, `home_xgf_pct`, `away_xgf_pct` but `predict_nhl_game()` never reads them. They exist in the feature dict but have zero effect on the forecast.
- **Evidence**: `src/abba/engine/hockey.py:build_nhl_features()` — sets these keys. `predict_nhl_game()` — reads `points_pct`, `pythag`, `gd_per_game`, `recent_form`, `goaltender_edge` but never `corsi` or `xgf`. Dossier section "Feature Ghost Claims".
- **Impact**: Users and tool descriptions claim 8-model prediction including Corsi and xG. In reality, these features are decoration. Predictions are less sophisticated than advertised.
- **Money path?**: Yes (by omission). Missing signal that should improve predictions.
- **Missing invariant**: Every feature in `NHL_FEATURE_SCHEMA` marked as model input must appear in at least one model's scoring path. Features not consumed must not appear in output.
- **Test**: `test_corsi_perturbation_changes_output()` — perturb `home_corsi_pct` by 20%, assert prediction changes. (Currently fails — proving the feature is dead.)
- **Code change**: Phase 2: remove ghost features from output. Phase 4: create AnalyticsModel component that actually consumes Corsi/xG when available.
- **Status**: open
- **Owner**: unassigned

---

### TG-010: Rest computed but never fed to model

- **Issue**: `WorkflowEngine` computes `rest_info` (rest days, back-to-back) and narrates it to the user. But calls `nhl_predict_game(game_id)` without passing rest data. Inside the model, `rest_edge` defaults to `0.0`.
- **Evidence**: `src/abba/workflows/engine.py:145` — computes `rest_info`. Line 148 — calls `nhl_predict_game(gid)` with no rest parameter. `src/abba/engine/hockey.py:build_nhl_features()` — `rest_edge` defaults to `0.0` when not provided.
- **Impact**: User sees "Team A is on a back-to-back" in the narrative but the prediction doesn't account for it. Misleading.
- **Money path?**: Yes. Rest advantage is a known NHL edge (1-3% per studies). Users may bet based on narrated rest info that isn't in the model.
- **Missing invariant**: If a feature is narrated to the user as relevant, it must affect the prediction.
- **Test**: `test_rest_affects_prediction()` — predict same game with 0 rest days vs 3 rest days, assert probability differs.
- **Code change**: Pass `rest_info` from workflow to `nhl_predict_game()`. Wire `rest_edge` into the composite model scoring.
- **Status**: open
- **Owner**: unassigned

---

### TG-011: Confidence theater — hard-coded baselines as calibration

- **Issue**: `build_prediction_meta()` uses `_DEFAULT_ACCURACY_HISTORY` — a hard-coded dict of fake backtest results — to compute confidence intervals and reliability grades. These numbers have no connection to the current model version.
- **Evidence**: `src/abba/engine/confidence.py` — `_DEFAULT_ACCURACY_HISTORY` with entries like `"nhl": {"log_loss": 0.67, "brier": 0.24, "n_predictions": 500}`. These are presented as calibration data but are fabricated baselines.
- **Impact**: Confidence intervals and reliability grades appear scientifically grounded but are fiction. Users trust the "B" grade and "80% CI [0.42, 0.68]" as if they were empirically validated.
- **Money path?**: Yes. Users size bets based on confidence metadata.
- **Missing invariant**: Confidence metadata must be linked to a versioned calibration artifact from a leakage-free backtest, or must be labeled `"uncalibrated"`.
- **Test**: `test_confidence_declares_uncalibrated()` — call `build_prediction_meta()` without loading a calibration artifact, assert output contains `"calibration_status": "uncalibrated"`.
- **Code change**: Replace `_DEFAULT_ACCURACY_HISTORY` with a `CalibrationArtifact` loader. If no artifact exists, return `reliability_grade: "U"` (uncalibrated) and `calibration_status: "uncalibrated"`. Keep CI computation but label it as estimated, not calibrated.
- **Status**: open
- **Owner**: unassigned

---

### TG-012: Ensemble mislabeled as inverse-variance weighting

- **Issue**: `_weighted_combine()` docstring says "Inverse-variance weighting" but the implementation computes consensus-proximity weights (inverse distance from group mean). These are fundamentally different statistical methods.
- **Evidence**: `src/abba/engine/ensemble.py:_weighted_combine()` — weights are `1 / (distance_from_mean + epsilon)`, not `1 / variance_of_model_i`. Module docstring: "weighted: default, weights by inverse variance".
- **Impact**: Anyone reviewing the code or extending it will make wrong assumptions about the statistical properties of the ensemble. Inverse-variance weighting has known optimality properties; consensus weighting does not.
- **Money path?**: Indirect. The weighting method affects the final probability shown to the user.
- **Missing invariant**: Algorithm names in docstrings and comments must match the implementation.
- **Test**: `test_ensemble_method_name_matches_implementation()` — verify that the method's behavior matches its documented name by comparing against a reference implementation of actual inverse-variance weighting.
- **Code change**: Rename to `"consensus"` method. Update all docstrings. If true inverse-variance is desired, implement it separately with per-model variance estimates.
- **Status**: open
- **Owner**: unassigned

---

### TG-013: Workflow math contamination — invented confidence

- **Issue**: `WorkflowEngine` invents its own confidence score: `abs(pred_val - 0.5) * 200`, applies ad-hoc risk heuristics, and formats narratives referencing features the model doesn't use.
- **Evidence**: `src/abba/workflows/engine.py` — confidence calculation independent of `build_prediction_meta()`. Risk scaling applied to Kelly sizing outside of `KellyEngine`.
- **Impact**: Two competing confidence numbers: one from the confidence engine, one from the workflow. User sees workflow-invented confidence which has no statistical meaning.
- **Money path?**: Yes. Workflow confidence feeds into narrative recommendations and risk assessments.
- **Missing invariant**: Only one code path should produce confidence metadata. Workflows must consume, not invent, confidence values.
- **Test**: `test_workflow_uses_prediction_confidence()` — run workflow, assert the confidence value in output matches the one from `predict_game()`, not an independently computed value.
- **Code change**: Remove `abs(pred_val - 0.5) * 200` formula. Use `prediction["model_agreement"]` (renamed from `prediction["confidence"]`) directly.
- **Status**: open
- **Owner**: unassigned

---

### TG-014: Snapshot incoherence — hybrid fresh/stale/absent data

- **Issue**: A single prediction may combine fresh standings (just refreshed), stale seed goalie data (from initialization), absent advanced stats (never populated), and missing odds (no API key). No provenance tracks which inputs are fresh vs stale.
- **Evidence**: Dossier section "Snapshot Incoherence". `nhl_predict_game()` fetches from storage without checking data freshness per table. `_last_refresh_ts` is a single scalar, not per-table.
- **Impact**: Prediction built on internally inconsistent data. E.g., standings show 40 wins but goalie stats are from when team had 20 wins.
- **Money path?**: Yes. Stale inputs produce stale predictions presented as current.
- **Missing invariant**: `NhlGameContext` must record `as_of` timestamp per data source. Predictions must flag when inputs have >24h age spread.
- **Test**: `test_context_detects_stale_data()` — set team stats `as_of` to today, goalie stats `as_of` to 30 days ago, assert context flags data staleness.
- **Code change**: Add `captured_at` column to goaltender_stats, team_stats, roster tables. `build_game_context()` computes max age spread and flags it.
- **Status**: open
- **Owner**: unassigned

---

### TG-015: `explain_prediction` uses hard-coded neutrals, not model attribution

- **Issue**: Feature importance in `explain_prediction()` is computed as deviation from hard-coded "neutral" baselines (0.500, 0.250, etc.), not from actual model gradients or SHAP values.
- **Evidence**: `src/abba/server/tools/analytics.py:explain_prediction()` — computes importance as `abs(feature_value - neutral_value)` for each feature. Dossier section "Abstraction Risk".
- **Impact**: Explanation says "Corsi% is the top driver" when Corsi isn't even consumed by the model. Users make decisions based on fake explanations.
- **Money path?**: Yes. Explanations inform user's trust in the prediction and willingness to bet.
- **Missing invariant**: Feature importance must be computed from the actual model, not from distance to arbitrary baselines.
- **Test**: `test_explanation_reflects_model()` — zero out a feature, assert its importance drops to near-zero in explanation. (Currently fails because explanation is baseline-distance, not model-based.)
- **Code change**: Short-term: add caveat `"method": "baseline_deviation", "note": "not model attribution"`. Long-term: implement permutation importance or sensitivity analysis from actual model components.
- **Status**: open
- **Owner**: unassigned

---

### TG-016: Workflow `_confidence` schema inconsistency

- **Issue**: `WorkflowEngine.run()` sometimes assigns a scalar (model agreement float) to `_confidence` and sometimes assigns a dict (from `build_workflow_meta`). Downstream code may treat it as either type.
- **Evidence**: Dossier lines 1413, 1471. `_confidence` can be `prediction["confidence"]` (float) or `build_workflow_meta(...)` (dict with `reliability_grade` key).
- **Impact**: Type confusion causes potential runtime errors or silent wrong behavior when accessing `_confidence["reliability_grade"]` on a float.
- **Money path?**: Indirect. Confidence metadata displayed to user may be wrong or cause errors.
- **Missing invariant**: `_confidence` must always be a dict with at least `reliability_grade` and `model_agreement` keys.
- **Test**: `test_workflow_confidence_always_dict()` — run every workflow, assert `isinstance(result["_confidence"], dict)` and `"reliability_grade" in result["_confidence"]`.
- **Code change**: Normalize `_confidence` assignment in `WorkflowEngine.run()` to always produce a dict.
- **Status**: open
- **Owner**: unassigned

---

## NUMERICAL STABILITY

### TG-017: xG sigmoid overflow on extreme inputs

- **Issue**: `expected_goals()` uses `1.0 / (1.0 + math.exp(-z))` which overflows for large negative `z` values (e.g., very long-distance shots).
- **Evidence**: `src/abba/engine/hockey.py:expected_goals()` — raw `math.exp(-z)` without clipping.
- **Impact**: `OverflowError` or `inf` propagates through xG computation into prediction.
- **Money path?**: Yes (if xG is wired in). Currently a latent bug since xG features are ghost features.
- **Missing invariant**: Sigmoid must return finite values for all real inputs.
- **Test**: `test_xg_sigmoid_extreme_inputs()` — pass z = -1000 and z = 1000, assert result is finite and in [0, 1].
- **Code change**: Replace with `scipy.special.expit(z)` or manual branch-stable implementation: `if z >= 0: 1/(1+exp(-z)) else: exp(z)/(1+exp(z))`.
- **Status**: open
- **Owner**: unassigned

---

### TG-018: Ensemble near-consensus instability

- **Issue**: `_weighted_combine()` computes `1 / (distance_from_mean + epsilon)` weights. When all predictions are nearly identical, distances approach 0, weights approach infinity, and floating-point noise dominates.
- **Evidence**: `src/abba/engine/ensemble.py:_weighted_combine()` — `epsilon` is small, near-consensus predictions produce extreme weight ratios.
- **Impact**: Ensemble output becomes numerically unstable when models agree. Paradoxically, high agreement produces less reliable computation.
- **Money path?**: Yes. Ensemble output is the final probability.
- **Missing invariant**: When `std(predictions) < 1e-6`, return `mean(predictions)` exactly.
- **Test**: `test_near_consensus_returns_mean()` — pass 6 predictions all within 1e-8 of 0.55, assert output equals 0.55 within 1e-10.
- **Code change**: Add early return: `if np.std(preds) < 1e-6: return float(np.mean(preds))`.
- **Status**: open
- **Owner**: unassigned

---

### TG-019: Elo overflow for extreme rating differences

- **Issue**: `_win_probability()` computes `10^(delta/400)` which overflows for extreme rating differences (>2000 points).
- **Evidence**: `src/abba/engine/elo.py:_win_probability()` — `10 ** (rating_diff / 400)`.
- **Impact**: `OverflowError` or `inf` when teams have extreme rating gaps (possible after long win/loss streaks with high K).
- **Money path?**: Yes. Elo probability is one of 8 model inputs.
- **Missing invariant**: Win probability must return finite value in (0, 1) for all rating inputs.
- **Test**: `test_elo_extreme_ratings()` — compute probability with 3000-point gap, assert result is finite and in (0, 1).
- **Code change**: Use logistic form: `1 / (1 + 10^(-delta/400))` via `scipy.special.expit(delta * log(10) / 400)`.
- **Status**: open
- **Owner**: unassigned

---

### TG-020: Log5 conditioning near singularity

- **Issue**: Log5 formula has denominator `p_a * p_b + (1 - p_a) * (1 - p_b)` which approaches 0 when both teams are near 0 or near 1 win%.
- **Evidence**: `src/abba/engine/features.py` and `src/abba/engine/hockey.py` — log5 implementations with 0.5 fallback when denominator is zero.
- **Impact**: Division by near-zero produces extreme values. Hard 0.5 fallback creates discontinuity.
- **Money path?**: Yes. Log5 is the foundation of points-based and Pythagorean models.
- **Missing invariant**: Log5 inputs must be clipped to [0.001, 0.999]. Output must be continuous.
- **Test**: `test_log5_near_singularity()` — compute log5 with inputs 0.001, 0.999, 0.0, 1.0 — assert finite output, no discontinuity.
- **Code change**: Clip inputs to [0.001, 0.999] before computing denominator. Remove hard 0.5 fallback.
- **Status**: open
- **Owner**: unassigned

---

### TG-021: Playoff simulation not reproducible

- **Issue**: `playoff_probability()` uses Monte Carlo simulation with no seed parameter. Same inputs produce different outputs each call.
- **Evidence**: `src/abba/engine/hockey.py:playoff_probability()` — uses `random.random()` without seeding.
- **Impact**: Cannot write deterministic tests. Users get different numbers on repeated queries. Confusing and untestable.
- **Money path?**: Indirect. Playoff odds inform season-level decisions.
- **Missing invariant**: Stochastic functions must accept an optional seed for reproducibility.
- **Test**: `test_playoff_probability_reproducible()` — call twice with same seed, assert identical output.
- **Code change**: Add `seed: int | None = None` parameter. Use `np.random.default_rng(seed)`. Record seed in output.
- **Status**: open
- **Owner**: unassigned

---

## ARCHITECTURE

### TG-022: Two competing architectures — toolkit vs analytics stack

- **Issue**: `src/abba/analytics/` contains a parallel system (AnalyticsManager, ModelFactory, ensemble, graph, personalization, biometrics) that duplicates toolkit functionality. No import from the toolkit reaches it, but it exists and confuses understanding.
- **Evidence**: `src/abba/analytics/__init__.py`, `manager.py`, `ensemble.py`, `graph.py`, `models.py`, `model_factory.py`, `personalization.py`, `biometrics.py`. Dossier section "Two Competing Architectures".
- **Impact**: Engineers don't know which is canonical. Both may be maintained independently, diverging over time.
- **Money path?**: No (analytics stack is unused). But its existence is a maintenance/confusion hazard.
- **Missing invariant**: One canonical runtime path. All other code archived or deleted.
- **Test**: `test_no_imports_from_analytics()` — grep active source for `from abba.analytics`, assert zero matches.
- **Code change**: Move entire `src/abba/analytics/` to `_archived/analytics/`. Remove from any `__init__.py`.
- **Status**: open
- **Owner**: unassigned

---

### TG-023: God-object toolkit

- **Issue**: `ABBAToolkit` is a mutable container for storage, 6 engines, Elo ratings, session state, caching, and tracking — all reachable from one object. Every mixin has access to every other mixin's state via `self`.
- **Evidence**: `src/abba/server/toolkit.py` — inherits from 7 mixins, holds `self.storage`, `self.ensemble`, `self.features`, `self.kelly`, `self.value`, `self.graph`, `self.hockey`, `self.elo`, `self._session_id`, `self._last_refresh_ts`.
- **Impact**: Impossible to test any tool in isolation. State mutations from one tool affect others. Tight coupling prevents any component from being replaced or tested independently.
- **Money path?**: Indirect. Architectural debt slows all correctness improvements.
- **Missing invariant**: Each service should have explicit, minimal dependencies. No service should access another service's state directly.
- **Test**: `test_services_have_minimal_dependencies()` — instantiate each service independently, assert it works without the full toolkit.
- **Code change**: Extract application services (ForecastService, MarketService, DataService, IngestionService). Toolkit becomes a thin facade delegating to services.
- **Status**: open
- **Owner**: unassigned

---

### TG-024: Dict-and-JSON boundaries erase domain semantics

- **Issue**: Every boundary in the system (storage → tool → engine → response) uses `dict[str, Any]`. Type information exists only in developer knowledge. A goalie stats dict and a team stats dict have the same type signature.
- **Evidence**: All storage `query_*` methods return `list[dict]`. All engine methods accept and return `dict`. `_track()` annotates `result: dict`. Dossier section "Dict-and-JSON Boundaries".
- **Impact**: Wrong dict passed to wrong function produces no error — just wrong output. Refactoring is dangerous because there's no compiler help.
- **Money path?**: Indirect. Enables silent degradation bugs across the system.
- **Missing invariant**: Domain objects (TeamStats, GoalieStats, OddsSnapshot, etc.) should be typed dataclasses or pydantic models.
- **Test**: `test_storage_returns_typed_objects()` — call each query method, assert return type is the expected domain model, not raw dict.
- **Code change**: Define domain models in `src/abba/domain/models.py`. Update storage methods to return typed objects. Update consumers to use typed fields.
- **Status**: open
- **Owner**: unassigned

---

### TG-025: Storage JSON blob schema — untyped stats column

- **Issue**: Team stats, goalie stats, roster entries all store their actual data in a `stats JSON` column. No schema validation on write or read. Any key can be present or absent.
- **Evidence**: `src/abba/storage/duckdb.py` — `stats JSON` column in team_stats, goaltender_stats, roster tables. `json.dumps(stats)` on write, `json.loads()` on read with no validation.
- **Impact**: Seed data and live data may have different keys in the JSON blob. Missing keys cause `KeyError` or silent `None` defaults.
- **Money path?**: Yes. Stats are the raw inputs to all models.
- **Missing invariant**: Stats JSON must conform to a declared schema per table. Unknown keys rejected. Required keys enforced.
- **Test**: `test_stats_schema_validation()` — attempt to store stats dict missing required keys, assert error raised.
- **Code change**: Define required/optional keys per table. Validate on `upsert_*`. Long-term: promote frequently-used JSON keys to typed columns.
- **Status**: open
- **Owner**: unassigned

---

### TG-026: `fetchdf().to_dict("records")` anti-pattern

- **Issue**: Storage query methods materialize a full pandas DataFrame just to convert it to a list of dicts. Unnecessary memory allocation and dependency on pandas for simple row fetching.
- **Evidence**: `src/abba/storage/duckdb.py` — pattern appears in multiple `query_*` methods: `self.conn.execute(sql).fetchdf().to_dict("records")`.
- **Impact**: Performance overhead for large result sets. Unnecessary pandas dependency in hot paths.
- **Money path?**: No. Performance/quality issue.
- **Missing invariant**: Query methods should use `fetchall()` + dict comprehension for simple queries.
- **Test**: N/A (performance improvement, not correctness).
- **Code change**: Replace `fetchdf().to_dict("records")` with `fetchall()` + column-name mapping.
- **Status**: open
- **Owner**: unassigned

---

### TG-027: `query_session_replay` uses pandas merge instead of SQL

- **Issue**: Session replay merges tool_call_log and reasoning_log tables in Python using pandas concat/sort. This should be a SQL UNION ALL with ORDER BY.
- **Evidence**: `src/abba/storage/duckdb.py:query_session_replay()` — fetches two DataFrames, concatenates with `pd.concat`, sorts with `sort_values("created_at")`.
- **Impact**: Inefficient for large sessions. Logic that belongs in the database is in Python.
- **Money path?**: No.
- **Missing invariant**: Set operations on DB tables should be SQL, not Python.
- **Test**: N/A (refactor, not correctness).
- **Code change**: Rewrite as SQL `UNION ALL` with `ORDER BY created_at` and `LIMIT`.
- **Status**: open
- **Owner**: unassigned

---

### TG-028: BFS uses list.pop(0) instead of deque

- **Issue**: `_betweenness_centrality()` in graph engine uses `queue = [s]` / `queue.pop(0)` which is O(n) per pop. Should be `collections.deque` with `popleft()`.
- **Evidence**: `src/abba/engine/graph.py:_betweenness_centrality()` — `queue.pop(0)`.
- **Impact**: O(n^2) BFS instead of O(n). Slow for large team interaction graphs.
- **Money path?**: No. Graph analysis is supplementary.
- **Missing invariant**: BFS implementations must use deque.
- **Test**: N/A (performance, correctness is same).
- **Code change**: `from collections import deque`, replace `queue = [s]` with `queue = deque([s])`, replace `queue.pop(0)` with `queue.popleft()`.
- **Status**: open
- **Owner**: unassigned

---

### TG-029: No live connector for advanced stats

- **Issue**: `nhl_advanced_stats` table (Corsi, Fenwick, xG, PDO) is only populated by seed data. No live connector fetches these from any API. After seed data ages out, advanced stats are empty.
- **Evidence**: `src/abba/connectors/live.py` — no method for advanced stats. `src/abba/connectors/seed.py:290-320` — hardcoded advanced stats for 8 teams. `src/abba/storage/duckdb.py:query_nhl_advanced_stats()` — exists but data source is seed-only.
- **Impact**: Advanced analytics queries return stale or empty data. When Corsi/xG are wired into models (Phase 4), they'll have no live data source.
- **Money path?**: Not yet (ghost features). Will be when features are wired in.
- **Missing invariant**: Every table consumed by the model must have a live data source or an explicit "seed-only" warning.
- **Test**: `test_advanced_stats_refreshable()` — call refresh, assert `nhl_advanced_stats` has data for current season teams.
- **Code change**: Either build a connector for NHL advanced stats API, or clearly mark these features as "seed-only, not available in production" in tool descriptions and model output.
- **Status**: open
- **Owner**: unassigned

---

## TEST QUALITY

### TG-030: Tests prove shape, not mathematical correctness

- **Issue**: Existing tests verify that functions return dicts with expected keys, not that the math is correct. A model returning `0.5` for every game would pass all tests.
- **Evidence**: Dossier section "Test Quality". Tests check `"home_win_prob" in result` and `0 <= result["home_win_prob"] <= 1` but never compare against a known-correct value.
- **Impact**: Bugs in mathematical logic are invisible. Silent degradation cannot be caught.
- **Money path?**: Yes. All math is on the money path.
- **Missing invariant**: Critical math functions must have oracle tests with known-correct reference values.
- **Test**: This IS the test gap. Create oracle tests per TG-031.
- **Code change**: See Phase 6 of remediation plan.
- **Status**: open
- **Owner**: unassigned

---

### TG-031: No oracle tests for critical math

- **Issue**: No test compares xG, Elo, log5, ensemble aggregation, or playoff simulation against independently computed reference values.
- **Evidence**: `tests/` directory — no test file performs oracle comparison. All tests are "does it run and return the right shape?"
- **Impact**: Cannot detect when a code change breaks mathematical correctness. The backtest (TG-032) was the only attempt at validation and it's contaminated.
- **Money path?**: Yes.
- **Missing invariant**: Each critical formula must have at least one hand-computed oracle test case.
- **Test**: Create `test_xg_oracle.py`, `test_elo_oracle.py`, `test_ensemble_oracle.py`, `test_playoff_oracle.py` with hand-verified examples.
- **Code change**: Write the oracle test suite (Phase 6 of remediation plan).
- **Status**: open
- **Owner**: unassigned

---

### TG-032: Contaminated backtest has lookahead bias

- **Issue**: `test_backtest.py` uses data that may include future information in feature construction. The backtest's reported accuracy is unreliable.
- **Evidence**: Dossier section "Statistical Rigor". Backtest does not enforce temporal partitioning or feature-time alignment.
- **Impact**: Any accuracy claims derived from this backtest are suspect. Cannot be used to validate model improvements.
- **Money path?**: Indirect. Backtest results may have been used to justify model confidence baselines.
- **Missing invariant**: Backtest must enforce strict temporal ordering — features computed only from data available before game time.
- **Test**: Archive current backtest. Build replacement with leakage-free temporal partitioning.
- **Code change**: Archive `test_backtest.py`. Build new backtest infrastructure in Phase 6.
- **Status**: open
- **Owner**: unassigned

---

### TG-033: test_mcp.py imports deleted API

- **Issue**: `test_mcp.py` imports `create_mcp_server`, `handle_mcp_request`, `tools_to_mcp_schema` — functions that no longer exist after the FastMCP rewrite.
- **Evidence**: `tests/test_mcp.py` — imports fail at module load time.
- **Impact**: Test file is dead code. MCP server has no test coverage.
- **Money path?**: No. But MCP is the primary user interface.
- **Missing invariant**: Every module should have passing tests or be explicitly excluded.
- **Test**: Rewrite `test_mcp.py` to test FastMCP tool registration and basic tool invocation.
- **Code change**: Delete old test, write new one that imports from `abba.server.mcp` and verifies tool count and basic invocation.
- **Status**: open
- **Owner**: unassigned

---

### TG-034: FeatureEngine pseudo-models (MLB path)

- **Issue**: `FeatureEngine.predict_from_features()` contains heuristic "models" for the general/MLB path that wear model interfaces but are just weighted feature sums.
- **Evidence**: `src/abba/engine/features.py:predict_from_features()`. Dossier section "Algorithm Verification", lines 867-908.
- **Impact**: MLB predictions (when expanded) will inherit the same overclaiming problem as NHL.
- **Money path?**: Not yet (MLB not active). Will be when MLB is added.
- **Missing invariant**: Functions labeled as models must implement actual statistical/ML models, not heuristic sums.
- **Test**: `test_feature_engine_documents_heuristic()` — assert output metadata includes `"method": "heuristic"`, not `"method": "model"`.
- **Code change**: Either clearly label as heuristic or replace with actual model. Deferred until MLB expansion.
- **Status**: open
- **Owner**: unassigned

---

## OVERCLAIMING

### TG-035: Workflow narrates features not used by model

- **Issue**: Workflow narratives describe Corsi%, xG%, rest advantage, injury impact as if they drove the prediction. The model doesn't consume them. User reads analysis that doesn't match the actual math.
- **Evidence**: `src/abba/workflows/engine.py` — narrative templates reference Corsi, xG, rest, injuries. Cross-reference with TG-009 and TG-010.
- **Impact**: User trusts narrative explanations that are disconnected from the actual prediction. Decisions based on false reasoning.
- **Money path?**: Yes. Narratives inform betting decisions.
- **Missing invariant**: Narratives must only reference features that affected the prediction. Unused features should be noted as "available but not yet integrated."
- **Test**: `test_narrative_matches_model_inputs()` — extract feature names from narrative, assert each one is in the model's actual input set.
- **Code change**: Conditionally include narrative sections only for features with non-zero model contribution.
- **Status**: open
- **Owner**: unassigned

---

### TG-036: Tool descriptions overclaim capabilities

- **Issue**: MCP tool descriptions and registry entries advertise capabilities the system doesn't have (e.g., "8-model prediction: Corsi, xG, goaltender matchup, Elo, player impact, special teams, and rest" when several of these are ghost features).
- **Evidence**: `src/abba/server/mcp.py:133-134` — `nhl_predict_game` description. `src/abba/server/tools/registry.py` — tool schema descriptions.
- **Impact**: Users (and Claude) believe the system is more capable than it is. Decisions based on inflated trust.
- **Money path?**: Yes. Tool descriptions set user expectations about prediction quality.
- **Missing invariant**: Tool descriptions must accurately reflect current capabilities, not aspirational ones.
- **Test**: `test_tool_descriptions_match_reality()` — for each feature claimed in tool descriptions, verify a test proves that feature affects output.
- **Code change**: Update descriptions to reflect actual active model inputs. Add "coming soon" or remove references to inactive features.
- **Status**: open
- **Owner**: unassigned

---

### TG-037: Confidence interval not empirically calibrated

- **Issue**: 80% confidence intervals are computed from `_DEFAULT_ACCURACY_HISTORY` baselines, not from empirical coverage analysis. Actual coverage rate is unknown.
- **Evidence**: `src/abba/engine/confidence.py:_compute_confidence_interval()` — uses `calibration_error` from hard-coded history. Dossier section "Statistical Rigor". Related to TG-011.
- **Impact**: "80% CI [0.42, 0.68]" may actually cover outcomes 50% or 95% of the time. User cannot trust interval width for decision-making.
- **Money path?**: Yes. CI width affects bet sizing and risk assessment.
- **Missing invariant**: Reported CI coverage must be within 5% of actual coverage on a holdout set, or CI must be labeled "uncalibrated."
- **Test**: `test_interval_empirical_coverage()` — over holdout set, check that 80% CIs contain actual outcomes between 75-85% of the time. (Expected to fail currently — proving the need for recalibration.)
- **Code change**: Build calibration infrastructure (Phase 6). Until then, label all CIs as `"estimated, not empirically validated"`.
- **Status**: open
- **Owner**: unassigned

---

### TG-038: `analytics/graph.py` GraphAnalyzer is dangerous reinvention

- **Issue**: The parallel analytics stack contains `GraphAnalyzer` which labels standard graph metrics (betweenness, closeness, eigenvector centrality) but computes approximations that may not match `networkx` reference implementations.
- **Evidence**: `src/abba/analytics/graph.py`. Dossier ranks this as "#1 dangerous reinvention" in Library-vs-Custom section. Distinct from `engine/graph.py` (TG-028).
- **Impact**: If anyone uses the analytics graph module (it's importable), they get approximate metrics labeled as exact.
- **Money path?**: No (analytics stack is unused).
- **Missing invariant**: Graph metrics must match `networkx` reference implementation within tolerance, or be clearly labeled as approximations.
- **Test**: Covered by quarantine in TG-022. Post-quarantine: no test needed.
- **Code change**: Archive with the rest of the analytics stack (TG-022).
- **Status**: open
- **Owner**: unassigned

---

### TG-039: Personalization maps risk tolerance to RF hyperparameters

- **Issue**: `analytics/personalization.py` maps user risk tolerance preferences to Random Forest hyperparameters (n_estimators, max_depth). This has no statistical basis.
- **Evidence**: `src/abba/analytics/personalization.py`. Dossier section "Personalization".
- **Impact**: If used, it would produce different predictions based on user preference, not data. Predictions become subjective.
- **Money path?**: No (analytics stack is unused).
- **Missing invariant**: Model hyperparameters must not be set by user preferences.
- **Test**: Covered by quarantine in TG-022.
- **Code change**: Archive with analytics stack.
- **Status**: open
- **Owner**: unassigned

---

### TG-040: Biometric fatigue/recovery logic is untrusted heuristic physiology

- **Issue**: `analytics/biometrics.py` models player fatigue and recovery using made-up physiological formulas with no citation or validation.
- **Evidence**: `src/abba/analytics/biometrics.py`. Dossier section "Biometrics".
- **Impact**: If used, fatigue adjustments would be pseudoscience.
- **Money path?**: No (analytics stack is unused).
- **Missing invariant**: Physiological models must cite peer-reviewed sources or be clearly labeled as speculative.
- **Test**: Covered by quarantine in TG-022.
- **Code change**: Archive with analytics stack.
- **Status**: open
- **Owner**: unassigned

---

## Summary

| Category | Count | Money Path | Priority |
|----------|-------|------------|----------|
| Data Pipeline (Empty Results) | 8 | 7/8 yes | P0 — fix first |
| Silent Degradation / Correctness | 8 | 8/8 yes | P0 — fix with data pipeline |
| Numerical Stability | 5 | 4/5 yes | P1 — fix before trusting math |
| Architecture | 8 | 2/8 direct | P2 — enables long-term fixes |
| Test Quality | 5 | 3/5 yes | P1 — needed to verify fixes |
| Overclaiming | 6 | 4/6 yes | P1 — stop misleading users |
| **Total** | **40** | **28/40** | |
