# Tests

The test suite is a mix of:

- fast unit tests around the active toolkit path
- integration-style tests for selected flows
- contract-style trust checks for high-risk invariants
- legacy or experimental tests tied to older subsystems

If you are new to the repo, start with the active toolkit tests rather than the older zero-mock material still present in some files.

## Quick Start

From the repository root:

```bash
pip install -e ".[dev]"
pytest
```

For a smaller local run:

```bash
pytest tests/test_storage.py tests/test_engine.py tests/test_toolkit.py
```

## Useful Slices

### Core toolkit path

```bash
pytest tests/test_storage.py tests/test_engine.py tests/test_toolkit.py tests/test_workflows.py
```

### Trust-contract checks

```bash
pytest -m contract
```

### Specific high-risk invariants

```bash
pytest \
  tests/test_predictor_family_invariants.py \
  tests/test_schema_contracts_nhl_live.py \
  tests/test_odds_schedule_identity.py \
  tests/test_snapshot_provenance.py \
  tests/test_goalie_selection_invariants.py \
  tests/test_season_selection_invariants.py
```

## Notes

- `tests/conftest.py` bootstraps `src/` onto `sys.path`, so plain `pytest` works from the repo root.
- Some files under `tests/integration/` exercise older analytics paths and are not the best entry point for understanding the current public toolkit.
- The most relevant tests for external users are the ones covering `ABBAToolkit`, storage, engines, workflows, and contract invariants.
