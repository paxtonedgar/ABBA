# Documentation

This directory is a mix of:

- current implementation notes
- audit material
- planning documents
- older project writeups that are still useful as context but are not canonical

Do not assume every file here reflects the current runtime.

## Start Here

- `../README.md` for the public repo overview
- `architecture-dossier.md` for the most detailed current audit
- `invariants.md` and `trust-gaps.md` for current correctness-focused notes

## Useful Current Docs

| Document | What it is |
|---|---|
| `architecture-dossier.md` | Deep audit of architecture, math, invariants, trust, and abstraction risk |
| `invariants.md` | Focused invariant notes |
| `trust-gaps.md` | Trust and reliability gaps |
| `database-setup.md` | DuckDB setup and schema notes |
| `data-pipeline.md` | Data flow notes |
| `validation-testing.md` | Validation/testing notes |

## Historical / Planning Material

These files may still be useful, but they should be read as planning or historical context rather than guaranteed current behavior:

- `PROJECT_SPECIFICATION.md`
- `IMPLEMENTATION_SUMMARY.md`
- `REFACTOR_SUMMARY.md`
- `implementation-plans.md`
- `system-analysis.md`
- `audit_report.md`
- `pipeline_blueprint.md`
- `ARCHITECTURE_PLAN.md`

## Documentation Hygiene

If you are updating the repo for external readers:

- prefer the top-level `README.md` for the public story
- prefer `architecture-dossier.md` for the current technical audit
- avoid marking documents "canonical" unless they are maintained against the active runtime path
