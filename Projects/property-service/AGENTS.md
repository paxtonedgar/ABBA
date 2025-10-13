# Repository Guidelines

## Project Structure & Module Organization
The GraphQL API bootstraps from `cmd/server/main.go`; the compiled `server` binary stays out of version control. Domain logic lives under `internal/` across `graphql`, `service`, `database`, `dataloader`, `middleware`, and `logger` packages. Ent schemas and generated code reside in `ent/`, shared docs in `docs/`, helper scripts in `scripts/`, and tests in `test/{unit,integration,e2e}` (Playwright).

## Build, Test, and Development Commands
Initialize or reset PostgreSQL with `./scripts/setup_database.sh`, then build via `go build -o server ./cmd/server` and run `./server` (listens on `:8080`). Execute Go tests with `go test ./test/...`, narrowing to `./test/unit/...` or `./test/integration/...` while iterating; add `-short` to skip integration. Drive end-to-end coverage with `npx playwright test`. Use `docker-compose up -d` for supporting services and `docker-compose down -v` to clean volumes.

## Coding Style & Naming Conventions
Format Go code using `gofmt` or `goimports` before committing. Prefer idiomatic Go naming: exported APIs use `CamelCase`, locals are lower camel, and package names remain short (`service`, `graphql`). GraphQL schema edits belong in `internal/graphql/schema.graphql` with matching resolver updates in the `resolver_*.go` files; keep shell scripts POSIX-compliant and executable.

## Testing Guidelines
Place focused unit cases in `test/unit` and resolver/service flows in `test/integration`; seed data through the scripts when integration tests need fixtures. Use `_test.go` files with `TestFeature_Scenario` naming. Capture GraphQL contract changes in Playwright specs inside `test/e2e`. Run `go test ./test/...` (and Playwright when applicable) before pushing.

## Commit & Pull Request Guidelines
Write commits in imperative mood (`docs: update README navigation`) and keep subjects near 72 characters. Each PR should describe intent, link related issues, and flag database, Ent, or GraphQL schema changes. Include screenshots or curl examples for user-visible work and confirm `go test ./test/...` plus `npx playwright test` in the checklist.

## Environment & Data Setup
Start Docker Desktop before running any database scripts. After launching the server, open `http://localhost:8080/` and add the `X-User-Email` header to simulate owners or agents. Use `./scripts/seed_test_data.sh` to populate sample properties or reset with `docker-compose down -v && ./scripts/setup_database.sh`.
