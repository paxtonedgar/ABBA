# Testing Structure

This directory contains all tests for the property service.

## Directory Structure

- `unit/` - Go unit tests that don't require external dependencies
- `integration/` - Go integration tests that require database connections
- `e2e/` - End-to-end tests using Playwright (GraphQL API testing)

## Running Tests

### Unit Tests Only
```bash
go test ./test/unit/... -v
```

### Integration Tests Only
```bash
go test ./test/integration/... -v
```

### All Go Tests
```bash
go test ./test/... -v
```

### E2E Tests (Playwright)
```bash
npx playwright test
```

### All Tests (Go + E2E)
```bash
go test ./test/... -v && npx playwright test
```

### Skip Integration Tests (for CI/CD)
```bash
go test -short ./test/... -v
```

## Environment Variables

- `DATABASE_URL` - PostgreSQL connection string (defaults to local docker setup)
- `TEST_SHORT` - Set to true to skip integration tests

## Prerequisites for Integration Tests

1. Docker must be running
2. PostgreSQL container must be started:
   ```bash
   docker-compose up -d postgres
   ```
3. Database must be initialized:
   ```bash
   ./scripts/setup_database.sh
   ```
