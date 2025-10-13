# Property Service

A GraphQL API for vacation rental property management with owner and agent workflows.

## Quick Start

**Brand new setup?** See [docs/SETUP.md](./docs/SETUP.md)

```bash
# 1. Setup database (first time only)
./scripts/setup_database.sh

# 2. Build and run server
go build -o server ./cmd/server
./server

# 3. Visit GraphQL Playground
open http://localhost:8080/
```

## Testing

```bash
# Go tests
go test ./test/...
```

## Documentation

📚 **[See docs/ for complete documentation](./docs/)**

- **[SETUP.md](./docs/SETUP.md)** - Complete setup guide
- **[GRAPHQL_EXAMPLES.md](./docs/GRAPHQL_EXAMPLES.md)** - Query examples
- **[ARCHITECTURE.md](./docs/ARCHITECTURE.md)** - Technical details

## Project Structure

```
property-service/
├── cmd/server/          # Main application entry point
├── ent/                 # Ent ORM schema and generated code
├── internal/
│   ├── graphql/        # GraphQL resolvers and handlers
│   └── loaders/        # DataLoaders for N+1 prevention
├── test/               # Go tests (unit & integration)
└── docs/               # All documentation
```

## Key Features

- ✅ GraphQL API with Playground
- ✅ Owner → Agent review workflow
- ✅ Advanced property search & filtering
- ✅ DataLoaders for N+1 query prevention
- ✅ PostgreSQL with Ent ORM
- ✅ Comprehensive test coverage

---

**Tech Stack:** Go • GraphQL (gqlgen) • PostgreSQL • Ent
