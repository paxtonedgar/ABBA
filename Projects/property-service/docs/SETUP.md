# Setup Guide

Complete setup instructions for the property service.

## Prerequisites

- Docker Desktop running
- Go 1.25+

## Fresh Start (Brand New Setup)

```bash
# 1. Setup database
./scripts/setup_database.sh
# └─ Starts PostgreSQL
# └─ Creates database
# └─ Seeds users + amenities

# 2. Build server
go build -o server ./cmd/server

# 3. Start server (auto-creates tables via Ent)
./server

# 4. Optional: Add test properties
./scripts/seed_test_data.sh
```

**That's it!** Server runs on http://localhost:8080/

## How It Works

### Ent Auto-Migration (No Manual SQL!)

When the server starts, Ent **automatically creates** all database tables:

```go
// cmd/server/main.go:35
client.Schema.Create(context.Background())
```

Ent reads your schema from `ent/schema/*.go` and generates:
- ✅ All tables (users, properties, rooms, amenities, property_amenities)
- ✅ All columns with correct types
- ✅ All foreign keys and constraints
- ✅ All indexes

**You never write SQL migrations!**

### What Gets Seeded

**setup_database.sh** creates:
- 2 test users:
  - `owner@example.com` (role: owner)
  - `agent@example.com` (role: agent)
- 10 common amenities:
  - wifi, parking, hot_tub, fireplace, pool, kitchen, washer, dryer, ac, heating

**seed_test_data.sh** (optional) adds 4 test properties:
- Mountain Cabin (CO, 6 guests, wifi+hot_tub)
- Miami Condo (FL, 2 guests, parking)
- Paris Apartment (FR, 4 guests, wifi)
- Ski Lodge (CO, 8 guests, wifi+hot_tub+fireplace)

## Verify It's Working

### Check GraphQL Endpoint

**Option 1: Single-line curl (copy-paste friendly)**
```bash
curl -X POST http://localhost:8080/graphql -H 'content-type: application/json' --data '{"query":"query { __typename }"}'
```

**Option 2: Multi-line (type manually)**
```bash
curl -X POST http://localhost:8080/graphql \
  -H 'content-type: application/json' \
  --data '{"query":"query { __typename }"}'
```

**Expected response:**
```json
{"data":{"__typename":"Query"}}
```

### Test a Real Query

**Create a Property (as owner@example.com)**
```bash
curl -X POST http://localhost:8080/graphql \
  -H 'content-type: application/json' \
  -H 'X-User-Email: owner@example.com' \
  --data '{"query":"mutation { createProperty(input: {title: \"Test Cabin\", type: CABIN, maxGuests: 4, bathroomsTotal: 1, addressLine1: \"123 Main St\", city: \"Denver\", region: \"CO\", postalCode: \"80202\", country: \"US\"}) { id title status } }"}'
```

**Or use the single-line version:**
```bash
curl -X POST http://localhost:8080/graphql -H 'content-type: application/json' -H 'X-User-Email: owner@example.com' --data '{"query":"mutation { createProperty(input: {title: \"Test Cabin\", type: CABIN, maxGuests: 4, bathroomsTotal: 1, addressLine1: \"123 Main St\", city: \"Denver\", region: \"CO\", postalCode: \"80202\", country: \"US\"}) { id title status } }"}'
```

**Query Properties:**
```bash
curl -X POST http://localhost:8080/graphql -H 'content-type: application/json' --data '{"query":"{ properties { id title city region } }"}'
```

> **Note:** The `X-User-Email` header authenticates you as a specific user.
> - Use `owner@example.com` for owner operations
> - Use `agent@example.com` for agent operations
> - If omitted, defaults to `owner@example.com`

### Open GraphQL Playground
```bash
open http://localhost:8080/
```

**In the Playground, add HTTP headers (click "HTTP HEADERS" at bottom):**
```json
{
  "X-User-Email": "owner@example.com"
}
```

Then try this mutation:
```graphql
mutation CreateProperty {
  createProperty(input: {
    title: "Sunny Cabin"
    type: CABIN
    descriptionShort: "Cozy place with pines"
    descriptionLong: "A lovely 2BR/1BA cabin near the lake."
    maxGuests: 6
    bathroomsTotal: 1
    addressLine1: "100 Pine Way"
    city: "Evergreen"
    region: "CO"
    postalCode: "80439"
    country: "US"
    petsAllowed: true
    smokingAllowed: false
  }) {
    id
    title
    status
    city
    region
  }
}
```

## Reset Everything

```bash
# Complete clean slate
docker compose down -v
./scripts/setup_database.sh
go build -o server ./cmd/server
./server
```

## Troubleshooting

### PostgreSQL not running
```bash
docker ps | grep postgres
# If empty:
docker compose up -d postgres
```

### Port 8080 already in use
```bash
# Find and kill the process
lsof -ti:8080 | xargs kill -9

# Or use different port
PORT=8081 ./server
```

### Database connection errors
Check `docker-compose.yml` credentials match:
- User: `postgres`
- Password: `password`
- Database: `property_service`
- Port: `5432`

**Note:** Modern Docker Desktop uses `docker compose` (not `docker-compose`)

### Tables not created
Tables are created when server starts. Check logs for:
```
Server starting on port 8080
```

If you see errors, the database might not be accessible.

## Development Workflow

```bash
# Make changes to code
vim internal/graphql/resolvers.go

# Rebuild
go build -o server ./cmd/server

# Restart server
pkill server && ./server
```

## GraphQL Schema Changes

If you modify `schema.graphql`:

```bash
# Regenerate GraphQL code
go run github.com/99designs/gqlgen generate

# Rebuild
go build -o server ./cmd/server
```

