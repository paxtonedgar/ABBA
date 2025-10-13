# Neo4j Dual-Write Integration Plan

**Status**: READ-ONLY EVALUATION
**Goal**: Add Neo4j as a dual-write graph database alongside PostgreSQL/Ent for rich relationship queries and visual demos

---

## 🎯 Executive Summary

### Why Neo4j + PostgreSQL?

**PostgreSQL (via Ent)**:
- ✅ ACID transactions
- ✅ Complex filtering/aggregations
- ✅ Strong consistency
- ✅ Production workload workhorse

**Neo4j**:
- ✅ Graph traversals (6-degree relationships in milliseconds)
- ✅ Visual demos for stakeholders (Neo4j Browser/Bloom)
- ✅ Path analysis (guest journeys, amenity networks)
- ✅ Real-time relationship insights

### The Dual-Write Pattern

```
GraphQL Mutation
      ↓
Service Layer (single source of truth)
      ↓
   ┌─────┴─────┐
   ↓           ↓
PostgreSQL   Neo4j
(Primary)   (Graph View)
```

---

## 📊 Current Architecture Analysis

### Current Data Model

```
User (owner/agent)
  ↓ owns
Property (house/apartment/villa)
  ↓ has
Room (bed_mix: JSONB)

Property ←→ Amenity (many-to-many)
```

### Key Entities & Relationships

| Entity | PostgreSQL Schema | Neo4j Node Labels |
|--------|------------------|-------------------|
| User | users table | `:User:Owner` or `:User:Agent` |
| Property | properties table | `:Property:House/Apartment/Villa` |
| Room | rooms table | `:Room` |
| Amenity | amenities table | `:Amenity` |

**Current Edges**:
- `User -[:OWNS]-> Property`
- `User -[:REVIEWED]-> Property`
- `Property -[:HAS_ROOM]-> Room`
- `Property -[:HAS_AMENITY]-> Amenity`

---

## 🏗️ Proposed Architecture

### 1. Docker Integration

**Current**: `docker-compose.yml` only has PostgreSQL

**Proposed Addition**:

```yaml
services:
  postgres:
    # ... existing config ...

  neo4j:
    image: neo4j:5.15-community
    environment:
      NEO4J_AUTH: neo4j/property_service_password
      NEO4J_PLUGINS: '["apoc", "graph-data-science"]'
      NEO4J_dbms_security_procedures_unrestricted: apoc.*,gds.*
      NEO4J_dbms_memory_heap_max__size: 2G
    ports:
      - "7474:7474"   # Browser UI
      - "7687:7687"   # Bolt protocol
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - ./scripts/neo4j-init:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "cypher-shell -u neo4j -p property_service_password 'RETURN 1'"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
  neo4j_data:
  neo4j_logs:
```

**Key Features**:
- APOC library for advanced graph operations
- Graph Data Science plugin for analytics
- Health checks for orchestration
- Persistent storage

---

### 2. Service Layer Integration Points

#### Current Service Architecture

```
internal/service/
├── property_service.go   ← Writes to Ent
├── room_service.go       ← Writes to Ent
├── amenity_service.go    ← Writes to Ent
└── auth_service.go
```

#### Proposed: Add Graph Sync Layer

```
internal/
├── service/
│   ├── property_service.go
│   ├── room_service.go
│   └── amenity_service.go
│
├── graph/                    ← NEW
│   ├── neo4j_client.go      # Connection management
│   ├── property_sync.go     # Property → Neo4j sync
│   ├── room_sync.go         # Room → Neo4j sync
│   ├── amenity_sync.go      # Amenity → Neo4j sync
│   ├── query_builder.go     # Cypher query builder
│   └── errors.go            # Graph-specific errors
│
└── middleware/
    └── graph_sync_middleware.go  # Optional: async sync
```

---

## 🔄 Dual-Write Strategy

### Option A: Synchronous Dual-Write (Recommended for Start)

**Pros**:
- Simple to implement
- Immediate consistency
- Easy to debug

**Cons**:
- Slightly slower writes
- Neo4j downtime affects writes

**Implementation**:

```go
// internal/service/property_service.go

type PropertyService struct {
    client      *ent.Client
    authService *AuthService
    graphSync   *graph.PropertySync  // NEW
}

func (s *PropertyService) CreateProperty(ctx context.Context, input CreatePropertyInput) (*ent.Property, error) {
    // 1. PostgreSQL write (primary)
    prop, err := s.client.Property.Create().
        SetTitle(input.Title).
        // ... all fields ...
        Save(ctx)
    if err != nil {
        return nil, fmt.Errorf("create property: %w", err)
    }

    // 2. Neo4j sync (graph view)
    if s.graphSync != nil {
        if err := s.graphSync.SyncCreate(ctx, prop); err != nil {
            // Log error but don't fail the request
            log.Error("neo4j sync failed", "property_id", prop.ID, "error", err)
            // Optional: queue for retry
        }
    }

    return prop, nil
}
```

**Write Flow**:
```
1. ✅ Write to PostgreSQL (ACID transaction)
2. ✅ Commit
3. ⚠️  Sync to Neo4j (best-effort)
   - Success → done
   - Failure → log + queue for retry
4. ✅ Return success to client
```

---

### Option B: Asynchronous Event-Driven Sync

**Pros**:
- Fast writes (no Neo4j latency)
- Neo4j downtime doesn't affect app
- Can batch operations

**Cons**:
- Eventual consistency
- Need message queue (Redis/Kafka)
- More complex

**Implementation**:

```go
// internal/graph/event_publisher.go

type GraphEvent struct {
    Type      string      // "property.created", "room.updated"
    EntityID  uuid.UUID
    Data      interface{}
    Timestamp time.Time
}

// Publish to Redis stream
func (p *EventPublisher) PublishPropertyCreated(ctx context.Context, prop *ent.Property) error {
    event := GraphEvent{
        Type:     "property.created",
        EntityID: prop.ID,
        Data:     prop,
    }
    return p.redis.XAdd(ctx, "graph:events", event)
}

// Worker consumes and syncs
func (w *GraphSyncWorker) ProcessEvents(ctx context.Context) {
    for event := range w.events {
        switch event.Type {
        case "property.created":
            w.propertySync.SyncCreate(ctx, event.Data)
        case "property.updated":
            w.propertySync.SyncUpdate(ctx, event.Data)
        }
    }
}
```

**Recommendation**: Start with Option A, migrate to Option B if Neo4j write latency becomes an issue.

---

## 📝 Neo4j Schema Design

### Node Structure

```cypher
// User nodes (dual-labeled for role)
(:User:Owner {
  id: "uuid",
  name: "John Doe",
  email: "john@example.com",
  created_at: datetime()
})

(:User:Agent {
  id: "uuid",
  name: "Jane Smith",
  email: "jane@example.com"
})

// Property nodes (dual-labeled for type)
(:Property:Villa {
  id: "uuid",
  title: "Luxury Beach Villa",
  status: "approved",
  max_guests: 8,
  city: "Miami",
  region: "FL",
  country: "US",
  latitude: 25.7617,
  longitude: -80.1918,
  created_at: datetime(),
  updated_at: datetime()
})

// Room nodes
(:Room {
  id: "uuid",
  name: "Master Bedroom",
  floor_label: "2nd Floor",
  area_sq_ft: 350.0,
  ensuite_bath: true
})

// Bed nodes (extracted from JSONB)
(:Bed {
  type: "queen",  // queen, king, twin, etc.
})

// Amenity nodes
(:Amenity {
  id: "uuid",
  code: "wifi",
  name: "High-Speed WiFi",
  category: "connectivity"
})
```

### Relationship Structure

```cypher
// Ownership
(:User:Owner)-[:OWNS {since: datetime()}]->(:Property)

// Review workflow
(:User:Agent)-[:REVIEWED {
  at: datetime(),
  decision: "approved",
  notes: "Beautiful property"
}]->(:Property)

// Property structure
(:Property)-[:HAS_ROOM {
  added_at: datetime()
}]->(:Room)

(:Room)-[:HAS_BED {
  count: 2
}]->(:Bed)

// Amenities
(:Property)-[:HAS_AMENITY {
  added_at: datetime(),
  featured: false
}]->(:Amenity)

// Location proximity (for "similar properties")
(:Property)-[:NEAR {
  distance_km: 2.5
}]->(:Property)
```

### Indexes & Constraints

```cypher
// Uniqueness constraints
CREATE CONSTRAINT user_id IF NOT EXISTS
FOR (u:User) REQUIRE u.id IS UNIQUE;

CREATE CONSTRAINT property_id IF NOT EXISTS
FOR (p:Property) REQUIRE p.id IS UNIQUE;

CREATE CONSTRAINT room_id IF NOT EXISTS
FOR (r:Room) REQUIRE r.id IS UNIQUE;

CREATE CONSTRAINT amenity_code IF NOT EXISTS
FOR (a:Amenity) REQUIRE a.code IS UNIQUE;

// Search indexes
CREATE INDEX property_location IF NOT EXISTS
FOR (p:Property) ON (p.city, p.region, p.country);

CREATE INDEX property_status IF NOT EXISTS
FOR (p:Property) ON (p.status);

CREATE FULLTEXT INDEX property_search IF NOT EXISTS
FOR (p:Property) ON EACH [p.title, p.description_short, p.description_long];
```

---

## 🔌 Implementation Details

### 1. Neo4j Client Setup

```go
// internal/graph/neo4j_client.go

package graph

import (
    "context"
    "fmt"
    "github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

type Client struct {
    driver neo4j.DriverWithContext
}

func NewClient(uri, username, password string) (*Client, error) {
    driver, err := neo4j.NewDriverWithContext(
        uri,
        neo4j.BasicAuth(username, password, ""),
        func(config *neo4j.Config) {
            config.MaxConnectionPoolSize = 50
            config.ConnectionAcquisitionTimeout = 30 * time.Second
        },
    )
    if err != nil {
        return nil, fmt.Errorf("neo4j driver: %w", err)
    }

    // Verify connectivity
    ctx := context.Background()
    if err := driver.VerifyConnectivity(ctx); err != nil {
        return nil, fmt.Errorf("neo4j verify: %w", err)
    }

    return &Client{driver: driver}, nil
}

func (c *Client) Close(ctx context.Context) error {
    return c.driver.Close(ctx)
}

func (c *Client) Session(ctx context.Context) neo4j.SessionWithContext {
    return c.driver.NewSession(ctx, neo4j.SessionConfig{
        AccessMode: neo4j.AccessModeWrite,
    })
}
```

### 2. Property Sync Implementation

```go
// internal/graph/property_sync.go

package graph

import (
    "context"
    "fmt"
    "github.com/evolve-interviews/paxtonedgar-case-study/ent"
)

type PropertySync struct {
    client *Client
}

func NewPropertySync(client *Client) *PropertySync {
    return &PropertySync{client: client}
}

func (s *PropertySync) SyncCreate(ctx context.Context, prop *ent.Property) error {
    session := s.client.Session(ctx)
    defer session.Close(ctx)

    query := `
        MERGE (p:Property {id: $id})
        SET p.title = $title,
            p.type = $type,
            p.status = $status,
            p.max_guests = $max_guests,
            p.city = $city,
            p.region = $region,
            p.country = $country,
            p.latitude = $latitude,
            p.longitude = $longitude,
            p.created_at = datetime($created_at),
            p.updated_at = datetime($updated_at)

        WITH p
        MERGE (owner:User {id: $owner_id})
        MERGE (owner)-[:OWNS {since: datetime($created_at)}]->(p)

        RETURN p.id
    `

    params := map[string]interface{}{
        "id":          prop.ID.String(),
        "title":       prop.Title,
        "type":        string(prop.Type),
        "status":      string(prop.Status),
        "max_guests":  prop.MaxGuests,
        "city":        prop.City,
        "region":      prop.Region,
        "country":     prop.Country,
        "latitude":    prop.Latitude,
        "longitude":   prop.Longitude,
        "owner_id":    prop.OwnerID.String(),
        "created_at":  prop.CreatedAt.Format(time.RFC3339),
        "updated_at":  prop.UpdatedAt.Format(time.RFC3339),
    }

    _, err := session.Run(ctx, query, params)
    if err != nil {
        return fmt.Errorf("neo4j sync create: %w", err)
    }

    return nil
}

func (s *PropertySync) SyncUpdate(ctx context.Context, prop *ent.Property) error {
    // Similar pattern with MATCH instead of MERGE
    // ...
}

func (s *PropertySync) SyncDelete(ctx context.Context, propertyID uuid.UUID) error {
    session := s.client.Session(ctx)
    defer session.Close(ctx)

    query := `
        MATCH (p:Property {id: $id})
        DETACH DELETE p
    `

    _, err := session.Run(ctx, query, map[string]interface{}{
        "id": propertyID.String(),
    })
    return err
}
```

### 3. Service Layer Integration

```go
// internal/service/property_service.go (modified)

type PropertyService struct {
    client      *ent.Client
    authService *AuthService
    graphSync   *graph.PropertySync  // NEW: optional graph sync
}

func NewPropertyService(
    client *ent.Client,
    authService *AuthService,
    graphSync *graph.PropertySync,  // Can be nil if Neo4j disabled
) *PropertyService {
    return &PropertyService{
        client:      client,
        authService: authService,
        graphSync:   graphSync,
    }
}

func (s *PropertyService) CreateProperty(ctx context.Context, input CreatePropertyInput) (*ent.Property, error) {
    // ... existing validation ...

    // Primary write to PostgreSQL
    prop, err := s.client.Property.Create().
        // ... all fields ...
        Save(ctx)
    if err != nil {
        return nil, fmt.Errorf("create property: %w", err)
    }

    // Best-effort sync to Neo4j
    if s.graphSync != nil {
        if err := s.graphSync.SyncCreate(ctx, prop); err != nil {
            // Log but don't fail the request
            log.Error("neo4j sync failed",
                "property_id", prop.ID,
                "error", err,
            )
            // TODO: Queue for async retry
        }
    }

    return prop, nil
}
```

---

## 🎨 Demo Queries for Neo4j Browser

### 1. Property Network Overview

```cypher
// Visualize entire property ecosystem
MATCH (owner:Owner)-[:OWNS]->(p:Property)-[:HAS_ROOM]->(r:Room)
OPTIONAL MATCH (p)-[:HAS_AMENITY]->(a:Amenity)
RETURN owner, p, r, a
LIMIT 50

// Style it
:style
node.Property {
  color: #FFA500;
  size: 60px;
  caption: '{title}';
}
node.Owner {
  color: #4CAF50;
  size: 40px;
  caption: '{name}';
}
node.Room {
  color: #2196F3;
  size: 30px;
  caption: '{name}';
}
node.Amenity {
  color: #9C27B0;
  size: 25px;
  caption: '{name}';
}
relationship.OWNS {
  color: #333;
  shaft-width: 3px;
}
relationship.HAS_ROOM {
  color: #4B8BF5;
  shaft-width: 2px;
}
relationship.HAS_AMENITY {
  color: #9C27B0;
  shaft-width: 1px;
}
```

### 2. Review Workflow Visualization

```cypher
// Show agent review paths
MATCH (owner:Owner)-[:OWNS]->(p:Property)
OPTIONAL MATCH (agent:Agent)-[r:REVIEWED]->(p)
WHERE p.status IN ['submitted', 'under_review', 'approved', 'rejected']
RETURN owner, p, agent, r

// Highlight status with colors
:style
node.Property {
  color: case(property.status)
    when 'draft' then '#9E9E9E'
    when 'submitted' then '#FFC107'
    when 'under_review' then '#2196F3'
    when 'approved' then '#4CAF50'
    when 'rejected' then '#F44336'
  end;
}
```

### 3. Amenity Overlap Analysis

```cypher
// Find properties with similar amenity profiles
MATCH (p1:Property)-[:HAS_AMENITY]->(a:Amenity)<-[:HAS_AMENITY]-(p2:Property)
WHERE p1.id < p2.id  // Avoid duplicates
WITH p1, p2, COUNT(a) as shared_amenities
WHERE shared_amenities >= 3
RETURN p1, p2, shared_amenities
ORDER BY shared_amenities DESC
LIMIT 20

// Business insight: Recommend similar properties to guests
```

### 4. Geographic Clustering

```cypher
// Properties in the same region
MATCH (p:Property)
WHERE p.region = 'FL'
WITH p.city as city, COLLECT(p) as properties, COUNT(p) as prop_count
WHERE prop_count > 1
RETURN city, properties, prop_count
ORDER BY prop_count DESC

// For creating "Near" relationships
MATCH (p1:Property), (p2:Property)
WHERE p1.id <> p2.id
  AND p1.region = p2.region
  AND point.distance(
        point({latitude: p1.latitude, longitude: p1.longitude}),
        point({latitude: p2.latitude, longitude: p2.longitude})
      ) < 5000  // 5km
MERGE (p1)-[r:NEAR]-(p2)
SET r.distance_km = point.distance(
      point({latitude: p1.latitude, longitude: p1.longitude}),
      point({latitude: p2.latitude, longitude: p2.longitude})
    ) / 1000
```

### 5. Owner Portfolio Analysis

```cypher
// Show all properties owned by top owners
MATCH (owner:Owner)-[:OWNS]->(p:Property)
WITH owner, COUNT(p) as property_count, COLLECT(p) as properties
WHERE property_count >= 2
RETURN owner, properties, property_count
ORDER BY property_count DESC
LIMIT 10

// Business insight: Identify power users for VIP treatment
```

---

## 📊 Neo4j Bloom Scenes (Stakeholder Presentations)

### Scene 1: "Our Property Network"
**Goal**: Show scale and diversity

**Setup**:
```cypher
MATCH (p:Property)
WITH p.type as type, COUNT(p) as count
RETURN type, count
```

**Bloom Category Rules**:
- Color by property type (villa=gold, apartment=blue, house=green)
- Size by max_guests
- Show total count badge

---

### Scene 2: "Guest Journey Paths"
**Goal**: Show how guests discover properties

**Setup**:
```cypher
// Create search journey
MATCH path = (amenity:Amenity)<-[:HAS_AMENITY]-(p:Property)-[:HAS_ROOM]->(r:Room)
WHERE amenity.code IN ['wifi', 'pool', 'hot_tub']
  AND p.status = 'approved'
RETURN path
LIMIT 30
```

**Bloom Settings**:
- Animate path highlighting
- Group by amenity category
- Show "search → filter → book" flow

---

### Scene 3: "Amenity Overlap Analysis"
**Goal**: Competitive analysis

**Setup**:
```cypher
MATCH (p:Property)-[rel:HAS_AMENITY]->(a:Amenity)
RETURN p, rel, a
```

**Bloom Features**:
- Cluster by amenity category
- Heat map by amenity popularity
- Identify gaps (amenities no properties have)

---

### Scene 4: "Revenue Flow Visualization"
**Goal**: Business metrics

**Setup** (requires booking data - future):
```cypher
// Placeholder - would show:
// Property → Bookings → Revenue
MATCH (owner:Owner)-[:OWNS]->(p:Property)
WITH owner, p, p.max_guests * 200 as potential_revenue  // Mock
RETURN owner, p, potential_revenue
ORDER BY potential_revenue DESC
```

---

## 🔍 Advanced Neo4j Queries

### Path Analysis: "Find similar properties"

```cypher
// Given a property, find similar ones based on:
// - Same region
// - Similar amenities
// - Similar capacity

MATCH (target:Property {id: $property_id})
MATCH (similar:Property)
WHERE target.id <> similar.id
  AND target.region = similar.region
  AND ABS(target.max_guests - similar.max_guests) <= 2

// Calculate amenity overlap
OPTIONAL MATCH (target)-[:HAS_AMENITY]->(a1:Amenity)<-[:HAS_AMENITY]-(similar)
WITH target, similar, COUNT(a1) as shared_amenities

// Calculate all amenities for target
OPTIONAL MATCH (target)-[:HAS_AMENITY]->(a2:Amenity)
WITH target, similar, shared_amenities, COUNT(a2) as target_amenity_count

// Similarity score
WITH similar,
     CASE
       WHEN target_amenity_count = 0 THEN 0
       ELSE toFloat(shared_amenities) / target_amenity_count
     END as similarity_score
WHERE similarity_score > 0.3

RETURN similar, similarity_score
ORDER BY similarity_score DESC
LIMIT 10
```

### Recommendation Engine

```cypher
// "Guests who viewed this also viewed..."
MATCH (viewed:Property {id: $property_id})
MATCH (viewed)-[:HAS_AMENITY]->(a:Amenity)<-[:HAS_AMENITY]-(related:Property)
WHERE viewed.id <> related.id
  AND related.status = 'approved'

WITH related, COUNT(a) as shared_amenities, COLLECT(DISTINCT a.name) as amenities
RETURN related, shared_amenities, amenities
ORDER BY shared_amenities DESC
LIMIT 5
```

### Analytics: "Most connected properties"

```cypher
// Properties with most relationships (rooms + amenities)
MATCH (p:Property)
OPTIONAL MATCH (p)-[:HAS_ROOM]->(r:Room)
OPTIONAL MATCH (p)-[:HAS_AMENITY]->(a:Amenity)

WITH p, COUNT(DISTINCT r) as room_count, COUNT(DISTINCT a) as amenity_count
WITH p, room_count, amenity_count, (room_count + amenity_count) as connection_score

RETURN p.title, room_count, amenity_count, connection_score
ORDER BY connection_score DESC
LIMIT 20
```

---

## 🚀 Implementation Phases

### Phase 1: Foundation (Week 1)
**Goal**: Get Neo4j running alongside PostgreSQL

- [ ] Add Neo4j to `docker-compose.yml`
- [ ] Create `internal/graph/` package structure
- [ ] Implement `neo4j_client.go` with connection pooling
- [ ] Write Neo4j schema initialization script
- [ ] Add health checks

**Deliverable**: `docker compose up` starts both databases

---

### Phase 2: Sync Layer (Week 2)
**Goal**: Dual-write for core entities

- [ ] Implement `property_sync.go` (Create/Update/Delete)
- [ ] Implement `room_sync.go`
- [ ] Implement `amenity_sync.go`
- [ ] Add sync calls to service layer
- [ ] Error handling + retry logic

**Deliverable**: Creating a property writes to both databases

---

### Phase 3: Relationship Enrichment (Week 3)
**Goal**: Add graph-specific relationships

- [ ] Owner → Property edges with metadata
- [ ] Agent review workflow edges
- [ ] Property → Room → Bed expansion
- [ ] Amenity relationships
- [ ] Geographic proximity `:NEAR` relationships

**Deliverable**: Rich graph with business context

---

### Phase 4: Demo Queries (Week 4)
**Goal**: Prepare stakeholder demos

- [ ] Create saved Cypher queries for Neo4j Browser
- [ ] Style graph visualizations
- [ ] Create Bloom scenes (if Enterprise edition)
- [ ] Document demo workflows
- [ ] Record demo videos

**Deliverable**: Impressive live demos

---

### Phase 5: Advanced Features (Future)
**Goal**: Production-ready features

- [ ] Async event-driven sync (Redis/Kafka)
- [ ] Consistency checker (compare Postgres ↔ Neo4j)
- [ ] Backfill script (sync existing data)
- [ ] Read queries from Neo4j (path analysis)
- [ ] Monitoring + metrics
- [ ] Performance tuning

---

## ⚠️ Key Decisions & Trade-offs

### 1. Synchronous vs Asynchronous Sync
**Decision**: Start synchronous, move to async later
**Rationale**: Simpler debugging, easier to verify consistency

### 2. Error Handling: Fail Fast vs Best Effort
**Decision**: Best effort (log + continue)
**Rationale**: Neo4j is secondary; PostgreSQL is source of truth

### 3. Schema Duplication
**Decision**: Duplicate some data in Neo4j
**Rationale**: Graph queries need local data for performance

### 4. Consistency Model
**Decision**: Eventual consistency acceptable
**Rationale**: Neo4j is for analytics/demos, not transactional

---

## 🧪 Testing Strategy

### Unit Tests
```go
// internal/graph/property_sync_test.go

func TestPropertySync_SyncCreate(t *testing.T) {
    // Use Neo4j testcontainers
    ctx := context.Background()
    container, err := neo4jtest.StartContainer(ctx)
    require.NoError(t, err)
    defer container.Terminate(ctx)

    client := graph.NewClient(container.URI(), "neo4j", "test")
    sync := graph.NewPropertySync(client)

    // Create test property
    prop := &ent.Property{
        ID:    uuid.New(),
        Title: "Test Villa",
        // ...
    }

    err = sync.SyncCreate(ctx, prop)
    assert.NoError(t, err)

    // Verify in Neo4j
    session := client.Session(ctx)
    result, err := session.Run(ctx,
        "MATCH (p:Property {id: $id}) RETURN p.title",
        map[string]interface{}{"id": prop.ID.String()},
    )
    require.NoError(t, err)

    record, err := result.Single(ctx)
    require.NoError(t, err)
    assert.Equal(t, "Test Villa", record.Values[0])
}
```

### Integration Tests
```go
// test/integration/dual_write_test.go

func TestDualWrite_CreateProperty(t *testing.T) {
    // Start both databases
    pgContainer := postgres.StartContainer(t)
    neoContainer := neo4j.StartContainer(t)

    // Create services
    entClient := ent.NewClient(pgContainer.URL())
    graphClient := graph.NewClient(neoContainer.URI(), "neo4j", "test")

    propertyService := service.NewPropertyService(
        entClient,
        authService,
        graph.NewPropertySync(graphClient),
    )

    // Create property
    ctx := testauth.WithOwner(context.Background())
    prop, err := propertyService.CreateProperty(ctx, input)
    require.NoError(t, err)

    // Verify in PostgreSQL
    dbProp, err := entClient.Property.Get(ctx, prop.ID)
    assert.NoError(t, err)
    assert.Equal(t, "Test Property", dbProp.Title)

    // Verify in Neo4j (eventually consistent)
    time.Sleep(100 * time.Millisecond)
    neoProp := queryNeo4j(t, graphClient, prop.ID)
    assert.Equal(t, "Test Property", neoProp.Title)
}
```

---

## 📈 Monitoring & Observability

### Key Metrics to Track

```go
// internal/graph/metrics.go

var (
    neo4jSyncDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "neo4j_sync_duration_seconds",
            Help: "Time taken to sync to Neo4j",
        },
        []string{"entity_type", "operation"},
    )

    neo4jSyncErrors = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "neo4j_sync_errors_total",
            Help: "Number of Neo4j sync errors",
        },
        []string{"entity_type", "operation", "error_type"},
    )

    neo4jConnectionPool = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "neo4j_connection_pool_active",
            Help: "Active Neo4j connections",
        },
        []string{"state"},
    )
)
```

### Logging

```go
log.Info("neo4j sync started",
    "entity", "property",
    "operation", "create",
    "property_id", prop.ID,
)

if err := sync.SyncCreate(ctx, prop); err != nil {
    log.Error("neo4j sync failed",
        "entity", "property",
        "operation", "create",
        "property_id", prop.ID,
        "error", err,
        "postgres_state", "committed",  // Important for debugging
    )
}
```

---

## 🔒 Security Considerations

### 1. Access Control
```yaml
# docker-compose.yml
neo4j:
  environment:
    NEO4J_AUTH: neo4j/${NEO4J_PASSWORD:-changeme}
    NEO4J_dbms_security_auth__enabled: "true"
```

### 2. Network Isolation
```yaml
networks:
  backend:
    driver: bridge

services:
  neo4j:
    networks:
      - backend
    # Don't expose 7474/7687 in production
```

### 3. Sensitive Data
- Don't sync sensitive fields (passwords, tokens)
- Consider encryption at rest (Neo4j Enterprise)
- Audit access logs

---

## 💰 Cost Analysis

### Neo4j Community Edition (Free)
✅ Suitable for:
- Development
- Demos
- Internal analytics

❌ Limitations:
- No clustering
- No hot backups
- No Bloom (visual tool)
- Single database

### Neo4j Enterprise Edition ($$$)
✅ Features:
- Multi-database
- Bloom visual tool
- Clustering/HA
- Advanced security

**Recommendation**: Start with Community, upgrade when demos need Bloom.

---

## 🎓 Learning Resources

### For Developers
- [Neo4j Go Driver Docs](https://neo4j.com/docs/go-manual/current/)
- [Cypher Query Language](https://neo4j.com/docs/cypher-manual/current/)
- [Neo4j Graph Data Science](https://neo4j.com/docs/graph-data-science/current/)

### For Stakeholders
- [Neo4j Bloom User Guide](https://neo4j.com/docs/bloom-user-guide/current/)
- [Graph Database Patterns](https://graphacademy.neo4j.com/)

---

## 📋 Checklist for Implementation

### Pre-Implementation
- [ ] Review this plan with team
- [ ] Decide on sync strategy (sync vs async)
- [ ] Provision Neo4j license if needed (Bloom)
- [ ] Set up development environment
- [ ] Create feature branch

### During Implementation
- [ ] Follow phases 1-4 sequentially
- [ ] Write tests for each component
- [ ] Document Cypher queries
- [ ] Create demo scripts
- [ ] Monitor performance

### Post-Implementation
- [ ] Load test dual-write performance
- [ ] Backfill existing data
- [ ] Train stakeholders on Neo4j Browser
- [ ] Create demo videos
- [ ] Document troubleshooting guide

---

## 🚨 Rollback Plan

If Neo4j integration causes issues:

### Immediate Rollback (< 5 minutes)
```go
// Disable graph sync via feature flag
propertyService := service.NewPropertyService(
    entClient,
    authService,
    nil,  // Pass nil to disable Neo4j sync
)
```

### Clean Rollback (< 1 hour)
1. Stop Neo4j container: `docker compose stop neo4j`
2. Revert code changes
3. Remove graph sync calls from services
4. Deploy

### Data Cleanup
```bash
# Remove Neo4j data
docker compose down neo4j
docker volume rm property-service_neo4j_data
```

---

## 📞 Support & Troubleshooting

### Common Issues

**Issue**: Neo4j sync timeout
**Solution**: Increase `NEO4J_dbms_transaction_timeout`, check network

**Issue**: Inconsistent data between DBs
**Solution**: Run consistency checker, backfill from PostgreSQL

**Issue**: Neo4j out of memory
**Solution**: Increase `NEO4J_dbms_memory_heap_max__size`

### Debug Commands

```bash
# Check Neo4j health
docker compose exec neo4j cypher-shell -u neo4j -p password "CALL dbms.components()"

# Count nodes
docker compose exec neo4j cypher-shell -u neo4j -p password "MATCH (n) RETURN count(n)"

# View sync errors
docker compose logs neo4j | grep ERROR
```

---

## ✅ Success Criteria

### Technical Success
- [ ] All write operations dual-write successfully
- [ ] Neo4j sync latency < 100ms p95
- [ ] Zero data loss (PostgreSQL writes always succeed)
- [ ] < 1% sync failures
- [ ] Tests pass with >90% coverage

### Business Success
- [ ] Stakeholders can run live demos
- [ ] Graph queries answer business questions
- [ ] Demo videos showcase capabilities
- [ ] Team trained on Neo4j Browser

---

## 🎯 Final Recommendation

**GO/NO-GO Decision Factors**:

**✅ GO if**:
- Need visual demos for stakeholders
- Want to explore graph analytics
- Have development capacity (2-4 weeks)
- Comfortable with eventual consistency

**❌ NO-GO if**:
- PostgreSQL queries are sufficient
- No stakeholder demand for graphs
- Can't accept write latency increase
- Team unfamiliar with graph databases

**My Recommendation**: **GO** - The dual-write pattern is low-risk (PostgreSQL remains source of truth), and the demo/analytics value is high for a property management system with rich relationships.

---

**Document Status**: READ-ONLY EVALUATION
**Next Steps**: Review with team → Approve phases → Begin Phase 1

