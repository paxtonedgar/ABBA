#!/bin/bash
# Database Setup Script - From Scratch
# Run this for a brand new start (or to reset everything)

set -e

DB_URL="postgresql://postgres:password@localhost:5432/property_service"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Database Setup - From Scratch                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 1. Start PostgreSQL
echo "📦 Step 1: Starting PostgreSQL (docker compose)..."
docker compose up -d postgres
sleep 3
echo "✅ PostgreSQL running"
echo ""

# 2. Drop and recreate database (clean slate)
echo "🗑️  Step 2: Recreating database..."
psql postgresql://postgres:password@localhost:5432/postgres -c "DROP DATABASE IF EXISTS property_service;" 2>/dev/null || true
psql postgresql://postgres:password@localhost:5432/postgres -c "CREATE DATABASE property_service;"
echo "✅ Database created"
echo ""

# 3. Ent will auto-create tables when server starts
echo "⚙️  Step 3: Schema will be created by Ent auto-migration"
echo "   (happens automatically when server starts)"
echo ""

# 4. Seed essential data
echo "🌱 Step 4: Seeding essential data..."
psql $DB_URL <<'SQL'
-- Insert test users
INSERT INTO users (id, name, email, role, created_at, updated_at) VALUES
('550e8400-e29b-41d4-a716-446655440000', 'Test Owner', 'owner@example.com', 'owner', NOW(), NOW()),
('550e8400-e29b-41d4-a716-446655440001', 'Test Agent', 'agent@example.com', 'agent', NOW(), NOW())
ON CONFLICT DO NOTHING;

-- Insert common amenities
INSERT INTO amenities (code, name, category, created_at) VALUES
('wifi', 'WiFi', 'connectivity', NOW()),
('parking', 'Parking', 'property', NOW()),
('hot_tub', 'Hot Tub', 'leisure', NOW()),
('fireplace', 'Fireplace', 'comfort', NOW()),
('pool', 'Pool', 'leisure', NOW()),
('kitchen', 'Full Kitchen', 'property', NOW()),
('washer', 'Washer', 'property', NOW()),
('dryer', 'Dryer', 'property', NOW()),
('ac', 'Air Conditioning', 'comfort', NOW()),
('heating', 'Heating', 'comfort', NOW())
ON CONFLICT (code) DO NOTHING;

\echo ''
\echo '=== Seed Data Summary ==='
SELECT 'Users' as table_name, count(*) as count FROM users
UNION ALL SELECT 'Amenities', count(*) FROM amenities;
SQL

echo ""
echo "✅ Seed data loaded"
echo ""

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Setup Complete!                                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Start the server:  go build -o server ./cmd/server && ./server"
echo "  2. Open playground:   http://localhost:8080/"
echo "  3. Run tests:         npx playwright test"
echo ""

