#!/bin/bash
# Seed Test Data - Contrasting Properties for Filter Testing

set -e

DB_URL="postgresql://postgres:password@localhost:5432/property_service"

echo "🌱 Seeding test properties for filter validation..."

psql $DB_URL <<'SQL'
-- Insert contrasting properties for filter testing
INSERT INTO properties (id, title, type, description_short, description_long, max_guests, bathrooms_total, 
  address_line1, city, region, postal_code, country, pets_allowed, smoking_allowed, status, owner_id, created_at, updated_at) VALUES
('a1000000-0000-0000-0000-000000000001', 'Mountain Cabin Retreat', 'cabin', 'Cozy mountain getaway', 'Beautiful cabin with stunning mountain views and modern amenities', 6, 2.0,
  '100 Pine St', 'Evergreen', 'CO', '80439', 'US', true, false, 'approved', '550e8400-e29b-41d4-a716-446655440000', NOW(), NOW()),
('a2000000-0000-0000-0000-000000000002', 'Miami Beach Condo', 'condo', 'Beachfront luxury', 'Modern condo steps from the ocean with parking', 2, 1.0,
  '200 Ocean Dr', 'Miami', 'FL', '33139', 'US', false, false, 'submitted', '550e8400-e29b-41d4-a716-446655440000', NOW(), NOW()),
('a3000000-0000-0000-0000-000000000003', 'Paris Apartment', 'apartment', 'Chic city living', 'Elegant apartment in the heart of Paris with wifi', 4, 1.5,
  '30 Rue de la Paix', 'Paris', 'IDF', '75001', 'FR', false, true, 'approved', '550e8400-e29b-41d4-a716-446655440000', NOW(), NOW()),
('a4000000-0000-0000-0000-000000000004', 'Colorado Ski Lodge', 'cabin', 'Ski-in ski-out', 'Luxury lodge with hot tub and fireplace near slopes', 8, 3.0,
  '500 Ski Run', 'Breckenridge', 'CO', '80424', 'US', true, false, 'approved', '550e8400-e29b-41d4-a716-446655440000', NOW(), NOW())
ON CONFLICT DO NOTHING;

-- Link amenities to properties
INSERT INTO property_amenities (property_id, amenity_id, details) 
SELECT 'a1000000-0000-0000-0000-000000000001'::uuid, id, NULL FROM amenities WHERE code='wifi'
UNION ALL
SELECT 'a1000000-0000-0000-0000-000000000001'::uuid, id, '{"private": true}'::jsonb FROM amenities WHERE code='hot_tub'
UNION ALL
SELECT 'a2000000-0000-0000-0000-000000000002'::uuid, id, NULL FROM amenities WHERE code='parking'
UNION ALL
SELECT 'a3000000-0000-0000-0000-000000000003'::uuid, id, NULL FROM amenities WHERE code='wifi'
UNION ALL
SELECT 'a4000000-0000-0000-0000-000000000004'::uuid, id, NULL FROM amenities WHERE code='wifi'
UNION ALL
SELECT 'a4000000-0000-0000-0000-000000000004'::uuid, id, '{"private": true}'::jsonb FROM amenities WHERE code='hot_tub'
UNION ALL
SELECT 'a4000000-0000-0000-0000-000000000004'::uuid, id, NULL FROM amenities WHERE code='fireplace'
ON CONFLICT DO NOTHING;

-- Add rooms
INSERT INTO rooms (id, property_id, name, bed_mix, created_at, updated_at) VALUES
(gen_random_uuid(), 'a1000000-0000-0000-0000-000000000001'::uuid, 'Master Bedroom', '[{"type":"QUEEN","count":1}]'::jsonb, NOW(), NOW()),
(gen_random_uuid(), 'a1000000-0000-0000-0000-000000000001'::uuid, 'Loft', '[{"type":"TWIN","count":2}]'::jsonb, NOW(), NOW()),
(gen_random_uuid(), 'a2000000-0000-0000-0000-000000000002'::uuid, 'Bedroom', '[{"type":"KING","count":1}]'::jsonb, NOW(), NOW()),
(gen_random_uuid(), 'a4000000-0000-0000-0000-000000000004'::uuid, 'Master Suite', '[{"type":"KING","count":1}]'::jsonb, NOW(), NOW()),
(gen_random_uuid(), 'a4000000-0000-0000-0000-000000000004'::uuid, 'Guest Room 1', '[{"type":"QUEEN","count":1}]'::jsonb, NOW(), NOW()),
(gen_random_uuid(), 'a4000000-0000-0000-0000-000000000004'::uuid, 'Guest Room 2', '[{"type":"TWIN","count":2}]'::jsonb, NOW(), NOW())
ON CONFLICT DO NOTHING;

\echo ''
\echo '=== Test Data Summary ==='
SELECT 'Properties' as table_name, count(*) as count FROM properties
UNION ALL SELECT 'Rooms', count(*) FROM rooms
UNION ALL SELECT 'Property-Amenities', count(*) FROM property_amenities;
SQL

echo "✅ Test data seeded"

