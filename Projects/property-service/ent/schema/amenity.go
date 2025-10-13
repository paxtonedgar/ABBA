// Package schema defines the ent entity schemas for the property service domain model.
package schema

import (
	"time"

	"entgo.io/ent"
	"entgo.io/ent/schema"
	"entgo.io/ent/schema/edge"
	"entgo.io/ent/schema/field"
	"entgo.io/ent/schema/index"
	"github.com/google/uuid"
)

// Amenity holds the schema definition for the Amenity entity.
type Amenity struct {
	ent.Schema
}

// Annotations of the Amenity.
func (Amenity) Annotations() []schema.Annotation {
	return nil
}

// Fields of the Amenity.
func (Amenity) Fields() []ent.Field {
	return []ent.Field{
		field.UUID("id", uuid.UUID{}).
			Default(uuid.New).
			StorageKey("id"),
		field.String("code").
			Unique().
			NotEmpty().
			Comment("Unique amenity code (e.g., 'PARKING', 'HOT_TUB', 'WIFI')"),
		field.String("name").
			NotEmpty().
			Comment("Human-readable amenity name (e.g., 'Parking', 'Hot Tub', 'Wi-Fi')"),
		field.String("category").
			NotEmpty().
			Comment("Amenity category (e.g., 'Parking', 'Wellness', 'Connectivity')"),
		field.Time("created_at").
			Default(time.Now).
			Comment("When the amenity was created"),
	}
}

// Edges of the Amenity.
func (Amenity) Edges() []ent.Edge {
	return []ent.Edge{
		edge.From("properties", Property.Type).
			Ref("amenities").
			Through("property_amenities", PropertyAmenity.Type).
			Comment("Properties that have this amenity"),
	}
}

// Indexes of the Amenity.
func (Amenity) Indexes() []ent.Index {
	return []ent.Index{
		index.Fields("code").
			Unique(),
	}
}
