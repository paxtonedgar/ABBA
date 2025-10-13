package schema

import (
	"entgo.io/ent"
	"entgo.io/ent/schema/edge"
	"entgo.io/ent/schema/field"
	"entgo.io/ent/schema/index"
	"github.com/google/uuid"
)

// PropertyAmenity holds the schema definition for the PropertyAmenity entity.
// This is a join table for the many-to-many relationship between Property and Amenity.
type PropertyAmenity struct {
	ent.Schema
}

// Fields of the PropertyAmenity.
func (PropertyAmenity) Fields() []ent.Field {
	return []ent.Field{
		field.UUID("property_id", uuid.UUID{}).
			Comment("ID of the property"),
		field.UUID("amenity_id", uuid.UUID{}).
			Comment("ID of the amenity"),
		field.JSON("details", map[string]interface{}{}).
			Optional().
			Comment("Additional amenity details (e.g., {\"spaces\":2} for parking)"),
	}
}

// Edges of the PropertyAmenity.
func (PropertyAmenity) Edges() []ent.Edge {
	return []ent.Edge{
		edge.To("property", Property.Type).
			Field("property_id").
			Required().
			Unique().
			Comment("The property"),
		edge.To("amenity", Amenity.Type).
			Field("amenity_id").
			Required().
			Unique().
			Comment("The amenity"),
	}
}

// Indexes of the PropertyAmenity.
func (PropertyAmenity) Indexes() []ent.Index {
	return []ent.Index{
		index.Fields("property_id"),
		index.Fields("amenity_id"),
		index.Fields("property_id", "amenity_id").
			Unique(),
	}
}
