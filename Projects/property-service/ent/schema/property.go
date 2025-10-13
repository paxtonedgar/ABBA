package schema

import (
	"time"

	"entgo.io/contrib/entgql"
	"entgo.io/ent"
	"entgo.io/ent/schema"
	"entgo.io/ent/schema/edge"
	"entgo.io/ent/schema/field"
	"entgo.io/ent/schema/index"
	"github.com/google/uuid"
)

// Property holds the schema definition for the Property entity.
type Property struct {
	ent.Schema
}

// Annotations of the Property.
func (Property) Annotations() []schema.Annotation {
	return []schema.Annotation{
		entgql.QueryField(),
		entgql.Mutations(entgql.MutationCreate(), entgql.MutationUpdate()),
	}
}

// Fields of the Property.
func (Property) Fields() []ent.Field {
	return []ent.Field{
		field.UUID("id", uuid.UUID{}).
			Default(uuid.New).
			StorageKey("id").
			Annotations(entgql.Type("ID")),
		field.UUID("owner_id", uuid.UUID{}).
			Comment("ID of the property owner"),
		field.String("title").
			NotEmpty().
			Comment("Property title").
			Annotations(entgql.OrderField("TITLE")),
		field.Enum("type").
			Values("house", "apartment", "condo", "cabin", "villa", "townhouse", "other").
			Default("other").
			Comment("Type of property"),
		field.Text("description_short").
			Optional().
			Comment("Short description of the property"),
		field.Text("description_long").
			Optional().
			Comment("Long description of the property"),
		field.Int("max_guests").
			Positive().
			Optional().
			Comment("Maximum number of guests").
			Annotations(entgql.OrderField("MAX_GUESTS")),
		field.Float("bathrooms_total").
			Min(0).
			Optional().
			Comment("Total number of bathrooms"),
		// Address fields
		field.String("address_line1").
			NotEmpty().
			Comment("Primary address line"),
		field.String("address_line2").
			Optional().
			Comment("Secondary address line"),
		field.String("city").
			NotEmpty().
			Comment("City"),
		field.String("region").
			NotEmpty().
			Comment("State/province/region"),
		field.String("postal_code").
			NotEmpty().
			Comment("Postal/ZIP code"),
		field.String("country").
			NotEmpty().
			Comment("Country"),
		// Geo coordinates
		field.Float("latitude").
			Optional().
			Comment("Latitude coordinate"),
		field.Float("longitude").
			Optional().
			Comment("Longitude coordinate"),
		// Policy fields
		field.Time("checkin_time").
			Optional().
			Comment("Check-in time"),
		field.Time("checkout_time").
			Optional().
			Comment("Check-out time"),
		field.Int("min_nights").
			Min(1).
			Optional().
			Comment("Minimum number of nights"),
		field.Int("max_nights").
			Min(1).
			Optional().
			Comment("Maximum number of nights"),
		field.Bool("pets_allowed").
			Default(false).
			Comment("Whether pets are allowed"),
		field.Bool("smoking_allowed").
			Default(false).
			Comment("Whether smoking is allowed"),
		field.Text("house_rules").
			Optional().
			Comment("House rules"),
		// Lifecycle fields
		field.Enum("status").
			Values("draft", "submitted", "under_review", "approved", "rejected").
			Default("draft").
			Comment("Property status"),
		field.Time("submitted_at").
			Optional().
			Comment("When the property was submitted for review"),
		field.UUID("reviewed_by", uuid.UUID{}).
			Optional().
			Comment("ID of the user who reviewed the property"),
		field.Time("reviewed_at").
			Optional().
			Comment("When the property was reviewed"),
		field.Text("review_notes").
			Optional().
			Comment("Notes from the review"),
		field.Time("created_at").
			Default(time.Now).
			Comment("When the property was created").
			Annotations(entgql.OrderField("CREATED_AT")),
		field.Time("updated_at").
			Default(time.Now).
			UpdateDefault(time.Now).
			Comment("When the property was last updated").
			Annotations(entgql.OrderField("UPDATED_AT")),
	}
}

// Edges of the Property.
func (Property) Edges() []ent.Edge {
	return []ent.Edge{
		edge.From("owner", User.Type).
			Ref("properties").
			Field("owner_id").
			Required().
			Unique().
			Comment("Property owner"),
		edge.From("reviewer", User.Type).
			Ref("reviewed_properties").
			Field("reviewed_by").
			Comment("User who reviewed this property"),
		edge.To("rooms", Room.Type).
			Comment("Rooms in this property"),
		edge.To("amenities", Amenity.Type).
			Through("property_amenities", PropertyAmenity.Type).
			Comment("Amenities available at this property"),
	}
}

// Indexes of the Property.
func (Property) Indexes() []ent.Index {
	return []ent.Index{
		index.Fields("owner_id", "status"),
		index.Fields("reviewed_by"),
		index.Fields("status", "reviewed_by"),
	}
}
