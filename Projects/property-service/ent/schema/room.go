package schema

import (
	"time"

	"entgo.io/ent"
	"entgo.io/ent/schema"
	"entgo.io/ent/schema/edge"
	"entgo.io/ent/schema/field"
	"github.com/google/uuid"
)

// Room holds the schema definition for the Room entity.
type Room struct {
	ent.Schema
}

// Annotations of the Room.
func (Room) Annotations() []schema.Annotation {
	return nil
}

// Fields of the Room.
func (Room) Fields() []ent.Field {
	return []ent.Field{
		field.UUID("id", uuid.UUID{}).
			Default(uuid.New).
			StorageKey("id"),
		field.UUID("property_id", uuid.UUID{}).
			Comment("ID of the property this room belongs to"),
		field.String("name").
			NotEmpty().
			Comment("Room name (e.g., 'Living Room', 'Bedroom 1')"),
		field.String("floor_label").
			Optional().
			Comment("Floor label (e.g., '1st', 'Garden', 'Loft')"),
		field.Int("area_sq_ft").
			Min(0).
			Optional().
			Comment("Room area in square feet"),
		field.JSON("bed_mix", map[string]int{}).
			Default(map[string]int{}).
			Comment("Bed configuration as JSON (e.g., {\"king\":1,\"queen\":1,\"twin\":0})"),
		field.Bool("ensuite_bath").
			Optional().
			Comment("Whether the room has an ensuite bathroom"),
		field.Text("notes").
			Optional().
			Comment("Additional room notes"),
		field.Time("created_at").
			Default(time.Now).
			Comment("When the room was created"),
		field.Time("updated_at").
			Default(time.Now).
			UpdateDefault(time.Now).
			Comment("When the room was last updated"),
	}
}

// Edges of the Room.
func (Room) Edges() []ent.Edge {
	return []ent.Edge{
		edge.From("property", Property.Type).
			Ref("rooms").
			Field("property_id").
			Required().
			Unique().
			Comment("Property this room belongs to"),
	}
}
