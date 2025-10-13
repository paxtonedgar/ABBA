package schema

import (
	"time"

	"entgo.io/ent"
	"entgo.io/ent/schema/edge"
	"entgo.io/ent/schema/field"
	"entgo.io/ent/schema/index"
	"github.com/google/uuid"
)

// User holds the schema definition for the User entity.
type User struct {
	ent.Schema
}

// Fields of the User.
func (User) Fields() []ent.Field {
	return []ent.Field{
		field.UUID("id", uuid.UUID{}).
			Default(uuid.New).
			StorageKey("id"),
		field.Enum("role").
			Values("owner", "agent").
			Comment("User role: owner or agent"),
		field.String("name").
			NotEmpty().
			Comment("User's full name"),
		field.String("email").
			Unique().
			NotEmpty().
			Comment("User's email address"),
		field.String("phone").
			Optional().
			Comment("User's phone number"),
		field.Time("created_at").
			Default(time.Now).
			Comment("When the user was created"),
		field.Time("updated_at").
			Default(time.Now).
			UpdateDefault(time.Now).
			Comment("When the user was last updated"),
	}
}

// Edges of the User.
func (User) Edges() []ent.Edge {
	return []ent.Edge{
		edge.To("properties", Property.Type).
			Comment("Properties owned by this user"),
		edge.To("reviewed_properties", Property.Type).
			Comment("Properties reviewed by this user").
			Unique(),
	}
}

// Indexes of the User.
func (User) Indexes() []ent.Index {
	return []ent.Index{
		index.Fields("email").
			Unique(),
	}
}
