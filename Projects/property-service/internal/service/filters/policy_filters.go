package filters

import (
	"context"

	"github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/property"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/graphql"
)

// PolicyFilter filters properties by pet and smoking policies.
type PolicyFilter struct{}

func (f *PolicyFilter) Apply(_ context.Context, q *ent.PropertyQuery, filter *graphql.PropertyFilter) bool {
	applied := false

	if filter.PetsAllowed != nil {
		q.Where(property.PetsAllowedEQ(*filter.PetsAllowed))
		applied = true
	}

	if filter.SmokingAllowed != nil {
		q.Where(property.SmokingAllowedEQ(*filter.SmokingAllowed))
		applied = true
	}

	return applied
}

// CapacityFilter filters properties by guest capacity and bathrooms.
type CapacityFilter struct{}

func (f *CapacityFilter) Apply(_ context.Context, q *ent.PropertyQuery, filter *graphql.PropertyFilter) bool {
	applied := false

	if filter.MinGuestsGte != nil {
		q.Where(property.MaxGuestsGTE(*filter.MinGuestsGte))
		applied = true
	}

	if filter.MinBathroomsGte != nil {
		q.Where(property.BathroomsTotalGTE(*filter.MinBathroomsGte))
		applied = true
	}

	return applied
}
