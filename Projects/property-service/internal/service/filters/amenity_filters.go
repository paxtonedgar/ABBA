package filters

import (
	"context"

	"github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/amenity"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/property"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/propertyamenity"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/graphql"
)

// AmenityFilter filters properties by amenities.
// Supports both ANY (OR) and ALL (AND) logic.
type AmenityFilter struct{}

func (f *AmenityFilter) Apply(_ context.Context, q *ent.PropertyQuery, filter *graphql.PropertyFilter) bool {
	applied := false

	// AmenitiesAnyOf: Properties that have at least ONE of the specified amenities (OR logic)
	if len(filter.AmenitiesAnyOf) > 0 {
		q.Where(property.HasPropertyAmenitiesWith(
			propertyamenity.HasAmenityWith(
				amenity.CodeIn(filter.AmenitiesAnyOf...),
			),
		))
		applied = true
	}

	// AmenitiesAllOf: Properties that have ALL of the specified amenities (AND logic)
	// We apply multiple predicates, each requiring one amenity
	if len(filter.AmenitiesAllOf) > 0 {
		for _, code := range filter.AmenitiesAllOf {
			q.Where(property.HasPropertyAmenitiesWith(
				propertyamenity.HasAmenityWith(
					amenity.CodeEQ(code),
				),
			))
		}
		applied = true
	}

	return applied
}
