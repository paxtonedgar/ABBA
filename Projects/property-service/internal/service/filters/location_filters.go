package filters

import (
	"context"

	"github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/property"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/graphql"
)

// LocationFilter filters properties by city, region, and country.
type LocationFilter struct{}

func (f *LocationFilter) Apply(_ context.Context, q *ent.PropertyQuery, filter *graphql.PropertyFilter) bool {
	applied := false

	if filter.CityEq != nil {
		q.Where(property.CityEqualFold(*filter.CityEq))
		applied = true
	}

	if filter.RegionEq != nil {
		q.Where(property.RegionEqualFold(*filter.RegionEq))
		applied = true
	}

	if filter.CountryEq != nil {
		q.Where(property.CountryEqualFold(*filter.CountryEq))
		applied = true
	}

	return applied
}

// GeoBoundsFilter filters properties by geographic bounding box.
type GeoBoundsFilter struct{}

func (f *GeoBoundsFilter) Apply(_ context.Context, q *ent.PropertyQuery, filter *graphql.PropertyFilter) bool {
	applied := false

	// Latitude bounds
	if filter.LatMin != nil && filter.LatMax != nil {
		q.Where(property.LatitudeGTE(*filter.LatMin), property.LatitudeLTE(*filter.LatMax))
		applied = true
	}

	// Longitude bounds
	if filter.LonMin != nil && filter.LonMax != nil {
		q.Where(property.LongitudeGTE(*filter.LonMin), property.LongitudeLTE(*filter.LonMax))
		applied = true
	}

	return applied
}
