package graphql

import (
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/domain"
	"context"
	"fmt"
	"strings"

	appent "github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/amenity"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/property"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/propertyamenity"
)

// =====================================================================================
// Property Filter & Sort Helpers
// =====================================================================================

// applyPropertyFilter applies all filter predicates to a property query.
// Returns an error if any ID is invalid (explicit error handling, no silent failures).
func applyPropertyFilter(_ context.Context, q *appent.PropertyQuery, f *PropertyFilter) error {
	if f == nil {
		return nil
	}

	// Property filters using entgql-generated predicates with explicit validation
	if len(f.Ids) > 0 {
		propertyIDs, err := domain.ParsePropertyIDs(f.Ids)
		if err != nil {
			return err // Explicit error - no silent failure
		}
		ids := domain.PropertyIDsToUUIDs(propertyIDs)
		q.Where(property.IDIn(ids...))
	}
	if f.OwnerID != nil {
		ownerID, err := domain.ParseUserID(*f.OwnerID)
		if err != nil {
			return fmt.Errorf("invalid owner id: %w", err) // Explicit error
		}
		q.Where(property.OwnerID(ownerID.UUID))
	}
	if len(f.StatusIn) > 0 {
		var sts []property.Status
		for _, s := range f.StatusIn {
			sts = append(sts, property.Status(strings.ToLower(string(s))))
		}
		q.Where(property.StatusIn(sts...))
	}
	if len(f.TypeIn) > 0 {
		var ts []property.Type
		for _, t := range f.TypeIn {
			ts = append(ts, property.Type(strings.ToLower(string(t))))
		}
		q.Where(property.TypeIn(ts...))
	}
	if f.CityEq != nil {
		q.Where(property.CityEqualFold(*f.CityEq))
	}
	if f.RegionEq != nil {
		q.Where(property.RegionEqualFold(*f.RegionEq))
	}
	if f.CountryEq != nil {
		q.Where(property.CountryEqualFold(*f.CountryEq))
	}
	if f.PetsAllowed != nil {
		q.Where(property.PetsAllowedEQ(*f.PetsAllowed))
	}
	if f.SmokingAllowed != nil {
		q.Where(property.SmokingAllowedEQ(*f.SmokingAllowed))
	}
	if f.MinGuestsGte != nil {
		q.Where(property.MaxGuestsGTE(*f.MinGuestsGte))
	}
	if f.MinBathroomsGte != nil {
		q.Where(property.BathroomsTotalGTE(*f.MinBathroomsGte))
	}

	// Geo bounding box filters (manual lat/long math)
	if f.LatMin != nil && f.LatMax != nil {
		q.Where(property.LatitudeGTE(*f.LatMin), property.LatitudeLTE(*f.LatMax))
	}
	if f.LonMin != nil && f.LonMax != nil {
		q.Where(property.LongitudeGTE(*f.LonMin), property.LongitudeLTE(*f.LonMax))
	}
	// Full-text search (multi-field LIKE)
	if f.FullText != nil && *f.FullText != "" {
		q.Where(
			property.Or(
				property.TitleContainsFold(*f.FullText),
				property.DescriptionShortContainsFold(*f.FullText),
				property.DescriptionLongContainsFold(*f.FullText),
				property.CityContainsFold(*f.FullText),
				property.RegionContainsFold(*f.FullText),
			),
		)
	}

	// Amenity filters (using property_amenities join table edge)
	if len(f.AmenitiesAnyOf) > 0 {
		// Properties that have at least one of the specified amenities (OR logic)
		q.Where(property.HasPropertyAmenitiesWith(
			propertyamenity.HasAmenityWith(
				amenity.CodeIn(f.AmenitiesAnyOf...),
			),
		))
	}
	if len(f.AmenitiesAllOf) > 0 {
		// Properties that have ALL of the specified amenities (AND logic)
		// Apply multiple HasPropertyAmenitiesWith predicates
		for _, code := range f.AmenitiesAllOf {
			q.Where(property.HasPropertyAmenitiesWith(
				propertyamenity.HasAmenityWith(
					amenity.CodeEQ(code),
				),
			))
		}
	}

	return nil
}

// applyPropertySort applies sort order to a property query.
func applyPropertySort(q *appent.PropertyQuery, s *PropertySortInput) {
	if s == nil {
		q.Order(appent.Desc(property.FieldCreatedAt))
		return
	}

	// Property sorting using entgql OrderField annotations
	fieldMap := map[PropertySortBy]string{
		PropertySortBySubmittedAt: property.FieldSubmittedAt,
		PropertySortByUpdatedAt:   property.FieldUpdatedAt,
		PropertySortByTitle:       property.FieldTitle,
		PropertySortByCity:        property.FieldCity,
		PropertySortByRegion:      property.FieldRegion,
		PropertySortByMaxGuests:   property.FieldMaxGuests,
	}

	field := fieldMap[s.By]
	if field == "" {
		field = property.FieldCreatedAt // default
	}

	if strings.EqualFold(string(s.Direction), "ASC") {
		if s.By == PropertySortBySubmittedAt {
			q.Order(appent.Asc(field), appent.Asc(property.FieldID)) // stable sort
		} else {
			q.Order(appent.Asc(field))
		}
	} else {
		if s.By == PropertySortBySubmittedAt {
			q.Order(appent.Desc(field), appent.Desc(property.FieldID)) // stable sort
		} else {
			q.Order(appent.Desc(field))
		}
	}
}
