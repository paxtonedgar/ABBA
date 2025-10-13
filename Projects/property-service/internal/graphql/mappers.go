package graphql

import (
	"strings"
	"time"

	// ent models
	appent "github.com/evolve-interviews/paxtonedgar-case-study/ent"
)

// timePtr converts a time.Time to *string in RFC3339 format
// Returns nil if the time is zero (unset)
func timePtr(t time.Time) *string {
	if t.IsZero() {
		return nil
	}
	s := t.Format(time.RFC3339)
	return &s
}

// toGraphQLEnum converts an ent enum string to uppercase GraphQL enum format.
// This eliminates duplicate strings.ToUpper(string(enum)) patterns.
func toGraphQLEnum[T ~string](entEnum string) T {
	return T(strings.ToUpper(entEnum))
}

// =====================================================================================
// Type conversion helpers (ent models → GraphQL types)
// =====================================================================================

// ConvertUserToGraphQL converts an ent User to a GraphQL User.
func ConvertUserToGraphQL(entUser *appent.User) *User {
	phone := entUser.Phone
	return &User{
		ID:        entUser.ID.String(),
		Role:      toGraphQLEnum[UserRole](string(entUser.Role)),
		Name:      entUser.Name,
		Email:     entUser.Email,
		Phone:     &phone,
		CreatedAt: entUser.CreatedAt.Format(time.RFC3339),
		UpdatedAt: entUser.UpdatedAt.Format(time.RFC3339),
	}
}

// ConvertPropertyToGraphQL converts an ent Property to a GraphQL Property.
func ConvertPropertyToGraphQL(entProperty *appent.Property) *Property {
	return &Property{
		ID:               entProperty.ID.String(),
		Title:            entProperty.Title,
		Type:             toGraphQLEnum[PropertyType](string(entProperty.Type)),
		DescriptionShort: &entProperty.DescriptionShort,
		DescriptionLong:  &entProperty.DescriptionLong,
		MaxGuests:        &entProperty.MaxGuests,
		BathroomsTotal:   &entProperty.BathroomsTotal,
		AddressLine1:     entProperty.AddressLine1,
		AddressLine2:     &entProperty.AddressLine2,
		City:             entProperty.City,
		Region:           entProperty.Region,
		PostalCode:       entProperty.PostalCode,
		Country:          entProperty.Country,
		Latitude:         &entProperty.Latitude,
		Longitude:        &entProperty.Longitude,
		CheckinTime:      &entProperty.CheckinTime,
		CheckoutTime:     &entProperty.CheckoutTime,
		MinNights:        &entProperty.MinNights,
		MaxNights:        &entProperty.MaxNights,
		PetsAllowed:      entProperty.PetsAllowed,
		SmokingAllowed:   entProperty.SmokingAllowed,
		HouseRules:       &entProperty.HouseRules,
		Status:           toGraphQLEnum[PropertyStatus](string(entProperty.Status)),
		SubmittedAt:      timePtr(entProperty.SubmittedAt),
		ReviewedAt:       timePtr(entProperty.ReviewedAt),
		ReviewNotes:      &entProperty.ReviewNotes,
		CreatedAt:        entProperty.CreatedAt.Format(time.RFC3339),
		UpdatedAt:        entProperty.UpdatedAt.Format(time.RFC3339),
	}
}

// ConvertRoomToGraphQL converts an ent Room to a GraphQL Room.
func ConvertRoomToGraphQL(entRoom *appent.Room) *Room {
	// Convert bed_mix map to RoomBed slice
	beds := make([]*RoomBed, 0, len(entRoom.BedMix))
	for bedType, count := range entRoom.BedMix {
		beds = append(beds, &RoomBed{
			Type:  BedType(bedType),
			Count: count,
		})
	}

	return &Room{
		ID:          entRoom.ID.String(),
		Name:        entRoom.Name,
		FloorLabel:  &entRoom.FloorLabel,
		AreaSqFt:    &entRoom.AreaSqFt,
		EnsuiteBath: &entRoom.EnsuiteBath,
		Beds:        beds,
		Notes:       &entRoom.Notes,
		CreatedAt:   entRoom.CreatedAt.Format(time.RFC3339),
		UpdatedAt:   entRoom.UpdatedAt.Format(time.RFC3339),
	}
}

// ConvertAmenityToGraphQL converts an ent Amenity to a GraphQL Amenity.
func ConvertAmenityToGraphQL(entAmenity *appent.Amenity) *Amenity {
	return &Amenity{
		ID:        entAmenity.ID.String(),
		Code:      entAmenity.Code,
		Name:      entAmenity.Name,
		Category:  entAmenity.Category,
		CreatedAt: entAmenity.CreatedAt.Format(time.RFC3339),
	}
}
