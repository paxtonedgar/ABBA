package graphql

import (
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/domain"
	"context"
	"fmt"

	"github.com/evolve-interviews/paxtonedgar-case-study/internal/service"
)

// =====================================================================================
// Mutation Resolvers
// =====================================================================================
// These resolvers are thin adapters that delegate to the service layer.

// CreateProperty creates a new property (owner-only).
func (resolver *mutationResolver) CreateProperty(ctx context.Context, input CreatePropertyInput) (*Property, error) {
	// Convert GraphQL input to service input
	serviceInput := service.CreatePropertyInput{
		Title:            input.Title,
		Type:             string(*input.Type),
		DescriptionShort: input.DescriptionShort,
		DescriptionLong:  input.DescriptionLong,
		MaxGuests:        input.MaxGuests,
		BathroomsTotal:   input.BathroomsTotal,
		AddressLine1:     input.AddressLine1,
		AddressLine2:     input.AddressLine2,
		City:             input.City,
		Region:           input.Region,
		PostalCode:       input.PostalCode,
		Country:          input.Country,
		Latitude:         input.Latitude,
		Longitude:        input.Longitude,
		CheckinTime:      input.CheckinTime,
		CheckoutTime:     input.CheckoutTime,
		MinNights:        input.MinNights,
		MaxNights:        input.MaxNights,
		PetsAllowed:      boolOr(input.PetsAllowed, false),
		SmokingAllowed:   boolOr(input.SmokingAllowed, false),
		HouseRules:       input.HouseRules,
	}

	// Delegate to service layer
	prop, err := resolver.propertyService.CreateProperty(ctx, serviceInput)
	if err != nil {
		return nil, err
	}

	return ConvertPropertyToGraphQL(prop), nil
}

// UpsertRooms creates or updates rooms by optional ID, all within a transaction.
func (resolver *mutationResolver) UpsertRooms(ctx context.Context, input UpsertRoomsInput) (*Property, error) {
	propertyID, err := domain.ParsePropertyID(input.PropertyID)
	if err != nil {
		return nil, err
	}

	// Convert GraphQL input to service input
	rooms := make([]service.RoomInput, len(input.Rooms))
	for i, ru := range input.Rooms {
		var roomID *domain.RoomID
		if ru.ID != nil && *ru.ID != "" {
			id, err := domain.ParseRoomID(*ru.ID)
			if err != nil {
				return nil, err
			}
			roomID = &id
		}

		rooms[i] = service.RoomInput{
			ID:          roomID,
			Name:        ru.Name,
			FloorLabel:  ru.FloorLabel,
			AreaSqFt:    ru.AreaSqFt,
			EnsuiteBath: ru.EnsuiteBath,
			BedMix:      roomBedsToJSONB(ru.Beds),
			Notes:       ru.Notes,
		}
	}

	// Delegate to service layer
	prop, err := resolver.roomService.UpsertRooms(ctx, propertyID, rooms)
	if err != nil {
		return nil, err
	}

	return ConvertPropertyToGraphQL(prop), nil
}

// SetAmenities attaches amenity codes (upsert edge); removes others not in the list if specified.
func (resolver *mutationResolver) SetAmenities(ctx context.Context, input SetAmenitiesInput) (*Property, error) {
	propertyID, err := domain.ParsePropertyID(input.PropertyID)
	if err != nil {
		return nil, err
	}

	// Convert GraphQL input to service input
	amenities := make([]service.AmenityInput, len(input.Amenities))
	for i, a := range input.Amenities {
		amenities[i] = service.AmenityInput{
			Code:    a.Code,
			Details: a.Details,
		}
	}

	// Delegate to service layer
	prop, err := resolver.amenityService.SetAmenities(ctx, propertyID, amenities)
	if err != nil {
		return nil, err
	}

	return ConvertPropertyToGraphQL(prop), nil
}

// SubmitProperty transitions a property from draft to submitted status (owner action).
func (resolver *mutationResolver) SubmitProperty(ctx context.Context, input SubmitPropertyInput) (*Property, error) {
	propertyID, err := domain.ParsePropertyID(input.PropertyID)
	if err != nil {
		return nil, err
	}

	// Delegate to service layer
	prop, err := resolver.propertyService.SubmitProperty(ctx, propertyID)
	if err != nil {
		return nil, err
	}

	return ConvertPropertyToGraphQL(prop), nil
}

// SaveDraft is a stub implementation - drafts not implemented.
func (resolver *mutationResolver) SaveDraft(_ context.Context, _ SaveDraftInput) (*PropertyDraft, error) {
	return nil, fmt.Errorf("drafts not enabled in this build")
}

// ReviewProperty handles agent review decision (approve/reject).
func (resolver *mutationResolver) ReviewProperty(ctx context.Context, input ReviewPropertyInput) (*Property, error) {
	propertyID, err := domain.ParsePropertyID(input.PropertyID)
	if err != nil {
		return nil, err
	}

	// Convert GraphQL input to service input
	serviceInput := service.ReviewPropertyInput{
		PropertyID: propertyID,
		Decision:   string(input.Decision),
		Notes:      input.Notes,
	}

	// Delegate to service layer
	prop, err := resolver.propertyService.ReviewProperty(ctx, serviceInput)
	if err != nil {
		return nil, err
	}

	return ConvertPropertyToGraphQL(prop), nil
}

// =====================================================================================
// Helper functions
// =====================================================================================

// boolOr returns the value of b if not nil, otherwise returns fallback.
func boolOr(b *bool, fallback bool) bool {
	if b == nil {
		return fallback
	}
	return *b
}

// roomBedsToJSONB converts a slice of RoomBedInput to a JSONB-compatible map.
func roomBedsToJSONB(beds []*RoomBedInput) map[string]int {
	// store as {"queen":1,"sofa_bed_full":1} etc.
	m := map[string]int{}
	for _, b := range beds {
		if b != nil {
			m[string(b.Type)] += b.Count
		}
	}
	return m
}
