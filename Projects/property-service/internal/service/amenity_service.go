package service

import (
	"context"
	"encoding/json"
	"fmt"

	"entgo.io/ent/dialect/sql"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/amenity"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/propertyamenity"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/domain"
)

// AmenityService handles amenity business logic.
type AmenityService struct {
	client      *ent.Client
	authService *AuthService
}

// NewAmenityService creates a new amenity service.
func NewAmenityService(client *ent.Client, authService *AuthService) *AmenityService {
	return &AmenityService{
		client:      client,
		authService: authService,
	}
}

// AmenityInput represents input for attaching an amenity to a property.
type AmenityInput struct {
	Code    string
	Details *string // JSON string
}

// SetAmenities replaces all amenities for a property.
func (service *AmenityService) SetAmenities(ctx context.Context, propertyID domain.PropertyID, amenities []AmenityInput) (*ent.Property, error) {
	// Get property and verify permission
	prop, err := getPropertyWithAuthCheck(ctx, service.client, service.authService, propertyID)
	if err != nil {
		return nil, err
	}

	// Resolve amenity codes to IDs
	codes := make([]string, len(amenities))
	for i, a := range amenities {
		codes[i] = a.Code
	}

	ams, err := service.client.Amenity.Query().Where(amenity.CodeIn(codes...)).All(ctx)
	if err != nil {
		return nil, fmt.Errorf("fetch amenities: %w", err)
	}

	codeToID := make(map[string]domain.AmenityID)
	for _, a := range ams {
		codeToID[a.Code] = domain.AmenityID{UUID: a.ID}
	}

	// Execute transaction
	_, err = withTransaction(ctx, service.client, func(tx *ent.Tx) (interface{}, error) {
		// Upsert each amenity
		for _, a := range amenities {
			aid, ok := codeToID[a.Code]
			if !ok {
				// Skip unknown amenity codes
				continue
			}

			// Parse details if provided
			var details map[string]interface{}
			if a.Details != nil {
				if err := json.Unmarshal([]byte(*a.Details), &details); err != nil {
					// Skip invalid JSON
					continue
				}
			}

			// Upsert property amenity
			_, err := tx.PropertyAmenity.
				Create().
				SetPropertyID(propertyID.UUID).
				SetAmenityID(aid.UUID).
				SetDetails(details).
				OnConflict(
					sql.ConflictColumns(propertyamenity.FieldPropertyID, propertyamenity.FieldAmenityID),
				).
				UpdateNewValues().
				ID(ctx)

			if err != nil {
				return nil, fmt.Errorf("upsert amenity %s: %w", a.Code, err)
			}
		}
		return nil, nil
	})
	if err != nil {
		return nil, err
	}

	return prop, nil
}
