package service

import (
	"context"
	"fmt"

	"github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/domain"
)

// RoomService handles room business logic.
type RoomService struct {
	client      *ent.Client
	authService *AuthService
}

// NewRoomService creates a new room service.
func NewRoomService(client *ent.Client, authService *AuthService) *RoomService {
	return &RoomService{
		client:      client,
		authService: authService,
	}
}

// RoomInput represents input for creating/updating a room.
type RoomInput struct {
	ID          *domain.RoomID
	Name        string
	FloorLabel  *string
	AreaSqFt    *int
	EnsuiteBath *bool
	BedMix      map[string]int
	Notes       *string
}

// UpsertRooms creates or updates rooms for a property.
func (service *RoomService) UpsertRooms(ctx context.Context, propertyID domain.PropertyID, rooms []RoomInput) (*ent.Property, error) {
	// Get property and verify permission
	prop, err := getPropertyWithAuthCheck(ctx, service.client, service.authService, propertyID)
	if err != nil {
		return nil, err
	}

	// Execute transaction
	_, err = withTransaction(ctx, service.client, func(tx *ent.Tx) (interface{}, error) {
		// Upsert each room
		for _, room := range rooms {
			if room.ID != nil && !room.ID.IsZero() {
				// Update existing room
				update := tx.Room.UpdateOneID(room.ID.UUID).
					SetName(room.Name).
					SetNillableFloorLabel(room.FloorLabel).
					SetNillableAreaSqFt(room.AreaSqFt).
					SetNillableEnsuiteBath(room.EnsuiteBath).
					SetNillableNotes(room.Notes)

				if room.BedMix != nil {
					update = update.SetBedMix(room.BedMix)
				}

				if _, err := update.Save(ctx); err != nil {
					return nil, fmt.Errorf("update room %s: %w", room.ID.String(), err)
				}
			} else {
				// Create new room
				create := tx.Room.Create().
					SetPropertyID(propertyID.UUID).
					SetName(room.Name).
					SetNillableFloorLabel(room.FloorLabel).
					SetNillableAreaSqFt(room.AreaSqFt).
					SetNillableEnsuiteBath(room.EnsuiteBath).
					SetNillableNotes(room.Notes).
					SetBedMix(room.BedMix)

				if _, err := create.Save(ctx); err != nil {
					return nil, fmt.Errorf("create room: %w", err)
				}
			}
		}
		return nil, nil
	})
	if err != nil {
		return nil, err
	}

	// Return updated property
	return prop, nil
}
