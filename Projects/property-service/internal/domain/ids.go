// Package domain contains pure domain types with no external dependencies.
// This package is at the bottom of the dependency hierarchy and can be imported
// by any other layer (service, graphql, etc.).
package domain

import (
	"fmt"

	"github.com/google/uuid"
)

// =====================================================================================
// Domain ID Value Objects
// =====================================================================================
// These types provide type safety and centralized validation for entity IDs.

// PropertyID represents a validated property identifier.
type PropertyID struct {
	UUID uuid.UUID
}

// ParsePropertyID validates and creates a PropertyID from a string.
func ParsePropertyID(s string) (PropertyID, error) {
	id, err := uuid.Parse(s)
	if err != nil {
		return PropertyID{}, fmt.Errorf("invalid property id: %w", err)
	}
	return PropertyID{UUID: id}, nil
}

// MustParsePropertyID parses a PropertyID and panics on error (use in tests only).
func MustParsePropertyID(s string) PropertyID {
	id, err := ParsePropertyID(s)
	if err != nil {
		panic(err)
	}
	return id
}

// String returns the string representation of the PropertyID.
func (id PropertyID) String() string {
	return id.UUID.String()
}

// IsZero returns true if the PropertyID is the zero value.
func (id PropertyID) IsZero() bool {
	return id.UUID == uuid.Nil
}

// UserID represents a validated user identifier.
type UserID struct {
	UUID uuid.UUID
}

// ParseUserID validates and creates a UserID from a string.
func ParseUserID(s string) (UserID, error) {
	id, err := uuid.Parse(s)
	if err != nil {
		return UserID{}, fmt.Errorf("invalid user id: %w", err)
	}
	return UserID{UUID: id}, nil
}

// String returns the string representation of the UserID.
func (id UserID) String() string {
	return id.UUID.String()
}

// IsZero returns true if the UserID is the zero value.
func (id UserID) IsZero() bool {
	return id.UUID == uuid.Nil
}

// RoomID represents a validated room identifier.
type RoomID struct {
	UUID uuid.UUID
}

// ParseRoomID validates and creates a RoomID from a string.
func ParseRoomID(s string) (RoomID, error) {
	id, err := uuid.Parse(s)
	if err != nil {
		return RoomID{}, fmt.Errorf("invalid room id: %w", err)
	}
	return RoomID{UUID: id}, nil
}

// String returns the string representation of the RoomID.
func (id RoomID) String() string {
	return id.UUID.String()
}

// IsZero returns true if the RoomID is the zero value.
func (id RoomID) IsZero() bool {
	return id.UUID == uuid.Nil
}

// AmenityID represents a validated amenity identifier.
type AmenityID struct {
	UUID uuid.UUID
}

// ParseAmenityID validates and creates an AmenityID from a string.
func ParseAmenityID(s string) (AmenityID, error) {
	id, err := uuid.Parse(s)
	if err != nil {
		return AmenityID{}, fmt.Errorf("invalid amenity id: %w", err)
	}
	return AmenityID{UUID: id}, nil
}

// String returns the string representation of the AmenityID.
func (id AmenityID) String() string {
	return id.UUID.String()
}

// IsZero returns true if the AmenityID is the zero value.
func (id AmenityID) IsZero() bool {
	return id.UUID == uuid.Nil
}

// =====================================================================================
// Batch Validation Helpers
// =====================================================================================

// ParsePropertyIDs validates a slice of property ID strings and returns typed IDs.
// Returns an error if any ID is invalid.
func ParsePropertyIDs(ids []string) ([]PropertyID, error) {
	result := make([]PropertyID, 0, len(ids))
	for _, id := range ids {
		parsed, err := ParsePropertyID(id)
		if err != nil {
			return nil, err
		}
		result = append(result, parsed)
	}
	return result, nil
}

// PropertyIDsToUUIDs converts a slice of PropertyIDs to uuid.UUIDs.
func PropertyIDsToUUIDs(ids []PropertyID) []uuid.UUID {
	result := make([]uuid.UUID, len(ids))
	for i, id := range ids {
		result[i] = id.UUID
	}
	return result
}
